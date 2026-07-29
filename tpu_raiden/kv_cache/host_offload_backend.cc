// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_raiden/kv_cache/host_offload_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

HostOffloadBackend::HostOffloadBackend(
    size_t capacity, std::optional<KVCacheMetadata> metadata,
    RaidenId raiden_id, controller::RaidenController* raiden_controller)
    : lru_cache_(capacity),
      metadata_(std::move(metadata)),
      raiden_id_(std::move(raiden_id)),
      raiden_controller_(raiden_controller) {}

absl::StatusOr<BlockSliceList> HostOffloadBackend::Lookup(
    absl::Span<const std::string> block_hashes,
    const LookupOptions& /*options*/) {
  absl::MutexLock lock(&mutex_);
  BlockSliceList results;
  results.reserve(block_hashes.size());
  for (const auto& hash : block_hashes) {
    const RaidenBlockID* existing = lru_cache_.Peek(hash);
    if (!existing) {
      break;  // First miss
    }
    results.push_back(std::make_pair(hash, *existing));
  }
  return results;
}

std::pair<bool, BlockSliceList> HostOffloadBackend::Insert(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool /*on_host*/) {
  absl::MutexLock lock(&mutex_);
  BlockSliceList evicted_entries;
  bool all_inserted = true;

  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    if (lru_cache_.Contains(hash)) {
      all_inserted = false;
      if (i < slices.size()) {
        if (RaidenBlockID* existing = lru_cache_.PeekMutable(hash)) {
          *existing = slices[i];
          SetMetadataEntry(hash, slices[i]);
        }
      }
      continue;
    }
    if (metadata_.has_value()) {
      if (const RaidenBlockID* stale =
              lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*stale);
      }
    }
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
      SetMetadataEntry(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, RaidenBlockID());
    }
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

bool HostOffloadBackend::InsertAndLock(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool /*on_host*/) {
  absl::MutexLock lock(&mutex_);

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return false;
    }
  }

  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return false;
  }

  size_t eviction_count = 0;
  for (auto it = new_indices.rbegin(); it != new_indices.rend(); ++it) {
    size_t i = *it;
    const std::string& hash = block_hashes[i];
    if (metadata_.has_value()) {
      if (const RaidenBlockID* stale =
              lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*stale);
      }
    }
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
      SetMetadataEntry(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, RaidenBlockID());
    }
    if (evicted.has_value()) {
      eviction_count++;
    }
  }

  for (size_t idx = 0; idx < new_indices.size(); ++idx) {
    size_t i = new_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[new_indices[j]]);
      }
      for (size_t j : existing_indices) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      for (size_t j : new_indices) {
        if (const RaidenBlockID* val =
                lru_cache_.PeekIncludingCandidates(block_hashes[j])) {
          ClearMetadataEntry(*val);
        }
        lru_cache_.Erase(block_hashes[j]);
      }
      for (size_t j = 0; j < eviction_count; ++j) {
        lru_cache_.RestoreLastCandidate();
      }
      return false;
    }
  }

  if (eviction_count > 0) {
    pending_eviction_counts_[GetSortedHashes(block_hashes)] = eviction_count;
  }
  return true;
}

size_t HostOffloadBackend::ReleaseAndDelete(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(&mutex_);
  size_t deleted_blocks = 0;

  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }

  for (const std::string& hash : block_hashes) {
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
        val->status != BlockStatus::HOST &&
        val->status != BlockStatus::HOST_AND_HBM) {
      lru_cache_.Erase(hash);
      deleted_blocks++;
    }
  }

  size_t restoration_count = 0;
  auto it = pending_eviction_counts_.find(GetSortedHashes(block_hashes));
  if (it != pending_eviction_counts_.end()) {
    restoration_count = it->second;
    pending_eviction_counts_.erase(it);
  }

  size_t to_restore = std::min(deleted_blocks, restoration_count);
  for (size_t i = 0; i < to_restore; ++i) {
    lru_cache_.RestoreLastCandidate();
  }

  return deleted_blocks;
}

void HostOffloadBackend::Delete(absl::Span<const std::string> block_hashes,
                                absl::Span<const RaidenBlockID> /*slices*/) {
  absl::MutexLock lock(&mutex_);
  for (const std::string& hash : block_hashes) {
    if (lru_cache_.GetPinCount(hash) > 0) {
      LOG(WARNING) << "Delete skipped pinned block hash (release it first): "
                   << absl::BytesToHexString(hash);
      continue;
    }
    if (const RaidenBlockID* val = lru_cache_.PeekIncludingCandidates(hash)) {
      ClearMetadataEntry(*val);
    }
    lru_cache_.Erase(hash);
  }
}

bool HostOffloadBackend::Pin(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(&mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void HostOffloadBackend::Release(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(&mutex_);
  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }
  pending_eviction_counts_.erase(GetSortedHashes(block_hashes));
}

int HostOffloadBackend::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t HostOffloadBackend::GetCapacity() const {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.capacity();
}

size_t HostOffloadBackend::GetSize() const {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.size();
}

size_t HostOffloadBackend::GetAvailableSpace() const {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.available_space();
}

absl::StatusOr<size_t> HostOffloadBackend::RecoverFromLocalManifest() {
  absl::MutexLock lock(&mutex_);
  if (!metadata_.has_value()) {
    return absl::FailedPreconditionError(
        "KVCacheMetadata is required for crash recovery");
  }
  std::vector<KVCacheMetadata::Entry> entries = metadata_->ValidEntries();
  if (entries.empty()) {
    return 0;
  }

  absl::flat_hash_map<absl::string_view, const KVCacheMetadata::Entry*> newest;
  newest.reserve(entries.size());
  for (const KVCacheMetadata::Entry& entry : entries) {
    auto [it, inserted] = newest.try_emplace(entry.hash, &entry);
    if (!inserted && entry.seq > it->second->seq) {
      it->second = &entry;
    }
  }

  std::vector<const KVCacheMetadata::Entry*> recoverable;
  recoverable.reserve(newest.size());
  for (const auto& [hash, entry] : newest) {
    recoverable.push_back(entry);
  }
  std::sort(recoverable.begin(), recoverable.end(),
            [](const KVCacheMetadata::Entry* a,
               const KVCacheMetadata::Entry* b) { return a->seq < b->seq; });

  if (raiden_controller_ != nullptr) {
    std::vector<int> block_ids;
    block_ids.reserve(recoverable.size());
    for (const KVCacheMetadata::Entry* entry : recoverable) {
      block_ids.push_back(entry->block_id);
    }
    absl::Status allocate_status =
        raiden_controller_->AllocateTargetBlockIds(block_ids);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
  }

  uint64_t max_seq = 0;
  for (const KVCacheMetadata::Entry* entry : recoverable) {
    lru_cache_.Put(entry->hash, RaidenBlockID(raiden_id_, entry->block_id,
                                              BlockStatus::HOST));
    max_seq = std::max(max_seq, entry->seq);
  }
  next_metadata_seq_ = max_seq + 1;

  for (const KVCacheMetadata::Entry& entry : entries) {
    if (newest.at(entry.hash)->block_id != entry.block_id) {
      absl::Status status = metadata_->Clear(entry.block_id);
      if (!status.ok()) {
        LOG(WARNING) << "Failed to clear the stale metadata entry for block "
                     << entry.block_id << ": " << status.message();
      }
    }
  }

  return recoverable.size();
}

bool HostOffloadBackend::ValidateAndPinHostBlocks(
    absl::Span<const int> host_block_ids) {
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> keys_to_pin;
  keys_to_pin.reserve(host_block_ids.size());
  for (int host_id : host_block_ids) {
    bool found = false;
    for (const auto& [key, it] : lru_cache_.map()) {
      if (it->value.host_block_id == host_id &&
          (it->value.status == BlockStatus::HOST ||
           it->value.status == BlockStatus::HOST_AND_HBM)) {
        keys_to_pin.push_back(key);
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  for (size_t i = 0; i < keys_to_pin.size(); ++i) {
    if (!lru_cache_.Pin(keys_to_pin[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(keys_to_pin[j]);
      }
      return false;
    }
  }
  return true;
}

std::vector<std::string> HostOffloadBackend::GetEvictableKeys(size_t count) {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.GetEvictableKeys(count);
}

std::vector<int> HostOffloadBackend::Evict(
    const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(&mutex_);
  std::vector<int> host_ids_to_deallocate;
  for (const std::string& hash : block_hashes) {
    const RaidenBlockID* block = lru_cache_.PeekIncludingCandidates(hash);
    if (block != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
        (block->status == BlockStatus::HOST ||
         block->status == BlockStatus::HOST_AND_HBM)) {
      host_ids_to_deallocate.push_back(block->host_block_id);
      ClearMetadataEntry(*block);
      lru_cache_.Erase(hash);
    }
  }
  return host_ids_to_deallocate;
}

std::vector<std::string> HostOffloadBackend::GetEvictCandidateKeys() const {
  absl::MutexLock lock(&mutex_);
  return lru_cache_.GetEvictCandidateKeys();
}

void HostOffloadBackend::SetRaidenController(
    controller::RaidenController* controller) {
  absl::MutexLock lock(&mutex_);
  raiden_controller_ = controller;
}

void HostOffloadBackend::SetMetadataEntry(absl::string_view hash,
                                          const RaidenBlockID& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status =
      metadata_->Set(block.host_block_id, hash, next_metadata_seq_++);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to set the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

void HostOffloadBackend::ClearMetadataEntry(const RaidenBlockID& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status = metadata_->Clear(block.host_block_id);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to clear the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

std::vector<std::string> HostOffloadBackend::GetSortedHashes(
    absl::Span<const std::string> hashes) const {
  std::vector<std::string> sorted(hashes.begin(), hashes.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

REGISTER_KV_CACHE_STORE_BACKEND(
    "HostOffloadBackend",
    [](const ::tpu_raiden::kv_cache::BackendConfig& config)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      if (config.capacity == 0) {
        return absl::InvalidArgumentError(
            "host_offload backend requires capacity > 0");
      }
      return std::make_shared<::tpu_raiden::kv_cache::HostOffloadBackend>(
          config.capacity, config.metadata, config.raiden_id);
    });

}  // namespace kv_cache
}  // namespace tpu_raiden
