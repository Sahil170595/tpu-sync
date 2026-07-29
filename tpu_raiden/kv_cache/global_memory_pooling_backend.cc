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

#include "tpu_raiden/kv_cache/global_memory_pooling_backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

GlobalMemoryPoolingBackend::GlobalMemoryPoolingBackend(
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
    RaidenId raiden_id)
    : registry_client_(std::move(registry_client)),
      raiden_id_(std::move(raiden_id)) {}

absl::StatusOr<BlockSliceList> GlobalMemoryPoolingBackend::Lookup(
    absl::Span<const std::string> block_hashes,
    const LookupOptions& /*options*/) {
  BlockSliceList results;
  if (!registry_client_ || block_hashes.empty()) {
    return results;
  }

  auto global_results_or = registry_client_->Lookup(
      std::vector<std::string>(block_hashes.begin(), block_hashes.end()));
  if (!global_results_or.ok()) {
    LOG(WARNING) << "Global registry lookup failed: "
                 << global_results_or.status().message();
    return global_results_or.status();
  }

  const auto& global_results = global_results_or.value();
  size_t num_results = std::min(global_results.size(), block_hashes.size());
  results.reserve(num_results);

  for (size_t i = 0; i < num_results; ++i) {
    const auto& metadata = global_results[i];
    const auto& proto_id = metadata.raiden_id();
    RaidenId remote_id{
        .job_name = proto_id.job_name(),
        .job_replica_id = proto_id.job_replica_id(),
        .data_name = proto_id.data_name(),
        .data_replica_idx = proto_id.data_replica_idx(),
    };
    results.push_back(std::make_pair(
        block_hashes[i],
        RaidenBlockID(remote_id, metadata.block_id(), BlockStatus::REMOTE)));
  }

  return results;
}

std::pair<bool, BlockSliceList> GlobalMemoryPoolingBackend::Insert(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool /*on_host*/) {
  if (!registry_client_ || block_hashes.empty()) {
    return std::make_pair(true, BlockSliceList{});
  }

  std::vector<global_registry::Registration> registrations;
  registrations.reserve(block_hashes.size());
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (i < slices.size()) {
      registrations.push_back({
          .prefix_hash = block_hashes[i],
          .raiden_id = raiden_id_,
          .block_id = slices[i].host_block_id,
      });
    }
  }

  if (!registrations.empty()) {
    auto status = registry_client_->Register(registrations);
    if (!status.ok()) {
      LOG(WARNING) << "Global registry register failed: " << status.message();
    }
  }

  return std::make_pair(true, BlockSliceList{});
}

bool GlobalMemoryPoolingBackend::InsertAndLock(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool on_host) {
  Insert(block_hashes, slices, on_host);
  return true;
}

size_t GlobalMemoryPoolingBackend::ReleaseAndDelete(
    absl::Span<const std::string> block_hashes) {
  Delete(block_hashes, {});
  return 0;
}

void GlobalMemoryPoolingBackend::Delete(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> /*slices*/) {
  if (registry_client_ && !block_hashes.empty()) {
    auto status = registry_client_->Unregister(
        std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
        raiden_id_);
    if (!status.ok()) {
      LOG(WARNING) << "Global registry unregister failed: " << status.message();
    }
  }
}

bool GlobalMemoryPoolingBackend::Pin(
    absl::Span<const std::string> /*block_hashes*/) {
  return true;
}

void GlobalMemoryPoolingBackend::Release(
    absl::Span<const std::string> /*block_hashes*/) {}

int GlobalMemoryPoolingBackend::GetPinCount(const std::string& /*hash*/) const {
  return 0;
}

REGISTER_KV_CACHE_STORE_BACKEND(
    "GlobalMemoryPoolingBackend",
    [](const ::tpu_raiden::kv_cache::BackendConfig& config)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      if (config.global_registry_address.empty()) {
        return absl::InvalidArgumentError(
            "global_memory_pooling backend requires non-empty "
            "global_registry_address");
      }
      auto channel = grpc::CreateChannel(config.global_registry_address,
                                         grpc::InsecureChannelCredentials());
      auto client = std::make_shared<
          ::tpu_raiden::kv_cache::global_registry::GlobalRegistryClient>(
          channel);
      return std::make_shared<
          ::tpu_raiden::kv_cache::GlobalMemoryPoolingBackend>(client,
                                                              config.raiden_id);
    });

}  // namespace kv_cache
}  // namespace tpu_raiden
