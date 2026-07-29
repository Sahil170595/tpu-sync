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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_GLOBAL_MEMORY_POOLING_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_GLOBAL_MEMORY_POOLING_BACKEND_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

class GlobalMemoryPoolingBackend : public KVCacheStoreBackend {
 public:
  explicit GlobalMemoryPoolingBackend(
      std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
      RaidenId raiden_id = {});

  ~GlobalMemoryPoolingBackend() override = default;

  std::string name() const override { return "GlobalMemoryPoolingBackend"; }

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockID> slices, bool on_host) override;

  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockID> slices,
                     bool on_host) override;

  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override;

  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockID> slices) override;

  bool Pin(absl::Span<const std::string> block_hashes) override;

  void Release(absl::Span<const std::string> block_hashes) override;

  int GetPinCount(const std::string& hash) const override;

  size_t GetCapacity() const override {
    return std::numeric_limits<size_t>::max();
  }

  size_t GetSize() const override { return 0; }

  size_t GetAvailableSpace() const override {
    return std::numeric_limits<size_t>::max();
  }

 private:
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_;
  RaidenId raiden_id_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_GLOBAL_MEMORY_POOLING_BACKEND_H_
