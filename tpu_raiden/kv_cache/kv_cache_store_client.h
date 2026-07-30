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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/proto/kv_cache_store_service.grpc.pb.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStoreClient {
 public:
  explicit KVCacheStoreClient(
      std::shared_ptr<::grpc::ChannelInterface> channel);
  explicit KVCacheStoreClient(
      std::unique_ptr<proto::KVCacheStoreService::StubInterface> stub);

  // Fetches remote KV cache blocks.
  //
  // Parameters:
  // - block_hashes: Hashes of the blocks to fetch.
  // - device_block_ids: Optional target TPU device block IDs for HBM fetch.
  //   If empty, data is landed into host DRAM staging memory.
  // - host_block_ids: Optional host block IDs for sanity check verification.
  //   If non-empty, must match block_hashes count.
  //
  // Returns:
  // - On success: Vector of successfully fetched block hashes.
  // - On failure: absl::Status describing the error.
  absl::StatusOr<std::vector<std::string>> Fetch(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> device_block_ids = {},
      absl::Span<const int32_t> host_block_ids = {});

 private:
  std::unique_ptr<proto::KVCacheStoreService::StubInterface> stub_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_CLIENT_H_
