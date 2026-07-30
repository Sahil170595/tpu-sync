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

#include "tpu_raiden/kv_cache/kv_cache_store_client.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "tpu_raiden/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStoreClient::KVCacheStoreClient(
    std::shared_ptr<::grpc::ChannelInterface> channel)
    : stub_(proto::KVCacheStoreService::NewStub(channel)) {}

KVCacheStoreClient::KVCacheStoreClient(
    std::unique_ptr<proto::KVCacheStoreService::StubInterface> stub)
    : stub_(std::move(stub)) {}

absl::StatusOr<std::vector<std::string>> KVCacheStoreClient::Fetch(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const int32_t> host_block_ids) {
  if (block_hashes.empty()) {
    return std::vector<std::string>{};
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Mismatched device_block_ids count (", device_block_ids.size(),
        ") vs block_hashes count (", block_hashes.size(), ")."));
  }

  if (!host_block_ids.empty() && host_block_ids.size() != block_hashes.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched host_block_ids count (", host_block_ids.size(),
                     ") vs block_hashes count (", block_hashes.size(), ")."));
  }

  proto::FetchRequest request;
  for (const auto& hash : block_hashes) {
    request.add_block_hashes(hash);
  }
  for (int32_t dev_id : device_block_ids) {
    request.add_device_block_ids(dev_id);
  }
  for (int32_t host_id : host_block_ids) {
    request.add_host_block_ids(host_id);
  }

  proto::FetchResponse response;
  ::grpc::ClientContext context;
  ::grpc::Status status = stub_->Fetch(&context, request, &response);
  if (!status.ok()) {
    return status;
  }

  if (response.failed_block_hashes_size() > 0) {
    return absl::InternalError(
        absl::StrCat("Fetch failed for ", response.failed_block_hashes_size(),
                     " out of ", block_hashes.size(), " blocks."));
  }

  std::vector<std::string> done_hashes;
  done_hashes.reserve(response.done_block_hashes_size());
  for (const auto& hash : response.done_block_hashes()) {
    done_hashes.push_back(hash);
  }
  return done_hashes;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
