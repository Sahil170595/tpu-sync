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
#include "grpcpp/impl/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStoreClient::KVCacheStoreClient(
    std::shared_ptr<::grpc::ChannelInterface> channel)
    : stub_(proto::KVCacheStoreService::NewStub(channel)) {}

KVCacheStoreClient::KVCacheStoreClient(
    std::unique_ptr<proto::KVCacheStoreService::StubInterface> stub)
    : stub_(std::move(stub)) {}

tsl::Future<proto::FetchResponse> KVCacheStoreClient::Fetch(
    absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids,
    absl::Span<const int32_t> host_block_ids,
    const rpc::RaidenIdProto& client_raiden_id,
    absl::Span<const ::tpu_raiden::proto::RaidenWorkerEndpointsProto>
        client_worker_endpoints) {
  if (block_hashes.empty()) {
    return tsl::Future<proto::FetchResponse>(proto::FetchResponse{});
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return tsl::Future<proto::FetchResponse>(
        absl::InvalidArgumentError(absl::StrCat(
            "Mismatched device_block_ids count (", device_block_ids.size(),
            ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  if (!host_block_ids.empty() && host_block_ids.size() != block_hashes.size()) {
    return tsl::Future<proto::FetchResponse>(absl::InvalidArgumentError(
        absl::StrCat("Mismatched host_block_ids count (", host_block_ids.size(),
                     ") vs block_hashes count (", block_hashes.size(), ").")));
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
  *request.mutable_client_raiden_id() = client_raiden_id;
  request.mutable_client_worker_endpoints()->Reserve(
      client_worker_endpoints.size());
  for (const auto& group : client_worker_endpoints) {
    *request.add_client_worker_endpoints() = group;
  }

  auto [promise, future] = tsl::MakePromise<proto::FetchResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<proto::FetchResponse>();

  stub_->async()->Fetch(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) mutable {
        if (!status.ok()) {
          promise->Set(absl::Status(
              static_cast<absl::StatusCode>(status.error_code()),
              absl::StrCat("Fetch RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
