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

// Service implementation for KVCacheStoreService.

#include "tpu_raiden/kv_cache/kv_cache_store_service.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/raiden_transfer_endpoint.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

template <typename ProtoContainer>
std::vector<RaidenWorkerEndpoints> UnpackWorkerEndpointsProto(
    const ProtoContainer& proto_groups) {
  std::vector<RaidenWorkerEndpoints> result;
  result.reserve(proto_groups.size());
  for (const auto& group_proto : proto_groups) {
    std::vector<RaidenTransferEndpoint> eps;
    eps.reserve(group_proto.endpoints_size());
    for (const auto& ep_proto : group_proto.endpoints()) {
      eps.push_back({ep_proto.endpoint(),
                     std::vector<int64_t>(ep_proto.shards().begin(),
                                          ep_proto.shards().end())});
    }
    result.push_back(
        {group_proto.node_id(), group_proto.worker_id(), std::move(eps)});
  }
  return result;
}

}  // namespace

KVCacheStoreServiceImpl::KVCacheStoreServiceImpl(
    KVCacheStoreBackend* backend,
    tpu_raiden::controller::RaidenController* controller)
    : backend_(backend), controller_(controller) {}

::grpc::Status KVCacheStoreServiceImpl::Fetch(
    ::grpc::ServerContext* context, const proto::FetchRequest* request,
    proto::FetchResponse* response) {
  if (backend_ == nullptr || controller_ == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION,
                          "Backend or RaidenController non-initialized");
  }

  const std::vector<std::string> block_hashes(request->block_hashes().begin(),
                                              request->block_hashes().end());
  const std::vector<int32_t> dst_host_block_ids(
      request->host_block_ids().begin(), request->host_block_ids().end());

  if (block_hashes.empty()) {
    return ::grpc::Status::OK;
  }

  if (dst_host_block_ids.size() != block_hashes.size()) {
    return ::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Mismatched host_block_ids count vs block_hashes count.");
  }

  // =========================================================================
  // STEP 1: Validation
  // Validate block_hashes exist in local index and reside in host memory.
  // =========================================================================
  auto lookup_or = backend_->Lookup(block_hashes);
  if (!lookup_or.ok()) {
    return ::grpc::Status(
        ::grpc::StatusCode::NOT_FOUND,
        absl::StrCat("Validation failed: ", lookup_or.status().message()));
  }
  const auto& lookup_slices = lookup_or.value();
  if (lookup_slices.size() < block_hashes.size()) {
    return ::grpc::Status(
        ::grpc::StatusCode::NOT_FOUND,
        absl::StrCat("Partial block match: found ", lookup_slices.size(),
                     " out of ", block_hashes.size()));
  }

  std::vector<int32_t> src_host_block_ids;
  src_host_block_ids.reserve(lookup_slices.size());
  for (const auto& [hash, slice] : lookup_slices) {
    if (slice.status != BlockStatus::HOST &&
        slice.status != BlockStatus::HOST_AND_HBM) {
      return ::grpc::Status(
          ::grpc::StatusCode::FAILED_PRECONDITION,
          absl::StrCat("Block hash '", hash, "' is not resident in host DRAM"));
    }
    src_host_block_ids.push_back(slice.host_block_id);
  }

  // Cross-node validation: require worker endpoints if request is from peer controller.
  RaidenId client_id{
      request->client_raiden_id().job_name(),
      request->client_raiden_id().job_replica_id(),
      request->client_raiden_id().data_name(),
      request->client_raiden_id().data_replica_idx(),
  };
  const auto& server_unit = controller_->unit();
  bool is_cross_node =
      !client_id.empty() &&
      (client_id.job_name != server_unit.job_name() ||
       client_id.job_replica_id != server_unit.job_replica_id() ||
       client_id.data_name != server_unit.data_name() ||
       client_id.data_replica_idx != server_unit.data_replica_idx());
  if (is_cross_node && request->client_worker_endpoints().empty()) {
    return ::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Cross-node FetchRequest requires non-empty client_worker_endpoints.");
  }

  // =========================================================================
  // STEP 2: Pinning
  // Protect source host blocks against LRU eviction during DMA transfer.
  // =========================================================================
  if (!backend_->Pin(block_hashes)) {
    return ::grpc::Status(::grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Failed to pin host blocks; blocks may be locked or "
                          "undergoing eviction.");
  }

  // =========================================================================
  // STEP 3: Unpinning Guarantee (RAII / absl::Cleanup)
  // Ensure unpinning ALWAYS executes when exiting this scope, even on error
  // or RPC cancellation.
  // =========================================================================
  auto unpin_cleanup = absl::MakeCleanup(
      [this, &block_hashes]() { backend_->Release(block_hashes); });

  // =========================================================================
  // STEP 4: Transfer Execution
  // Transfer data from local source host DRAM directly to destination host DRAM
  // using RaidenController::TransferBuffers with Buffer structs.
  // =========================================================================
  std::vector<RaidenWorkerEndpoints> client_groups =
      UnpackWorkerEndpointsProto(request->client_worker_endpoints());

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int id : src_host_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
  }

  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(dst_host_block_ids.size());
  for (int id : dst_host_block_ids) {
    Buffer dst_buf(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
    if (!client_groups.empty()) {
      dst_buf.set_remote_worker_endpoints(client_groups);
    }
    dst_buffers.push_back(std::move(dst_buf));
  }

  tsl::Future<> transfer_future =
      controller_->TransferBuffers(src_buffers, dst_buffers);

  absl::Status transfer_status = transfer_future.Await();

  // Check RPC Cancellation Safety
  if (context != nullptr && context->IsCancelled()) {
    return ::grpc::Status(::grpc::StatusCode::CANCELLED,
                          "Fetch request cancelled by client context");
  }

  if (!transfer_status.ok()) {
    for (const auto& hash : block_hashes) {
      response->add_failed_block_hashes(hash);
    }
    return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                          absl::StrCat("Host-to-Host transfer failed: ",
                                       transfer_status.message()));
  }

  // =========================================================================
  // STEP 5: Response Generation
  // =========================================================================
  for (const auto& hash : block_hashes) {
    response->add_done_block_hashes(hash);
  }

  return ::grpc::Status::OK;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
