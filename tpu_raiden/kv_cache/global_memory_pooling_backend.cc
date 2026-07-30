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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

GlobalMemoryPoolingBackend::GlobalMemoryPoolingBackend(
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
    RaidenId raiden_id)
    : registry_client_(std::move(registry_client)),
      raiden_id_(std::move(raiden_id)),
      server_(KVCacheStoreServer::Create()) {}

GlobalMemoryPoolingBackend::~GlobalMemoryPoolingBackend() {
  if (server_) {
    server_->Shutdown();
  }
}

void GlobalMemoryPoolingBackend::SetRaidenController(
    tpu_raiden::controller::RaidenController* controller) {
  controller_ = controller;
  if (server_ && server_->GetGrpcPort() == 0) {
    auto status = server_->StartServer(this, controller_, "[::]:0");
    if (!status.ok()) {
      LOG(WARNING)
          << "Failed to start KVCacheStoreServer in SetRaidenController: "
          << status.message();
    }
  }
}

std::string GlobalMemoryPoolingBackend::GetServerAddress() const {
  return server_ ? server_->GetServerAddress() : "";
}

int GlobalMemoryPoolingBackend::GetGrpcPort() const {
  return server_ ? server_->GetGrpcPort() : 0;
}

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
    BlockStatus status =
        (remote_id == raiden_id_) ? BlockStatus::HOST : BlockStatus::REMOTE;
    results.push_back(
        std::make_pair(block_hashes[i],
                       RaidenBlockID(remote_id, metadata.block_id(), status)));
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

void GlobalMemoryPoolingBackend::SetStoreClient(
    const RaidenId& remote_id, std::shared_ptr<KVCacheStoreClient> client) {
  absl::MutexLock lock(mutex_);
  store_clients_[remote_id] = std::move(client);
}

absl::StatusOr<std::shared_ptr<KVCacheStoreClient>>
GlobalMemoryPoolingBackend::GetKVCacheStoreClient(const RaidenId& remote_id) {
  {
    absl::MutexLock lock(mutex_);
    auto it = store_clients_.find(remote_id);
    if (it != store_clients_.end()) {
      return it->second;
    }
  }

  if (controller_ == nullptr) {
    return absl::FailedPreconditionError("RaidenController is null");
  }

  tpu_raiden::rpc::RaidenIdProto remote_proto;
  remote_proto.set_job_name(remote_id.job_name);
  remote_proto.set_job_replica_id(remote_id.job_replica_id);
  remote_proto.set_data_name(remote_id.data_name);
  remote_proto.set_data_replica_idx(remote_id.data_replica_idx);

  ASSIGN_OR_RETURN(std::string address,
                   controller_->ResolvePeerController(remote_proto));
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto client = std::make_shared<KVCacheStoreClient>(channel);
  {
    absl::MutexLock lock(mutex_);
    store_clients_[remote_id] = client;
  }
  return client;
}

tsl::Future<> GlobalMemoryPoolingBackend::Load(
    const RaidenId& remote_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids) {
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "Mismatched device_block_ids count (", device_block_ids.size(),
        ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  if (controller_ == nullptr) {
    return tsl::Future<>(
        absl::FailedPreconditionError("RaidenController is null"));
  }

  auto client_or = GetKVCacheStoreClient(remote_id);
  if (!client_or.ok()) {
    return tsl::Future<>(client_or.status());
  }
  std::shared_ptr<KVCacheStoreClient> client = std::move(client_or.value());

  auto host_blocks_or = controller_->AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    return tsl::Future<>(host_blocks_or.status());
  }
  std::vector<int32_t> dst_host_block_ids(host_blocks_or.value().begin(),
                                          host_blocks_or.value().end());

  auto [load_promise, load_future] = tsl::MakePromise<>();

  tpu_raiden::rpc::RaidenIdProto client_raiden_id = controller_->unit();
  tsl::Future<proto::FetchResponse> fetch_future = client->Fetch(
      block_hashes, device_block_ids, dst_host_block_ids, client_raiden_id);

  // TODO: The H2D command should not wait until the Fetch is done.
  fetch_future.OnReady(
      [this, dst_host_block_ids,
       dev_ids_vec = std::vector<int32_t>(device_block_ids.begin(),
                                          device_block_ids.end()),
       load_promise = std::move(load_promise)](
          const absl::StatusOr<proto::FetchResponse>& response_or) mutable {
        if (!response_or.ok()) {
          if (controller_) {
            (void)controller_->DeallocateBlockIds(dst_host_block_ids);
          }
          load_promise.Set(response_or.status());
          return;
        }

        const auto& response = response_or.value();
        if (!response.failed_block_hashes().empty()) {
          if (controller_) {
            (void)controller_->DeallocateBlockIds(dst_host_block_ids);
          }
          std::string err_msg = response.error_message().empty()
                                    ? "Fetch RPC returned failed blocks"
                                    : response.error_message();
          load_promise.Set(absl::InternalError(err_msg));
          return;
        }

        if (dev_ids_vec.empty()) {
          if (controller_) {
            (void)controller_->DeallocateBlockIds(dst_host_block_ids);
          }
          load_promise.Set(absl::OkStatus());
          return;
        }

        std::vector<Buffer> src_buffers;
        src_buffers.reserve(dst_host_block_ids.size());
        for (int id : dst_host_block_ids) {
          src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   rpc::MEMORY_TYPE_DRAM);
        }

        std::vector<Buffer> dst_buffers;
        dst_buffers.reserve(dev_ids_vec.size());
        for (int id : dev_ids_vec) {
          dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   rpc::MEMORY_TYPE_HBM);
        }

        tsl::Future<> h2d_future =
            controller_->TransferBuffers(src_buffers, dst_buffers);

        h2d_future.OnReady(
            [this, dst_host_block_ids, load_promise = std::move(load_promise)](
                absl::Status status) mutable {
              if (controller_) {
                (void)controller_->DeallocateBlockIds(dst_host_block_ids);
              }
              load_promise.Set(status);
            });
      });

  return load_future;
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
