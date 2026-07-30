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

#include "tpu_raiden/kv_cache/kv_cache_store_service.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStoreServiceImpl::KVCacheStoreServiceImpl(KVCacheStore* store)
    : store_(store) {}

KVCacheStoreServiceImpl::KVCacheStoreServiceImpl(
    std::shared_ptr<KVCacheStore> store)
    : store_(store.get()), store_shared_ptr_(std::move(store)) {}

void KVCacheStoreServiceImpl::SetStore(KVCacheStore* store) {
  absl::MutexLock lock(&mutex_);
  store_ = store;
  store_shared_ptr_.reset();
}

void KVCacheStoreServiceImpl::SetStore(std::shared_ptr<KVCacheStore> store) {
  absl::MutexLock lock(&mutex_);
  store_ = store.get();
  store_shared_ptr_ = std::move(store);
}

::grpc::Status KVCacheStoreServiceImpl::Fetch(
    ::grpc::ServerContext* context, const proto::FetchRequest* request,
    proto::FetchResponse* response) {
  KVCacheStore* store_ptr = nullptr;
  std::shared_ptr<KVCacheStore> store_shared;
  {
    absl::MutexLock lock(&mutex_);
    store_ptr = store_;
    store_shared = store_shared_ptr_;
  }

  if (store_ptr == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::FAILED_PRECONDITION,
                          "KVCacheStore is null");
  }

  std::vector<std::string> block_hashes(request->block_hashes().begin(),
                                        request->block_hashes().end());
  std::vector<int32_t> device_block_ids(request->device_block_ids().begin(),
                                        request->device_block_ids().end());

  if (block_hashes.empty()) {
    return ::grpc::Status::OK;
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return ::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat("Mismatched device_block_ids count (",
                     device_block_ids.size(), ") vs block_hashes count (",
                     block_hashes.size(), ")."));
  }

  if (!request->host_block_ids().empty() &&
      request->host_block_ids().size() != request->block_hashes().size()) {
    return ::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        "Mismatched host_block_ids count vs block_hashes count.");
  }

  absl::Status status = store_ptr->ReadRemote(block_hashes, device_block_ids);
  if (!status.ok()) {
    return status;
  }

  absl::flat_hash_set<std::string> requested_set(block_hashes.begin(),
                                                 block_hashes.end());
  absl::flat_hash_set<std::string> done_set;
  absl::flat_hash_set<std::string> failed_set;

  {
    absl::MutexLock lock(&fetch_mutex_);
    while (done_set.size() + failed_set.size() < requested_set.size()) {
      if (context != nullptr && context->IsCancelled()) {
        for (const auto& hash : requested_set) {
          completed_hashes_.erase(hash);
          failed_hashes_.erase(hash);
        }
        return ::grpc::Status(::grpc::StatusCode::CANCELLED,
                              "Client cancelled fetch request");
      }

      auto [done, failed, pending] = store_ptr->PollRemoteReadStatus();
      bool new_events = !done.empty() || !failed.empty();
      for (const auto& hash : done) {
        completed_hashes_.insert(hash);
      }
      for (const auto& hash : failed) {
        failed_hashes_.insert(hash);
      }
      if (new_events) {
        fetch_cv_.SignalAll();
      }

      for (const auto& hash : requested_set) {
        if (!done_set.contains(hash) && !failed_set.contains(hash)) {
          if (completed_hashes_.contains(hash)) {
            done_set.insert(hash);
          } else if (failed_hashes_.contains(hash)) {
            failed_set.insert(hash);
          }
        }
      }

      if (done_set.size() + failed_set.size() < requested_set.size()) {
        fetch_cv_.WaitWithTimeout(&fetch_mutex_, absl::Milliseconds(10));
      }
    }

    for (const auto& hash : requested_set) {
      completed_hashes_.erase(hash);
      failed_hashes_.erase(hash);
    }
  }

  for (const auto& hash : done_set) {
    response->add_done_block_hashes(hash);
  }
  for (const auto& hash : failed_set) {
    response->add_failed_block_hashes(hash);
  }

  return ::grpc::Status::OK;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
