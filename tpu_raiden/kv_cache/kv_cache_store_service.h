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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStoreServiceImpl : public proto::KVCacheStoreService::Service {
 public:
  explicit KVCacheStoreServiceImpl(KVCacheStore* store);
  explicit KVCacheStoreServiceImpl(std::shared_ptr<KVCacheStore> store);

  void SetStore(KVCacheStore* store);
  void SetStore(std::shared_ptr<KVCacheStore> store);

  ::grpc::Status Fetch(::grpc::ServerContext* context,
                       const proto::FetchRequest* request,
                       proto::FetchResponse* response) override;

 private:
  mutable absl::Mutex mutex_;
  KVCacheStore* store_ ABSL_GUARDED_BY(mutex_) = nullptr;
  std::shared_ptr<KVCacheStore> store_shared_ptr_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex fetch_mutex_;
  absl::CondVar fetch_cv_ ABSL_GUARDED_BY(fetch_mutex_);
  absl::flat_hash_set<std::string> completed_hashes_
      ABSL_GUARDED_BY(fetch_mutex_);
  absl::flat_hash_set<std::string> failed_hashes_ ABSL_GUARDED_BY(fetch_mutex_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_SERVICE_H_
