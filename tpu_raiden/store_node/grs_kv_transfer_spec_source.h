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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {

// KVTransferSpec from the global registry (GetKVTransferSpec RPC).
class GrsKVTransferSpecSource : public KVTransferSpecSource {
 public:
  explicit GrsKVTransferSpecSource(absl::string_view global_registry_address);

  // NotFound until a serving host has published the spec; Unavailable while
  // the registry is unreachable. Both are retried by WaitForSpec.
  absl::StatusOr<KVTransferSpec> Get() override;

 private:
  kv_cache::global_registry::GlobalRegistryClient client_;
};

}  // namespace store_node
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_STORE_NODE_GRS_KV_TRANSFER_SPEC_SOURCE_H_
