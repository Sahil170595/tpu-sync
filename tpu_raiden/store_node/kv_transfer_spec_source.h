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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tpu_raiden {
namespace store_node {

// Everything a host store node must match about the deployment's serving hosts
// to interoperate with their KV transfers. The authority for these values is
// the serving host's manager runtime (which has the real device buffers);
// a host store node only ever receives them, it never derives them.
//
// Today this carries the block geometry: the uniform-slice block shape,
// where every (layer, shard) slice has the same byte size. Planned
// extensions -- the serving hosts' transfer worker topology, and per-pool
// block shapes for hybrid models -- grow this struct without changing the
// source interface below.
struct KVTransferSpec {
  size_t num_layers = 0;
  size_t num_shards = 0;
  size_t slice_byte_size = 0;
};

// Returns InvalidArgument unless every field is positive.
inline absl::Status ValidateSpec(const KVTransferSpec& spec) {
  if (spec.num_layers == 0 || spec.num_shards == 0 ||
      spec.slice_byte_size == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "KVTransferSpec fields must all be positive, got num_layers=",
        spec.num_layers, " num_shards=", spec.num_shards,
        " slice_byte_size=", spec.slice_byte_size));
  }
  return absl::OkStatus();
}

// Where a booting host store node obtains the deployment's KVTransferSpec.
//
// Get() contract:
//  - OK: the spec is known. The value is fixed for the lifetime of the
//    deployment; callers read it once at boot.
//  - NotFound: nothing published yet. Expected during turnup, when the store
//    node can come up before any serving host has published its spec; the
//    caller retries.
//  - Unavailable: the source itself is not reachable yet (also expected
//    during turnup); the caller retries.
//  - anything else: fatal.
class KVTransferSpecSource {
 public:
  virtual ~KVTransferSpecSource() = default;

  virtual absl::StatusOr<KVTransferSpec> Get() = 0;
};

// KVTransferSpec fixed at construction, e.g. from flags. Stopgap until the
// global-registry-backed source lands: the registry will hold the spec
// published by the serving hosts at their own registration, and a source
// implementation will poll it here through the same interface.
class StaticKVTransferSpecSource : public KVTransferSpecSource {
 public:
  explicit StaticKVTransferSpecSource(KVTransferSpec spec) : spec_(spec) {}

  absl::StatusOr<KVTransferSpec> Get() override { return spec_; }

 private:
  KVTransferSpec spec_;
};

}  // namespace store_node
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_STORE_NODE_KV_TRANSFER_SPEC_SOURCE_H_
