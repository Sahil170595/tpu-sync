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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_IOV_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_IOV_H_

#include <sys/uio.h>

#include <cstddef>
#include <memory>

#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal::testing {

// `OwnedIoVec` describes a continuous buffer of data owned by the caller.
struct OwnedIoVec {
  std::unique_ptr<Byte[]> data;
  size_t size;
};

// Linearizes a sequence of `IoVecs` into a newly created contiguous buffer.
// Returns the buffer with its ownership moved to the caller.
OwnedIoVec TestOnly_Linearize(absl::Span<const IoVec> iovecs);

}  // namespace peregrine::internal::testing

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_IOV_H_
