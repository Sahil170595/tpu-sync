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

#include "tpu_raiden/transport/peregrine/src/internal/util/test_iov.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/util.h"

namespace peregrine::internal::testing {

OwnedIoVec TestOnly_Linearize(const absl::Span<const IoVec> iovecs) {
  // Calculate the total length of all the iovecs.
  const size_t size = TotalLength(iovecs);
  if ABSL_PREDICT_FALSE (size <= 0) {
    return OwnedIoVec{.data = nullptr, .size = 0};
  }

  // Allocate a buffer to hold all the data.
  auto buf = std::make_unique_for_overwrite<Byte[]>(size);
  DCHECK_NE(buf, nullptr);

  // Copy the data from the iovecs into the buffer.
  size_t offset = 0;
  for (const auto& v : iovecs) {
    void* __restrict const dst = buf.get() + offset;
    const void* __restrict const src = v.iov_base;
    const size_t n = v.iov_len;
    DCHECK(IsValid(v));
    if (src != nullptr && n > 0) {
      std::memcpy(dst, src, n);
      offset += n;
    }
  }
  DCHECK_EQ(offset, size);

  // Return the buffer, together with its ownership.
  return OwnedIoVec{.data = std::move(buf), .size = size};
}

}  // namespace peregrine::internal::testing
