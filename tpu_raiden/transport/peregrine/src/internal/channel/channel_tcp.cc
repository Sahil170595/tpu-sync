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

#include "tpu_raiden/transport/peregrine/src/internal/channel/channel_tcp.h"

#include <string>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/util.h"

namespace peregrine::internal {

ssize_t TcpChannel::WriteV(const absl::Span<const IoVec> iovecs) {
  DCHECK(IsValid(iovecs));
  DCHECK_GE(iovecs.size(), 1);
  DCHECK_LE(iovecs.size(), IOV_MAX);
  DCHECK_GE(TotalLength(iovecs), 1);

  if ABSL_PREDICT_FALSE (iovecs.size() == 1) {
    const auto [buf, len] = BufLen(iovecs[0]);
    return socket_->Send(buf, len);
  } else {
    DCHECK_GE(iovecs.size(), 2);
    return socket_->SendV(iovecs);
  }
}

ssize_t TcpChannel::ReadV(absl::Span<IoVec> iovecs) {
  DCHECK(IsValid(iovecs));
  DCHECK_GE(iovecs.size(), 1);
  DCHECK_LE(iovecs.size(), IOV_MAX);
  DCHECK_GE(TotalLength(iovecs), 1);

  if ABSL_PREDICT_FALSE (iovecs.size() == 1) {
    const auto [buf, len] = BufLen(iovecs[0]);
    return socket_->Recv(buf, len);
  } else {
    DCHECK_GE(iovecs.size(), 2);
    return socket_->RecvV(iovecs);
  }
}

std::string TcpChannel::ToString() const {
  return absl::StrCat("TcpChannel: ", socket_->ToString());
}

}  // namespace peregrine::internal
