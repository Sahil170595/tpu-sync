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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_TCP_SOCKET_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_TCP_SOCKET_UTIL_H_

// NOTE: Do __not__ modify this file.
// It is temporarily used by tpu raiden.
// It will soon be replaced by the `TcpSocket` class.

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>

#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

// This utility class implements tcp socket send/recv functions.
// This class is thread-safe since it has no data members.
class TcpSocketUtil final {
 public:
  // Sends on the socket `fd` exactly `len` bytes of data from the `buf`.
  // Returns OK if all the bytes are sent successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status Send(fd_t fd, const Byte* buf, size_t len);

  // Receives on the socket `fd` exactly `len` bytes of data into the `buf`.
  // Returns OK if all the bytes are received successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status Recv(fd_t fd, Byte* buf, size_t len);

  // Sends on the socket `fd` exactly all the data from the `iovecs` buffers.
  // Returns OK if all the bytes are sent successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status SendV(fd_t fd, absl::Span<const IoVec> iovecs);

  // Receives on the socket `fd` exactly all the data into the `iovecs` buffers.
  // Returns OK if all the bytes are received successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status RecvV(fd_t fd, absl::Span<const IoVec> iovecs);
};

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_TCP_SOCKET_UTIL_H_
