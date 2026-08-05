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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UDP_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UDP_H_

#include <sys/socket.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel_types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"

namespace peregrine::internal {

// A udp socket based channel: unreliable, message.
// It is thread-compatible but not thread-safe.
class UdpChannel final : public Channel {
 public:
  // Constructor.
  explicit UdpChannel(std::unique_ptr<UdpSocket> socket)
      : socket_(std::move(socket)) {
    DCHECK_NE(socket_, nullptr);
  }

  // Returns the channel type.
  constexpr ChannelType Type() const override {
    return ChannelType::kUnreliableMessage;
  }

  // Writes a buffer of `len` bytes to the channel.
  // Returns the number of bytes actually written if successful. Zero byte means
  // no data has been written due to non-error reasons. Returns -1 on error.
  ssize_t Write(const Byte* buf, size_t len) override {
    DCHECK_NE(buf, nullptr);
    DCHECK_GE(len, 1);
    return socket_->Send(buf, len);
  }

  // Writes data from the `iovecs` buffers to the channel.
  // Returns the number of bytes actually written if successful. Zero byte means
  // no data has been written due to non-error reasons. Returns -1 on error.
  ssize_t WriteV(absl::Span<const IoVec> iovecs) override;

  // Reads one message of up to `len` bytes from the channel into the `buf`.
  // Returns the number of bytes actually read if successful. Returns 0 if the
  // received packet has no payload. Returns -1 on error.
  ssize_t Read(Byte* buf, size_t len) override {
    DCHECK_NE(buf, nullptr);
    DCHECK_GE(len, 1);
    return socket_->Recv(buf, len);
  }

  // Reads one message of up to `length(iovecs)` bytes from the channel into
  // the buffers. Returns the number of bytes actually read if successful.
  // Returns 0 if the received packet has no payload. Returns -1 on error.
  ssize_t ReadV(absl::Span<IoVec> iovecs) override;

  // Shuts down the channel. After the channel is shutdown, write calls will
  // return -1, so no more data can be injected into the channel. Read calls
  // will continue to read the remaining data in the channel, if any.
  void Shutdown() override { socket_->Shutdown(); }

  // Returns a string representation for the channel.
  std::string ToString() const override;

 private:
  std::unique_ptr<UdpSocket> socket_;
};

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UDP_H_
