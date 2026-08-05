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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UTIL_H_

#include <memory>
#include <utility>
#include <vector>

#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel_udp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"

namespace peregrine::internal {

// Creates a tcp channel.
inline std::unique_ptr<Channel> CreateTcpChannel(
    std::unique_ptr<TcpSocket> socket) {
  return std::make_unique<TcpChannel>(std::move(socket));
}

// Creates a udp channel.
inline std::unique_ptr<Channel> CreateUdpChannel(
    std::unique_ptr<UdpSocket> socket) {
  return std::make_unique<UdpChannel>(std::move(socket));
}

using Channels = std::vector<std::unique_ptr<Channel>>;

// Creates `n` channels connected to the `peer`.
Channels Create(const Endpoint& peer, int n);

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_UTIL_H_
