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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TEST_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TEST_UTIL_H_

#include <memory>
#include <utility>

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"

namespace peregrine::internal::testing {

// Creates a connected tcp socket pair in the given address family.
std::pair<std::unique_ptr<TcpSocket>, std::unique_ptr<TcpSocket>>
CreateTcpSocketPair(int family);

// Creates a connected udp socket pair in the given address family.
std::pair<std::unique_ptr<UdpSocket>, std::unique_ptr<UdpSocket>>
CreateUdpSocketPair(int family);

}  // namespace peregrine::internal::testing

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TEST_UTIL_H_
