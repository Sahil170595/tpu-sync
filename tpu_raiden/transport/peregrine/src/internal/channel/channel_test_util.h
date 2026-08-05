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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TEST_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TEST_UTIL_H_

#include <sys/socket.h>

#include <memory>
#include <string>

#include "tpu_raiden/transport/peregrine/src/internal/channel/channel.h"

namespace peregrine::internal::testing {

enum class TestChannelType {
  kTcp,
  kUdp,
  kMemStream,
  kMemMsg,
};

// Returns a string representation for the test channel type.
std::string ToString(TestChannelType t);

// A pair of connected channels.
struct ConnectedChannelPair final {
  std::unique_ptr<Channel> sndr;
  std::unique_ptr<Channel> rcvr;
};

// Creates a tcp channel pair in the given address family.
ConnectedChannelPair CreateTcpChannelPair(int family);

// Creates a udp channel pair in the given address family.
ConnectedChannelPair CreateUdpChannelPair(int family);

// Creates a memory stream channel pair.
ConnectedChannelPair CreateMemStreamChannelPair(int error_rate);

// Creates a memory message channel pair.
ConnectedChannelPair CreateMemMsgChannelPair(int error_rate);

// Creates a test channel pair with the given type and error rate.
ConnectedChannelPair CreateTestChannelPair(TestChannelType type,
                                           int family = AF_INET,
                                           int error_rate = 0);

}  // namespace peregrine::internal::testing

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TEST_UTIL_H_
