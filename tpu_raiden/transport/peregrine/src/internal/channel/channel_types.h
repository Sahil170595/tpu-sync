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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TYPES_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TYPES_H_

#include <cstdint>

namespace peregrine::internal {

// An enum that specifies the type of a channel.
enum class ChannelType : uint8_t {
  kReliableStream,     // e.g., tcp
  kReliableMessage,    // e.g., rdma
  kUnreliableMessage,  // e.g., udp
};

// Returns true iff the channel type is reliable stream.
constexpr bool IsReliableStream(ChannelType t) {
  return t == ChannelType::kReliableStream;
}

// Returns true iff the channel type is reliable message.
constexpr bool IsReliableMessage(ChannelType t) {
  return t == ChannelType::kReliableMessage;
}

// Returns true iff the channel type is unreliable message.
constexpr bool IsUnreliableMessage(ChannelType t) {
  return t == ChannelType::kUnreliableMessage;
}

// Returns true iff the channel type is reliable or unreliable message.
constexpr bool IsMessageChannel(ChannelType t) {
  return IsReliableMessage(t) || IsUnreliableMessage(t);
}

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_CHANNEL_CHANNEL_TYPES_H_
