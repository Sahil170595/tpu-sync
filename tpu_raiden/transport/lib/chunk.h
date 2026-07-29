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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_

#include <cstdint>

namespace tpu_raiden::transport::lib {

// Compact 32-byte binary chunk header layout.
// TODO(swasthi): serialization using flatbuffer.
// TODO(swasthi): add version field to prevent breaking changes.
struct alignas(8) ChunkHeader {
  uint8_t op;     // 3=BytePull, 5=ByteSlicePush, 1,2,4,6=HigherLevelBlockOps
  uint8_t flags;  // Holds major_order or protocol flags
  uint16_t buffer_id;      // Multidimensional Buffer / Layer ID coordinate
  uint16_t reserved;       // Holds parallelism/expected chunks count
  uint16_t padding;        // Unused padding to align fields
  uint32_t remote_id;      // Remote block ID or linear memory offset
  uint32_t local_id;       // Local block ID or target shard index
  uint32_t count_or_size;  // Number of blocks or continuous payload bytes
  uint64_t uuid;           // Globally unique transaction routing ID
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_
