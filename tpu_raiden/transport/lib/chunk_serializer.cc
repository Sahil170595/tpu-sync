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

#include "tpu_raiden/transport/lib/chunk_serializer.h"

#include <cstdint>
#include <cstring>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/lib/chunk.h"
#include "tpu_raiden/transport/lib/chunk_generated.h"

namespace tpu_raiden::transport::lib {

namespace {

absl::InlinedVector<char, kChunkHeaderSize> SerializeHeaderV1(
    const ChunkHeader& header) {
  constexpr uint16_t kVer = 1;
  const flatbuf::ChunkHeader h(
      kRaidenMagic, kVer, header.op, header.flags, header.buffer_id,
      header.reserved,
      /*metadata_size=*/static_cast<uint16_t>(GetChunkMetadataSize(kVer)),
      header.remote_id, header.local_id, header.count_or_size, header.uuid,
      /*padding0=*/0, /*padding1=*/0, /*padding2=*/0, /*padding3=*/0);
  absl::InlinedVector<char, kChunkHeaderSize> bytes(sizeof(h));
  std::memcpy(bytes.data(), &h, sizeof(h));
  return bytes;
}

void DeserializeHeaderV1(const flatbuf::ChunkHeader& h, ChunkHeader& header) {
  DCHECK_EQ(h.ver(), 1);
  header.op = h.op();
  header.flags = h.flags();
  header.buffer_id = h.buffer_id();
  header.reserved = h.reserved();
  header.remote_id = h.remote_id();
  header.local_id = h.local_id();
  header.count_or_size = h.count_or_size();
  header.uuid = h.uuid();
}

absl::InlinedVector<char, kMaxMetadataSize> SerializeMetadataV1(
    const ChunkMetadata& meta) {
  const flatbuf::ChunkMetadata m(meta.layer_idx, meta.dst_shard_idx,
                                 meta.dst_offset_bytes, meta.size_bytes);
  absl::InlinedVector<char, kMaxMetadataSize> bytes(sizeof(m));
  std::memcpy(bytes.data(), &m, sizeof(m));
  return bytes;
}

void DeserializeMetadataV1(const flatbuf::ChunkMetadata& m,
                           ChunkMetadata& meta) {
  meta.layer_idx = m.layer_idx();
  meta.dst_shard_idx = m.dst_shard_idx();
  meta.dst_offset_bytes = m.dst_offset_bytes();
  meta.size_bytes = m.size_bytes();
}

}  // namespace

absl::InlinedVector<char, kChunkHeaderSize> SerializeChunkHeader(
    const ChunkHeader& header) {
  const auto bytes = SerializeHeaderV1(header);
  DCHECK_EQ(bytes.size(), kChunkHeaderSize);
  return bytes;
}

absl::StatusOr<ChunkHeader> DeserializeChunkHeader(absl::Span<const char> s) {
  flatbuf::ChunkHeader h;
  DCHECK_EQ(sizeof(h), kChunkHeaderSize);
  if (s.size() != kChunkHeaderSize) {
    return absl::InvalidArgumentError("Invalid chunk header size");
  }

  std::memcpy(&h, s.data(), sizeof(h));

  if (h.magic() != kRaidenMagic) {
    return absl::InvalidArgumentError(
        absl::StrCat("Chunk header magic mismatch: expected ", kRaidenMagic,
                     ", got ", h.magic()));
  }

  const uint16_t ver = h.ver();
  switch (ver) {
    case 1: {
      ChunkHeader header = {};
      DeserializeHeaderV1(h, header);
      return header;
    }
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported chunk header flatbuf version: ", ver));
  }
}

absl::InlinedVector<char, kMaxMetadataSize> SerializeChunkMetadata(
    const ChunkMetadata& meta) {
  return SerializeMetadataV1(meta);
}

absl::StatusOr<ChunkMetadata> DeserializeChunkMetadata(absl::Span<const char> s,
                                                       uint16_t ver) {
  const size_t meta_size = GetChunkMetadataSize(ver);
  if (s.size() != meta_size) {
    return absl::InvalidArgumentError("Invalid chunk metadata size");
  }

  ChunkMetadata metadata = {};
  switch (ver) {
    case 1: {
      flatbuf::ChunkMetadata m = {};
      std::memcpy(&m, s.data(), meta_size);
      DeserializeMetadataV1(m, metadata);
      break;
    }
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported chunk metadata flatbuf version: ", ver));
  }
  return metadata;
}

}  // namespace tpu_raiden::transport::lib
