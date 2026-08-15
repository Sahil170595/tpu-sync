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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_

// Pool-reshard <-> TransferProgram bijection and validator.
//
// Design invariant: the transfer-program worker entry executes through the
// same manager operations the framed listener drives (PoolReshardPush /
// PoolReshardRegisterRecv, both consuming a whole StartTransferRequest).
// CompileStartTransfer and LowerToStartTransfer are exact inverses for the
// pool-reshard class, so the manager sees byte-identical inputs under either
// entry and preserves the receiver's validation. The replay differential
// program-roundtrip mode compiles and lowers recorded live worker payloads,
// then byte-compares them under deterministic serialization.

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {

// The supported lowering classes. NormalizeReshardProgram rejects
// AwaitInline / Barrier contracts as Unimplemented and malformed programs as
// InvalidArgument.
enum class ReshardLoweringClass {
  kPoolReshardSender,
  kPoolReshardArm,
};

// Normalizing validator. Admits a program into exactly one lowering class or
// rejects it. Dispatch keys on the completion contract and role, never on
// envelope.kind (kind is checked only for consistency).
absl::StatusOr<ReshardLoweringClass> NormalizeReshardProgram(
    const ::tpu_sync::proto::TransferProgramRequest& request);

// Lossless pool-reshard StartTransferRequest -> program conversion.
// Rejects non-pool requests (legacy dense plans stay on the framed door).
absl::StatusOr<::tpu_sync::proto::TransferProgramRequest> CompileStartTransfer(
    const ::tpu_sync::rpc::StartTransferRequest& request);

// Worker-door lowering (inverse of CompileStartTransfer). Callers must have
// run NormalizeReshardProgram first; this reconstructs the byte-identical
// StartTransferRequest the framed door would have delivered.
absl::StatusOr<::tpu_sync::rpc::StartTransferRequest> LowerToStartTransfer(
    const ::tpu_sync::proto::TransferProgramRequest& request);

// Listener-parity block-id derivations (kv_cache_listener.cc sender/receiver
// paths): the sender's local block list is the sorted-unique union of its
// schedule entries' source blocks; the receiver's flat destination list is
// the pool groups' dst_device_block_ids concatenation.
std::vector<int64_t> DeriveSenderSourceBlockIds(
    const ::tpu_sync::rpc::StartTransferRequest& request);
std::vector<int64_t> DeriveArmChipBlockIds(
    const ::tpu_sync::rpc::StartTransferRequest& request);

}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_PROGRAM_RESHARD_H_
