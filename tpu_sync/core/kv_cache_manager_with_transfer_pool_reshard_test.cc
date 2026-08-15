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

// Deviceless executor unit tier (X1/X4): the ValidatePoolReshardPlan
// accept/reject table, the device-only rejection at the public entry points,
// the tag-neutral skip summary, and the sender-side pool selection (including
// the no-bytes-owned sender completing without device work). The executor
// byte path and the pool receive lifecycle are device-only by design and are
// exercised on real chips by the D-series harness.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::StartTransferRequest;

class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s = 30.0)
      : KVCacheManagerWithTransfer(
            /*num_layers=*/1, /*num_shards=*/1,
            /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::make_optional(4),
            /*parallelism=*/1, /*node_id=*/0,
            /*local_control_port=*/-1, /*max_blocks=*/0, /*num_slots=*/0,
            timeout_s) {}

  using KVCacheManagerWithTransfer::ValidatePoolReshardPlan;

  // Passes the device-attached gate with no real device state. Only paths
  // that never touch the holds (validation and the no-bytes-owned sender
  // completion) may rely on it.
  void AttachPlaceholderDeviceHold() { buffer_holds_.emplace_back(); }
};

kv_cache::PoolSpec DensePool(std::string tag, int64_t block_stride = 128,
                             std::string dtype_tag = "bf16") {
  return kv_cache::PoolSpec{
      .tag = std::move(tag),
      .storage_index = 0,
      .base_offset_bytes = 0,
      .block_stride_bytes = block_stride,
      .num_blocks = 4,
      .regions = {kv_cache::RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = block_stride,
          .unit_bytes = block_stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = std::move(dtype_tag),
  };
}

StartTransferRequest ValidPlan(
    int64_t uuid, const std::vector<std::string>& dtype_tags = {"bf16"},
    const std::vector<int32_t>& transferred_pools = {0}) {
  StartTransferRequest plan;
  plan.set_uuid(uuid);
  plan.set_req_id("pool_reshard_req_" + std::to_string(uuid));
  plan.set_dst_mem_type(MEMORY_TYPE_HBM);
  plan.set_use_block_chunks(true);
  plan.set_parallelism(1);
  for (int32_t pool_idx : transferred_pools) {
    plan.add_transfer_pool_indices(pool_idx);
  }
  for (const std::string& tag : dtype_tags) {
    plan.add_pool_dtype_tags(tag);
  }
  auto* group = plan.add_pool_groups();
  for (int32_t pool_idx : transferred_pools) {
    group->add_pool_indices(pool_idx);
  }
  group->add_dst_device_block_ids(0);
  group->set_expected_pushes(1);
  group->add_dst_expected_extent_bytes(16);
  auto* entry = (*plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(16);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);
  return plan;
}

void ExpectInvalid(const absl::Status& status, const std::string& fragment) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
      << status.ToString();
  EXPECT_NE(std::string(status.message()).find(fragment), std::string::npos)
      << status.ToString();
}

TEST(PoolReshardValidationTest, AcceptsCanonicalPlanOnExplicitPools) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/true)
                  .ok());
}

TEST(PoolReshardValidationTest, AcceptsImplicitPools) {
  // A manager that never registered explicit pools exposes one implicit
  // opaque pool per storage (dtype_tag ""); a legacy whole-manager transfer
  // is expressible against it (N5).
  TestManager manager;
  StartTransferRequest plan = ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
}

TEST(PoolReshardValidationTest, HasNoTagPolicy) {
  // Pool selection is request data resolved by the controller. The executor
  // validates geometry and consistency for whatever pool set the plan
  // declares; no tag value is special-cased.
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("gdn.conv")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1003);

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
}

TEST(PoolReshardValidationTest, DeviceOnlyRejectionAtPublicEntryPoints) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1004);

  const absl::Status recv_status =
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0});
  EXPECT_EQ(recv_status.code(), absl::StatusCode::kFailedPrecondition)
      << recv_status.ToString();
  EXPECT_NE(std::string(recv_status.message()).find("device-attached"),
            std::string::npos);

  const absl::Status push_status =
      manager.PoolReshardPush(plan, std::vector<int64_t>{0});
  EXPECT_EQ(push_status.code(), absl::StatusCode::kFailedPrecondition)
      << push_status.ToString();
}

TEST(PoolReshardValidationTest, RejectsMissingIdentityAndPoolFields) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1005);
  plan.clear_req_id();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1006);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "transfer_pool_indices");

  plan = ValidPlan(/*uuid=*/1007);
  plan.clear_pool_groups();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "must declare pool_groups");

  plan = ValidPlan(/*uuid=*/1008);
  plan.set_use_block_chunks(false);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "use_block_chunks");
}

TEST(PoolReshardValidationTest, RejectsOutOfRangeDuplicateAndDtypeMismatch) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1009, /*dtype_tags=*/{"bf16"},
                                        /*transferred_pools=*/{1});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "out of range");

  plan = ValidPlan(/*uuid=*/1010, /*dtype_tags=*/{"bf16"},
                   /*transferred_pools=*/{0, 0});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "duplicate transfer pool index");

  plan = ValidPlan(/*uuid=*/1011, /*dtype_tags=*/{"fp8"});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "dtype tag mismatch");

  plan = ValidPlan(/*uuid=*/1012, /*dtype_tags=*/{"bf16", "bf16"});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "one dtype tag per pool");
}

TEST(PoolReshardValidationTest, ChecksEveryPoolSpanAndDestinationZeroCover) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa", 128), DensePool("fa", 64)}).ok());
  StartTransferRequest plan =
      ValidPlan(/*uuid=*/1013, /*dtype_tags=*/{"bf16", "bf16"},
                /*transferred_pools=*/{0, 1});
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(48);
  entry->set_size_bytes(32);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "destination span exceeds declared pool 1");

  plan = ValidPlan(/*uuid=*/1014, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(16);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "no transfer entry starting at offset 0");

  plan = ValidPlan(/*uuid=*/1015, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "no entries");
}

TEST(PoolReshardValidationTest, RejectsOverflowingSenderSpan) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1016);
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(96);
  entry->set_src_stride_bytes(std::numeric_limits<int64_t>::max());
  entry->set_dst_stride_bytes(1);
  entry->set_size_bytes(16);
  entry->set_count(2);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/true),
                "source span exceeds declared pool 0");
}

TEST(PoolReshardValidationTest,
     RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage) {
  TestManager manager;
  kv_cache::PoolSpec pool = DensePool("fa");
  pool.regions = {
      kv_cache::RegionSpec{
          .name = "head_group_0",
          .offset_bytes = 0,
          .stride_bytes = 8,
          .unit_bytes = 4,
          .num_units = 4,
          .units_per_stride = 2,
      },
      kv_cache::RegionSpec{
          .name = "head_group_1",
          .offset_bytes = 64,
          .stride_bytes = 8,
          .unit_bytes = 4,
          .num_units = 4,
          .units_per_stride = 2,
      },
  };
  ASSERT_TRUE(manager.RegisterPools({pool}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1019);
  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/true)
                  .ok());
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(32);
  entry->set_size_bytes(16);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/true),
                "source span exceeds declared pool 0 live regions");
}

ShardPushEntryProto* AddEntry(StartTransferRequest& plan, int32_t schedule_key,
                              int64_t dst_block_id, int64_t dst_offset,
                              int64_t size, int32_t group_idx = 0) {
  auto* entry =
      (*plan.mutable_shard_push_schedules())[schedule_key].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(dst_block_id);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(dst_offset);
  entry->set_size_bytes(size);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);
  entry->set_pool_group(group_idx);
  return entry;
}

TEST(PoolReshardReceiverCoverageTest, RejectsCoverageGap) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1101);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(48);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/32, /*size=*/16);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "coverage gap");
  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/true)
                  .ok());
}

TEST(PoolReshardReceiverCoverageTest, RejectsCoverageOverlap) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1102);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/8, /*size=*/16);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "coverage overlap");
}

TEST(PoolReshardReceiverCoverageTest, RejectsShortCoverage) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1103);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "does not cover the exact live bytes");
}

TEST(PoolReshardReceiverCoverageTest, RejectsNonPrefixExtents) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1104);
  auto* group = plan.mutable_pool_groups(0);
  group->add_dst_device_block_ids(1);
  group->clear_dst_expected_extent_bytes();
  group->add_dst_expected_extent_bytes(16);
  group->add_dst_expected_extent_bytes(16);
  AddEntry(plan, 0, /*dst_block_id=*/1, /*dst_offset=*/0, /*size=*/16);

  ExpectInvalid(
      manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0, 1},
                                      /*is_sender=*/false),
      "except the final one");
}

TEST(PoolReshardReceiverCoverageTest, RejectsMissingExtents) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1105);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "require dst_expected_extent_bytes");
}

TEST(PoolReshardReceiverCoverageTest, RejectsMissingParallelism) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1106);
  plan.set_parallelism(0);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "positive parallelism");
}

TEST(PoolReshardReceiverCoverageTest, RejectsExpectedPushMismatch) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1107);
  plan.mutable_pool_groups(0)->set_expected_pushes(2);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "do not match the received schedules");
}

TEST(PoolReshardReceiverCoverageTest, RejectsWriteBeyondDeclaredLiveTail) {
  // Two live runs of 32 bytes each (compact-live size 64) with padding in
  // between: a write that FITS the physical regions can still exceed the
  // block's declared compact-live extent.
  TestManager manager;
  kv_cache::PoolSpec pool = DensePool("fa");
  pool.regions = {
      kv_cache::RegionSpec{
          .name = "run_0",
          .offset_bytes = 0,
          .stride_bytes = 32,
          .unit_bytes = 32,
          .num_units = 1,
          .units_per_stride = 1,
      },
      kv_cache::RegionSpec{
          .name = "run_1",
          .offset_bytes = 64,
          .stride_bytes = 32,
          .unit_bytes = 32,
          .num_units = 1,
          .units_per_stride = 1,
      },
  };
  ASSERT_TRUE(manager.RegisterPools({pool}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1108);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(40);
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/0, /*size=*/32);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/64, /*size=*/16);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "declared live tail");

  // The same physical writes bounded to the declared 40 compact-live bytes
  // pass: [0, 32) in run_0 plus [64, 72) = compact [32, 40) in run_1.
  plan.mutable_shard_push_schedules()->at(0).mutable_entries(1)->set_size_bytes(
      8);
  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
}

TEST(PoolReshardReceiverCoverageTest, GroupScopesDestinationBlocks) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa", 128), DensePool("gdn", 64)}).ok());
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1109, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  plan.clear_pool_groups();
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  AddEntry(plan, 0, /*dst_block_id=*/2, /*dst_offset=*/0, /*size=*/16,
           /*group_idx=*/0);
  AddEntry(plan, 0, /*dst_block_id=*/3, /*dst_offset=*/0, /*size=*/8,
           /*group_idx=*/1);
  auto* fa_group = plan.add_pool_groups();
  fa_group->add_pool_indices(0);
  fa_group->add_dst_device_block_ids(2);
  fa_group->set_expected_pushes(1);
  fa_group->add_dst_expected_extent_bytes(16);
  auto* state_group = plan.add_pool_groups();
  state_group->add_pool_indices(1);
  state_group->add_dst_device_block_ids(3);
  state_group->set_expected_pushes(1);
  state_group->add_dst_expected_extent_bytes(8);
  state_group->set_order_rank(1);

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{2, 3},
                                           /*is_sender=*/false)
                  .ok());

  // An entry naming a destination block of the OTHER group is rejected even
  // though the id exists in the plan's flat destination list.
  AddEntry(plan, 0, /*dst_block_id=*/2, /*dst_offset=*/0, /*size=*/8,
           /*group_idx=*/1);
  ExpectInvalid(
      manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{2, 3},
                                      /*is_sender=*/false),
      "outside its group");
}

TEST(PoolReshardValidationTest, RejectsBlockIdsOutsideDeclaredPool) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1017);
  plan.mutable_pool_groups(0)->set_dst_device_block_ids(0, 9);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{9},
                                                /*is_sender=*/false),
                "out of range for pool");

  plan = ValidPlan(/*uuid=*/1018);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{8},
                                                /*is_sender=*/false),
                "must concatenate to the plan's local block ids");
  ExpectInvalid(
      manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0, 0},
                                      /*is_sender=*/false),
      "must cover the plan's local block ids");
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{},
                                                /*is_sender=*/false),
                "must not be empty");
}

// Extends ValidPlan's single-group shape with a second group holding pool 1;
// the base plan's only schedule entry stays in group 0.
StartTransferRequest TwoGroupPlan(int64_t uuid) {
  StartTransferRequest plan = ValidPlan(uuid, /*dtype_tags=*/{"bf16", "bf16"},
                                        /*transferred_pools=*/{0});
  auto* group = plan.add_pool_groups();
  group->add_pool_indices(1);
  group->add_dst_device_block_ids(0);
  group->set_expected_pushes(1);
  group->add_dst_expected_extent_bytes(16);
  return plan;
}

TEST(PoolReshardSendTest, SenderWithNoBytesForAnyTransferredPoolCompletes) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa"), DensePool("state")}).ok());
  manager.AttachPlaceholderDeviceHold();

  // The sender's only schedule entry names group 1, while the transfer set
  // holds just pool 0 (group 0) — the shape of a PCP rank owning no bytes of
  // any transferred pool. Such a sender must finish as done_sending with no
  // device work; failing the plan here is the regression that turned a
  // short-prefix transfer into a plan-wide INVALID_ARGUMENT.
  StartTransferRequest plan = TwoGroupPlan(/*uuid=*/2004);
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_pool_group(
      1);

  const absl::Status push_status =
      manager.PoolReshardPush(plan, std::vector<int64_t>{0});
  ASSERT_TRUE(push_status.ok()) << push_status.ToString();

  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_EQ(done_sending, std::vector<std::string>{plan.req_id()});
  EXPECT_TRUE(done_recving.empty()) << done_recving.size();
  EXPECT_TRUE(failed_recving.empty()) << failed_recving.size();
}

}  // namespace
}  // namespace tpu_raiden
