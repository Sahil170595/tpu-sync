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

// Unit tests for HostOffloadBackend.
#include "tpu_raiden/kv_cache/host_offload_backend.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(HostOffloadBackendTest, BasicInsertAndLookup) {
  HostOffloadBackend backend(/*capacity=*/2);
  EXPECT_EQ(backend.name(), "HostOffloadBackend");
  EXPECT_EQ(backend.GetCapacity(), 2);
  EXPECT_EQ(backend.GetSize(), 0);

  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST)};

  auto [all_new, evicted] = backend.Insert(hashes, slices, /*on_host=*/true);
  EXPECT_TRUE(all_new);
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(backend.GetSize(), 2);

  // Lookup both
  auto lookup_res = backend.Lookup({"h1", "h2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "h1");
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 10);
  EXPECT_EQ((*lookup_res)[1].first, "h2");
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 11);

  // Partial miss at end
  auto partial_res = backend.Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(partial_res.ok());
  EXPECT_EQ(partial_res->size(), 2);

  // Miss at start
  auto miss_res = backend.Lookup({"h3", "h1"});
  ASSERT_TRUE(miss_res.ok());
  EXPECT_TRUE(miss_res->empty());
}

TEST(HostOffloadBackendTest, LookupUnboundedByAvailableSpace) {
  HostOffloadBackend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST)};

  backend.Insert(hashes, slices, /*on_host=*/true);
  EXPECT_TRUE(backend.Pin(hashes));
  EXPECT_EQ(backend.GetAvailableSpace(), 0);

  // Lookup still succeeds completely despite available_space() == 0
  auto lookup_res = backend.Lookup({"h1", "h2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
}

TEST(HostOffloadBackendTest, InsertAndLockRollbackOnCapacityExceeded) {
  HostOffloadBackend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2", "h3"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST),
      RaidenBlockID(id, 12, BlockStatus::HOST)};

  // InsertAndLock for 3 items on capacity=2 must fail and rollback
  bool success = backend.InsertAndLock(hashes, slices, /*on_host=*/true);
  EXPECT_FALSE(success);
  EXPECT_EQ(backend.GetSize(), 0);

  // Partial InsertAndLock up to capacity works
  std::vector<std::string> sub_hashes = {"h1", "h2"};
  std::vector<RaidenBlockID> sub_slices = {slices[0], slices[1]};
  EXPECT_TRUE(backend.InsertAndLock(sub_hashes, sub_slices, /*on_host=*/true));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);

  // Attempt to InsertAndLock h3 should fail because available_space() is 0
  EXPECT_FALSE(backend.InsertAndLock({"h3"}, {slices[2]}, /*on_host=*/true));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
