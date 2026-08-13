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

#include "tpu_sync/kv_cache/kv_cache_metadata_shm.h"

#include <sys/mman.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

// Names a per-test shm segment and unlinks it on scope exit (and up front,
// in case a previously crashed run left one behind).
class ScopedShmKey {
 public:
  explicit ScopedShmKey(absl::string_view test_name)
      : key_(absl::StrCat("/raiden_md_shm_test_", getpid(), "_", test_name)) {
    shm_unlink(key_.c_str());
  }
  ~ScopedShmKey() { shm_unlink(key_.c_str()); }

  const std::string& key() const { return key_; }

 private:
  std::string key_;
};

TEST(KVCacheMetadataShmTest, ColdStartFormatsEmptyTable) {
  ScopedShmKey key("cold");
  auto region_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(region_or.ok()) << region_or.status().ToString();
  EXPECT_FALSE((*region_or)->warm());
  EXPECT_THAT((*region_or)->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, TableSurvivesProcessRestart) {
  ScopedShmKey key("restart");
  auto region_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(region_or.ok()) << region_or.status().ToString();
  KVCacheMetadata metadata = (*region_or)->metadata();
  ASSERT_TRUE(metadata.Set(1, "hash_b", /*seq=*/2).ok());
  ASSERT_TRUE(metadata.Set(3, "hash_d", /*seq=*/1).ok());

  // Dropping the region simulates the engine dying: the mapping goes away,
  // the segment survives.
  region_or->reset();

  auto revived_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(revived_or.ok()) << revived_or.status().ToString();
  EXPECT_TRUE((*revived_or)->warm());
  EXPECT_THAT(
      (*revived_or)->metadata().ValidEntries(),
      ElementsAre(FieldsAre(1, "hash_b", 2), FieldsAre(3, "hash_d", 1)));
}

TEST(KVCacheMetadataShmTest, NumBlocksMismatchRecreates) {
  ScopedShmKey key("num_blocks");
  auto region_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(region_or.ok()) << region_or.status().ToString();
  KVCacheMetadata metadata = (*region_or)->metadata();
  ASSERT_TRUE(metadata.Set(1, "hash_b", /*seq=*/2).ok());
  region_or->reset();

  // A different table geometry must not resurrect the old entries, whether
  // the surviving segment is too small (grown table) or its header disagrees
  // (shrunk table).
  auto grown_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 64, "model_a");
  ASSERT_TRUE(grown_or.ok()) << grown_or.status().ToString();
  EXPECT_FALSE((*grown_or)->warm());
  EXPECT_THAT((*grown_or)->metadata().ValidEntries(), IsEmpty());
  region_or = std::move(grown_or);
  KVCacheMetadata grown_metadata = (*region_or)->metadata();
  ASSERT_TRUE(grown_metadata.Set(1, "hash_b", /*seq=*/2).ok());
  region_or->reset();

  auto shrunk_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(shrunk_or.ok()) << shrunk_or.status().ToString();
  EXPECT_FALSE((*shrunk_or)->warm());
  EXPECT_THAT((*shrunk_or)->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, ModelUidMismatchRecreates) {
  ScopedShmKey key("model_uid");
  auto region_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(region_or.ok()) << region_or.status().ToString();
  KVCacheMetadata metadata = (*region_or)->metadata();
  ASSERT_TRUE(metadata.Set(1, "hash_b", /*seq=*/2).ok());
  region_or->reset();

  // A table recorded under another model must not resurrect its bindings.
  auto other_model_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_b");
  ASSERT_TRUE(other_model_or.ok()) << other_model_or.status().ToString();
  EXPECT_FALSE((*other_model_or)->warm());
  EXPECT_THAT((*other_model_or)->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, ReformatErasesEntries) {
  ScopedShmKey key("reformat");
  auto region_or =
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a");
  ASSERT_TRUE(region_or.ok()) << region_or.status().ToString();
  KVCacheMetadata metadata = (*region_or)->metadata();
  ASSERT_TRUE(metadata.Set(1, "hash_b", /*seq=*/2).ok());

  ASSERT_TRUE((*region_or)->Reformat().ok());
  EXPECT_FALSE((*region_or)->warm());
  EXPECT_THAT((*region_or)->metadata().ValidEntries(), IsEmpty());
  // Copies attached before the reformat observe the emptied table.
  EXPECT_THAT(metadata.ValidEntries(), IsEmpty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
