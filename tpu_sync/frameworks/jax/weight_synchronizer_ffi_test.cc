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

#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

// Keep ffi buffers alive during test scope
struct FfiBufferFixture {
  XLA_FFI_Buffer ffi_buf;
  std::vector<int64_t> dims;

  FfiBufferFixture(XLA_FFI_DataType dtype, void* data,
                   std::vector<int64_t> dimensions)
      : dims(dimensions) {
    ffi_buf.struct_size = sizeof(XLA_FFI_Buffer);
    ffi_buf.extension_start = nullptr;
    ffi_buf.dtype = dtype;
    ffi_buf.data = data;
    ffi_buf.rank = dims.size();
    ffi_buf.dims = dims.data();
  }

  xla::ffi::AnyBuffer AsAnyBuffer() { return xla::ffi::AnyBuffer(&ffi_buf); }
};

class WeightSynchronizerFfiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure Host platform is registered and active
    auto platform = stream_executor::PlatformManager::PlatformWithName("Host");
    ASSERT_TRUE(platform.ok())
        << "Host Platform must be available on CPU sandboxes";
  }

  void TearDown() override {
    // Clean up global registry between tests
    absl::flat_hash_set<WeightSynchronizerBase*> deleted;
    for (int i = 0; i < 32; ++i) {
      if (g_weight_synchronizers[i] != nullptr) {
        if (deleted.insert(g_weight_synchronizers[i]).second) {
          delete g_weight_synchronizers[i];
        }
        g_weight_synchronizers[i] = nullptr;
      }
      g_streams[i].reset();
    }
    ClearSharedWsMap();
  }
};

enum class FfiType { kInit, kInitAndD2h };

class WeightSynchronizerFfiParamTest : public WeightSynchronizerFfiTest,
                                       public ::testing::WithParamInterface<FfiType> {
 protected:
  xla::ffi::Error CallInit(xla::ffi::AnyBuffer x,
                           xla::ffi::AnyBuffer shard_idx_buf,
                           xla::ffi::AnyBuffer slice_byte_sizes_buf,
                           int32_t local_port, int32_t parallelism,
                           int32_t num_layers,
                           xla::ffi::Result<xla::ffi::AnyBuffer> out,
                           int32_t listener_port = 0, int32_t num_shards = 1) {
    if (GetParam() == FfiType::kInit) {
      return TriggerWeightSynchronizerInitImpl(
          x, shard_idx_buf, slice_byte_sizes_buf, local_port, parallelism,
          num_layers, listener_port, num_shards, out);
    } else {
      std::vector<xla::ffi::AnyBuffer> jax_arrays = {x};
      return TriggerWeightSynchronizerInitAndD2hHelper(
          shard_idx_buf, slice_byte_sizes_buf, jax_arrays, local_port,
          parallelism, num_layers, listener_port, num_shards, out);
    }
  }
};

TEST_P(WeightSynchronizerFfiParamTest, TriggerWSInitSucceeds) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});

  std::vector<int32_t> anchor(256);
  for (int i = 0; i < 256; ++i) anchor[i] = i + 1;
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});

  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  int32_t slice_byte_size = 1024;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &slice_byte_size, {1});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  int32_t local_port = 0;  // Allocate dynamic free port
  int32_t parallelism = 1;
  int32_t num_layers = 1;

  // Output buffer for IP (16 bytes) and port (4 bytes) -> 20 bytes -> 5 int32
  std::vector<int32_t> out_data(5, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {5});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err = CallInit(x, shard_idx_buf, slice_byte_sizes_buf,
                                 local_port, parallelism, num_layers, out);

  EXPECT_TRUE(err.success()) << "WS Init failed: " << err.message();
  EXPECT_NE(g_weight_synchronizers[0], nullptr);
  EXPECT_NE(g_streams[0], nullptr);

  // Verify output contains some IP (can be all zeros if no external IP, but
  // port should be > 0)
  int32_t port = out_data[4];
  EXPECT_GT(port, 0) << "Assigned port should be positive";
  VLOG(1) << "Assigned port: " << port;
}

TEST_F(WeightSynchronizerFfiTest, TriggerMultiH2DSucceeds) {
  // 1. Initialize WeightSynchronizer
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  // We will have 2 layers, each of size 1024 bytes (256 int32)
  std::vector<int32_t> slice_byte_sizes = {1024, 1024};
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, slice_byte_sizes.data(),
                                 {2});
  xla::ffi::AnyBuffer slice_byte_sizes_buf = sizes_fixture.AsAnyBuffer();

  // Anchor is not really used in Init (Host mode) except to pass to
  // TriggerWSInit We can just pass dummy anchor
  std::vector<int32_t> dummy_anchor(1);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, dummy_anchor.data(),
                                  {1});
  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();

  // Output buffer for Init
  std::vector<int32_t> out_data(6, 0);  // Use 6 to get listener_port too
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {6});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err = TriggerWeightSynchronizerInitImpl(
      x, shard_idx_buf, slice_byte_sizes_buf,
      /*local_port=*/0, /*parallelism=*/1, /*num_layers=*/2,
      /*listener_port=*/0, /*num_shards=*/1, out);
  ASSERT_TRUE(err.success()) << "Init failed: " << err.message();

  WeightSynchronizerBase* ws = g_weight_synchronizers[shard_idx];
  ASSERT_NE(ws, nullptr);

  // 2. Populate Host Staging Buffers with dummy data
  // Layer 0: [0, 1, 2, ... 255]
  // Layer 1: [256, 257, ... 511]
  uint8_t* host_ptr_0 = const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 0));
  uint8_t* host_ptr_1 = const_cast<uint8_t*>(ws->GetHostBufferPtr(1, 0));
  ASSERT_NE(host_ptr_0, nullptr);
  ASSERT_NE(host_ptr_1, nullptr);

  std::vector<int32_t> src_data_0(256);
  std::vector<int32_t> src_data_1(256);
  for (int i = 0; i < 256; ++i) {
    src_data_0[i] = i;
    src_data_1[i] = i + 256;
  }
  std::memcpy(host_ptr_0, src_data_0.data(), 1024);
  std::memcpy(host_ptr_1, src_data_1.data(), 1024);

  // 3. Prepare Output Buffers for H2D
  std::vector<int32_t> dst_data_0(256, 0);
  std::vector<int32_t> dst_data_1(256, 0);
  FfiBufferFixture dst_fixture_0(XLA_FFI_DataType_S32, dst_data_0.data(),
                                 {256});
  FfiBufferFixture dst_fixture_1(XLA_FFI_DataType_S32, dst_data_1.data(),
                                 {256});

  XLA_FFI_Buffer ffi_buf_0 = dst_fixture_0.ffi_buf;
  XLA_FFI_Buffer ffi_buf_1 = dst_fixture_1.ffi_buf;

  XLA_FFI_RetType types[] = {XLA_FFI_RetType_BUFFER, XLA_FFI_RetType_BUFFER};
  void* rets_ptrs[] = {&ffi_buf_0, &ffi_buf_1};

  XLA_FFI_Rets rets;
  rets.struct_size = sizeof(XLA_FFI_Rets);
  rets.extension_start = nullptr;
  rets.size = 2;
  rets.types = types;
  rets.rets = rets_ptrs;

  xla::ffi::RemainingRets remaining_rets(&rets, 0);

  // 4. Call TriggerMultiH2DImpl
  err = TriggerMultiH2DImpl(shard_idx_buf, remaining_rets);
  EXPECT_TRUE(err.success()) << "MultiH2D failed: " << err.message();

  // 5. Verify data was copied
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(dst_data_0[i], src_data_0[i])
        << "Mismatch at layer 0 index " << i;
    EXPECT_EQ(dst_data_1[i], src_data_1[i])
        << "Mismatch at layer 1 index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(FfiTypeTests, WeightSynchronizerFfiParamTest,
                         ::testing::Values(FfiType::kInit, FfiType::kInitAndD2h));

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
