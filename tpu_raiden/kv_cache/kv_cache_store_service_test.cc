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

#include "tpu_raiden/kv_cache/kv_cache_store_service.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "grpcpp/grpcpp.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/host_offload_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::UnorderedElementsAre;

class KVCacheStoreServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up test worker server and mock transfer manager
    test_worker_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    dst_transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    test_worker_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(dst_transfer_mock_.get()));

    // Set up orchestrator server
    orchestrator_service_ =
        std::make_unique<::tpu_raiden::RaidenOrchestrator>();
    ::grpc::ServerBuilder orch_builder;
    int orch_port = 0;
    orch_builder.AddListeningPort(
        "0.0.0.0:0", ::grpc::InsecureServerCredentials(), &orch_port);
    orch_builder.RegisterService(orchestrator_service_.get());
    orchestrator_server_ = orch_builder.BuildAndStart();
    std::string orchestrator_address = "localhost:" + std::to_string(orch_port);

    // Set up src controller server
    src_controller_server_ = core::controller::CreateTestControllerServer();

    RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
    RaidenId dst_raiden_id{"dst_job", "0", "dst_data", 0};

    rpc::RaidenIdProto src_unit;
    src_unit.set_job_name(src_raiden_id.job_name);
    src_unit.set_job_replica_id(src_raiden_id.job_replica_id);
    src_unit.set_data_name(src_raiden_id.data_name);
    src_unit.set_data_replica_idx(src_raiden_id.data_replica_idx);

    ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
        ::grpc::CreateChannel(orchestrator_address,
                              ::grpc::InsecureChannelCredentials()));
    ASSERT_OK(orchestrator_client.RegisterController(
        src_unit, src_controller_server_->server_address));

    ASSERT_OK(src_controller_server_->client->RegisterWorker(
        "worker_0", test_worker_server_->server_address,
        {{test_worker_server_->server_address, {}}}));

    src_controller_server_->service->SetReadRemoteHooks(
        [&](absl::Span<const std::string> h)
            -> absl::StatusOr<std::vector<int32_t>> {
          return std::vector<int32_t>(h.size(), 42);
        },
        [&](absl::Span<const std::string> /*h*/) {});

    // Create dst KVCacheStore
    store_ = std::make_unique<KVCacheStore>(
        /*capacity=*/100, orchestrator_address, dst_raiden_id, /*num_shards=*/1,
        /*shard_size_bytes=*/1024, orchestrator_address,
        src_controller_server_->server_address);

    ::tpu_raiden::core::controller::RaidenControllerClient
        dst_controller_client(store_->raiden_controller_address());
    ASSERT_OK(dst_controller_client.RegisterWorker(
        "dst_worker_0", test_worker_server_->server_address,
        {{test_worker_server_->server_address, {}}}));

    // Pre-populate and pin test blocks in store so ReadRemote succeeds
    std::vector<std::string> test_hashes = {
        "block_hash_1", "block_hash_2", "block_hash_dev_1", "block_hash_dev_2"};
    std::vector<RaidenBlockID> slices = {
        RaidenBlockID(src_raiden_id, 10, BlockStatus::REMOTE),
        RaidenBlockID(src_raiden_id, 11, BlockStatus::REMOTE),
        RaidenBlockID(src_raiden_id, 12, BlockStatus::REMOTE),
        RaidenBlockID(src_raiden_id, 13, BlockStatus::REMOTE),
    };
    ASSERT_TRUE(store_->InsertAndLock(test_hashes, slices, /*on_host=*/true));

    // Setup KVCacheStoreServiceImpl & gRPC server
    service_ = std::make_unique<KVCacheStoreServiceImpl>(store_.get());
    ::grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                             &selected_port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();

    std::string server_address = "localhost:" + std::to_string(selected_port);
    auto channel = ::grpc::CreateChannel(server_address,
                                         ::grpc::InsecureChannelCredentials());
    client_ = std::make_unique<KVCacheStoreClient>(channel);
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
    }
    if (orchestrator_server_) {
      orchestrator_server_->Shutdown();
    }
  }

  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer>
      test_worker_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      dst_transfer_mock_;
  std::unique_ptr<::tpu_raiden::RaidenOrchestrator> orchestrator_service_;
  std::unique_ptr<::grpc::Server> orchestrator_server_;
  std::unique_ptr<::tpu_raiden::core::controller::TestControllerServer>
      src_controller_server_;
  std::unique_ptr<KVCacheStore> store_;
  std::unique_ptr<KVCacheStoreServiceImpl> service_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<KVCacheStoreClient> client_;
};

TEST_F(KVCacheStoreServiceTest, FetchEmptyRequest) {
  std::vector<std::string> empty_hashes;
  auto fetch_res = client_->Fetch(empty_hashes);
  ASSERT_OK(fetch_res.status());
  EXPECT_TRUE(fetch_res->empty());
}

TEST_F(KVCacheStoreServiceTest, FetchToHostBuffer) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  // Test fetching into host DRAM buffer (device_block_ids empty)
  auto fetch_res = client_->Fetch(hashes);
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(*fetch_res, UnorderedElementsAre("block_hash_1", "block_hash_2"));
}

TEST_F(KVCacheStoreServiceTest, FetchToDeviceBuffer) {
  std::vector<std::string> hashes = {"block_hash_dev_1", "block_hash_dev_2"};
  std::vector<int32_t> device_block_ids = {100, 101};
  // Test fetching into device HBM buffer with target device block IDs
  auto fetch_res = client_->Fetch(hashes, device_block_ids);
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(*fetch_res,
              UnorderedElementsAre("block_hash_dev_1", "block_hash_dev_2"));
}

TEST_F(KVCacheStoreServiceTest, FetchMismatchedDeviceBlockCount) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> device_block_ids = {100};  // Mismatched size!
  auto fetch_res = client_->Fetch(hashes, device_block_ids);
  EXPECT_THAT(fetch_res.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(KVCacheStoreServiceTest, FetchWithValidHostBlockIdsSanityCheck) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_block_ids = {10, 20};  // Valid matching size
  auto fetch_res =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(*fetch_res, UnorderedElementsAre("block_hash_1", "block_hash_2"));
}

TEST_F(KVCacheStoreServiceTest, FetchMismatchedHostBlockCount) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_block_ids = {10};  // Mismatched size!
  auto fetch_res =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  EXPECT_THAT(fetch_res.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(KVCacheStoreServiceTest, FetchNullStoreHandling) {
  // Test server behavior when store pointer is null
  auto null_service = std::make_unique<KVCacheStoreServiceImpl>(nullptr);
  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(null_service.get());
  auto null_server = builder.BuildAndStart();

  auto null_client = std::make_unique<KVCacheStoreClient>(
      ::grpc::CreateChannel("localhost:" + std::to_string(port),
                            ::grpc::InsecureChannelCredentials()));

  std::vector<std::string> hashes = {"block_hash_1"};
  auto fetch_res = null_client->Fetch(hashes);
  EXPECT_THAT(fetch_res.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  null_server->Shutdown();
}

TEST_F(KVCacheStoreServiceTest, ConcurrentFetchRPCs) {
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  constexpr int kNumThreads = 8;
  std::vector<std::string> extra_hashes;
  std::vector<RaidenBlockID> extra_slices;
  extra_hashes.reserve(kNumThreads * 2);
  extra_slices.reserve(kNumThreads * 2);

  for (int i = 0; i < kNumThreads * 2; ++i) {
    extra_hashes.push_back("concurrent_hash_" + std::to_string(i));
    extra_slices.push_back(
        RaidenBlockID(src_raiden_id, 100 + i, BlockStatus::REMOTE));
  }
  ASSERT_TRUE(
      store_->InsertAndLock(extra_hashes, extra_slices, /*on_host=*/true));

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<absl::StatusOr<std::vector<std::string>>> results(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this, i, &results]() {
      std::vector<std::string> hashes = {
          "concurrent_hash_" + std::to_string(2 * i),
          "concurrent_hash_" + std::to_string(2 * i + 1)};
      results[i] = client_->Fetch(hashes);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < kNumThreads; ++i) {
    ASSERT_OK(results[i].status());
    EXPECT_THAT(
        *results[i],
        UnorderedElementsAre("concurrent_hash_" + std::to_string(2 * i),
                             "concurrent_hash_" + std::to_string(2 * i + 1)));
  }
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
