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

// Unit tests for GlobalMemoryPoolingBackend.
#include "tpu_raiden/kv_cache/global_memory_pooling_backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::UnorderedElementsAre;

TEST(GlobalMemoryPoolingBackendTest, LookupReturnsRemoteDescriptors) {
  // Setup local gRPC registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);

  RaidenId remote_node_id{"remote_job", "1", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "r_hash1", .raiden_id = remote_node_id, .block_id = 42},
      {.prefix_hash = "r_hash2", .raiden_id = remote_node_id, .block_id = 43},
  };
  ASSERT_TRUE(registry_client->Register(regs).ok());

  RaidenId local_node_id{"local_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  controller::RaidenController controller(unit_proto, /*num_blocks=*/100,
                                          /*num_shards=*/1,
                                          /*shard_size_bytes=*/1024);

  BackendConfig config;
  config.type = "GlobalMemoryPoolingBackend";
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  auto backend_or = GlobalMemoryPoolingBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = *backend_or;
  EXPECT_EQ(backend->name(), "GlobalMemoryPoolingBackend");

  auto lookup_res = backend->Lookup({"r_hash1", "r_hash2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "r_hash1");
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 42);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, remote_node_id);

  EXPECT_EQ((*lookup_res)[1].first, "r_hash2");
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 43);

  // Lookup with miss stops at miss
  auto partial_res = backend->Lookup({"r_hash1", "missing_hash"});
  ASSERT_TRUE(partial_res.ok());
  EXPECT_EQ(partial_res->size(), 1);

  server->Shutdown();
}

TEST(GlobalMemoryPoolingBackendTest,
     ServerLifecycleAndControllerInitialization) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  RaidenId node_id{"node_job", "0", "data", 0};
  BackendConfig config;
  config.type = "GlobalMemoryPoolingBackend";
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  // Create RaidenController
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  controller::RaidenController controller(unit_proto, /*num_blocks=*/100,
                                          /*num_shards=*/1,
                                          /*shard_size_bytes=*/1024);

  auto backend_or = GlobalMemoryPoolingBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);
  EXPECT_GT(backend->GetGrpcPort(), 0);
  EXPECT_FALSE(backend->GetServerAddress().empty());

  server->Shutdown();
}

TEST(GlobalMemoryPoolingBackendTest, StartServerStripsControllerPort) {
  RaidenId node_id{"node_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  controller::RaidenController controller(
      unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
      /*shard_size_bytes=*/1024, /*raiden_orchestrator_address=*/"",
      /*raiden_controller_address=*/"127.0.0.1:12345");

  BackendConfig config;
  config.type = "GlobalMemoryPoolingBackend";
  config.global_registry_address = "localhost:0";
  config.raiden_id = node_id;

  auto backend_or = GlobalMemoryPoolingBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);
  EXPECT_GT(backend->GetGrpcPort(), 0);
  EXPECT_NE(backend->GetGrpcPort(), 12345);
}

TEST(GlobalMemoryPoolingBackendTest, EndToEndFetchRPC) {
  // 1. Setup global registry server
  auto reg_service =
      std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder reg_builder;
  int reg_port = 0;
  reg_builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                               &reg_port);
  reg_builder.RegisterService(reg_service.get());
  auto reg_server = reg_builder.BuildAndStart();
  std::string reg_address = "localhost:" + std::to_string(reg_port);

  auto reg_channel =
      grpc::CreateChannel(reg_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(reg_channel);

  // 2. Setup mock worker server & transfer manager
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  // 3. Setup orchestrator server
  auto orchestrator_service = std::make_unique<RaidenOrchestrator>();
  grpc::ServerBuilder orch_builder;
  int orch_port = 0;
  orch_builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                                &orch_port);
  orch_builder.RegisterService(orchestrator_service.get());
  auto orchestrator_server = orch_builder.BuildAndStart();
  std::string orchestrator_address = "localhost:" + std::to_string(orch_port);

  // 4. Setup src controller server
  auto src_controller_server = core::controller::CreateTestControllerServer();

  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  RaidenId dst_raiden_id{"dst_job", "0", "dst_data", 0};

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name(src_raiden_id.job_name);
  src_unit.set_job_replica_id(src_raiden_id.job_replica_id);
  src_unit.set_data_name(src_raiden_id.data_name);
  src_unit.set_data_replica_idx(src_raiden_id.data_replica_idx);

  controller::OrchestratorServiceClient orchestrator_client(grpc::CreateChannel(
      orchestrator_address, grpc::InsecureChannelCredentials()));
  ASSERT_OK(orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address));

  ASSERT_OK(src_controller_server->client->RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});

  // 5. Register remote blocks in GlobalRegistry
  std::vector<global_registry::Registration> registrations = {
      {.prefix_hash = "fetch_hash_1",
       .raiden_id = dst_raiden_id,
       .block_id = 101},
      {.prefix_hash = "fetch_hash_2",
       .raiden_id = dst_raiden_id,
       .block_id = 102},
  };
  ASSERT_OK(registry_client->Register(registrations));

  // 6. Create destination GlobalMemoryPoolingBackend & RaidenController
  rpc::RaidenIdProto dst_unit_proto;
  dst_unit_proto.set_job_name(dst_raiden_id.job_name);
  dst_unit_proto.set_job_replica_id(dst_raiden_id.job_replica_id);
  dst_unit_proto.set_data_name(dst_raiden_id.data_name);
  dst_unit_proto.set_data_replica_idx(dst_raiden_id.data_replica_idx);

  controller::RaidenController dst_controller(
      dst_unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
      /*shard_size_bytes=*/1024, orchestrator_address,
      /*raiden_controller_address=*/"");

  BackendConfig dst_config;
  dst_config.type = "GlobalMemoryPoolingBackend";
  dst_config.global_registry_address = reg_address;
  dst_config.raiden_id = dst_raiden_id;

  auto backend_or =
      GlobalMemoryPoolingBackend::Create(dst_config, &dst_controller);
  ASSERT_OK(backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  core::controller::RaidenControllerClient dst_controller_client(
      dst_controller.controller_address());
  ASSERT_OK(dst_controller_client.RegisterWorker(
      "dst_worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  EXPECT_GT(backend->GetGrpcPort(), 0);

  // 7. Issue Fetch RPC using KVCacheStoreClient
  auto client_channel = grpc::CreateChannel(backend->GetServerAddress(),
                                            grpc::InsecureChannelCredentials());
  KVCacheStoreClient client(client_channel);

  std::vector<std::string> hashes = {"fetch_hash_1", "fetch_hash_2"};
  std::vector<int32_t> host_ids = {201, 202};
  auto fetch_res = client
                       .Fetch(hashes, /*device_block_ids=*/{}, host_ids,
                              dst_controller.unit())
                       .Await();
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(fetch_res->done_block_hashes(),
              UnorderedElementsAre("fetch_hash_1", "fetch_hash_2"));

  orchestrator_server->Shutdown();
  reg_server->Shutdown();
}


TEST(GlobalMemoryPoolingBackendTest, LoadMismatchedDeviceBlockCount) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);
  RaidenId node_id{"node_job", "0", "data", 0};

  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  controller::RaidenController controller(unit_proto, /*num_blocks=*/100,
                                          /*num_shards=*/1,
                                          /*shard_size_bytes=*/1024);
  BackendConfig config;
  config.type = "GlobalMemoryPoolingBackend";
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  auto backend_or = GlobalMemoryPoolingBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  std::vector<std::string> hashes = {"hash1", "hash2"};
  std::vector<int32_t> dev_ids = {10};  // Mismatched count
  auto load_future = backend->Load(node_id, hashes, dev_ids);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  server->Shutdown();
}

TEST(GlobalMemoryPoolingBackendTest, LoadSuccess) {
  // Setup GlobalRegistry and register remote block
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);

  RaidenId remote_node_id{"remote_job", "0", "remote_data", 0};
  RaidenId local_node_id{"local_job", "0", "local_data", 0};

  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "load_hash_1",
       .raiden_id = remote_node_id,
       .block_id = 42},
  };
  ASSERT_OK(registry_client->Register(regs));

  // Setup local RaidenController
  rpc::RaidenIdProto local_unit;
  local_unit.set_job_name(local_node_id.job_name);
  local_unit.set_job_replica_id(local_node_id.job_replica_id);
  local_unit.set_data_name(local_node_id.data_name);
  local_unit.set_data_replica_idx(local_node_id.data_replica_idx);

  controller::RaidenController controller(local_unit, /*num_blocks=*/100,
                                          /*num_shards=*/1,
                                          /*shard_size_bytes=*/1024);

  // Setup fake server for remote node to process Fetch
  BackendConfig remote_config;
  remote_config.type = "GlobalMemoryPoolingBackend";
  remote_config.global_registry_address = server_address;
  remote_config.raiden_id = remote_node_id;

  auto remote_backend_or =
      GlobalMemoryPoolingBackend::Create(remote_config, &controller);
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  BackendConfig local_config;
  local_config.type = "GlobalMemoryPoolingBackend";
  local_config.global_registry_address = server_address;
  local_config.raiden_id = local_node_id;

  auto local_backend_or =
      GlobalMemoryPoolingBackend::Create(local_config, &controller);
  ASSERT_OK(local_backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<GlobalMemoryPoolingBackend>(*local_backend_or);
  ASSERT_NE(backend, nullptr);

  // Inject client connected directly to remote server
  auto client_channel = grpc::CreateChannel(remote_backend->GetServerAddress(),
                                            grpc::InsecureChannelCredentials());
  auto store_client = std::make_shared<KVCacheStoreClient>(client_channel);
  backend->SetStoreClient(remote_node_id, store_client);

  // Register local worker in controller
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller.controller_address());
  ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  // Perform Load
  std::vector<std::string> hashes = {"load_hash_1"};
  std::vector<int32_t> dev_ids = {5};
  auto load_future = backend->Load(remote_node_id, hashes, dev_ids);
  EXPECT_OK(load_future.Await());

  server->Shutdown();
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
