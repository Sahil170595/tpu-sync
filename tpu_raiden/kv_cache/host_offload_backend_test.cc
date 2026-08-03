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
#include "grpcpp/security/server_credentials.h"
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
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::UnorderedElementsAre;

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

TEST(HostOffloadBackendTest, LookupReturnsRemoteDescriptors) {
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
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  auto backend_or = HostOffloadBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = *backend_or;
  EXPECT_EQ(backend->name(), "HostOffloadBackend");

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

TEST(HostOffloadBackendTest, ServerLifecycleAndControllerInitialization) {
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
  config.type = "HostOffloadBackend";
  config.capacity = 100;
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

  auto backend_or = HostOffloadBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ASSERT_OK(store_server->StartServer(backend.get(), &controller, "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_FALSE(store_server->GetServerAddress().empty());
  store_server->Shutdown();

  server->Shutdown();
}

TEST(HostOffloadBackendTest, StartServerStripsControllerPort) {
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
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = "localhost:0";
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  std::string ctrl_addr = controller.controller_address();
  std::string target_host = ctrl_addr.substr(0, ctrl_addr.rfind(':'));
  ASSERT_OK(store_server->StartServer(backend.get(), &controller, target_host));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_NE(store_server->GetGrpcPort(), 12345);
  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, EndToEndFetchRPC) {
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

  // 6. Create destination HostOffloadBackend & RaidenController
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
  dst_config.type = "HostOffloadBackend";
  dst_config.capacity = 100;
  dst_config.global_registry_address = reg_address;
  auto backend_or = HostOffloadBackend::Create(dst_config, &dst_controller);
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  std::vector<RaidenBlockID> dst_slices = {
      RaidenBlockID(dst_raiden_id, 101, BlockStatus::HOST),
      RaidenBlockID(dst_raiden_id, 102, BlockStatus::HOST),
  };
  backend->Insert({"fetch_hash_1", "fetch_hash_2"}, dst_slices,
                  /*on_host=*/true);

  core::controller::RaidenControllerClient dst_controller_client(
      dst_controller.controller_address());
  ASSERT_OK(dst_controller_client.RegisterWorker(
      "dst_worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ASSERT_OK(
      store_server->StartServer(backend.get(), &dst_controller, "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);

  // 7. Issue Fetch RPC using KVCacheStoreClient
  auto client_channel = grpc::CreateChannel(store_server->GetServerAddress(),
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

  store_server->Shutdown();
  orchestrator_server->Shutdown();
  reg_server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadMismatchedDeviceBlockCount) {
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
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  std::vector<std::string> hashes = {"hash1", "hash2"};
  std::vector<int32_t> dev_ids = {10};  // Mismatched count
  auto load_future = backend->Load(node_id, hashes, dev_ids);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadSuccess) {
  // Setup GlobalRegistry and register remote block
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // Setup orchestrator server
  auto orchestrator_service = std::make_unique<RaidenOrchestrator>();
  grpc::ServerBuilder orch_builder;
  int orch_port = 0;
  orch_builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                                &orch_port);
  orch_builder.RegisterService(orchestrator_service.get());
  auto orchestrator_server = orch_builder.BuildAndStart();
  std::string orchestrator_address = "localhost:" + std::to_string(orch_port);

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

  controller::RaidenController controller(
      local_unit, /*num_blocks=*/100, /*num_shards=*/1,
      /*shard_size_bytes=*/1024, orchestrator_address,
      /*raiden_controller_address=*/"");

  // Setup fake server for remote node to process Fetch
  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = server_address;
  remote_config.raiden_id = remote_node_id;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, &controller);
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(remote_node_id, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_hash_1"}, remote_slices, /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host -- this test publishes it below.
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), &controller,
                                       "127.0.0.1"));

  ASSERT_OK(registry_client->RegisterStore(
      remote_node_id, remote_server->GetServerAddress(), orchestrator_address));

  BackendConfig local_config;
  local_config.type = "HostOffloadBackend";
  local_config.capacity = 100;
  local_config.global_registry_address = server_address;
  local_config.raiden_id = local_node_id;

  auto local_backend_or = HostOffloadBackend::Create(local_config, &controller);
  ASSERT_OK(local_backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*local_backend_or);
  ASSERT_NE(backend, nullptr);

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

  remote_server->Shutdown();
  orchestrator_server->Shutdown();
  server->Shutdown();
}

TEST(HostOffloadBackendTest, StoreServerOverride) {
  RaidenId local_node_id{"override_job", "0", "cache", 0};
  rpc::RaidenIdProto local_unit;
  local_unit.set_job_name(local_node_id.job_name);
  local_unit.set_job_replica_id(local_node_id.job_replica_id);
  local_unit.set_data_name(local_node_id.data_name);
  local_unit.set_data_replica_idx(local_node_id.data_replica_idx);

  controller::RaidenController controller(
      local_unit, /*num_blocks=*/100, /*num_shards=*/1,
      /*shard_size_bytes=*/1024, /*raiden_orchestrator_address=*/"",
      /*raiden_controller_address=*/"");

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  auto backend_or = HostOffloadBackend::Create(config, &controller);
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->store_server(), nullptr);
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  EXPECT_NE(backend->store_server(), nullptr);
  backend->store_server()->Shutdown();
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
