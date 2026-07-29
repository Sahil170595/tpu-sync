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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

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

  GlobalMemoryPoolingBackend backend(registry_client, remote_node_id);
  EXPECT_EQ(backend.name(), "GlobalMemoryPoolingBackend");

  auto lookup_res = backend.Lookup({"r_hash1", "r_hash2"});
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
  auto partial_res = backend.Lookup({"r_hash1", "missing_hash"});
  ASSERT_TRUE(partial_res.ok());
  EXPECT_EQ(partial_res->size(), 1);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
