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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

// Holds an in-process GlobalRegistry service, gRPC server, channel, and client.
struct TestGlobalRegistryServer {
  std::unique_ptr<GlobalRegistryServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<GlobalRegistryClient> client;

  ~TestGlobalRegistryServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestGlobalRegistryServer hosting
// GlobalRegistryServiceImpl on an ephemeral port ("localhost:0").
inline std::unique_ptr<TestGlobalRegistryServer> CreateTestGlobalRegistryServer(
    absl::Duration default_ttl = absl::Seconds(60),
    absl::Duration cleanup_interval = absl::Seconds(10),
    int64_t pull_owned_batch_size =
        GlobalRegistryServiceImpl::kDefaultPullOwnedBatchSize) {
  auto test_server = std::make_unique<TestGlobalRegistryServer>();
  test_server->service = std::make_unique<GlobalRegistryServiceImpl>(
      default_ttl, cleanup_interval, pull_owned_batch_size);

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(test_server->service.get());
  test_server->server = builder.BuildAndStart();

  test_server->server_address = absl::StrCat("localhost:", selected_port);
  test_server->channel = grpc::CreateChannel(
      test_server->server_address, grpc::InsecureChannelCredentials());
  test_server->client =
      std::make_unique<GlobalRegistryClient>(test_server->channel);
  return test_server;
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_TEST_UTIL_H_
