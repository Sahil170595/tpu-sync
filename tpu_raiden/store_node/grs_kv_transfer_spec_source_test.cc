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

#include "tpu_raiden/store_node/grs_kv_transfer_spec_source.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {
namespace {

namespace gr = ::tpu_raiden::kv_cache::global_registry;

class GrsKVTransferSpecSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    service_ = std::make_unique<gr::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &port_);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    address_ = absl::StrCat("localhost:", port_);
  }

  void TearDown() override { server_->Shutdown(); }

  // Publishes a spec the way a serving host will.
  void Publish(const gr::KVTransferSpec& spec) {
    gr::GlobalRegistryClient client(grpc::CreateChannel(
        address_, grpc::InsecureChannelCredentials()));
    ASSERT_TRUE(client.RegisterKVTransferSpec(spec).ok());
  }

  std::unique_ptr<gr::GlobalRegistryServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  int port_ = 0;
  std::string address_;
};

TEST_F(GrsKVTransferSpecSourceTest, NotFoundBeforePublish) {
  GrsKVTransferSpecSource source(address_);
  EXPECT_TRUE(absl::IsNotFound(source.Get().status()));
}

TEST_F(GrsKVTransferSpecSourceTest, RoundTripsPublishedSpec) {
  gr::KVTransferSpec proto;
  proto.add_block_arrays()->set_block_bytes(4096);
  proto.add_block_arrays()->set_block_bytes(512);
  proto.set_num_kv_shards(2);
  proto.set_num_workers(1);
  Publish(proto);

  GrsKVTransferSpecSource source(address_);
  absl::StatusOr<KVTransferSpec> spec = source.Get();
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_EQ(spec->block_array_bytes, (std::vector<uint64_t>{4096, 512}));
  EXPECT_EQ(spec->num_kv_shards, 2u);
  EXPECT_EQ(spec->num_workers, 1u);
}

TEST_F(GrsKVTransferSpecSourceTest, UnavailableWhenRegistryUnreachable) {
  server_->Shutdown();
  GrsKVTransferSpecSource source(address_);
  EXPECT_TRUE(absl::IsUnavailable(source.Get().status()));
}

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden
