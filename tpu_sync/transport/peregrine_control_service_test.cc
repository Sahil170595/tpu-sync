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

#include "tpu_sync/transport/peregrine_control_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/proto/peregrine_control_service.grpc.pb.h"
#include "tpu_sync/transport/proto/peregrine_control_service.pb.h"

namespace tpu_raiden::transport {
namespace {

using ::testing::NotNull;

class FakeRawDelegate : public lib::RawBufferTransportDelegate {
 public:
  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return nullptr;
  }
  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override { return 0; }
};

class FakeBlockDelegate : public BlockTransportDelegate {
 public:
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    return std::vector<int>(num_blocks, 0);
  }
  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return nullptr;
  }
  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override { return 0; }
  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }
  size_t num_layers() const override { return 1; }
  size_t num_shards() const override { return 1; }
  size_t slice_byte_size() const override { return 1024; }
  size_t shard_factor() const override { return 1; }
};

TEST(PeregrineControlServiceTest, BlockTransportOwnsPeregrineControlService) {
  FakeBlockDelegate block_delegate;
  BlockTransport transport(&block_delegate, /*local_port=*/0);

  EXPECT_THAT(transport.peregrine_control_service(), NotNull());
}

TEST(PeregrineControlServiceTest, InProcessGrpcExchangePspKey) {
  FakeRawDelegate raw_delegate;
  lib::RawBufferTransport transport(&raw_delegate, /*local_port=*/0);
  PeregrineControlServiceImpl service(&transport);

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  std::shared_ptr<grpc::Channel> channel =
      server->InProcessChannel(grpc::ChannelArguments());
  auto stub = proto::PeregrineControlService::NewStub(channel);

  proto::PspKeyExchangeRequest req;
  req.set_client_spi(0x12345678);
  req.set_client_key(std::string(16, 'z'));
  proto::PspKeyExchangeResponse resp;
  grpc::ClientContext ctx;

  grpc::Status status = stub->ExchangePspKey(&ctx, req, &resp);
  // In Phase 1, RawBufferTransport::RegisterPspPeer returns UnimplementedError
  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);

  server->Shutdown();
}

}  // namespace
}  // namespace tpu_raiden::transport
