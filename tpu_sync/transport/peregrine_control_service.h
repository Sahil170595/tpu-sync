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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_PEREGRINE_CONTROL_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_PEREGRINE_CONTROL_SERVICE_H_

#include "absl/status/statusor.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/proto/peregrine_control_service.grpc.pb.h"
#include "tpu_sync/transport/proto/peregrine_control_service.pb.h"

namespace tpu_raiden::transport {

// Server-side gRPC implementation for PeregrineControlService.
// Handles incoming ExchangePspKey RPCs from connecting peers, delegates key
// registration to data plane RawBufferTransport, and returns the server's
// allocated RX SPI and key.
class PeregrineControlServiceImpl final
    : public proto::PeregrineControlService::Service {
 public:
  explicit PeregrineControlServiceImpl(lib::RawBufferTransport* transport)
      : transport_(transport) {}

  grpc::Status ExchangePspKey(
      grpc::ServerContext* context, const proto::PspKeyExchangeRequest* request,
      proto::PspKeyExchangeResponse* response) override {
    if (transport_ == nullptr) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "RawBufferTransport is not initialized");
    }
    if (request->client_spi() == 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "client_spi must be non-zero");
    }
    if (request->client_key().size() != 16) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "client_key must be exactly 16 bytes");
    }

    auto server_rx_key_or = transport_->RegisterPspPeer(request->client_spi(),
                                                        request->client_key());
    if (!server_rx_key_or.ok()) {
      return grpc::Status(server_rx_key_or.status());
    }

    response->set_server_spi(server_rx_key_or->spi);
    response->set_server_key(server_rx_key_or->key);
    return grpc::Status::OK;
  }

 private:
  lib::RawBufferTransport* const transport_;
};

}  // namespace tpu_raiden::transport

#endif  // TPU_SYNC_TRANSPORT_PEREGRINE_CONTROL_SERVICE_H_
