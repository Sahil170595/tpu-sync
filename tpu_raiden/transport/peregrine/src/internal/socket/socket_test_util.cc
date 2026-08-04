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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_test_util.h"

#include <memory>
#include <thread>  // NOLINT
#include <utility>

#include "absl/log/check.h"
#include "absl/synchronization/notification.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/acceptor.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/connector.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

namespace peregrine::internal::testing {

std::pair<std::unique_ptr<TcpSocket>, std::unique_ptr<TcpSocket>>
CreateTcpSocketPair(int family) {
  const Endpoint a(TestOnly_LocalEndpoint(family, /*tcp=*/true));
  std::unique_ptr<TcpAcceptor> acceptor = TcpAcceptor::Create(a);
  CHECK_NE(acceptor, nullptr);

  std::unique_ptr<TcpSocket> sa = nullptr;
  absl::Notification acceptor_started;
  absl::Notification socket_accepted;
  auto accept = [&](std::unique_ptr<TcpSocket> socket) {
    sa = std::move(socket);
    socket_accepted.Notify();
  };
  std::jthread acceptor_thread([&]() {
    DCHECK(acceptor->Socket().IsBlocking());
    acceptor_started.Notify();
    acceptor->Start(accept);
  });

  acceptor_started.WaitForNotification();
  std::unique_ptr<TcpSocket> sb = TcpConnector::Create(/*peer=*/a);

  socket_accepted.WaitForNotification();
  acceptor->Stop();
  acceptor_thread.join();

  CHECK_NE(sa, nullptr);
  CHECK_NE(sb, nullptr);
  return {std::move(sa), std::move(sb)};
}

std::pair<std::unique_ptr<UdpSocket>, std::unique_ptr<UdpSocket>>
CreateUdpSocketPair(int family) {
  const Endpoint a(TestOnly_LocalEndpoint(family, /*tcp=*/false));
  const Endpoint b(TestOnly_LocalEndpoint(family, /*tcp=*/false));

  std::unique_ptr<UdpSocket> sa = TestOnly_CreateUdpSocket(family);
  std::unique_ptr<UdpSocket> sb = TestOnly_CreateUdpSocket(family);

  CHECK(sa->Bind(a));
  CHECK(sb->Bind(b));

  CHECK(sa->Connect(b));
  CHECK(sb->Connect(a));

  return {std::move(sa), std::move(sb)};
}

}  // namespace peregrine::internal::testing
