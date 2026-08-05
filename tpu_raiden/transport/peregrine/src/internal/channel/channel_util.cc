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

#include "tpu_raiden/transport/peregrine/src/internal/channel/channel_util.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/channel/channel.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/connector.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

namespace peregrine::internal {

Channels Create(const Endpoint& peer, const int n) {
  DCHECK(peer.IsValid());
  DCHECK_GE(n, 1);

  Channels chs;
  chs.reserve(n);
  for (int i = 0; i < 2 * n; ++i) {
    std::unique_ptr<TcpSocket> socket = TcpConnector::Create(peer);
    if (socket == nullptr) continue;
    DCHECK(socket->IsBlocking());

    std::unique_ptr<Channel> ch = CreateTcpChannel(std::move(socket));
    DCHECK_NE(ch, nullptr);

    chs.emplace_back(std::move(ch));
    if (chs.size() >= n) break;
  }
  return chs;
}

}  // namespace peregrine::internal
