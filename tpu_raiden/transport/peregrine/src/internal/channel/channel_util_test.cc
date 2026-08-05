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
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/acceptor.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

namespace peregrine::internal::testing {
namespace {

using ::testing::TestParamInfo;
using ::testing::Values;

using Param = std::tuple</*family=*/int>;

std::string ToString(const TestParamInfo<Param>& info) {
  const int family = std::get<0>(info.param);
  DCHECK(family == AF_INET || family == AF_INET6);
  return absl::StrFormat("IPv%d", family == AF_INET ? 4 : 6);
}

class ChannelUtilTest : public ::testing::TestWithParam<Param> {
 protected:
  ChannelUtilTest()
      : family_(std::get<0>(GetParam())),
        local_(TestOnly_LocalEndpoint(family_, /*tcp=*/true)),
        peer_(local_),
        acceptor_(TcpAcceptor::Create(local_)) {
    CHECK_EQ(local_, peer_);
    CHECK_NE(acceptor_, nullptr);
  }

  static void Accept(std::unique_ptr<TcpSocket> socket) {
    auto x = std::move(socket);
    CHECK_NE(x, nullptr);
  }

 protected:
  const int family_;
  const Endpoint local_;
  const Endpoint peer_;
  std::unique_ptr<TcpAcceptor> acceptor_;
};

INSTANTIATE_TEST_SUITE_P(, ChannelUtilTest,
                         /*family=*/Values(AF_INET, AF_INET6), ToString);

TEST_P(ChannelUtilTest, Create) {
  std::jthread ta([&]() {
    DCHECK(acceptor_->Socket().IsBlocking());
    acceptor_->Start(Accept);
  });

  constexpr int kNumChannels = 8;
  Channels chs = Create(peer_, kNumChannels);
  EXPECT_EQ(chs.size(), kNumChannels);

  acceptor_->Stop();
}

}  // namespace
}  // namespace peregrine::internal::testing
