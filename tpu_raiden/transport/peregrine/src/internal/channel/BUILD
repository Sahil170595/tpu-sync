# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "channel_types",
    hdrs = ["channel_types.h"],
)

cc_library(
    name = "channel",
    hdrs = ["channel.h"],
    deps = [
        ":channel_types",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "channel_test_util",
    testonly = True,
    srcs = ["channel_test_util.cc"],
    hdrs = ["channel_test_util.h"],
    deps = [
        ":channel",
        ":channel_msg",
        ":channel_stream",
        ":channel_util",
        ":pipe",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_test_util",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "channel_tcp",
    srcs = ["channel_tcp.cc"],
    hdrs = ["channel_tcp.h"],
    deps = [
        ":channel",
        ":channel_types",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_tcp",
        "//tpu_raiden/transport/peregrine/src/internal/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "channel_udp",
    srcs = ["channel_udp.cc"],
    hdrs = ["channel_udp.h"],
    deps = [
        ":channel",
        ":channel_types",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_udp",
        "//tpu_raiden/transport/peregrine/src/internal/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pipe",
    testonly = True,
    srcs = ["pipe.cc"],
    hdrs = ["pipe.h"],
    deps = [
        "//tpu_raiden/transport/peregrine/src/internal/util:test_iov",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "channel_msg",
    testonly = True,
    srcs = ["channel_msg.cc"],
    hdrs = ["channel_msg.h"],
    deps = [
        ":channel",
        ":channel_types",
        ":pipe",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "//tpu_raiden/transport/peregrine/src/internal/util",
        "//tpu_raiden/transport/peregrine/src/internal/util:test_iov",
        "//tpu_raiden/transport/peregrine/src/util",
        "//util/random:shared_bit_gen",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "channel_stream",
    testonly = True,
    srcs = ["channel_stream.cc"],
    hdrs = ["channel_stream.h"],
    deps = [
        ":channel",
        ":channel_types",
        ":pipe",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "//tpu_raiden/transport/peregrine/src/internal/util",
        "//tpu_raiden/transport/peregrine/src/internal/util:test_iov",
        "//tpu_raiden/transport/peregrine/src/util",
        "//util/random:shared_bit_gen",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "channel_test",
    srcs = ["channel_test.cc"],
    shard_count = 6,
    deps = [
        ":channel",
        ":channel_test_util",
        "//tpu_raiden/transport/peregrine/src/internal/base:types",
        "//tpu_raiden/transport/peregrine/src/util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "channel_util",
    srcs = ["channel_util.cc"],
    hdrs = ["channel_util.h"],
    deps = [
        ":channel",
        ":channel_tcp",
        ":channel_udp",
        "//tpu_raiden/transport/peregrine/src/internal/base:endpoint",
        "//tpu_raiden/transport/peregrine/src/internal/socket:connector",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_tcp",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_udp",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "channel_util_test",
    srcs = ["channel_util_test.cc"],
    shard_count = 2,
    deps = [
        ":channel_util",
        "//tpu_raiden/transport/peregrine/src/internal/base:endpoint",
        "//tpu_raiden/transport/peregrine/src/internal/socket:acceptor",
        "//tpu_raiden/transport/peregrine/src/internal/socket:socket_tcp",
        "//tpu_raiden/transport/peregrine/src/internal/util:test_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)
