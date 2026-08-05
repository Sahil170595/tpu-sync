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

#include "tpu_raiden/transport/peregrine/src/internal/channel/pipe.h"

#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"

namespace peregrine::internal::testing {

MemPipe::MemPipe() : mu(), shutdown(false), queue() {}

MemPipe::~MemPipe() {
  absl::MutexLock lock(mu);
  shutdown = true;
  queue.clear();
}

bool MemPipe::HasData() const { return !queue.empty() || shutdown; }

void MemPipe::Shutdown() {
  absl::MutexLock lock(mu);
  shutdown = true;
}

std::pair<BidiPipe, BidiPipe> BidiPipe::Create() {
  auto p1 = std::make_shared<MemPipe>();
  auto p2 = std::make_shared<MemPipe>();
  return {BidiPipe{/*in_pipe=*/p1, /*out_pipe=*/p2},
          BidiPipe{/*in_pipe=*/p2, /*out_pipe=*/p1}};
}

}  // namespace peregrine::internal::testing
