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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/transport/block_transport_delegate.h"
#include "tpu_raiden/transport/buffer_push_task.h"
#include "tpu_raiden/transport/lib/chunk.h"
#include "tpu_raiden/transport/lib/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

// Marks a pull whose ordering is already guaranteed by a KVCacheStore read
// lease, so the source must NOT gate it on device-to-host readiness.
//
// Why a sentinel rather than 0: the readiness gate
// (KVCacheManagerWithTransfer::RegisterBlockReadinessCallback) identifies a
// transfer by uuid. Pulls historically sent 0, which is indistinguishable from
// "no uuid supplied", so the gate could never match a pull to its own transfer
// and always fell through to scanning every live send entry -- coupling the
// pull to an unrelated transfer's D2H future. This value says something
// specific and true instead: "the lease already ordered this read."
//
// Why THIS value cannot collide. Both uuid generators produce strictly smaller
// numbers, so nothing in the system can ever mint it:
//   * the disagg connector: uuid4().int >> 78, i.e. 50 bits, range
//     [0, 2^50 - 1]   (tpu-inference: distributed/tpu_raiden_connector.py,
//     get_uuid())
//   * the raiden python controller: random.randint(1, 2**63 - 1), range
//     [1, 2^63 - 1]   (tpu_raiden/rpc/raiden_controller.py)
// Anything with the top bit set is therefore unreachable by either; all-ones is
// also unmistakable in a log line or packet dump.
//
// If a third uuid source is ever added, it MUST stay below 2^63 or this
// sentinel loses its meaning silently.
inline constexpr uint64_t kLeaseAuthorizedPullUuid = 0xFFFFFFFFFFFFFFFFull;

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback = std::function<absl::Status(
    size_t layer_idx, size_t shard_idx, int block_id, size_t size_bytes)>;

// High-speed Key-Value block transport engine extending RawBufferTransport.
class BlockTransport final {
 public:
  BlockTransport(BlockTransportDelegate* delegate, int local_port,
                 const std::vector<std::string>& local_ips = {},
                 int parallelism = 1);
  ~BlockTransport();

  // Asynchronous Scatter-Gather Push
  void AsyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, int parallelism,
      MajorOrder major_order, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> on_complete);

  // Synchronous Scatter-Gather Push (op = 1 / op = 6)
  absl::StatusOr<std::vector<int>> SyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1);

  // Synchronous Scatter-Gather Pull (op = 2)
  // When explicit_dst_ptrs is supplied it contains one base pointer per
  // (block array, shard), in block-array-major order.
  absl::StatusOr<std::vector<int>> SyncPull(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {}, uint64_t uuid = 0);

  // Drops receive-progress counters belonging to a finished, failed, or
  // timed-out plan so the UUID can be safely reused.
  void ForgetPushProgress(uint64_t uuid);

  int local_port() const { return raw_transport_.local_port(); }
  const std::string& bound_ip() const { return raw_transport_.bound_ip(); }

  absl::Status PullBuffer(absl::string_view peer, size_t buffer_id,
                          size_t src_shard_idx, size_t src_offset_bytes,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          size_t size_bytes) {
    return raw_transport_.PullBuffer(peer, buffer_id, src_shard_idx,
                                     src_offset_bytes, dst_shard_idx,
                                     dst_offset_bytes, size_bytes);
  }

  absl::Status PushBuffer(absl::string_view peer, size_t buffer_id,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          const uint8_t* data_ptr, size_t size_bytes,
                          uint64_t uuid = 0) {
    return raw_transport_.PushBuffer(peer, buffer_id, dst_shard_idx,
                                     dst_offset_bytes, data_ptr, size_bytes,
                                     uuid);
  }

  absl::Status PushBuffers(const std::vector<BufferPushTask>& tasks,
                           int parallelism, uint64_t uuid) {
    return raw_transport_.PushBuffers(tasks, parallelism, uuid);
  }

  absl::Status RegisterExpectedChunks(uint64_t uuid, uint32_t expected_chunks) {
    return raw_transport_.RegisterExpectedChunks(uuid, expected_chunks);
  }

 private:
  struct WriteTask {
    uint64_t uuid;
    int layer_idx;
    int stream_idx;
    std::string peer;
    std::function<void()> run;
  };

  struct PeerQueue {
    std::deque<std::unique_ptr<WriteTask>> tasks;
    int active_streams = 0;
  };

  void SocketWorkerLoop();
  std::unique_ptr<WriteTask> SelectNextTask()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);

  void H2hWriteWorker(int stream_idx, absl::string_view peer,
                      absl::string_view local_ip, size_t block_offset,
                      size_t block_count, const std::vector<int>& src_block_ids,
                      const std::vector<int>& dst_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses,
                      MajorOrder major_order, uint64_t uuid = 0,
                      int layer_idx = -1, int parallelism = 1);

  void H2hReadWorker(int stream_idx, absl::string_view peer,
                     absl::string_view local_ip, size_t local_block_offset,
                     size_t local_block_count, size_t remote_block_offset,
                     size_t remote_block_count,
                     const std::vector<int>& src_block_ids,
                     const std::vector<int>& allocated_ids,
                     const std::vector<uint8_t*>& explicit_dst_ptrs,
                     std::vector<absl::Status>& statuses,
                     MajorOrder major_order,
                     BlockReceivedCallback on_block_received,
                     uint64_t uuid = 0);

  absl::Status HandleIncomingPush(int client_fd,
                                  const lib::ChunkHeader& header);
  absl::Status HandleIncomingPull(int client_fd,
                                  const lib::ChunkHeader& header);

  struct SendStreamState {
    int client_fd;
    uint64_t uuid;
    int remote_id;
    size_t count_or_size;
    MajorOrder major_order;
    size_t current_step = 0;
    size_t total_steps = 0;
  };

  void TriggerNextSendStep(std::shared_ptr<SendStreamState> state);
  void ResolveStepCoordinates(const std::shared_ptr<SendStreamState>& state,
                              size_t* layer, size_t* shard, size_t* block_idx);
  uint32_t GetChunksTotalSize(const std::vector<BlockChunk>& chunks);
  absl::Status HandleCustomRequest(int client_fd,
                                   const lib::ChunkHeader& header);

 private:
  absl::Mutex active_sends_mu_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<SendStreamState>> active_sends_
      ABSL_GUARDED_BY(active_sends_mu_);

  BlockTransportDelegate* block_delegate_;

  struct LayerProgress {
    size_t completed_chunks = 0;
    bool on_layer_received_called = false;
  };
  absl::Mutex progress_mu_;
  absl::flat_hash_map<std::pair<uint64_t, int>, LayerProgress> layer_progress_
      ABSL_GUARDED_BY(progress_mu_);

  absl::Mutex scheduler_mu_;
  absl::CondVar scheduler_cv_;
  absl::flat_hash_map<std::string, PeerQueue> peer_queues_
      ABSL_GUARDED_BY(scheduler_mu_);
  std::vector<std::string> active_peers_ ABSL_GUARDED_BY(scheduler_mu_);
  size_t rr_index_ ABSL_GUARDED_BY(scheduler_mu_) = 0;
  int parallelism_ = 1;
  std::atomic<bool> scheduler_stopping_{false};

  lib::RawBufferTransport raw_transport_;
  std::vector<std::thread> socket_workers_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
