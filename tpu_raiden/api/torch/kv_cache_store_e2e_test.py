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

"""E2E test for JAX KVCacheStore with TPUs."""

import os
import socket
import subprocess
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

resources = None
from tpu_raiden.api.torch import kv_cache_manager
from tpu_raiden.api.torch import kv_cache_store

# Set XLA flags to force CPU/Host platform devices if running locally on
# simulator
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def _pick_unused_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def find_free_port() -> int:
  return _pick_unused_port()


def get_local_ip() -> str:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
  except OSError:
    ip = "127.0.0.1"
  finally:
    s.close()
  return ip


# Global variables for subprocesses
_orchestrator_process = None
_registry_process = None
_orchestrator_port = None
_registry_port = None


def start_servers():
  global _orchestrator_process, _registry_process
  global _orchestrator_port, _registry_port

  _orchestrator_port = _pick_unused_port()
  _registry_port = _pick_unused_port()

  this_dir = os.path.dirname(os.path.abspath(__file__))
  orchestrator_binary = os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "core",
          "controller",
          "raiden_orchestrator_main",
      )
  )
  registry_binary = os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "kv_cache",
          "global_registry",
          "global_registry_server",
      )
  )
  extra_flags = []

  print(f"Starting Orchestrator on port {_orchestrator_port}")
  orch_log = open("/tmp/raiden_orchestrator.log", "w")
  _orchestrator_process = subprocess.Popen(
      [
          orchestrator_binary,
          f"--port={_orchestrator_port}",
      ]
      + extra_flags,
      stdout=orch_log,
      stderr=subprocess.STDOUT,
  )

  print(f"Starting Registry on port {_registry_port}")
  reg_log = open("/tmp/raiden_registry.log", "w")
  _registry_process = subprocess.Popen(
      [
          registry_binary,
          f"--port={_registry_port}",
      ]
      + extra_flags,
      stdout=reg_log,
      stderr=subprocess.STDOUT,
  )

  # Give them some time to start
  time.sleep(2)


def stop_servers():
  global _orchestrator_process, _registry_process
  if _orchestrator_process:
    code = _orchestrator_process.poll()
    if code is not None and code != 0:
      print(f"--- Orchestrator exited with {code} ---")
      try:
        with open("/tmp/raiden_orchestrator.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read orchestrator log: {e}")
    _orchestrator_process.terminate()
    _orchestrator_process.wait()
    _orchestrator_process = None
  if _registry_process:
    code = _registry_process.poll()
    if code is not None and code != 0:
      print(f"--- Registry exited with {code} ---")
      try:
        with open("/tmp/raiden_registry.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read registry log: {e}")
    _registry_process.terminate()
    _registry_process.wait()
    _registry_process = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"


def tearDownModule():
  pass


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.controller_port = find_free_port()

  def setUp(self):
    super().setUp()
    start_servers()
    try:
      self.device = torch.device("tpu")
    except RuntimeError:
      self.device = torch.device("cpu")

    self.num_devices = 1  # E2E tests for PyTorch currently use single device logic for kv caches
    self.num_layers = 1
    self.skip_lock = True

  def tearDown(self):
    stop_servers()
    super().tearDown()

  def test_e2e_save_and_load(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    # np.arange creates unique values for each element, ensuring different
    # values for different shards
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)

    expected_ref = host_data

    # 2. Get free port for controller
    controller_port = find_free_port()

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # 3. Create KVCacheStore (Controller)
    rid = kv_cache_store.RaidenId("e2e_job", "0", "e2e_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=num_blocks,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        raiden_controller_address=f"localhost:{controller_port}",
    )

    # 4. Create KVCacheManager (Worker)
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=0,
        # Must match the address the store's controller binds
        # ("localhost:{controller_port}", see the KVCacheStore above); using
        # get_local_ip() here dials a LAN IP the controller is not listening on,
        # so RegisterWorker never lands and Save fails with "No registered
        # workers available for TransferBuffers".
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_0",
    )

    # 5. Insert HBM blocks to KVCacheStore
    hashes = [b"hash_0", b"hash_1"]
    slices = [
        kv_cache_store.RaidenBlockID(
            rid,
            host_block_id=-1,
            device_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        ),
        kv_cache_store.RaidenBlockID(
            rid,
            host_block_id=-1,
            device_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        ),
    ]
    inserted, evicted = store.insert(hashes, slices, on_host=False)
    self.assertTrue(inserted)
    self.assertEmpty(evicted)

    # Verify status in store is HBM
    lookup_res = store.lookup(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[0][1].device_block_id, 0)
    self.assertEqual(lookup_res[1][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[1][1].device_block_id, 1)

    # 6. Save HBM blocks to host memory
    self.assertTrue(store.pin(hashes))

    def get_slice_e2e(x):
      return x[0, 0, 0, 0, 0:16].cpu().numpy()

    print(f"DEBUG: test_e2e tpu_cache before Save: {get_slice_e2e(tpu_cache)}")

    store.save(hashes)

    # Wait for save completion
    done = False
    while not done:
      save_done, save_failed, _ = store.poll_save_status()
      if save_failed:
        raise RuntimeError(f"Async Save failed: {save_failed}")
      if save_done:
        done = True
      if not done:
        time.sleep(0.01)

    # Release them so we can test pinning before load
    store.release(hashes)

    # Verify status in store is updated to HOST_AND_HBM
    lookup_res = store.lookup(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(
        lookup_res[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[0][1].host_block_id, 0)
    self.assertEqual(
        lookup_res[1][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[1][1].host_block_id, 1)

    # 7. Overwrite device memory with zeros
    # host blocks 2 and 3 are empty/uninitialized, containing zeros
    manager.h2d([2, 2], [0, 1]).wait()

    # Verify they are indeed zeros using JIT sum to avoid host caching
    # sum_val = jax.jit(jnp.sum)(tpu_cache)
    # self.assertEqual(float(sum_val), 0.0)
    # print(f"DEBUG: tpu_cache.cpu().numpy() after overwrite with zeros: {tpu_cache.cpu().numpy()[0, 0, 0, 0, 0:5]}")

    # 8. Load from host DRAM back to device HBM
    self.assertTrue(store.pin(hashes))
    store.load(hashes, [0, 1])

    # Wait for load completion
    done = False
    while not done:
      load_done, load_failed, _ = store.poll_load_status()
      if load_failed:
        raise RuntimeError(f"Async Load failed: {load_failed}")
      if load_done:
        done = True
      if not done:
        time.sleep(0.01)

    # Release at the very end
    store.release(hashes)

    # 9. Verify device memory contains the original random data
    np.testing.assert_array_equal(tpu_cache.cpu().numpy(), expected_ref)


if __name__ == "__main__":
  absltest.main()
