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

"""High-performance JAX Weight Synchronizer for RL Trainer-Inference Pipelines."""

from typing import List, Optional

# Import Nanobind binary library directly E2E!
from tpu_raiden.frameworks.jax import _tpu_raiden_jax as _weight_synchronizer


class WeightSynchronizer:
  """Zero-copy distributed Weight Synchronizer for JAX."""

  def __init__(
      self,
      jax_arrays: List[any],
      local_port: Optional[int] = None,
      parallelism: int = 1,
      unsafe_skip_buffer_lock: bool = False,
      listener_port: Optional[int] = None,
      bind_ip: Optional[str] = None,
  ):
    """Instantiates the Weight Synchronizer on a JAX weights list.

    Args:
      jax_arrays: A list of JAX arrays representing the sharded model weights.
      local_port: Sockets server port for incoming pulls (inference mode).
      parallelism: Number of parallel network stream TCP sockets workers.
      unsafe_skip_buffer_lock: Skip PJRT buffer locks during weights unpack.
      listener_port: Sockets server port for incoming C++ Listener commands.
      bind_ip: Sockets server bind IP address.
    """
    self._impl = _weight_synchronizer.WeightSynchronizer(
        jax_arrays,
        local_port,
        parallelism,
        unsafe_skip_buffer_lock,
        listener_port,
        bind_ip,
    )

  def d2h(self) -> None:
    """Triggers asynchronous Device-to-Host (D2H) copy of current weights to Host buffer."""
    self._impl.D2h()

  def h2d(self) -> None:
    """Triggers asynchronous Host-to-Device (H2D) copy of staged host buffer back to Device memory E2E."""
    self._impl.H2d()

  def bind_weights(self, jax_arrays: List[any]) -> None:
    """Binds the JAX arrays to the weight synchronizer in-place.

    Args:
      jax_arrays: A list of JAX arrays representing the updated sharded model
        weights.
    """
    self._impl.bind_weights(jax_arrays)

  def get_host_buffer(self, layer_idx: int = 0, shard_idx: int = 0) -> any:
    """Returns a zero-copy Host-side CPU NumPy ndarray view of the C++ staging buffer.

    Args:
      layer_idx: Target layer index to fetch.
      shard_idx: Target shard index to fetch.
    """
    return self._impl.get_host_buffer(layer_idx, shard_idx)

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active local port assigned to the transceiving sockets server."""
    return self._impl.local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the active local port assigned to the C++ Listener."""
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ Listener is actively running."""
    return self._impl.is_listener_active

  @property
  def num_layers(self) -> int:
    """Returns the total number of model weight layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns the sharded devices count per layer."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns the slice capacity per device block."""
    return self._impl.slice_byte_size
