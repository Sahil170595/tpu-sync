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

"""Local-scope loader for the torch_tpu common shared library."""

import ctypes
import importlib
import os
import pathlib
import re

import torch_tpu

_torch_tpu_loader = importlib.import_module("torch_tpu._loader")


_LOADED = False


def load_torch_tpu_common() -> None:
  """Loads torch_tpu and its common library without global XLA symbols."""
  global _LOADED
  if _LOADED:
    return

  _torch_tpu_loader.load()
  common = pathlib.Path(torch_tpu.__file__).resolve().parent / "common"
  lib = common / "libpywrap_torch_tpu_common.so"
  if not lib.exists():
    # Per-torch-version glue layout: common/glue_<v>/libpywrap_<v>_common.so.
    # Load the glue matching the installed torch; its SONAME is what the
    # raiden extension's NEEDED entry names.
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    # pylint: enable=g-import-not-at-top
    built = [
        m.group(1)
        for d in common.glob("glue_*")
        if (m := re.match(r"glue_(\d+_\d+_\d+)$", d.name)) is not None
    ]
    if built:
      suffix = torch_abi.resolve_suffix(torch_abi.running_torch_suffix(), built)
      lib = common / f"glue_{suffix}" / f"libpywrap_{suffix}_common.so"
  if lib.exists():
    ctypes.CDLL(str(lib), mode=os.RTLD_LOCAL | os.RTLD_NOW)
  _LOADED = True
