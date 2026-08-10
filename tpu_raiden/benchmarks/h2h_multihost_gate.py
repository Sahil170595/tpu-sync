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

"""Cross-host H2H gate: runs the C++ h2h_benchmark_runner between two hosts.

The single-host gate (h2h_cpp_gate.py) spawns sender and receiver as two local
processes over loopback, so one Python process owns both stdouts and can gate on
the receiver's integrity verdict directly. Across hosts that stops being true --
sender and receiver are separate processes on separate machines -- so this driver
adds the three things the single-host one never needed:

  * ROLE ASSIGNMENT, from the GKE TPU slice topology (TPU_WORKER_ID /
    TPU_WORKER_HOSTNAMES, injected by the TPU device plugin). Worker 0 sends,
    worker 1 receives, workers >= 2 exit 0: H2H needs exactly two hosts, and a
    v5e-16 slice hands us four.

  * A READINESS SIGNAL. The single-host `time.sleep(3)` is meaningless here: the
    two hosts pull images and run bazel independently and can start minutes
    apart. We do NOT poll the runner's control port to find out -- a TCP probe is
    indistinguishable from a sender connecting, and the receiver would burn a
    handshake round rejecting it as a scanner. Instead we watch the runner's own
    stdout for the line it prints once its control server is bound, and the
    receiver's driver republishes that readiness to the sender's driver over a
    separate peer channel.

  * A VERDICT CHANNEL. The receiver serves its per-config integrity verdicts over
    that same peer channel once done; the sender folds them into the gate.
    Without it, "Data integrity verification FAILED" gets printed on a host whose
    exit code nobody reads, and the gate goes green on corrupt data.

Modes (--mode=auto detects):
  spmd    -- every host in the slice runs this driver. Roles come from worker id.
  kubectl -- only the leader runs it; the receiver is staged onto a sibling pod
             and launched with `kubectl exec`.
  local   -- loopback, for exercising this driver without a slice.

Throughput is derived from the runner's own H2H_TOTAL_BYTES line when present,
which keeps kNumLayers/kNumShards/active-NIC-count from being duplicated (and
drifting) on the Python side.
"""

import csv
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time

from absl import app
from absl import flags

from tpu_raiden.benchmarks import bap_metrics

_MODE = flags.DEFINE_enum(
    'mode', 'auto', ['auto', 'spmd', 'kubectl', 'local'],
    'How the receiver is launched. "auto" picks spmd when the slice topology '
    'env vars are present, else local.')
_SUITE = flags.DEFINE_enum(
    'suite', 'correctness', ['correctness', 'perf', 'both'],
    'correctness: the single-host bug-class configs, re-run over a real NIC. '
    'perf: a parallelism sweep for the saturation curve.')

_PEERS = flags.DEFINE_string(
    'peers', '', 'Comma-separated peer hostnames; overrides '
    'TPU_WORKER_HOSTNAMES.')
_WORKER_ID = flags.DEFINE_integer(
    'worker_id', -1, 'This host index in --peers. Overrides TPU_WORKER_ID.')
_SENDER_IDX = flags.DEFINE_integer('sender_index', 0, 'Worker index that sends.')
_RECEIVER_IDX = flags.DEFINE_integer('receiver_index', 1,
                                     'Worker index that receives.')

_CONTROL_IFACE = flags.DEFINE_string(
    'control_interface', 'eth0',
    'Interface for the control-plane handshake. Must be set explicitly in GKE.')
_DATA_IFACE = flags.DEFINE_string(
    'data_interface', '',
    'Comma-separated data-plane interfaces. Empty auto-discovers every active '
    'secondary interface -- correct on a multi-NIC (DRANET) node. On a '
    'single-NIC node such as v5e, set this to eth0 explicitly.')
_NUMA_NODE = flags.DEFINE_integer(
    'numa_node', -1,
    'Pin to a NUMA node. -1 lets the runner pin per-NIC, which is what you want '
    'on real hardware; the single-host gate only forces 0/1 to make loopback '
    'cross the socket interconnect.')

_CONTROL_PORT = flags.DEFINE_integer(
    'control_port', 9099, 'Base runner control port; config i uses base + i.')
_PEER_PORT = flags.DEFINE_integer(
    'peer_port', 9299,
    'Driver-to-driver channel on the receiver host: readiness and verdicts. '
    'Distinct from the runner control ports on purpose.')
_STARTUP_TIMEOUT_S = flags.DEFINE_integer(
    'startup_timeout_s', 1800,
    'How long to wait for the peer driver and each receiver process. Sized for '
    'an independent image pull plus bazel build on the peer, not the transfer.')
_TIMEOUT_S = flags.DEFINE_integer('timeout_s', 1800,
                                  'Per-process hard timeout.')
_ITERS = flags.DEFINE_integer(
    'iters', 5,
    'Timed iterations per config. Correctness only needs to exercise the path; '
    'raise to a few hundred for perf work.')

_RECORD = flags.DEFINE_bool('record', False,
                            'Write baselines instead of gating.')
_ANALYZE = flags.DEFINE_bool(
    'analyze', False,
    'Collect and print the distribution only. Never fails. Use this before '
    'choosing a throughput floor.')
_GATE_THROUGHPUT = flags.DEFINE_bool(
    'gate_throughput', False,
    'Also fail when throughput drops below the recorded floor. Leave off until '
    'a baseline exists and its spread is understood; integrity always gates.')
_BASELINES = flags.DEFINE_string('baselines', None,
                                 'Baselines JSON path. Default: runfiles copy.')
_SIGMA_K = flags.DEFINE_float(
    'sigma_k', 3.5, 'Robust sigmas (MAD) below the median for the floor.')
_MAX_MARGIN = flags.DEFINE_float(
    'max_margin', 0.05,
    'Floor is never looser than this fractional drop. Looser than the '
    'single-host 0.03: a real NIC has more run-to-run spread than a memcpy.')
_DUMP = flags.DEFINE_string('dump', None, 'CSV of every sample.')

# (block_size_bytes, num_blocks, parallelism)
#
# Correctness: the exact configs the single-host gate uses, so that a green
# single-host run next to a red multi-host run isolates the difference to the
# real socket path. Each reaches a bug class the others cannot -- see
# H2H_SINGLE_HOST_CORRECTNESS_TEST.md.
_CORRECTNESS_CONFIGS = [
    (1048576, 64, 1),   # baseline block mapping / offset / copy
    (1048576, 64, 8),   # races, write interleaving, ordering, stream partition
    (1048573, 64, 4),   # alignment / boundary / partial write
]
# Perf: a parallelism sweep at a fixed 2 MiB block, matching the block size in
# H2H.md's reference table so the numbers are directly comparable. What matters
# is the shape of the curve and where it bends, not any single value.
_PERF_CONFIGS = [(2097152, 64, p) for p in (1, 2, 4, 8, 16)]

# The runner prints this once its control server is bound and it is blocking in
# accept(). It is the readiness edge we wait on.
_READY_MARKER = 'Waiting for sender connection on control plane'

_RE_P50 = re.compile(r'p50:\s*([0-9.]+)')
_RE_P90 = re.compile(r'p90:\s*([0-9.]+)')
_RE_P99 = re.compile(r'p99:\s*([0-9.]+)')
_RE_MEAN_GBS = re.compile(r'Throughput:\s*([0-9.]+)')
_RE_RAW_MS = re.compile(r'H2H_ITER_MS\s+([0-9.]+)')
_RE_TOTAL_BYTES = re.compile(r'H2H_TOTAL_BYTES\s+(\d+)')
_RE_IFACES = re.compile(r'Interfaces:\s*(\d+)\s*\(active:\s*(\d+)\)')

_INTEG_PASS = 'Data integrity verification PASSED'
_INTEG_FAIL = 'Data integrity verification FAILED'

# Only used to reconstruct total bytes when the runner is too old to print
# H2H_TOTAL_BYTES. Mirrors kNumLayers/kNumShards in h2h_benchmark_runner.cc.
_FALLBACK_LAYERS = 32
_FALLBACK_SHARDS = 1


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _discover_topology():
  """Returns (worker_id, [hostnames]).

  On GKE the TPU device plugin injects TPU_WORKER_HOSTNAMES (every host in the
  slice, comma-separated, index-aligned) and TPU_WORKER_ID. That solves peer
  discovery with no Kubernetes API access and no external rendezvous. Flags win
  so the driver stays runnable off-cluster.
  """
  raw = _PEERS.value or os.environ.get('TPU_WORKER_HOSTNAMES', '')
  hosts = [h.strip() for h in raw.split(',') if h.strip()]

  wid = _WORKER_ID.value
  if wid < 0:
    for var in ('TPU_WORKER_ID', 'JOB_COMPLETION_INDEX', 'MEGASCALE_SLICE_ID'):
      if os.environ.get(var, '').strip().isdigit():
        wid = int(os.environ[var].strip())
        break
  return wid, hosts


def _detect_mode(worker_id, hosts):
  if _MODE.value != 'auto':
    return _MODE.value
  # Two or more addressable hosts AND a usable index means every host is running
  # this same driver -- the SPMD case.
  if len(hosts) >= 2 and worker_id >= 0:
    return 'spmd'
  return 'local'


# ---------------------------------------------------------------------------
# Runfiles / process launch
# ---------------------------------------------------------------------------


def _runfiles_root():
  """Runfiles ROOT, not this binary's dir: the C++ runner is a data dep in a
  sibling tree (_main/examples/microbenchmarks/)."""
  if os.environ.get('RUNFILES_DIR'):
    return os.environ['RUNFILES_DIR']
  d = os.path.dirname(os.path.abspath(__file__))
  main_root = None
  while d != os.path.dirname(d):
    if d.endswith('.runfiles'):
      return d
    if os.path.basename(d) == '_main':
      main_root = d
    d = os.path.dirname(d)
  return main_root or os.path.dirname(os.path.abspath(__file__))


def _locate(basename):
  root = _runfiles_root()
  for dirpath, _, files in os.walk(root, followlinks=True):
    if basename in files:
      cand = os.path.join(dirpath, basename)
      if os.access(cand, os.X_OK):
        return cand
  raise FileNotFoundError(f'could not locate {basename} under {root}')


def _base_argv(cc, bs, nb, p, port):
  return [
      cc,
      f'--control_interface={_CONTROL_IFACE.value}',
      f'--data_interface={_DATA_IFACE.value}',
      f'--peer_control_port={port}',
      f'--block_size={bs}',
      f'--num_blocks={nb}',
      f'--parallelism={p}',
      f'--numa_node={_NUMA_NODE.value}',
      f'--iterations={_ITERS.value}',
  ]


def _popen(argv):
  return subprocess.Popen(argv, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})


class _StreamWatcher(threading.Thread):
  """Drains a process's stdout, flagging when the readiness marker appears.

  Draining in a thread does double duty: it turns the runner's own "waiting for
  sender" line into a precise readiness edge, and it keeps the pipe from filling
  while the caller does something else (run the sender, wait on a peer).
  """

  def __init__(self, proc, marker):
    super().__init__(daemon=True)
    self._proc = proc
    self._marker = marker
    self._lines = []
    self.ready = threading.Event()

  def run(self):
    try:
      for raw in self._proc.stdout:
        line = raw.decode('utf-8', 'replace')
        self._lines.append(line)
        if self._marker in line:
          self.ready.set()
    finally:
      # Unblock anyone waiting on readiness for a process that died instead.
      self.ready.set()
      try:
        self._proc.stdout.close()
      except OSError:
        pass

  @property
  def output(self):
    """Only valid after join(); the reader thread owns the list until then."""
    return ''.join(self._lines)


def _finish(proc, watcher, timeout):
  """Waits for a watched process to exit and returns its full output."""
  try:
    proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
  watcher.join(timeout=60)
  return watcher.output


# ---------------------------------------------------------------------------
# Driver-to-driver channel (receiver side serves, sender side queries)
# ---------------------------------------------------------------------------


class _PeerServer(threading.Thread):
  """Line protocol on the receiver host, for the sender host's driver.

      READY <i>  -> "1" once config i's receiver is bound, else "0"
      VERDICTS   -> the per-config integrity verdicts as JSON

  One long-lived port rather than probing the runner's control ports, which
  would collide with the sender's handshake.
  """

  def __init__(self, port):
    super().__init__(daemon=True)
    self._port = port
    self._lock = threading.Lock()
    self._ready = set()
    self._verdicts = None
    self.fetched = threading.Event()
    self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._srv.bind(('0.0.0.0', port))
    self._srv.listen(8)

  def mark_ready(self, index):
    with self._lock:
      self._ready.add(index)

  def set_verdicts(self, verdicts):
    with self._lock:
      self._verdicts = dict(verdicts)

  def run(self):
    while True:
      try:
        conn, _ = self._srv.accept()
      except OSError:
        return
      with conn:
        try:
          req = conn.recv(4096).decode('utf-8', 'replace').strip()
          if req.startswith('READY'):
            idx = int(req.split()[1])
            with self._lock:
              ok = idx in self._ready
            conn.sendall(b'1' if ok else b'0')
          elif req == 'VERDICTS':
            with self._lock:
              payload = self._verdicts
            if payload is None:
              conn.sendall(b'')
            else:
              conn.sendall(json.dumps(payload).encode('utf-8'))
              self.fetched.set()
          else:
            conn.sendall(b'?')
        except (OSError, ValueError, IndexError):
          continue


def _peer_query(host, port, request, timeout=30):
  """One request/response against _PeerServer. Returns None if unreachable."""
  try:
    with socket.create_connection((host, port), timeout=timeout) as s:
      s.settimeout(timeout)
      s.sendall(request.encode('utf-8'))
      chunks = []
      while True:
        chunk = s.recv(65536)
        if not chunk:
          break
        chunks.append(chunk)
    return b''.join(chunks).decode('utf-8')
  except OSError:
    return None


def _await_peer(host, port, request, want, timeout, what):
  """Polls `request` until the reply is satisfactory. Returns it, or None.

  `want` is the exact reply to wait for, or None to accept any non-empty reply
  (the server answers an early VERDICTS query with an empty body).
  """
  deadline = time.time() + timeout
  attempt = 0
  while time.time() < deadline:
    reply = _peer_query(host, port, request)
    if reply:
      if want is None or reply == want:
        return reply
    attempt += 1
    if attempt % 12 == 0:
      print(f'[barrier] still waiting for {what} '
            f'({int(deadline - time.time())}s left)', flush=True)
    time.sleep(5)
  print(f'[barrier] TIMEOUT waiting for {what}', flush=True)
  return None


# ---------------------------------------------------------------------------
# Stats (mirrors h2h_cpp_gate.py so a floor means the same in both gates)
# ---------------------------------------------------------------------------


def _pct(xs, q):
  xs = sorted(xs)
  if len(xs) == 1:
    return xs[0]
  pos = (len(xs) - 1) * (q / 100.0)
  lo = int(pos)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _core_floor(samples, k, max_margin):
  """floor = max(median - k * MAD_sigma, median * (1 - max_margin))."""
  med = statistics.median(samples)
  mad = (statistics.median([abs(x - med) for x in samples])
         if len(samples) > 1 else 0.0)
  return max(med - k * 1.4826 * mad, med * (1.0 - max_margin))


def _f(regex, text):
  m = regex.search(text or '')
  return float(m.group(1)) if m else -1.0


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _configs():
  if _SUITE.value == 'correctness':
    return list(_CORRECTNESS_CONFIGS)
  if _SUITE.value == 'perf':
    return list(_PERF_CONFIGS)
  return list(_CORRECTNESS_CONFIGS) + list(_PERF_CONFIGS)


def _parse_sender(out, bs, nb):
  """Turns the sender's stdout into throughput samples.

  total_bytes comes from the runner's own H2H_TOTAL_BYTES when available. The
  fallback recomputes it -- and MUST include the active-NIC count, because the
  runner's total is `active_managers.size() * layers * shards * blocks * size`.
  Dropping that factor is invisible on one NIC and wrong by N on a DRANET node.
  """
  m = _RE_TOTAL_BYTES.search(out or '')
  if m:
    total_bytes = float(m.group(1))
  else:
    ifm = _RE_IFACES.search(out or '')
    active = int(ifm.group(2)) if ifm else 1
    total_bytes = float(active * _FALLBACK_LAYERS * _FALLBACK_SHARDS * nb * bs)

  p50 = _f(_RE_P50, out)
  raw_gbs = [(total_bytes / 1e9) / (float(ms) / 1000.0)
             for ms in _RE_RAW_MS.findall(out or '') if float(ms) > 0]
  mean_gbs = _f(_RE_MEAN_GBS, out)

  gbs = -1.0
  if p50 > 0:
    gbs = (total_bytes / 1e9) / (p50 / 1000.0)
  elif raw_gbs:
    gbs = statistics.median(raw_gbs)
  elif mean_gbs > 0:
    gbs = mean_gbs

  return {'gbs': gbs, 'mean_gbs': mean_gbs, 'p50_ms': p50,
          'p90_ms': _f(_RE_P90, out), 'p99_ms': _f(_RE_P99, out),
          'raw_gbs': raw_gbs, 'total_bytes': total_bytes}


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def _run_receiver(cc, configs):
  """SPMD receiver: run every config, publish readiness, then serve verdicts.

  Config order and ports are a pure function of the config list, so both hosts
  walk the same sequence without exchanging a schedule.
  """
  srv = _PeerServer(_PEER_PORT.value)
  srv.start()
  print(f'[receiver] peer channel up on :{_PEER_PORT.value}', flush=True)

  verdicts = {}
  for i, (bs, nb, p) in enumerate(configs):
    label = _label(bs, nb, p)
    port = _CONTROL_PORT.value + i
    print(f'[receiver] ({i + 1}/{len(configs)}) {label} on :{port}', flush=True)

    proc = _popen(_base_argv(cc, bs, nb, p, port) + ['--role=receiver'])
    watcher = _StreamWatcher(proc, _READY_MARKER)
    watcher.start()
    if not watcher.ready.wait(timeout=_STARTUP_TIMEOUT_S.value):
      print(f'[receiver] {label}: never reached "{_READY_MARKER}"',
            file=sys.stderr)
    srv.mark_ready(i)

    # The receiver self-exits after its byte-compare; never terminate it early
    # or the integrity check is cut short and reports a false CORRUPT.
    out = _finish(proc, watcher, _TIMEOUT_S.value + _STARTUP_TIMEOUT_S.value)
    ok = (_INTEG_PASS in out) and (_INTEG_FAIL not in out)
    verdicts[label] = ok
    print(f'[receiver] {label}: integrity={"OK" if ok else "CORRUPT"}',
          flush=True)
    if not ok:
      print(f'--- receiver tail ({label}) ---\n{out[-4000:]}', flush=True)

  srv.set_verdicts(verdicts)
  print('[receiver] all configs done; waiting for the sender to collect '
        'verdicts ...', flush=True)
  if not srv.fetched.wait(timeout=_TIMEOUT_S.value):
    print('[receiver] sender never collected the verdicts.', file=sys.stderr)
    return 1
  return 0


def _run_sender(cc, configs, peer_host, spawn_receiver=None):
  """Runs the sender for every config.

  spawn_receiver is None in spmd mode (the peer host runs its own driver and
  publishes readiness over the peer channel); in local/kubectl mode it is a
  callable(argv) -> Popen and this process owns both ends.
  """
  results = {}
  local_verdicts = {}
  aborted = False

  for i, (bs, nb, p) in enumerate(configs):
    label = _label(bs, nb, p)
    port = _CONTROL_PORT.value + i
    argv = _base_argv(cc, bs, nb, p, port)

    recv_proc = recv_watcher = None
    if spawn_receiver is None:
      print(f'[sender] ({i + 1}/{len(configs)}) {label}: waiting for peer '
            f'receiver on {peer_host}', flush=True)
      ready = _await_peer(peer_host, _PEER_PORT.value, f'READY {i}', '1',
                          _STARTUP_TIMEOUT_S.value,
                          f'receiver readiness for {label}')
      if ready != '1':
        print(f'[sender] {label}: peer receiver never became ready. If this is '
              f'the FIRST config, the peer is probably not running this driver '
              f'at all -- re-check the probe output and --mode.',
              file=sys.stderr)
        results[label] = {'gbs': -1.0, 'samples': []}
        # Abort rather than advance: the peer walks the same config list, so
        # moving to config i+1 while it is stuck on config i trades one timeout
        # for a cascade. Configs left absent report NO MEASUREMENT and fail the
        # gate, which is the correct verdict.
        aborted = True
        break
    else:
      print(f'[sender] ({i + 1}/{len(configs)}) {label}: starting local '
            f'receiver on :{port}', flush=True)
      recv_proc = spawn_receiver(argv + ['--role=receiver'])
      recv_watcher = _StreamWatcher(recv_proc, _READY_MARKER)
      recv_watcher.start()
      if not recv_watcher.ready.wait(timeout=_STARTUP_TIMEOUT_S.value):
        print(f'[sender] {label}: receiver never signalled ready.',
              file=sys.stderr)
        recv_proc.kill()
        results[label] = {'gbs': -1.0, 'samples': []}
        aborted = True
        break

    send_proc = _popen(argv + ['--role=sender',
                               f'--peer_control_ip={peer_host}'])
    send_watcher = _StreamWatcher(send_proc, _READY_MARKER)
    send_watcher.start()
    out = _finish(send_proc, send_watcher, _TIMEOUT_S.value)
    m = _parse_sender(out, bs, nb)

    if recv_proc is not None:
      recv_out = _finish(recv_proc, recv_watcher, _TIMEOUT_S.value)
      local_verdicts[label] = ((_INTEG_PASS in recv_out) and
                               (_INTEG_FAIL not in recv_out))
      if not local_verdicts[label]:
        print(f'--- receiver tail ({label}) ---\n{recv_out[-4000:]}', flush=True)

    samples = m['raw_gbs'] or ([m['gbs']] if m['gbs'] > 0 else [])
    if not samples:
      print(f'[sender] {label} produced no throughput reading; output:\n{out}',
            flush=True)
    results[label] = {'gbs': _pct(samples, 50) if samples else -1.0,
                      'samples': samples, 'raw': m}

  return results, local_verdicts, aborted


def _kubectl_spawner(peer_pod, cc):
  """Stages the runner onto a sibling pod and returns a spawn callable.

  The binary lives in this host's bazel runfiles, so the peer -- same image, no
  build of its own -- has to be handed a copy before it can be exec'd.
  """
  remote = f'/tmp/{os.path.basename(cc)}'
  subprocess.run(['kubectl', 'cp', cc, f'{peer_pod}:{remote}'], check=True)
  subprocess.run(['kubectl', 'exec', peer_pod, '--', 'chmod', '+x', remote],
                 check=True)

  def spawn(argv):
    return _popen(['kubectl', 'exec', peer_pod, '--', remote] + argv[1:])

  return spawn


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_multihost_baselines.json')


# ---------------------------------------------------------------------------


def main(_):
  worker_id, hosts = _discover_topology()
  mode = _detect_mode(worker_id, hosts)
  configs = _configs()
  cc = _locate('h2h_benchmark_runner')

  # Loopback has no eth0. Only override flags left at their defaults.
  if mode == 'local':
    for name in ('control_interface', 'data_interface'):
      if flags.FLAGS[name].using_default_value:
        setattr(flags.FLAGS, name, 'lo')

  print(f'[topology] mode={mode} worker_id={worker_id} hosts={hosts}',
        flush=True)
  print(f'[topology] suite={_SUITE.value} configs={len(configs)} '
        f'iters={_ITERS.value} runner={cc}', flush=True)

  if mode == 'spmd':
    if worker_id == _RECEIVER_IDX.value:
      sys.exit(_run_receiver(cc, configs))
    if worker_id != _SENDER_IDX.value:
      # A v5e-16 slice gives four hosts; H2H uses two. The rest must exit 0 or
      # they hold the job open for nothing.
      print(f'[topology] worker {worker_id} is neither sender '
            f'({_SENDER_IDX.value}) nor receiver ({_RECEIVER_IDX.value}); '
            f'nothing to do.', flush=True)
      return
    if len(hosts) <= _RECEIVER_IDX.value:
      print(f'GATE FAIL: --receiver_index={_RECEIVER_IDX.value} but only '
            f'{len(hosts)} host(s) known: {hosts}', file=sys.stderr)
      sys.exit(1)
    peer_host = hosts[_RECEIVER_IDX.value]
    results, _, aborted = _run_sender(cc, configs, peer_host)
    if aborted:
      # The peer is wedged or absent; do not sit on the full verdict timeout
      # waiting for a host that already failed to show up.
      print('GATE FAIL: the peer receiver never became ready, so the run is '
            'incomplete. Check that worker '
            f'{_RECEIVER_IDX.value} is running this same target, and re-read '
            'the probe output.', file=sys.stderr)
      sys.exit(1)
    raw = _await_peer(peer_host, _PEER_PORT.value, 'VERDICTS', None,
                      _TIMEOUT_S.value, 'receiver verdicts')
    if not raw:
      print("GATE FAIL: could not read the receiver's integrity verdicts. The "
            'transfer may well have succeeded, but nothing proves the bytes '
            'are correct, so this is a failure.', file=sys.stderr)
      sys.exit(1)
    verdicts = json.loads(raw)

  elif mode == 'kubectl':
    if len(hosts) <= _RECEIVER_IDX.value:
      print('GATE FAIL: --mode=kubectl needs --peers or TPU_WORKER_HOSTNAMES '
            'to name the sibling pod.', file=sys.stderr)
      sys.exit(1)
    peer_host = hosts[_RECEIVER_IDX.value]
    results, verdicts, _ = _run_sender(cc, configs, peer_host,
                                       _kubectl_spawner(peer_host, cc))

  else:  # local
    peer_host = '127.0.0.1'
    results, verdicts, _ = _run_sender(cc, configs, peer_host, _popen)

  # --- reporting ----------------------------------------------------------
  scalars = {}
  dump = writer = dump_path = None
  if _DUMP.value:
    dump_path = _DUMP.value
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      dump_path = os.path.join(adir, os.path.basename(dump_path))
    dump = open(dump_path, 'w', newline='')
    writer = csv.writer(dump)
    writer.writerow(['config', 'iter', 'gbs', 'mean_gbs', 'p50_ms', 'p90_ms',
                     'p99_ms', 'total_bytes', 'integrity'])

  for (bs, nb, p) in configs:
    label = _label(bs, nb, p)
    r = results.get(label, {'gbs': -1.0, 'samples': []})
    s = r['samples']
    integ = verdicts.get(label, False)
    if s:
      print(f'[measured] {label:<22} n={len(s):<4} median={_pct(s, 50):8.3f}  '
            f'p10={_pct(s, 10):7.3f}  p90={_pct(s, 90):7.3f}  '
            f'min={min(s):7.3f}  max={max(s):7.3f}  '
            f'stdev={statistics.pstdev(s):6.3f} GB/s  '
            f'integrity={"OK" if integ else "CORRUPT"}', flush=True)
    else:
      print(f'[measured] {label}: no samples', flush=True)
    scalars[f'{label}/cpp_gbs'] = r['gbs']
    if writer:
      raw = r.get('raw', {})
      for j, g in enumerate(s):
        writer.writerow([label, j, f'{g:.4f}',
                         f'{raw.get("mean_gbs", -1):.4f}',
                         f'{raw.get("p50_ms", -1):.4f}',
                         f'{raw.get("p90_ms", -1):.4f}',
                         f'{raw.get("p99_ms", -1):.4f}',
                         int(raw.get('total_bytes', -1)), int(integ)])

  bap_metrics.emit(scalars)
  if dump:
    dump.close()
    print(f'\nWrote samples -> {dump_path}', flush=True)

  if _ANALYZE.value:
    print('\nanalyze mode: data collected, no gate/baseline. Done.', flush=True)
    return

  if _RECORD.value:
    cfg = {'sigma_k': _SIGMA_K.value, 'max_margin': _MAX_MARGIN.value,
           'configs': {}}
    for (bs, nb, p) in configs:
      label = _label(bs, nb, p)
      s = results.get(label, {}).get('samples', [])
      cfg['configs'][label] = {
          'baseline_gbs': round(_pct(s, 50), 3) if s else 0.0,
          'floor_gbs': round(_core_floor(s, _SIGMA_K.value, _MAX_MARGIN.value),
                             3) if s else 0.0,
          'integrity': verdicts.get(label, False),
          'n_samples': len(s),
      }
    out_path = _baselines_path()
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, 'h2h_multihost_baselines.json')
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'\nRecorded {len(cfg["configs"])} baselines -> {out_path}',
          flush=True)
    return

  # --- gate ---------------------------------------------------------------
  floors = {}
  if _GATE_THROUGHPUT.value:
    try:
      with open(_baselines_path()) as f:
        floors = {k: v.get('floor_gbs', 0.0)
                  for k, v in json.load(f).get('configs', {}).items()}
    except (OSError, ValueError) as e:
      print(f'GATE FAIL: --gate_throughput set but baselines unreadable: {e}',
            file=sys.stderr)
      sys.exit(1)

  bad = []
  print('\nCross-host H2H gate\n')
  print('config                    median    floor   integrity  verdict')
  print('-' * 70)
  for (bs, nb, p) in configs:
    label = _label(bs, nb, p)
    r = results.get(label, {'gbs': -1.0, 'samples': []})
    integ = verdicts.get(label, False)
    floor = floors.get(label, 0.0)
    reasons = []
    if not r['samples']:
      reasons.append('NO MEASUREMENT')
    if not integ:
      reasons.append('DATA CORRUPTION')
    if (_GATE_THROUGHPUT.value and r['gbs'] > 0 and floor > 0 and
        r['gbs'] < floor):
      reasons.append(f'BELOW FLOOR {floor:.3f}')
    print(f'{label:<22} {r["gbs"]:8.3f} {floor:8.3f}  '
          f'{"OK" if integ else "CORRUPT":<9} '
          f'{"PASS" if not reasons else "FAIL <-- " + ", ".join(reasons)}')
    if reasons:
      bad.append(label)

  if bad:
    print(f'\nGATE FAIL on {len(bad)} config(s): {bad}', file=sys.stderr)
    sys.exit(1)
  print('\nGATE PASS: all configs byte-exact across hosts.', flush=True)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
