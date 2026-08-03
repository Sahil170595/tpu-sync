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

"""C++-only H2H record/gate driver.

A THIN orchestrator: it only spawns the C++ h2h_benchmark_runner (sender +
receiver) and parses its stdout. It never imports the JAX bindings, so it needs
NO change to tpu_raiden_jax_module.cc.

Two topologies:

  single-host (default)
    Both roles run on this machine over loopback, sender on NUMA 0 and receiver
    on NUMA 1 so the copy crosses the socket interconnect. Throughput here is a
    kernel/memcpy number, not a product number, so the gate is byte-integrity
    only.

  --multihost
    The ml-actions multi-host runner executes every workflow step on all hosts
    at once, so this same script runs everywhere and splits by rank: rank 0 is
    the receiver, rank 1 the sender, ranks >= 2 exit immediately. The two roles
    never need to talk to each other outside the C++ handshake, because the
    verdicts partition cleanly -- rank 0 owns the integrity check, rank 1 owns
    the throughput. Either failing exits non-zero, which fails the job.

Per config it captures:
  * throughput: derived from the C++ runner's p50 latency (median GB/s), and
  * integrity: the C++ receiver's own "Data integrity verification PASSED/FAILED".

Modes:
  --record : run each config, write {throughput, integrity} to --baselines.
  --analyze: collect + --dump samples, print distribution stats, always exit 0.
  (gate)   : FAIL if any config reports integrity FAILED (and, once baselines
             exist, if throughput is below floor).

Metrics are emitted to TENSORBOARD_OUTPUT_DIR so BAP ingests them; every tag has
a matching metrics{name:...} in the registry pbtxt.
"""

import csv
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time

from absl import app
from absl import flags

from tpu_raiden.benchmarks import bap_metrics

_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for the sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1,
                                      'NUMA node for the receiver.')
_TIMEOUT_S = flags.DEFINE_integer(
    'timeout_s', 900,
    'Per-process hard timeout. High-parallelism configs contend on a single '
    'loopback and can take several minutes for 50 iterations. Cross-host runs '
    'with large --iters need this raised.')
_RECORD = flags.DEFINE_bool('record', False,
                            'Record baselines to --baselines instead of gating.')
_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Baselines JSON path. Default: alongside this binary (runfiles).')
_MAX_MARGIN = flags.DEFINE_float(
    'max_margin', 0.03,
    'Floor is never looser than this fractional drop below the median.')
_SIGMA_K = flags.DEFINE_float(
    'sigma_k', 3.5, 'Robust sigmas (MAD) below the median for the gate floor.')
_CONTROL_PORT = flags.DEFINE_integer('control_port', 9099,
                                     'Base control port for the C++ handshake.')
_RUNS_PER_CONFIG = flags.DEFINE_integer(
    'runs_per_config', 1,
    'Independent C++ processes to spawn per config. Raw samples per config = '
    'runs_per_config * iters. Prefer raising --iters: a single process amortizes '
    'the handshake/warmup. Raise this only to measure process-to-process spread.')
_ITERS = flags.DEFINE_integer(
    'iters', 50,
    'Timed iterations PER process, passed to the C++ runner as --iterations. One '
    'process emits this many raw H2H_ITER_MS samples. e.g. --iters=500 '
    '--runs_per_config=1 gives 500 raw samples in a single handshake.')
_DUMP = flags.DEFINE_string(
    'dump', None,
    'CSV to write every sample for your own analysis (config,run,iter,gbs,'
    'integrity). On BAP it is redirected into WORKLOAD_ARTIFACTS_DIR so the '
    'workflow uploads it as a downloadable artifact.')
_ANALYZE = flags.DEFINE_bool(
    'analyze', False,
    'Collect + --dump + print distribution stats only. No gate, no baseline '
    'write; always exits 0. Use for the data-collection workflow.')

# --- cross-host -------------------------------------------------------------
_MULTIHOST = flags.DEFINE_bool(
    'multihost', False,
    'Cross-host mode: derive the role from this hostrank instead of spawning '
    'both roles locally.')
_PEER_HOST = flags.DEFINE_string(
    'peer_host', None,
    'Receiver hostname override. Default: the rank-0 sibling of this hostname.')
_CONTROL_IFACE = flags.DEFINE_string(
    'control_interface', 'eth0',
    'Control-plane NIC for the TCP handshake. Must be eth0 on GKE.')
_DATA_IFACE = flags.DEFINE_string(
    'data_interface', '',
    'Comma-separated data NICs. Empty = auto-discover all secondary interfaces '
    '(the DRANET multi-NIC path).')
_SPAWN_RETRIES = flags.DEFINE_integer(
    'spawn_retries', 30,
    'Sender re-spawn attempts while the peer receiver is still coming up. We '
    'retry the process rather than probing the port, because a bare TCP probe '
    'would be accepted by the C++ control server and corrupt the handshake.')

_LAYERS = 32   # C++ kNumLayers (fixed in the runner)
_SHARDS = 1    # C++ kNumShards (fixed in the runner)

# (block_size_bytes, num_blocks, parallelism). On a single host both processes
# live on the same machine, so 1MB keeps 32 x 64 x 1MB = 2GB/process from OOMing
# the pod (16MB/128MB drop BAP's :50051 channel). Cross-host each side holds only
# one buffer, so larger blocks become affordable -- but keep the same list until
# the first green cross-host run, so the two topologies stay comparable.
_CONFIGS = [
    (1048576, 64, 1),
    (1048576, 64, 8),
    (1048573, 64, 4),
]

_CPP_P50_RE = re.compile(r'p50:\s*([0-9.]+)')          # "p50:   X ms"
_CPP_P90_RE = re.compile(r'p90:\s*([0-9.]+)')          # "p90:   X ms"
_CPP_P99_RE = re.compile(r'p99:\s*([0-9.]+)')          # "p99:   X ms"
_CPP_MEAN_RE = re.compile(r'Throughput:\s*([0-9.]+)')  # mean GB/s
# OPTIONAL per-iteration raw latency. If the runner prints one line per iter like
# "H2H_ITER_MS <ms>" (a ~3-line print loop over the latency vector it ALREADY
# builds to compute p50/p90/p99), this captures every raw sample -> one run gives
# all 50 points, no re-running. Absent, we fall back to the per-run median.
_CPP_RAW_RE = re.compile(r'H2H_ITER_MS\s+([0-9.]+)')
_INTEG_PASS = 'Data integrity verification PASSED'
_INTEG_FAIL = 'Data integrity verification FAILED'


def _baselines_path():
  """Baselines file: --baselines if given, else alongside this binary (runfiles).

  A CWD-relative path breaks under `bazel run` (CWD is the runfiles tree, not the
  workspace), so default to the copy shipped next to this .py via the BUILD data
  dep -- which is what the gate reads on BAP.
  """
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_cpp_baselines.json')


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _total_bytes(bs, nb):
  return _LAYERS * _SHARDS * nb * bs


def _runfiles_root():
  """Runfiles ROOT, not just this binary's dir. The C++ runner is a data dep in
  a SIBLING tree (_main/examples/microbenchmarks/), so searching from
  dirname(__file__) (=.../benchmarks) misses it. Walk up to the enclosing
  '<name>.runfiles' (covers every repo) or the '_main' workspace root."""
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
  """Find the C++ runner binary in the bazel runfiles by basename."""
  root = _runfiles_root()
  for dirpath, _, files in os.walk(root, followlinks=True):
    if basename in files:
      cand = os.path.join(dirpath, basename)
      if os.access(cand, os.X_OK):
        return cand
  raise FileNotFoundError(f'could not locate {basename} under {root}')


def _run(cmd, timeout):
  """Run a subprocess to completion; return (rc, combined_output)."""
  env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
  try:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, env=env)
    return p.returncode, p.stdout.decode('utf-8', 'replace')
  except subprocess.TimeoutExpired as e:
    out = e.output.decode('utf-8', 'replace') if e.output else ''
    return 124, out + f'\n[timeout after {timeout}s]'


def _f(regex, text):
  m = regex.search(text or '')
  return float(m.group(1)) if m else -1.0


def _samples_from(sender_out, total_bytes):
  """(gbs, raw_gbs, mean_gbs, p50, p90, p99) parsed from the sender's stdout."""
  p50 = _f(_CPP_P50_RE, sender_out)
  p90 = _f(_CPP_P90_RE, sender_out)
  p99 = _f(_CPP_P99_RE, sender_out)
  mean_gbs = _f(_CPP_MEAN_RE, sender_out)
  raw_gbs = [(total_bytes / 1e9) / (float(ms) / 1000.0)
             for ms in _CPP_RAW_RE.findall(sender_out or '') if float(ms) > 0]
  gbs = -1.0
  if p50 > 0:
    gbs = (total_bytes / 1e9) / (p50 / 1000.0)
  elif raw_gbs:
    gbs = statistics.median(raw_gbs)
  elif mean_gbs > 0:
    gbs = mean_gbs
  return gbs, raw_gbs, mean_gbs, p50, p90, p99


# --- single-host ------------------------------------------------------------


def _run_cpp(cc, bs, nb, p, port):
  """Spawn the C++ receiver (bg) + sender (timed) for ONE run on this machine.

  Returns a dict: gbs (median throughput from p50 latency), mean_gbs, p50_ms,
  p90_ms, p99_ms, integrity (bool). gbs is -1.0 on failure.
  """
  base = [cc, '--data_interface=lo', f'--peer_control_port={port}',
          f'--block_size={bs}', f'--num_blocks={nb}', f'--parallelism={p}']
  # 50 is the runner's built-in default; only pass --iterations when overriding,
  # so the correctness gate works against a runner that lacks the flag. Custom
  # iters (e.g. analyze --iters=500) does require the runner's --iterations support.
  if _ITERS.value != 50:
    base.append(f'--iterations={_ITERS.value}')
  # Receiver: capture its stdout so we can read the integrity verdict.
  recv = subprocess.Popen(
      base + ['--role=receiver', f'--numa_node={_RECEIVER_NUMA.value}'],
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
      env={**os.environ, 'PYTHONUNBUFFERED': '1'})
  time.sleep(3)  # let the control server bind
  rc, sender_out = _run(
      base + ['--role=sender', '--peer_control_ip=127.0.0.1',
              f'--numa_node={_SENDER_NUMA.value}'],
      _TIMEOUT_S.value)
  # After the sender finishes, the receiver runs a byte-by-byte integrity check
  # over the whole 2GB buffer, prints PASSED/FAILED, then exits on its own.
  # Terminating it immediately races that check -> false CORRUPT. In record/gate
  # we wait for it to finish and self-exit; in analyze we don't gate on integrity,
  # so skip the (slow) wait and kill it right away.
  if _ANALYZE.value:
    try:
      recv.terminate()
      recv_out = recv.communicate(timeout=15)[0].decode('utf-8', 'replace')
    except Exception:  # pylint: disable=broad-exception-caught
      recv.kill()
      recv_out = ''
  else:
    try:
      recv_out = recv.communicate(timeout=_TIMEOUT_S.value)[0].decode('utf-8', 'replace')
    except subprocess.TimeoutExpired:
      recv.terminate()
      try:
        recv_out = recv.communicate(timeout=15)[0].decode('utf-8', 'replace')
      except Exception:  # pylint: disable=broad-exception-caught
        recv.kill()
        recv_out = ''

  integrity_ok = (_INTEG_PASS in recv_out) and (_INTEG_FAIL not in recv_out)
  gbs, raw_gbs, mean_gbs, p50, p90, p99 = _samples_from(
      sender_out, _total_bytes(bs, nb))

  # In analyze mode the receiver is killed before its integrity check finishes,
  # so integrity_ok is expectedly False -- that is NOT a failure here; only a
  # missing throughput reading is. In record/gate, bad integrity IS a failure.
  failed = gbs < 0 or (not _ANALYZE.value and not integrity_ok)
  if failed:
    print(f'[cpp] {_label(bs, nb, p)} FAILED (rc={rc}, gbs={gbs:.3f}, '
          f'integrity_ok={integrity_ok}); sender output:\n{sender_out}\n'
          f'--- receiver tail ---\n{recv_out[-2000:]}', flush=True)
  return {'gbs': gbs, 'mean_gbs': mean_gbs, 'p50_ms': p50, 'p90_ms': p90,
          'p99_ms': p99, 'integrity': integrity_ok, 'raw_gbs': raw_gbs}


# --- cross-host -------------------------------------------------------------


def _rank_and_prefix():
  """(rank, hostname_prefix) WITHOUT importing JAX -- h2h must stay JAX-free.

  JobSet names pods <jobset>-<replicatedjob>-<jobindex>-<podindex>, so the
  trailing integer is this host's rank and prefix + '-0' names the receiver. An
  explicit index env var wins if the runner exposes one.
  """
  hn = socket.gethostname()
  head, _, tail = hn.rpartition('-')
  for var in ('JOB_COMPLETION_INDEX', 'BATCH_TASK_INDEX', 'NODE_RANK', 'RANK'):
    v = os.environ.get(var, '')
    if v.isdigit():
      return int(v), head
  if tail.isdigit():
    return int(tail), head
  raise RuntimeError(
      f'cannot derive rank from hostname {hn!r} or any of '
      'JOB_COMPLETION_INDEX/BATCH_TASK_INDEX/NODE_RANK/RANK -- see the topology '
      'dump above for what this runner actually exposes')


def _resolve_peer(prefix):
  """Receiver IP. Pod DNS can lag pod startup, so retry. --peer_control_ip wants
  an address, not a name."""
  host = _PEER_HOST.value or f'{prefix}-0'
  for _ in range(60):
    try:
      return socket.gethostbyname(host)
    except socket.gaierror:
      time.sleep(2)
  raise RuntimeError(f'could not resolve receiver host {host!r}')


def _dump_topology():
  """Everything a first cross-host run needs to explain itself: rank source,
  NICs (eth1+ means Net DRA attached the physical NICs; eth0 alone means we are
  measuring pod overlay, not line rate), and NUMA layout."""
  print('===== topology =====', flush=True)
  print(f'hostname: {socket.gethostname()}', flush=True)
  keys = ('JOB_COMPLETION_INDEX', 'BATCH_TASK_INDEX', 'NODE_RANK', 'RANK',
          'JOBSET_NAME', 'POD_NAME', 'HOSTNAME')
  for k in keys:
    if k in os.environ:
      print(f'  env {k}={os.environ[k]}', flush=True)
  for cmd in (['ip', '-br', 'link'], ['numactl', '-H']):
    rc, out = _run(cmd, 30)
    print(f'--- {" ".join(cmd)} (rc={rc}) ---\n{out}', flush=True)
  print('====================', flush=True)


def _cpp_base(cc, bs, nb, p, port):
  """Flags shared by both cross-host roles."""
  base = [cc, f'--peer_control_port={port}', f'--block_size={bs}',
          f'--num_blocks={nb}', f'--parallelism={p}',
          f'--control_interface={_CONTROL_IFACE.value}',
          f'--data_interface={_DATA_IFACE.value}']
  if _ITERS.value != 50:
    base.append(f'--iterations={_ITERS.value}')
  return base


def _run_receiver(base):
  """Block until the sender finishes and the byte-integrity check prints."""
  rc, out = _run(base + ['--role=receiver', f'--numa_node={_RECEIVER_NUMA.value}'],
                 _TIMEOUT_S.value)
  ok = (_INTEG_PASS in out) and (_INTEG_FAIL not in out)
  if not ok:
    print(f'[recv] integrity NOT confirmed (rc={rc}); tail:\n{out[-3000:]}',
          flush=True)
  return {'integrity': ok}


def _run_sender(base, peer_ip, total_bytes):
  """Re-spawn while the peer receiver is still binding its control port."""
  cmd = base + ['--role=sender', f'--peer_control_ip={peer_ip}',
                f'--numa_node={_SENDER_NUMA.value}']
  for attempt in range(_SPAWN_RETRIES.value):
    rc, out = _run(cmd, _TIMEOUT_S.value)
    gbs, raw_gbs, mean_gbs, p50, p90, p99 = _samples_from(out, total_bytes)
    if gbs > 0:
      return {'gbs': gbs, 'raw_gbs': raw_gbs, 'mean_gbs': mean_gbs,
              'p50_ms': p50, 'p90_ms': p90, 'p99_ms': p99, 'integrity': True}
    print(f'[send] attempt {attempt} produced no timing (rc={rc}); peer likely '
          'not up yet, retrying', flush=True)
    time.sleep(2)
  print(f'[send] gave up after {_SPAWN_RETRIES.value} attempts; last output:\n'
        f'{out[-3000:]}', flush=True)
  return {'gbs': -1.0, 'raw_gbs': [], 'mean_gbs': -1.0, 'p50_ms': -1.0,
          'p90_ms': -1.0, 'p99_ms': -1.0, 'integrity': True}


def _artifact(name):
  """Land a file in WORKLOAD_ARTIFACTS_DIR when BAP provides one."""
  adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  return os.path.join(adir, name) if adir else name


def _main_multihost(cc):
  """rank 0 = receiver (owns integrity), rank 1 = sender (owns throughput).

  The verdicts partition, so the two hosts never exchange results: each exits
  non-zero on its own failure and GitHub fails the job if any host does.
  """
  _dump_topology()
  rank, prefix = _rank_and_prefix()
  if rank >= 2:
    print(f'[rank {rank}] H2H needs exactly one sender/receiver pair; '
          'nothing to do on this host.', flush=True)
    return
  role = 'receiver' if rank == 0 else 'sender'
  peer_ip = _resolve_peer(prefix) if rank == 1 else None
  print(f'[rank {rank}] role={role} peer={peer_ip} '
        f'control_iface={_CONTROL_IFACE.value} '
        f'data_iface={_DATA_IFACE.value or "(auto-discover)"}', flush=True)

  writer = dump = dump_path = None
  if _DUMP.value and rank == 1:
    dump_path = _artifact(os.path.basename(_DUMP.value))
    dump = open(dump_path, 'w', newline='')
    writer = csv.writer(dump)
    writer.writerow(['config', 'run', 'iter', 'gbs', 'mean_gbs', 'p50_ms',
                     'p90_ms', 'p99_ms', 'integrity'])

  results, scalars = {}, {}
  for i, (bs, nb, p) in enumerate(_CONFIGS):
    label = _label(bs, nb, p)
    # A distinct port per config is the whole synchronisation mechanism: both
    # hosts walk the same ordered list, so equal index == equal port == the same
    # pairing. The sender's re-spawn loop absorbs any startup skew.
    base = _cpp_base(cc, bs, nb, p, _CONTROL_PORT.value + i)
    print(f'[rank {rank}] ({i + 1}/{len(_CONFIGS)}) {label} ...', flush=True)
    series, integ_all = [], True
    for run in range(max(1, _RUNS_PER_CONFIG.value)):
      if rank == 0:
        m = _run_receiver(base)
        integ_all = integ_all and m['integrity']
        continue
      m = _run_sender(base, peer_ip, _total_bytes(bs, nb))
      samples = ([(j, g) for j, g in enumerate(m['raw_gbs'])] if m['raw_gbs']
                 else ([(-1, m['gbs'])] if m['gbs'] > 0 else []))
      series.extend(g for _, g in samples)
      if writer:
        for it, g in samples:
          writer.writerow([label, run, it, f'{g:.4f}', f'{m["mean_gbs"]:.4f}',
                           f'{m["p50_ms"]:.4f}', f'{m["p90_ms"]:.4f}',
                           f'{m["p99_ms"]:.4f}', 1])
        dump.flush()

    if rank == 0:
      print(f'[rank 0] {label:<22} integrity='
            f'{"OK" if integ_all else "CORRUPT"}', flush=True)
      results[label] = {'integrity': integ_all}
      continue

    if series:
      med = _pct(series, 50)
      print(f'[measured] {label:<22} n={len(series):<4} median={med:8.3f}  '
            f'p10={_pct(series, 10):7.3f}  p90={_pct(series, 90):7.3f}  '
            f'min={min(series):7.3f}  max={max(series):7.3f}  '
            f'stdev={statistics.pstdev(series):6.3f} GB/s', flush=True)
    else:
      med = -1.0
      print(f'[measured] {label}: no timing collected', flush=True)
    results[label] = {'gbs': med, 'samples': series}
    scalars[f'{label}/cpp_gbs'] = med

  if dump:
    dump.close()
    print(f'\nWrote samples -> {dump_path}', flush=True)

  if rank == 0:
    # Integrity is this host's whole job. Publish it as an artifact too so the
    # dist plot can annotate which configs were byte-exact.
    with open(_artifact('h2h_integrity.json'), 'w') as f:
      json.dump({k: v['integrity'] for k, v in results.items()}, f, indent=2)
    bad = [k for k, v in results.items() if not v['integrity']]
    if bad and not _ANALYZE.value:
      sys.exit(f'GATE FAIL: byte-integrity failed on {bad}')
    print(f'\n[rank 0] integrity {"OK" if not bad else "FAILED: %s" % bad}',
          flush=True)
    return

  # Sender: only this host emits throughput metrics, so BAP sees one value per
  # tag instead of one per host.
  bap_metrics.emit(scalars)

  if _ANALYZE.value:
    print('\nanalyze mode: data collected, no gate/baseline. Done.', flush=True)
    return

  if _RECORD.value:
    _write_baselines(results, _artifact('h2h_cpp_baselines.json'))
    return

  missing = [k for k, v in results.items() if v['gbs'] <= 0]
  if missing:
    sys.exit(f'GATE FAIL: no throughput measured for {missing}')
  print('\nGATE PASS (throughput collected; floor comparison is report-only '
        'until baselines are recorded on this runner).', flush=True)


def _write_baselines(results, out_path):
  """baseline = median of all samples; floor = median - k*MADsigma capped at
  max_margin. Record from MANY samples (large --iters) so MAD is stable."""
  k, cap = _SIGMA_K.value, _MAX_MARGIN.value
  cfg = {'sigma_k': k, 'max_margin': cap, 'configs': {}}
  for label, r in results.items():
    s = r.get('samples') or []
    cfg['configs'][label] = {
        'baseline_gbs': round(_pct(s, 50), 3) if s else 0.0,
        'floor_gbs': round(_core_floor(s, k, cap), 3) if s else 0.0,
        'integrity': r.get('integrity', True),
        'n_samples': len(s),
    }
  with open(out_path, 'w') as f:
    json.dump(cfg, f, indent=2)
  print(f'\nRecorded {len(results)} C++ H2H baselines+floors '
        f'(sigma_k={k}, cap={cap*100:.0f}%) -> {out_path}', flush=True)


# --- shared stats -----------------------------------------------------------


def _core_floor(samples, k, max_margin):
  """Gate floor: lower edge of the normal core, capped so it is never looser
  than max_margin (mirrors the d2h/h2d _core_floor exactly).

      floor = max(median - k * MAD_sigma,  median * (1 - max_margin))

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant stddev, so a
  low tail does not drag the bound down; the cap keeps a noisy config from ending
  up looser than a flat max_margin.
  """
  med = statistics.median(samples)
  mad = statistics.median([abs(x - med) for x in samples]) if len(samples) > 1 else 0.0
  sigma = 1.4826 * mad
  core = med - k * sigma
  cap = med * (1.0 - max_margin)
  return max(core, cap)


def _pct(xs, q):
  """q-th percentile (0..100) of non-empty xs via linear interpolation."""
  xs = sorted(xs)
  if len(xs) == 1:
    return xs[0]
  pos = (len(xs) - 1) * (q / 100.0)
  lo = int(pos)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def main(_):
  cc = _locate('h2h_benchmark_runner')
  if _MULTIHOST.value:
    print(f'[cpp-gate] cross-host; C++ runner: {cc}', flush=True)
    _main_multihost(cc)
    return

  runs = max(1, _RUNS_PER_CONFIG.value)
  print(f'[cpp-gate] C++ runner: {cc}  ({runs} process(es) per config)',
        flush=True)

  writer = dump = dump_path = None
  if _DUMP.value:
    dump_path = _artifact(os.path.basename(_DUMP.value)) if os.environ.get(
        'WORKLOAD_ARTIFACTS_DIR') else _DUMP.value
    dump = open(dump_path, 'w', newline='')
    writer = csv.writer(dump)
    # iter = per-iteration index when the runner emits raw H2H_ITER_MS samples,
    # else -1 (the row is that run's median). mean_gbs/p50_ms/p90_ms/p99_ms are
    # the runner's own summary of the 50 internal iters -- available every run,
    # repeated on each raw row so every row is self-describing.
    writer.writerow(['config', 'run', 'iter', 'gbs', 'mean_gbs', 'p50_ms',
                     'p90_ms', 'p99_ms', 'integrity'])

  # results[label] = representative {gbs (median across all samples), integrity}
  # scalars[tag] = value; collected across configs and written in ONE event file
  # after the loop (a writer per config would scatter N event files).
  results = {}
  scalars = {}
  for i, (bs, nb, p) in enumerate(_CONFIGS):
    label = _label(bs, nb, p)
    port = _CONTROL_PORT.value + i
    series, integ_all = [], True
    print(f'[cpp-gate] ({i + 1}/{len(_CONFIGS)}) {label}: {runs} process(es) ...',
          flush=True)
    for run in range(runs):
      m = _run_cpp(cc, bs, nb, p, port)
      integ_all = integ_all and m['integrity']
      # Prefer per-iteration raw samples (one run -> 50 points); else the run's
      # single median.
      if m['raw_gbs']:
        samples = [(j, g) for j, g in enumerate(m['raw_gbs'])]
      elif m['gbs'] > 0:
        samples = [(-1, m['gbs'])]
      else:
        samples = []
      series.extend(g for _, g in samples)
      if writer:
        for it, g in samples:
          writer.writerow([label, run, it, f'{g:.4f}', f'{m["mean_gbs"]:.4f}',
                           f'{m["p50_ms"]:.4f}', f'{m["p90_ms"]:.4f}',
                           f'{m["p99_ms"]:.4f}', int(m['integrity'])])
        dump.flush()
      if runs > 1 and (run + 1) % 10 == 0:
        print(f'    {label}: {run + 1}/{runs} runs done', flush=True)

    if series:
      med = _pct(series, 50)
      print(f'[measured] {label:<22} n={len(series):<4} median={med:8.3f}  '
            f'p10={_pct(series, 10):7.3f}  p90={_pct(series, 90):7.3f}  '
            f'min={min(series):7.3f}  max={max(series):7.3f}  '
            f'stdev={statistics.pstdev(series):6.3f} GB/s  '
            f'integrity={"n/a" if _ANALYZE.value else ("OK" if integ_all else "CORRUPT")}',
            flush=True)
    else:
      med = -1.0
      print(f'[measured] {label}: ALL {runs} process(es) failed', flush=True)
    results[label] = {'gbs': med, 'integrity': integ_all, 'samples': series}
    scalars[f'{label}/cpp_gbs'] = med

  bap_metrics.emit(scalars)

  if dump:
    dump.close()
    print(f'\nWrote samples for {len(_CONFIGS)} config(s) x {runs} process(es) '
          f'-> {dump_path}', flush=True)

  if _ANALYZE.value:
    print('\nanalyze mode: data collected, no gate/baseline. Done.', flush=True)
    return

  if _RECORD.value:
    # On BAP the runfiles tree is read-only, so write into WORKLOAD_ARTIFACTS_DIR
    # (BAP uploads it as a downloadable artifact you then commit); locally, write
    # straight to the file.
    out_path = _baselines_path()
    if os.environ.get('WORKLOAD_ARTIFACTS_DIR'):
      out_path = _artifact('h2h_cpp_baselines.json')
    _write_baselines(results, out_path)
    return

  # Gate mode: CORRECTNESS ONLY. On a single machine H2H runs over loopback, so
  # the throughput is not a product metric -- pass/fail is the receiver's
  # byte-integrity check alone. Throughput is printed + emitted for observability
  # but never fails the build, and no baseline/floor is needed (integrity
  # self-checks against the runner's deterministic byte pattern).
  bad = []
  print('\nH2H C++ correctness gate (single-machine loopback; throughput is informational only)\n')
  print('config                    median   integrity  verdict')
  print('-' * 58)
  for label, r in results.items():
    integ = r['integrity']
    print(f'{label:<22} {r["gbs"]:8.3f}  {"OK" if integ else "CORRUPT":<9} '
          f'{"PASS" if integ else "FAIL"}{"" if integ else " <-- DATA CORRUPTION"}')
    if not integ:
      bad.append(label)

  if bad:
    print(f'\nGATE FAIL: byte-integrity failed on {len(bad)} config(s): {bad}',
          file=sys.stderr)
    sys.exit(1)
  print('\nGATE PASS: all configs byte-exact.', flush=True)


if __name__ == '__main__':
