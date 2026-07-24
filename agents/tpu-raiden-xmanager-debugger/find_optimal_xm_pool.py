#!/usr/bin/env python3

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
import subprocess
import argparse
import re
from typing import Dict, List, Any

XM_TOOL = "/google/bin/releases/gemini-agents-xmanager/xmanager_tool"

def run_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout

def parse_allocs() -> Dict[str, Any]:
    stdout = run_cmd([XM_TOOL, "list_resource_allocs"])
    allocs = {}
    current_alloc = None
    for line in stdout.splitlines():
        if line.startswith("group:"):
            current_alloc = line[6:].strip()  # strip group:
            allocs[current_alloc] = 0 # default 0 implies flex pool bounds
        elif current_alloc and ":" in line and line.startswith("    "):
            # A restriction block (HighlyAvailable, etc.)
            parts = line.split(":")
            if len(parts) == 2:
                res_type = parts[0].strip()
                try:
                    count = int(parts[1].strip())
                    if not isinstance(allocs[current_alloc], dict):
                        allocs[current_alloc] = {}
                    allocs[current_alloc][res_type] = count
                except ValueError:
                    pass
    return allocs

def parse_pool_capacity(pool: str) -> Dict[str, Dict[str, float]]:
    stdout = run_cmd([XM_TOOL, "get_pool_capacity", f"--pool={pool}"])
    cells = {}
    current_cell = None
    for line in stdout.splitlines():
        if "/" in line and ":" in line:
            current_cell = line.split("/")[0].strip()
            if current_cell not in cells:
                cells[current_cell] = {}
        elif current_cell and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                res = parts[0].strip()
                val_str = parts[1].strip()
                match = re.match(r"([0-9.]+)", val_str)
                if match:
                    cells[current_cell][res] = float(match.group(1))
    return cells

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu_type", type=str, default="ghostfish", help="Type of TPU to search for (partial match allowed).")
    parser.add_argument("--min_tpus", type=int, default=4, help="Minimum number of TPU chips required.")
    parser.add_argument("--min_gcu", type=float, default=2.0)
    parser.add_argument("--min_memory_gb", type=float, default=8.0)
    args = parser.parse_args()

    print(f"Scanning for {args.min_tpus}x {args.tpu_type} with min {args.min_gcu} GCU and {args.min_memory_gb}GB RAM...")
    allocs = parse_allocs()
    
    candidates = []
    pool_capacities = {}

    for alloc, alloc_res in allocs.items():
        pool = alloc.split("/")[0] if "/" in alloc else alloc
        if pool not in pool_capacities:
            pool_capacities[pool] = parse_pool_capacity(pool)
        
        cells = pool_capacities[pool]
        
        for cell_name, cell_res in cells.items():
            cell_tpu_count = sum(v for k, v in cell_res.items() if args.tpu_type in k and "tpu" in k)
            alloc_tpu_count = cell_tpu_count 
            
            if isinstance(alloc_res, dict):
                matched_tpus = sum(v for k, v in alloc_res.items() if args.tpu_type in k and "tpu" in k)
                if matched_tpus > 0:
                   alloc_tpu_count = min(matched_tpus, cell_tpu_count)
                else:
                    # If this alloc is restricted to something else, and has no target TPUs listed.
                    alloc_tpu_count = 0 
            
            if alloc_tpu_count >= args.min_tpus:
                gcu = cell_res.get("gcu", 0.0)
                mem = cell_res.get("memory", 0.0)
                if gcu >= args.min_gcu and mem >= args.min_memory_gb:
                    candidates.append({
                        "alloc": alloc,
                        "cell": cell_name,
                        "tpus": alloc_tpu_count,
                        "gcu": gcu,
                        "memory": mem,
                        "score": gcu * mem 
                    })
                    
    if not candidates:
        print("No eligible allocations or cells found with the required resources!")
        return

    candidates.sort(key=lambda x: (x["tpus"], x["score"]), reverse=True)
    
    best = candidates[0]
    print(f"\n--- Best Found Allocation ---")
    print(f"Resource Alloc:  --xm_resource_alloc={best['alloc']}")
    print(f"Target Cell:     --cell={best['cell']}")
    print(f"Available TPUs:  {best['tpus']}")
    print(f"Available GCU:   {best['gcu']}")
    print(f"Available Mem:   {best['memory']}GiB")

if __name__ == "__main__":
    main()
