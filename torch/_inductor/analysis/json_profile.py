"""
JsonProfile class for analyzing JSON trace files from profiling.
"""

import json
import logging
import multiprocessing as mp
import os
import tempfile
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import torch
from torch._inductor.analysis.device_info import (
    compute_device_ridgepoint,
    lookup_device_info,
)
from torch._inductor.utils import tabulate_2d, zip_dicts
from torch.utils._ordered_set import OrderedSet

from tqdm import tqdm

from .dag_nodes import TraceDAG
from .types import _IdxEvt, Device, DeviceMap, KernelStats, Table
from .utils import (
    _augment_trace_helper,
    _calculate_flops,
    _create_extern_mapping,
    _dtype_map,
    _estimate_gb,
    _wrap_text,
    compute_stats_chunk,
    parse_list,
)


try:
    import graphviz
except ImportError:
    pass

log = logging.getLogger(__name__)


class JsonProfile:
    _devices: DeviceMap

    def __init__(
        self,
        path: str,
        benchmark_name: Optional[str] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ):
        """
        Convenience class for running common operations on chrome/perfetto json traces.
        """
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.benchmark_name = benchmark_name
        if dtype is None:
            self.dtype = None
        elif isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            if dtype in _dtype_map:
                self.dtype = _dtype_map[dtype]
            else:
                self.dtype = None
        self._create_devices()

    def convert_dtype(self, event: dict[str, Any]) -> Optional[torch.dtype]:
        """
        Each op has a list of dtypes for each input arg. We need to convert these into a single dtype for flop estimation.
        Issues:
         - converting the strings to concrete torch.dtypes
         - What if we have float32, float, float16 all in the inputs? Our choice is to use the largest buffer dtype.
        """

        if (
            "Input Dims" not in event["args"]
            or "Input type" not in event["args"]
            or "Concrete Inputs" not in event["args"]
        ):
            if "bfloat16" in event["name"]:
                return torch.bfloat16
            elif "float16" in event["name"]:
                return torch.float16
            else:
                return None

        input_sizes = event["args"]["Input Dims"]
        input_types = event["args"]["Input type"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        assert len(input_sizes) == len(input_types)
        assert len(input_types) == len(concrete_inputs)

        if len(input_sizes) == 0:
            raise RuntimeError("Empty input_sizes and input_types")

        biggest_size = 0
        biggest_index = 0
        for i in range(len(input_sizes)):
            if concrete_inputs[i] != "":
                # concrete inputs are usually small tensors, so we can just skip
                continue
            my_size = input_sizes[i]
            total_size = sum(parse_list(my_size))
            if total_size > biggest_size:
                biggest_size = total_size
                biggest_index = i
        ret_type = input_types[biggest_index]
        if ret_type in _dtype_map:
            return _dtype_map[ret_type]
        raise RuntimeError(f"Unknown type: {ret_type}. Please add to _dtype_map.")

    def _create_devices(self) -> None:
        self._devices = {}
        for dev in self.data["deviceProperties"]:
            name = dev["name"]
            device_info = lookup_device_info(name)

            if device_info is None:
                log.info(
                    "Unsupported device in profile: %s, please consider contributing to _device_mapping.",
                    name,
                )
            self._devices[dev["id"]] = Device(
                name, dev["id"], device_info, defaultdict(OrderedSet)
            )

    def calculate_flops(self, event: dict[str, Any]) -> int:
        return _calculate_flops(event)

    def estimate_gb(self, event: dict[str, Any]) -> float:
        return _estimate_gb(event)

    def augment_trace(self) -> None:
        self.data = _augment_trace_helper(self.data)

    def _compute_stats(self, use_parallel: bool = True) -> None:
        """populates the name -> stats map"""
        num_events = len(self.events)

        # Use parallel processing for large traces
        if use_parallel and num_events > 10000:
            self._compute_stats_parallel()
            return

        # Original single-threaded implementation for small traces
        for event in self.events:
            if "cat" not in event or "args" not in event or event["cat"] != "kernel":
                continue
            if "device" not in event["args"]:
                continue
            dev_tmp = event["args"]["device"]
            if dev_tmp not in self._devices:
                continue
            dev = self._devices[event["args"]["device"]]

            dur = event["dur"]  # us
            if "kernel_flop" in event["args"]:
                assert dur != 0
                # 1,000,000us/s * flop / us
                op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
            else:
                op_flops = 0

            if "kernel_num_gb" in event["args"]:
                assert dur != 0
                # 1,000,000us/s * gb  = gb/s
                op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
            else:
                op_gbps = 0

            if dev.info is not None:
                dtype = self.convert_dtype(event) or self.dtype
                if dtype is None:
                    raise RuntimeError(
                        "dtype is not found on tensor and default dtype is not set"
                    )
                achieved_flops = 100 * op_flops / (1e12 * dev.info.tops[dtype])
                achieved_bandwidth = 100 * op_gbps / dev.info.dram_bw_gbs

                # Calculate roofline bound type
                bound_type = self._calculate_bound_type(
                    dev.name, op_flops, op_gbps, dtype
                )
            else:
                achieved_flops = 0
                achieved_bandwidth = 0
                bound_type = "unknown"

            if "name" not in event:
                continue
            kernel_name = event["name"]
            dev.stats[kernel_name].add(
                KernelStats(
                    flops=op_flops,
                    bw=op_gbps,
                    latency=dur,
                    achieved_bandwidth=achieved_bandwidth,
                    achieved_flops=achieved_flops,
                    bound_type=bound_type,
                )
            )

    def _calculate_bound_type(
        self, device_name: str, op_flops: float, op_gbps: float, dtype: torch.dtype
    ) -> str:
        """
        Calculate whether a kernel is compute-bound or memory-bound based on roofline analysis.

        Args:
            device_name: Name of the device (e.g., "NVIDIA H100")
            op_flops: Achieved FLOPS per second for this kernel instance
            op_gbps: Achieved GB/s for this kernel instance
            dtype: Data type being used

        Returns:
            "compute" if compute-bound, "memory" if memory-bound, "unknown" if calculation fails
        """
        # Calculate the device ridgepoint B = TOPS / BW
        ridgepoint = compute_device_ridgepoint(device_name, dtype)
        if ridgepoint is None:
            return "unknown"

        # Convert op_flops to TOPS
        op_tops = op_flops / 1e12

        # Calculate kernel's operational intensity: TOPS / GB/s
        if op_gbps == 0:
            # No memory traffic - pure compute
            return "compute"

        kernel_intensity = op_tops / op_gbps

        # Compare to ridgepoint: if kernel intensity >= ridgepoint, it's compute-bound
        if kernel_intensity >= ridgepoint:
            return "compute"
        else:
            return "memory"

    def _compute_stats_parallel(self) -> None:
        """Parallel version of _compute_stats."""
        num_events = len(self.events)
        num_workers = min(mp.cpu_count(), max(1, num_events // 2000))
        chunk_size = max(1, num_events // num_workers)

        print(
            f"Computing statistics for {num_events} events using {num_workers} workers..."
        )

        # Split events into chunks
        chunks = []
        for i in range(0, num_events, chunk_size):
            chunk = self.events[i : i + chunk_size]
            chunks.append((chunk, self._devices, self.dtype))

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(compute_stats_chunk, chunk_args)
                for chunk_args in chunks
            ]

            # Merge results
            for future in as_completed(futures):
                local_device_stats = future.result()

                # Merge local_device_stats into self._devices
                for dev_id, local_stats in local_device_stats.items():
                    if dev_id in self._devices:
                        dev = self._devices[dev_id]
                        for kernel_name, stats_set in local_stats.items():
                            dev.stats[kernel_name].update(stats_set)

        print("Statistics computation completed.")

    def _create_single_table(self, dev: Device) -> Table:
        """Create a table with the devices mapped to indices."""
        headers = [
            "Kernel Name",
            "Kernel Count",
            "FLOPS",
            "Kernel Reads (GB)",
            "Dur (us)",
            "Achieved FLOPS %",
            "Achieved Bandwidth %",
        ]
        rows: dict[str, list[str]] = {}

        def safe_div_format(x: float, y: float) -> str:
            if y == 0:
                return "0.0"
            return f"{x / y:.4f}"

        for kernel_name, stats_set in dev.stats.items():
            ker_count = 0
            flops = 0
            flops_count = 0
            achieved_flops = 0.0
            bw = 0.0
            bw_count = 0
            achieved_bandwidth = 0.0
            latency = 0.0
            for stats in stats_set:
                if stats.flops != 0:
                    flops += stats.flops
                    achieved_flops += stats.achieved_flops
                    flops_count += 1
                if stats.bw != 0:
                    bw += stats.bw
                    achieved_bandwidth += stats.achieved_bandwidth
                    bw_count += 1
                latency += stats.latency
                ker_count += 1
            assert ker_count != 0
            rows[kernel_name] = [
                str(ker_count),
                safe_div_format(flops, flops_count),
                safe_div_format(bw, bw_count),
                safe_div_format(latency, ker_count),
                safe_div_format(achieved_flops, flops_count),
                safe_div_format(achieved_bandwidth, bw_count),
            ]

        return headers, rows

    def _create_tables(self, devs: DeviceMap) -> dict[int, Table]:
        return {idx: self._create_single_table(dev) for idx, dev in devs.items()}

    def _combine_tables(
        self, table1: Table, table1_name: str, table2: Table, table2_name: str
    ) -> Table:
        new_headers = (
            ["Kernel Name"]
            + [f"{table1_name} {head}" for head in table1[0][1:]]
            + [f"{table2_name} {head}" for head in table2[0][1:]]
        )
        t1_length = len(table1[0][1:])
        t2_length = len(table2[0][1:])
        new_rows = {}

        for key, row1, row2 in zip_dicts(
            table1[1],
            table2[1],
            d1_default=["Empty"] * t1_length,
            d2_default=["Empty"] * t2_length,
        ):
            assert row1 is not None
            assert row2 is not None
            new_rows[key] = row1 + row2
        return new_headers, new_rows

    def report(
        self, other: Optional["JsonProfile"] = None, name_limit: int = 40
    ) -> str:
        def create_ret(
            table_headers: list[str], table_rows: dict[str, list[str]]
        ) -> str:
            table_flattened = [
                [kernel_name[:name_limit], *kernel_vals]
                for kernel_name, kernel_vals in table_rows.items()
            ]
            return tabulate_2d(table_flattened, headers=table_headers)

        if other is not None:
            self._compute_stats()
            other._compute_stats()

            self_tables = self._create_tables(self._devices)
            other_tables = self._create_tables(other._devices)

            self_name = (
                self.benchmark_name if self.benchmark_name is not None else "Table 1"
            )
            other_name = (
                other.benchmark_name if other.benchmark_name is not None else "Table 2"
            )

            ret = []
            assert self._devices.keys() == other._devices.keys()
            for device_idx, t1, t2 in zip_dicts(
                self_tables, other_tables, d1_default=None, d2_default=None
            ):
                assert t1 is not None
                assert t2 is not None
                table_headers, table_rows = self._combine_tables(
                    t1, self_name, t2, other_name
                )
                tab_string = create_ret(table_headers, table_rows)
                ret.append(f"{self._devices[device_idx]}:\n{tab_string}")
            return "\n".join(ret)
        self._compute_stats()

        self_tables = self._create_tables(self._devices)

        ret = []
        for idx, table in self_tables.items():
            table_headers, table_rows = table
            tab_string = create_ret(table_headers, table_rows)
            ret.append(f"{self._devices[idx]}:\n{tab_string}")
        return "\n".join(ret)

    def _build_extern_and_kernel_maps(self, use_parallel: bool = True):
        """Build per-thread intervals with correct parent pointers.
        Also returns kernels and cudaLaunchKernel indices per thread.
        """
        extern_map = _create_extern_mapping(self.data)

        num_events = len(self.events)

        # Use parallel processing for large traces
        if use_parallel and num_events > 5000:
            return self._build_extern_and_kernel_maps_parallel(extern_map)

        # Original single-threaded implementation for small traces
        print("1) Collect intervals per tid (handle 'X' and match 'B'/'E')")

        per_tid_intervals: dict[int, list[_IdxEvt]] = defaultdict(list)
        open_stack: dict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)

        for gi, ev in enumerate(self.events):
            ph = ev.get("ph")
            tid = ev.get("tid", 0)
            if ph == "X":
                ts = ev.get("ts", 0)
                dur = ev.get("dur", 0)
                per_tid_intervals[tid].append(
                    _IdxEvt(
                        name=ev.get("name", ""),
                        cat=ev.get("cat", ""),
                        ts=ts,
                        end_ts=ts + dur,
                        tid=tid,
                        parent=None,  # will fill in below
                        idx=gi,
                    )
                )
            elif ph == "B":
                open_stack[tid].append((ev, gi))
            elif ph == "E":
                if open_stack[tid]:
                    beg_ev, beg_idx = open_stack[tid].pop()
                    per_tid_intervals[tid].append(
                        _IdxEvt(
                            name=beg_ev.get("name", ""),
                            cat=beg_ev.get("cat", ""),
                            ts=beg_ev.get("ts", 0),
                            end_ts=ev.get("ts", 0),
                            tid=tid,
                            parent=None,  # will fill in below
                            idx=beg_idx,
                        )
                    )

        print(
            "2) For each thread, sort by (ts, -end_ts) and rebuild parent pointers via sweep line"
        )
        per_tid_compact: dict[int, list[_IdxEvt]] = {}
        launches_per_tid: dict[int, list[_IdxEvt]] = defaultdict(list)
        kernels: list[dict[str, Any]] = []

        for tid, arr in per_tid_intervals.items():
            arr.sort(key=lambda x: (x.ts, -x.end_ts))
            stack: list[int] = []
            rebuilt: list[_IdxEvt] = []

            for ev in arr:
                while stack and rebuilt[stack[-1]].end_ts <= ev.ts:
                    stack.pop()
                parent_idx = stack[-1] if stack else None
                rebuilt.append(
                    _IdxEvt(
                        name=ev.name,
                        cat=ev.cat,
                        ts=ev.ts,
                        end_ts=ev.end_ts,
                        tid=tid,
                        parent=parent_idx,
                        idx=ev.idx,
                    )
                )
                stack.append(len(rebuilt) - 1)

            per_tid_compact[tid] = rebuilt

            # Collect launches and kernels from rebuilt intervals
            for it in rebuilt:
                if "cudaLaunchKer" in it.name:
                    launches_per_tid[tid].append(it)
                if it.cat == "kernel":
                    kernels.append(
                        {
                            "event": self.events[it.idx],
                            "ts": it.ts,
                            "tid": tid,
                            "name": it.name,
                        }
                    )

            # launches need to be time-sorted for bisect
            launches_per_tid[tid].sort(key=lambda x: x.ts)

        return extern_map, kernels, per_tid_compact, launches_per_tid

    def _build_extern_and_kernel_maps_parallel(self, extern_map):
        """Parallel version of _build_extern_and_kernel_maps."""
        from .utils import process_events_chunk

        num_events = len(self.events)
        num_workers = min(
            mp.cpu_count(), max(1, num_events // 1000)
        )  # At least 1000 events per worker
        chunk_size = max(1, num_events // num_workers)

        print(f"Processing {num_events} events using {num_workers} workers...")

        # Split events into chunks
        chunks = []
        for i in range(0, num_events, chunk_size):
            chunk = self.events[i : i + chunk_size]
            chunks.append((chunk, i, self._devices, self.dtype))

        # Process chunks in parallel using threads (since we're I/O bound on data structures)
        per_tid_intervals_combined = defaultdict(list)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_events_chunk, chunk_args)
                for chunk_args in chunks
            ]

            for future in as_completed(futures):
                per_tid_intervals, _ = future.result()

                # Merge results
                for tid, intervals in per_tid_intervals.items():
                    per_tid_intervals_combined[tid].extend(intervals)

        print(
            f"Event processing completed. Processing {len(per_tid_intervals_combined)} threads..."
        )

        # Continue with the rest of the processing (sorting, parent pointers, etc.)
        # This part is harder to parallelize due to dependencies
        per_tid_compact: dict[int, list[_IdxEvt]] = {}
        launches_per_tid: dict[int, list[_IdxEvt]] = defaultdict(list)
        kernels: list[dict[str, Any]] = []

        # Process each thread's intervals in parallel
        def process_tid_intervals(tid_data):
            tid, arr = tid_data
            arr.sort(key=lambda x: (x.ts, -x.end_ts))

            stack: list[int] = []
            rebuilt: list[_IdxEvt] = []

            for ev in arr:
                while stack and rebuilt[stack[-1]].end_ts <= ev.ts:
                    stack.pop()
                parent_idx = stack[-1] if stack else None
                rebuilt.append(
                    _IdxEvt(
                        name=ev.name,
                        cat=ev.cat,
                        ts=ev.ts,
                        end_ts=ev.end_ts,
                        tid=tid,
                        parent=parent_idx,
                        idx=ev.idx,
                    )
                )
                stack.append(len(rebuilt) - 1)

            # Collect launches and kernels
            launches = []
            tid_kernels = []

            for it in rebuilt:
                if "cudaLaunchKer" in it.name:
                    launches.append(it)
                if it.cat == "kernel":
                    tid_kernels.append(
                        {
                            "event": self.events[it.idx],
                            "ts": it.ts,
                            "tid": tid,
                            "name": it.name,
                        }
                    )

            launches.sort(key=lambda x: x.ts)

            return tid, rebuilt, launches, tid_kernels

        # Process each thread's data in parallel
        with ThreadPoolExecutor(
            max_workers=min(len(per_tid_intervals_combined), num_workers)
        ) as executor:
            futures = [
                executor.submit(process_tid_intervals, (tid, arr))
                for tid, arr in per_tid_intervals_combined.items()
            ]

            for future in as_completed(futures):
                tid, rebuilt, launches, tid_kernels = future.result()
                per_tid_compact[tid] = rebuilt
                launches_per_tid[tid] = launches
                kernels.extend(tid_kernels)

        print(f"Thread processing completed. Found {len(kernels)} kernels.")

        return extern_map, kernels, per_tid_compact, launches_per_tid

    def _find_launch_for_kernel(
        self, kernel_ev: dict, launches_per_tid: dict[int, list[_IdxEvt]]
    ) -> Optional[_IdxEvt]:
        """Find the cudaLaunchKernel that encloses kernel start, using bisect on the kernel's tid and also trying nearby CPU tids when needed."""
        ts_k = kernel_ev.get("ts", 0)
        # Try same tid first (some traces put the launch on the same logical tid as API calls)
        tid = kernel_ev.get("tid", 0)
        for try_tid in (tid,):
            la = launches_per_tid.get(try_tid)
            if not la:
                continue
            # Find rightmost launch with start_ts <= ts_k
            idx = bisect_right([x.ts for x in la], ts_k) - 1
            if idx >= 0:
                cand = la[idx]
                if cand.ts <= ts_k <= cand.end_ts:
                    return cand
        # Fallback: scan a tiny set of other tids (cheap) — prefer the nearest enclosing one by (end_ts - ts_k)
        best = None
        best_slack = 1 << 62
        for la in launches_per_tid.values():
            # binary search to nearest candidate
            idx = bisect_right([x.ts for x in la], ts_k) - 1
            if idx >= 0:
                cand = la[idx]
                if cand.ts <= ts_k <= cand.end_ts:
                    slack = cand.end_ts - ts_k
                    if slack < best_slack:
                        best, best_slack = cand, slack
        return best

    def _collect_chain_from(
        self,
        start_evt: _IdxEvt,
        per_tid_compact: dict[int, list[_IdxEvt]],
        include_all: bool = True,
    ) -> list[str]:
        """Walk to the root, returning names from outermost -> leaf.
        include_all=True keeps user annotations like 'expected' so you see the full tree.
        """
        arr = per_tid_compact[start_evt.tid]
        chain: list[str] = []
        cur = start_evt
        tmp: list[str] = []
        while cur is not None:
            nm = cur.name
            if include_all or (
                nm.startswith("aten::")
                or "contiguous" in nm
                or "clone" in nm
                or "copy" in nm
                or "empty" in nm
                or nm.startswith("torch::")
                or nm.startswith("c10::")
                or any(
                    op in nm
                    for op in ("linear", "conv", "matmul", "bmm", "addmm", "mm")
                )
            ):
                tmp.append(nm)
            cur = arr[cur.parent] if (cur.parent is not None) else None
        tmp.reverse()
        chain.extend(tmp)
        return chain

    def build_trace_dag(self) -> TraceDAG:
        """
        Fast DAG build:
        - Pre-index per-thread with parent pointers (O(N))
        - Resolve each kernel to a cpu site (External id or cudaLaunch via bisect) (O(log N) each)
        - Add edges only (set handles de-dupe)
        The slow bits are from (a) O(K·N) overlap scans and (b) de-duping whole chains. Below is a drop-in, sweep-line + parent-pointer approach that makes everything essentially O(N log N):
        Pre-index every cpu_op / user_annotation / cudaLaunchKernel by thread, with parent pointers built from the per-thread stack (no overlap scans).
        For each kernel, resolve its launching site fast:
        Prefer args["External id"] → cpu_op (your existing mapping).
        Else, find the cudaLaunchKernel whose interval contains the kernel start using bisect over a per-thread sorted list (O(log N)).
        Build the op chain by walking parent pointers from the launch (or external op) up to the root; don't de-dup chains—just add edges (the set takes care of uniqueness).
        Intern strings and store compact structs to keep memory/cache friendly.
        Notes on why this is fast
        No per-kernel O(N) overlap scans; ancestor resolution is O(log N) via bisect → O(K log N).
        Parent pointers are built in one linear sweep per thread.
        No chain de-duplication pass; sets make edge/node insertion idempotent.
        """
        dag = TraceDAG()

        # Compute stats first to have performance data available
        self._compute_stats()

        extern_map, kernels, per_tid_compact, launches_per_tid = (
            self._build_extern_and_kernel_maps()
        )

        # Track operation instance counts
        op_instance_counts = defaultdict(int)

        # Pre-intern node objects to reduce dict churn
        def _get_or_add(name: str, typ: str):
            node = dag.nodes.get(name)
            if node is None:
                node = dag.add_node(name, typ)
            return node

        # Process kernels sequentially
        for k in tqdm(kernels, desc="Processing kernels"):
            kev = k["event"]
            kname = kev.get("name", "unknown_kernel")
            kdur = kev.get("dur", 0.0)
            ktid = kev.get("tid", 0)

            # 1) Prefer External id mapping to a concrete cpu_op
            start_chain_names: list[str] = []
            ext_ok = False
            if "args" in kev and "External id" in kev["args"]:
                ext_id = kev["args"]["External id"]
                lst = extern_map.get(ext_id)
                if lst:
                    # Use the cpu_op we mapped; find its compact record via per-thread binary search
                    cpu_ev = lst[0]
                    tid = cpu_ev.get("tid", 0)
                    arr = per_tid_compact.get(tid, [])
                    if arr:
                        # binary search nearest exact match by ts
                        ts = cpu_ev.get("ts", 0)
                        idx = bisect_right([x.ts for x in arr], ts) - 1
                        # walk forward to the first with same ts/name if needed
                        found = None
                        for j in range(max(idx, 0), min(idx + 4, len(arr))):
                            if arr[j].ts == ts and arr[j].name == cpu_ev.get(
                                "name", ""
                            ):
                                found = arr[j]
                                break
                        if found:
                            start_chain_names = self._collect_chain_from(
                                found, per_tid_compact
                            )
                            ext_ok = True

            # 2) Else resolve launch site by bisect
            if not ext_ok:
                launch = self._find_launch_for_kernel(kev, launches_per_tid)
                if launch:
                    start_chain_names = self._collect_chain_from(
                        launch, per_tid_compact
                    )

            # If nothing found, skip — we only keep kernel-linked chains
            if not start_chain_names:
                continue

            # Count operation instances and add op nodes and edges
            for nm in start_chain_names:
                op_instance_counts[nm] += 1

            prev = None
            for nm in start_chain_names:
                _get_or_add(nm, "op")
                if prev is not None:
                    dag.add_edge(prev, nm)
                prev = nm

            # Add the kernel node + edge from last op
            kernel_node = _get_or_add(kname, "kernel")
            if prev is not None:
                dag.add_edge(prev, kname)
            dag.add_kernel_instance(kname, float(kdur), int(ktid))

            # Collect performance statistics for this kernel instance
            if "device" in kev.get("args", {}):
                device_id = kev["args"]["device"]
                if device_id in self._devices:
                    dev = self._devices[device_id]
                    if kname in dev.stats:
                        # For each kernel instance, add its performance stats
                        # Since the stats are collected from the same kernel events we're processing,
                        # we can use all stats for this kernel name
                        stats_list = list(dev.stats[kname])
                        if stats_list:
                            # Find the best matching stats by latency
                            best_stats = min(
                                stats_list, key=lambda s: abs(s.latency - kdur)
                            )
                            latency_diff = abs(best_stats.latency - kdur)
                            if (
                                latency_diff < 100.0
                            ):  # Increase tolerance to 100 microseconds
                                kernel_node.achieved_flops_list.append(
                                    best_stats.achieved_flops
                                )
                                kernel_node.achieved_bandwidth_list.append(
                                    best_stats.achieved_bandwidth
                                )
                                kernel_node.bound_type_list.append(
                                    best_stats.bound_type
                                )

        # Store operation instance counts in the DAG nodes
        for op_name, count in tqdm(
            op_instance_counts.items(), desc="Processing operation instances"
        ):
            if op_name in dag.nodes:
                node = dag.nodes[op_name]
                if node.node_type == "op":
                    # Add instance count to operation nodes
                    node.instance_count = count

        return dag

    def _trace_up_from_kernel(
        self, kernel_info: Dict, thread_stacks: Dict
    ) -> List[Dict]:
        """
        Trace up from a kernel to find the operation chain that led to it.
        This uses timing overlap to find parent operations.
        """
        chain = []
        kernel_event = kernel_info["event"]
        kernel_ts = kernel_info["ts"]
        kernel_dur = kernel_info.get("dur", 0)
        kernel_end_ts = kernel_ts + kernel_dur

        # Find events that overlap with the kernel timing
        overlapping_ops = []

        for event in self.events:
            if event.get("cat") not in ["cpu_op", "user_annotation"]:
                continue

            event_ts = event.get("ts", 0)
            event_dur = event.get("dur", 0)
            event_end_ts = event_ts + event_dur

            # Check if events overlap
            if event_ts <= kernel_end_ts and event_end_ts >= kernel_ts:
                op_name = event.get("name", "")

                # Filter for relevant operations (aten:: ops)
                if op_name.startswith("aten::") or any(
                    x in op_name for x in ["contiguous", "clone", "copy", "empty"]
                ):
                    overlapping_ops.append(
                        {
                            "name": op_name,
                            "ts": event_ts,
                            "dur": event_dur,
                            "end_ts": event_end_ts,
                            "event": event,
                        }
                    )

        # Sort by start time to build the chain
        overlapping_ops.sort(key=lambda x: x["ts"])

        # Build the operation chain
        for op in overlapping_ops:
            chain.append(op)

        # Add the kernel at the end
        chain.append(
            {
                "name": kernel_info["name"],
                "ts": kernel_ts,
                "dur": kernel_dur,
                "event": kernel_event,
            }
        )

        return chain

    def _find_connected_kernels(self, dag: TraceDAG, op_node_name: str) -> list[str]:
        """
        Find all kernel nodes that are reachable from the given operation node.
        Uses BFS to traverse the DAG and collect all connected kernels.
        """
        visited = OrderedSet()
        kernel_nodes = []
        queue = [op_node_name]

        print(f"DEBUG: Finding connected kernels for op: {op_node_name}")
        print(f"DEBUG: Total edges in DAG: {len(dag.edges)}")

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Find all children of current node
            children_found = []
            for parent, child in dag.edges:
                if parent == current and child not in visited:
                    children_found.append(child)
                    if dag.nodes[child].node_type == "kernel":
                        kernel_nodes.append(child)
                    else:
                        queue.append(child)

        print(f"DEBUG: Total kernels found: {len(kernel_nodes)}")
        return kernel_nodes

    def visualize_trace_dag(
        self,
        dag: TraceDAG,
        output_path: str = "trace_dag.png",
        format: str = "png",
        color_mode: Optional[str] = None,
        baseline_profile: Optional["JsonProfile"] = None,
        height: Optional[int] = None,
        compact: bool = True,
    ) -> None:
        """
        Create a PNG visualization of the trace DAG with operations at top and kernels at bottom.

        Args:
            height: If specified, limits the visualization to only show this many levels
                    of non-kernel nodes above the kernels. For example, height=1 shows
                    only the first level of operations directly connected to kernels.
        """
        # Note: Visualization libraries assumed to be available

        # Apply height filtering if specified
        if height is not None and height >= 0:
            dag = self._filter_dag_by_height(dag, height)
            print(f"Filtered DAG to height {height}: {len(dag.nodes)} nodes remaining")

        # Calculate total kernel runtime for each operation node
        op_kernel_runtimes = {}
        for node_name, node in dag.nodes.items():
            if node.node_type == "op":
                connected_kernels = self._find_connected_kernels(dag, node_name)
                total_runtime = 0.0
                for kernel_name in connected_kernels:
                    kernel_node = dag.nodes[kernel_name]
                    kernel_runtime = sum(dur for dur, _ in kernel_node.kernel_instances)
                    total_runtime += kernel_runtime
                op_kernel_runtimes[node_name] = total_runtime
                # Debug print to verify calculation
                if total_runtime > 0:
                    print(
                        f"Op '{node_name}' has {len(connected_kernels)} connected kernels with total runtime: {total_runtime:.2f}μs"
                    )

        # Calculate coloring for kernel nodes based on color_mode
        kernel_colors = {}
        if color_mode:
            kernel_colors = self._calculate_kernel_colors(
                dag, color_mode, baseline_profile
            )

        # Use graphviz for clean DAG layout
        try:
            import graphviz

            dot = graphviz.Digraph(comment="Trace DAG")
            dot.attr(rankdir="TB")  # Top to bottom layout
            dot.attr("node", shape="box")

            # Add nodes with different styles for ops vs kernels
            # Create a mapping for safe node names (graphviz doesn't like special chars)
            safe_names = {}
            for i, (node_name, node) in enumerate(dag.nodes.items()):
                # Create safe name for graphviz
                safe_name = f"node_{i}"
                safe_names[node_name] = safe_name

                if node.node_type == "kernel":
                    # Kernel nodes at bottom with instance count and performance stats
                    instance_count = len(node.kernel_instances)
                    total_duration = sum(dur for dur, _ in node.kernel_instances)

                    # Handle kernel name display based on compact setting
                    if compact:
                        # Use compact names without wrapping
                        display_name = node_name[:40] + "..."
                    else:
                        # Wrap long kernel names for display with proper line breaks
                        display_name = _wrap_text(node_name, 40)

                    label = f"{display_name}\\n{instance_count} inst\\n{total_duration:.2f}μs total"

                    # Calculate average duration per instance
                    avg_duration = (
                        total_duration / instance_count if instance_count > 0 else 0
                    )
                    label += f"\\n{avg_duration:.1f}μs avg"

                    # Add performance statistics if available and non-zero
                    if node.achieved_flops_list:
                        flops_min = min(node.achieved_flops_list)
                        flops_max = max(node.achieved_flops_list)
                        flops_avg = sum(node.achieved_flops_list) / len(
                            node.achieved_flops_list
                        )
                        # Only show FLOPS stats if they're not all zero
                        if flops_max > 0.0:
                            label += f"\\nFLOPS %: min={flops_min:.1f}, max={flops_max:.1f}, avg={flops_avg:.1f}"

                    if node.achieved_bandwidth_list:
                        bw_min = min(node.achieved_bandwidth_list)
                        bw_max = max(node.achieved_bandwidth_list)
                        bw_avg = sum(node.achieved_bandwidth_list) / len(
                            node.achieved_bandwidth_list
                        )
                        # Only show BW stats if they're not all zero
                        if bw_max > 0.0:
                            label += f"\\nBW %: min={bw_min:.1f}, max={bw_max:.1f}, avg={bw_avg:.1f}"

                    # Add bound type information if available
                    if node.bound_type_list:
                        # Count compute vs memory bound instances
                        compute_count = node.bound_type_list.count("compute")
                        memory_count = node.bound_type_list.count("memory")
                        total_bound = compute_count + memory_count

                        if total_bound > 0:
                            if compute_count > memory_count:
                                label += (
                                    f"\\nCompute Bound ({compute_count}/{total_bound})"
                                )
                            elif memory_count > compute_count:
                                label += (
                                    f"\\nMemory Bound ({memory_count}/{total_bound})"
                                )
                            else:
                                label += f"\\nMixed Bound (C:{compute_count}, M:{memory_count})"
                        else:
                            # All instances are unknown bound type
                            label += "\\nBound: Unknown"

                    # Use color from coloring algorithm or default
                    color = kernel_colors.get(node_name, "lightcoral")
                    dot.node(
                        safe_name,
                        label,
                        style="filled",
                        fillcolor=color,
                        shape="ellipse",
                    )
                else:
                    # Operation nodes with instance counts and total kernel runtime
                    instance_count = getattr(node, "instance_count", 0)
                    # Truncate long operation names for display
                    # Handle operation name display based on compact setting
                    if compact:
                        # Use compact names without wrapping
                        display_name = node_name[:40] + "..."
                    else:
                        # Wrap long operation names for display
                        display_name = _wrap_text(node_name, 40)
                    label = f"{display_name}"
                    if instance_count > 0:
                        label += f"\\n{instance_count} instances"

                    # Add total kernel runtime if this operation has connected kernels
                    kernel_runtime = op_kernel_runtimes.get(node_name, 0.0)
                    if kernel_runtime > 0.0:
                        label += f"\\nKernel time: {kernel_runtime:.2f}μs"

                    dot.node(
                        safe_name,
                        label,
                        style="filled",
                        fillcolor="lightblue",
                        shape="box",
                    )

            # Add edges using safe names
            for parent, child in dag.edges:
                if parent in safe_names and child in safe_names:
                    dot.edge(safe_names[parent], safe_names[child])

            # Add color legend based on color mode
            if color_mode:
                legend_text = self._get_color_legend_text(color_mode)
                if legend_text:
                    # Add legend as a separate node
                    dot.node(
                        "legend",
                        legend_text,
                        shape="note",
                        style="filled",
                        fillcolor="white",
                        pos="0,1!",
                        pin="true",
                    )

            # Render to specified format
            base_path = output_path.replace(".png", "").replace(".svg", "")
            dot.render(base_path, format=format, cleanup=True)
            print(f"DAG visualization saved to {output_path}")

        except Exception as e:
            print(f"Graphviz visualization failed: {e}")

    def _calculate_kernel_colors(
        self,
        dag: TraceDAG,
        color_mode: str,
        baseline_profile: Optional["JsonProfile"] = None,
    ) -> Dict[str, str]:
        """
        Calculate colors for kernel nodes based on the specified color mode.
        Always uses time-based gradients as the base, with intensity modified by color mode.
        Returns a dictionary mapping kernel names to color strings.
        """
        if color_mode not in [
            "time",
            "diff",
            "mem-utilization",
            "compute-utilization",
            "roofline",
        ]:
            return {}

        kernel_colors = {}
        kernel_nodes = {
            name: node for name, node in dag.nodes.items() if node.node_type == "kernel"
        }

        if not kernel_nodes:
            return {}

        base_time_gradients = dag.calculate_kernel_time_gradients()

        if color_mode == "time":
            return base_time_gradients

        if color_mode == "diff":
            if baseline_profile is None:
                print(
                    "Warning: diff coloring requested but no baseline profile provided"
                )
                return base_time_gradients

            baseline_dag = baseline_profile.build_trace_dag()
            baseline_durations = {}

            for name, node in baseline_dag.nodes.items():
                if node.node_type == "kernel":
                    total_duration = sum(dur for dur, _ in node.kernel_instances)
                    baseline_durations[name] = total_duration

            # Calculate duration differences and modify the base gradients
            for name, base_color in base_time_gradients.items():
                if name not in kernel_nodes:
                    continue

                current_duration = sum(
                    dur for dur, _ in kernel_nodes[name].kernel_instances
                )
                baseline_duration = baseline_durations.get(name, 0.0)

                if baseline_duration == 0.0:
                    kernel_colors[name] = self._tint_color(base_color, "red", 0.8)
                else:
                    diff_ratio = (
                        current_duration - baseline_duration
                    ) / baseline_duration
                    if diff_ratio > 0.1:
                        tint_intensity = min(1.0, diff_ratio)
                        kernel_colors[name] = self._tint_color(
                            base_color, "red", tint_intensity
                        )
                    elif diff_ratio < -0.1:
                        tint_intensity = min(1.0, abs(diff_ratio))
                        kernel_colors[name] = self._tint_color(
                            base_color, "blue", tint_intensity
                        )
                    else:
                        # Similar performance - use base time gradient
                        kernel_colors[name] = base_color

        elif color_mode == "mem-utilization":
            # Modify base gradients based on memory bandwidth utilization
            for name, base_color in base_time_gradients.items():
                if name not in kernel_nodes:
                    continue

                node = kernel_nodes[name]
                if node.achieved_bandwidth_list:
                    avg_utilization = sum(node.achieved_bandwidth_list) / len(
                        node.achieved_bandwidth_list
                    )
                    # Higher utilization -> more green tint
                    utilization_intensity = min(
                        1.0, avg_utilization / 100.0
                    )  # Normalize to [0,1]
                    kernel_colors[name] = self._tint_color(
                        base_color, "green", utilization_intensity
                    )
                else:
                    # No utilization data - use base gradient
                    kernel_colors[name] = base_color

        elif color_mode == "compute-utilization":
            # Modify base gradients based on compute (FLOPS) utilization
            for name, base_color in base_time_gradients.items():
                if name not in kernel_nodes:
                    continue

                node = kernel_nodes[name]
                if node.achieved_flops_list:
                    avg_utilization = sum(node.achieved_flops_list) / len(
                        node.achieved_flops_list
                    )
                    # Higher utilization -> more purple tint
                    utilization_intensity = min(
                        1.0, avg_utilization / 100.0
                    )  # Normalize to [0,1]
                    kernel_colors[name] = self._tint_color(
                        base_color, "purple", utilization_intensity
                    )
                else:
                    # No utilization data - use base gradient
                    kernel_colors[name] = base_color

        elif color_mode == "roofline":
            # Modify base gradients based on roofline analysis
            for name, base_color in base_time_gradients.items():
                if name not in kernel_nodes:
                    continue

                node = kernel_nodes[name]
                if (
                    node.bound_type_list
                    and node.achieved_flops_list
                    and node.achieved_bandwidth_list
                ):
                    # Calculate the average roofline score
                    bound_types = node.bound_type_list
                    flops_utils = node.achieved_flops_list
                    bw_utils = node.achieved_bandwidth_list

                    total_score = 0.0
                    valid_instances = 0

                    for i in range(
                        min(len(bound_types), len(flops_utils), len(bw_utils))
                    ):
                        bound_type = bound_types[i]
                        if bound_type == "compute":
                            total_score += flops_utils[i]
                            valid_instances += 1
                        elif bound_type == "memory":
                            total_score += bw_utils[i]
                            valid_instances += 1

                    if valid_instances > 0:
                        avg_score = total_score / valid_instances
                        # Lower utilization -> more red tint (worse performance)
                        utilization_intensity = 1.0 - min(1.0, avg_score / 100.0)
                        kernel_colors[name] = self._tint_color(
                            base_color, "red", utilization_intensity
                        )
                    else:
                        kernel_colors[name] = base_color
                else:
                    # No roofline data - use base gradient
                    kernel_colors[name] = base_color

        return kernel_colors

    def _tint_color(self, base_color: str, tint: str, intensity: float) -> str:
        """Apply a color tint to a base color with given intensity."""
        # Convert hex to RGB
        base_color = base_color.lstrip("#")
        r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (0, 2, 4))

        # Define tint colors
        tint_colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "purple": (255, 0, 255),
        }

        if tint not in tint_colors:
            return base_color

        tint_r, tint_g, tint_b = tint_colors[tint]

        # Blend base color with tint based on intensity
        final_r = int(r * (1 - intensity) + tint_r * intensity)
        final_g = int(g * (1 - intensity) + tint_g * intensity)
        final_b = int(b * (1 - intensity) + tint_b * intensity)

        # Ensure values are in valid range
        final_r = max(0, min(255, final_r))
        final_g = max(0, min(255, final_g))
        final_b = max(0, min(255, final_b))

        return f"#{final_r:02x}{final_g:02x}{final_b:02x}"

    def _filter_dag_by_height(self, dag: TraceDAG, height: int) -> TraceDAG:
        """
        Filter the DAG to only show nodes up to a specified height above kernel nodes.

        Args:
            dag: The original DAG
            height: Maximum levels of non-kernel nodes to show above kernels
                   (0 = only kernels, 1 = kernels + direct parents, etc.)

        Returns:
            A new filtered DAG containing only nodes within the height limit
        """
        if height < 0:
            return dag

        filtered_dag = TraceDAG()

        # Find all kernel nodes first
        kernel_nodes = OrderedSet(
            [name for name, node in dag.nodes.items() if node.node_type == "kernel"]
        )

        # If height is 0, only show kernels
        if height == 0:
            for kernel_name in kernel_nodes:
                kernel_node = dag.nodes[kernel_name]
                filtered_dag.add_node(kernel_name, "kernel")
                # Copy kernel instances
                for dur, tid in kernel_node.kernel_instances:
                    filtered_dag.add_kernel_instance(kernel_name, dur, tid)
                # Copy performance stats
                filtered_dag.nodes[kernel_name].achieved_flops_list = (
                    kernel_node.achieved_flops_list[:]
                )
                filtered_dag.nodes[kernel_name].achieved_bandwidth_list = (
                    kernel_node.achieved_bandwidth_list[:]
                )
                filtered_dag.nodes[kernel_name].bound_type_list = (
                    kernel_node.bound_type_list[:]
                )
            return filtered_dag

        # Perform reverse BFS from kernels to find nodes within height limit
        # We need to traverse backwards through the DAG edges
        # Build reverse edge mapping first
        reverse_edges = defaultdict(list)
        for parent, child in dag.edges:
            reverse_edges[child].append(parent)

        # BFS from kernel nodes going backwards (upwards in the DAG)
        nodes_to_include = OrderedSet(kernel_nodes)  # Always include kernels
        current_level = kernel_nodes.copy()

        for level in range(height):
            next_level = OrderedSet()
            for node in current_level:
                # Add all parents of current level nodes
                for parent in reverse_edges[node]:
                    if parent not in nodes_to_include:
                        next_level.add(parent)
                        nodes_to_include.add(parent)

            if not next_level:
                break  # No more levels to explore

            current_level = next_level

        # Build the filtered DAG with only the nodes we want to include
        for node_name in nodes_to_include:
            original_node = dag.nodes[node_name]
            filtered_dag.add_node(node_name, original_node.node_type)

            # Copy node-specific data
            if original_node.node_type == "kernel":
                # Copy kernel instances
                for dur, tid in original_node.kernel_instances:
                    filtered_dag.add_kernel_instance(node_name, dur, tid)
                # Copy performance stats
                filtered_dag.nodes[node_name].achieved_flops_list = (
                    original_node.achieved_flops_list[:]
                )
                filtered_dag.nodes[node_name].achieved_bandwidth_list = (
                    original_node.achieved_bandwidth_list[:]
                )
                filtered_dag.nodes[node_name].bound_type_list = (
                    original_node.bound_type_list[:]
                )
            else:
                # Copy operation node data
                if hasattr(original_node, "instance_count"):
                    filtered_dag.nodes[node_name].instance_count = (
                        original_node.instance_count
                    )

        # Add edges that connect nodes within our filtered set
        for parent, child in dag.edges:
            if parent in nodes_to_include and child in nodes_to_include:
                filtered_dag.add_edge(parent, child)

        return filtered_dag

    def _get_color_legend_text(self, color_mode: str) -> Optional[str]:
        """Get legend text for the specified color mode."""
        if color_mode == "diff":
            return "Legend\\nRed: Slower than baseline\\nBlue: Faster than baseline\\nWhite: Same as baseline"
        elif color_mode == "mem-utilization":
            return "Legend\\nGreen: Memory bandwidth utilization\\nDarker = lower utilization"
        elif color_mode == "compute-utilization":
            return "Legend\\nPurple: Compute utilization\\nDarker = lower utilization"
        elif color_mode == "roofline":
            return "Legend\\nRoofline Analysis\\nDark Red: Low utilization (worse)\\nLight Yellow: High utilization (better)\\nCompute-bound: Uses compute %\\nMemory-bound: Uses memory %"
        return None

    def _is_trace_augmented(self) -> bool:
        """Check if the trace has been augmented with performance information."""
        for event in self.events:
            if event.get("cat") == "kernel" and "args" in event:
                # If we find a kernel with performance data, the trace is augmented
                if "kernel_flop" in event["args"] or "kernel_num_gb" in event["args"]:
                    return True
        return False

    def create_trace_dag_visualization(
        self,
        output_path: str = "trace_dag.png",
        format: str = "png",
        color_mode: Optional[str] = None,
        baseline_profile: Optional["JsonProfile"] = None,
        height: Optional[int] = None,
        compact: bool = True,
    ) -> TraceDAG:
        """
        Convenience method to build and visualize the trace DAG in one step.
        Automatically augments the trace if it hasn't been augmented yet.
        Returns the created DAG for further analysis.
        """
        # Check if trace needs augmentation
        if not self._is_trace_augmented():
            print("Trace not augmented with performance data. Augmenting trace...")
            self.augment_trace()
            print("Trace augmentation completed.")
        else:
            print("Trace already augmented with performance data.")

        print("Building trace DAG from JSON profile...")
        dag = self.build_trace_dag()

        print(f"DAG created with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        print(
            f"Operations: {[name for name, node in dag.nodes.items() if node.node_type == 'op']}"
        )
        print(
            f"Kernels: {[name for name, node in dag.nodes.items() if node.node_type == 'kernel']}"
        )

        # Check if performance statistics are available
        kernels_with_perf_stats = sum(
            1
            for node in dag.nodes.values()
            if node.node_type == "kernel"
            and (
                (node.achieved_flops_list and max(node.achieved_flops_list) > 0)
                or (
                    node.achieved_bandwidth_list
                    and max(node.achieved_bandwidth_list) > 0
                )
            )
        )

        if kernels_with_perf_stats == 0:
            print(
                """
                Note: No performance statistics (FLOPS/bandwidth %) are displayed because the trace lacks tensor \
shape and type information needed for calculations. To include performance metrics, run torch.profiler.profile with record_shapes=True
                """
            )

        print(f"Visualizing DAG to {output_path}...")
        self.visualize_trace_dag(
            dag, output_path, format, color_mode, baseline_profile, height, compact
        )

        return dag

    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(self.data, f)

    def combine_with(self, other: "JsonProfile") -> "JsonProfile":
        """
        Combine this profile with another profile by merging their trace events.
        Returns a new JsonProfile object with combined data.
        """
        # Create a new combined data structure
        combined_data = {
            "traceEvents": self.data["traceEvents"] + other.data["traceEvents"],
            "deviceProperties": self.data.get("deviceProperties", []),
        }

        # Merge device properties, avoiding duplicates
        other_device_props = other.data.get("deviceProperties", [])
        existing_device_ids = OrderedSet(
            [dev["id"] for dev in combined_data["deviceProperties"]]
        )

        for device_prop in other_device_props:
            if device_prop["id"] not in existing_device_ids:
                combined_data["deviceProperties"].append(device_prop)

        # Copy any other top-level properties from the first profile
        for key, value in self.data.items():
            if key not in combined_data:
                combined_data[key] = value

        # Create a temporary file to write the combined data

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(combined_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Create new JsonProfile from the combined data
            combined_profile = JsonProfile(
                tmp_path,
                benchmark_name=f"{self.benchmark_name or 'Profile1'}_+_{other.benchmark_name or 'Profile2'}",
                dtype=self.dtype or other.dtype,
            )
            return combined_profile
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
