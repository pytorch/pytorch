"""
Utility functions for profile analysis.
"""

import hashlib
import logging
import math
import multiprocessing as mp
import os
import pickle
import tempfile
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry

from .types import _IdxEvt, KernelStats

if TYPE_CHECKING:
    from .json_profile import JsonProfile


try:
    import graphviz

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


log = logging.getLogger(__name__)


ATEN_PREFIX = "aten::"


# adapters convert the json trace into a format that works with flops_counter
ArgsType = tuple[tuple[Any, ...], dict[Any, Any]]
AdapterType = Callable[[tuple[Any, ...], tuple[Any, ...]], ArgsType]
adapters_map: dict[str, AdapterType] = {}


def parse_list(lst: str) -> list[int]:
    lst = lst.replace("[", "").replace("]", "")
    substrings = lst.split(",")

    return [int(substring.strip()) for substring in substrings]


def register_adapter(
    aten: Union[str, list[str]],
) -> Callable[
    [AdapterType],
    AdapterType,
]:
    def decorator(func: AdapterType) -> AdapterType:
        global _adapters_map

        if isinstance(aten, str):
            adapters_map[aten] = func
        else:
            for at in aten:
                adapters_map[at] = func
        return func

    return decorator


@register_adapter(["_slow_conv2d_forward"])
def _slow_conv2d_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    tmp.append(False)
    tmp2 = list(concrete)
    if len(tmp2) < 5:
        raise ParseException("slow conv2d has less than 5 concrete inputs")
    tmp2[3] = tmp2[4]
    return conv_adapter(tuple(tmp), tuple(tmp2))


@register_adapter(["flash_attention_forward"])
def _flash_attention_forward(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    breakpoint()


@register_adapter(["_efficient_attention_forward"])
def _efficient_attention_forward(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # Create fake tensors with FakeTensorMode
    with FakeTensorMode() as fake_mode:
        # Create query, key, value tensors from shapes[0], shapes[1], shapes[2]
        query = torch.zeros(shapes[0], dtype=torch.float32, device="cuda")
        key = torch.zeros(shapes[1], dtype=torch.float32, device="cuda")
        value = torch.zeros(shapes[2], dtype=torch.float32, device="cuda")

        # Create cum_seq_q tensor from shapes[3] if it's not empty
        if shapes[3]:  # Check if shapes[3] is not empty
            cum_seq_q = torch.zeros(shapes[3], dtype=torch.int32, device="cuda")
        else:
            cum_seq_q = torch.tensor([], dtype=torch.int32, device="cuda")

    # Return the tensors in the format (query, key, value, (), cum_seq_q)
    return tuple([query, key, value, cum_seq_q, None, None, None, None]), {}


@register_adapter(["convolution", "_convolution", "cudnn_convolution"])
def conv_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    if len(tmp) == 4:
        transposed = False
    elif len(tmp) > 6:
        transposed = bool(tmp[6])
        tmp[6] = transposed
    else:
        raise ParseException(f"Convolution has the wrong number of inputs: {len(tmp)}")

    kwargs: dict[Any, Any] = {}
    if not transposed:
        # calculate output shape if not transposed.
        def conv_out_dims(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(concrete[3])
        inp = shapes[0]
        w = shapes[1]
        out_x_y = [conv_out_dims(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y  # we only need the xy values
        kwargs["out_val"] = out

    return tuple(tmp), kwargs


def default_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


@register_adapter("addmm")
def addmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("bmm")
def bmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    return tuple(tmp[:2]), {}


@register_adapter("baddbmm")
def baddbmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("mm")
def mm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


def _parse_kernel_name(name: str) -> Optional[str]:
    """
    parse the name of the kernel from the event name.
    """
    if name.startswith(ATEN_PREFIX):
        return name[len(ATEN_PREFIX) :]
    elif "conv" in name:
        return "convolution"
    elif "addmm" in name:
        return "addmm"
    elif "bmm" in name:
        return "bmm"
    elif "baddbmm" in name:
        return "baddbmm"
    elif "_mm" in name:
        return "mm"
    else:
        return None


def _calculate_flops(event: dict[str, Any]) -> int:
    """
    This function has to parse the kernel name, which is error prone. There doesn't seem to be another solution that
    will support all the different backends that can generate kernels, so make sure to update this function when new
    ops and backends are desired.
    """
    name = event["name"]
    if "kernel_flop" in event["args"] and event["args"]["kernel_flop"] != 0:
        return event["args"]["kernel_flop"]
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return 0

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None or op_obj not in flop_registry:
        return 0

    flop_function = flop_registry[op_obj]

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return 0
    input_shapes = event["args"]["Input Dims"]
    concrete = event["args"]["Concrete Inputs"]
    if op_name in adapters_map:
        try:
            args, kwargs = adapters_map[op_name](input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    else:
        try:
            args, kwargs = default_adapter(input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    return flop_function(*args, **kwargs)


def _get_size_from_string(type_string: str) -> int:
    if not hasattr(torch, type_string):
        return 1
    else:
        return getattr(torch, type_string).itemsize


def _default_estimate_gb(event: dict[str, Any]) -> float:
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, typ in sizes_and_types:
        isize = _get_size_from_string(typ)
        bw += isize * math.prod(pytree.tree_flatten(size)[0])
    return bw / 1e9


def _estimate_gb(event: dict[str, Any]) -> float:
    """
    Our best effort to estimate the gb, should be refactored soon with MemoryCounter.
    """
    name = event["name"]
    if "kernel_num_gb" in event["args"] and event["args"]["kernel_num_gb"] != 0:
        return event["args"]["kernel_num_gb"]
    if "Input type" not in event["args"] or "Input Dims" not in event["args"]:
        return 0
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return _default_estimate_gb(event)

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None:
        return _default_estimate_gb(event)

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return _default_estimate_gb(event)
    input_shapes = event["args"]["Input Dims"]

    # NOTE these will be refactored into a similar object to FlopCounter soon
    def mm_formula(M: int, N: int, K: int, size: int) -> int:
        return 2 * (M * K + N * K + M * N) * size

    if op_name == "addmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        M = input_shapes[1][0]
        N = input_shapes[1][1]
        assert input_shapes[1][1] == input_shapes[2][0]
        K = input_shapes[2][1]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size) / 1e9
    elif op_name == "mm":
        M = input_shapes[0][0]
        N = input_shapes[0][1]
        assert input_shapes[0][1] == input_shapes[1][0]
        K = input_shapes[1][1]
        type_size = _get_size_from_string(event["args"]["Input type"][0])
        return mm_formula(M, N, K, type_size) / 1e9
    elif op_name == "baddbmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[1][1]
        N = input_shapes[1][2]
        K = input_shapes[2][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name == "bmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[0][1]
        N = input_shapes[0][2]
        K = input_shapes[1][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name in [
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
    ]:
        concrete = event["args"]["Concrete Inputs"]

        def conv_out_dim(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(
            concrete[3] if op_name != "_slow_conv2d_forward" else concrete[4]
        )
        inp = input_shapes[0]
        w = input_shapes[1]
        out_x_y = [conv_out_dim(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y
        # each output element reads in * w * w chunk
        input_reads = out[0] * out[1] * out[2] * out[3] * inp[1] * w[2] * w[3]
        # Assume weights are in cache, so only read once
        weight_reads = w[0] * w[1] * w[2] * w[3]
        return (input_reads + weight_reads) / 1e9

    return _default_estimate_gb(event)


def _create_extern_mapping(
    data: dict[str, Any],
) -> defaultdict[int, list[dict[str, Any]]]:
    """
    compute a mapping from external ids to non kernels, which contain the information we need to estimate flops etc
    """
    extern_mapping: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in data["traceEvents"]:
        if (
            "args" not in event
            or "External id" not in event["args"]
            or event["cat"] != "cpu_op"
        ):
            continue
        if len(extern_mapping[event["args"]["External id"]]) > 0:
            raise ParseException("duplicate external id in event")
        extern_mapping[event["args"]["External id"]].append(event)
    return extern_mapping


def _augment_trace_helper(data: dict[str, Any]) -> dict[str, Any]:
    extern_mapping = _create_extern_mapping(data)

    for event in data["traceEvents"]:
        if "cat" not in event or event["cat"] != "kernel":
            continue
        if "args" not in event:
            raise ParseException(f"kernel has no args: {event}")
        if "External id" not in event["args"]:
            event_str = f"kernel has no External id: {event}"
            log.info(event_str)
            continue

        external_op = extern_mapping[event["args"]["External id"]][0]
        flops = _calculate_flops(external_op)
        if flops == 0:
            flops = _calculate_flops(event)
        external_op["args"]["kernel_flop"] = flops
        external_op["args"]["kernel_num_gb"] = _estimate_gb(external_op)
        event["args"]["kernel_flop"] = external_op["args"]["kernel_flop"]
        event["args"]["kernel_num_gb"] = external_op["args"]["kernel_num_gb"]
    return data


_dtype_map = {
    "float": torch.float,
    "float32": torch.float,
    "int": torch.int,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int,
    "long": torch.long,
    "long int": torch.long,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.double,
}


def compute_stats_chunk(args):
    """Compute statistics for a chunk of events."""
    events_chunk, devices, dtype_obj = args

    local_device_stats = {}
    for dev_id, dev in devices.items():
        local_device_stats[dev_id] = defaultdict(OrderedSet)

    for event in events_chunk:
        if "cat" not in event or "args" not in event or event["cat"] != "kernel":
            continue
        if "device" not in event["args"]:
            continue

        dev_tmp = event["args"]["device"]
        if dev_tmp not in devices:
            continue

        dev = devices[dev_tmp]

        dur = event["dur"]  # us
        if "kernel_flop" in event["args"]:
            assert dur != 0
            op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
        else:
            op_flops = 0

        if "kernel_num_gb" in event["args"]:
            assert dur != 0
            op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
        else:
            op_gbps = 0

        if dev.info is not None:
            # Use DeviceInfo API to get FLOPS and bandwidth correctly
            from torch._inductor.analysis.device_info import DeviceInfo

            peak_flops = DeviceInfo.lookup_tops(dev.name, dtype_obj or torch.float32)
            peak_bandwidth_gbs = DeviceInfo.lookup_dram_bw_gbs(dev.name)

            if peak_flops is not None and peak_flops > 0:
                achieved_flops = 100 * op_flops / peak_flops
            else:
                achieved_flops = 0

            if peak_bandwidth_gbs is not None and peak_bandwidth_gbs > 0:
                achieved_bandwidth = 100 * op_gbps / peak_bandwidth_gbs
            else:
                achieved_bandwidth = 0
        else:
            achieved_flops = 0
            achieved_bandwidth = 0

        if "name" not in event:
            continue
        kernel_name = event["name"]
        local_device_stats[dev_tmp][kernel_name].add(
            KernelStats(
                flops=op_flops,
                bw=op_gbps,
                latency=dur,
                achieved_bandwidth=achieved_bandwidth,
                achieved_flops=achieved_flops,
            )
        )

    return local_device_stats


def process_events_chunk(args):
    """Process a chunk of events and return per-tid intervals."""
    events_chunk, start_idx, devices, dtype = args

    per_tid_intervals = defaultdict(list)
    open_stack = defaultdict(list)

    for i, ev in enumerate(events_chunk):
        gi = start_idx + i
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
                    parent=None,
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
                        parent=None,
                        idx=beg_idx,
                    )
                )

    return per_tid_intervals, open_stack


# DAG Caching Functions
def get_cache_key(file_path: str) -> str:
    """Generate cache key based on file path, modification time, and size."""
    try:
        stat = os.stat(file_path)
        return hashlib.md5(
            f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode()
        ).hexdigest()
    except OSError:
        # File doesn't exist or can't be accessed
        return hashlib.md5(file_path.encode()).hexdigest()


def get_cache_dir() -> str:
    """Get the cache directory, creating it if necessary."""
    cache_dir = os.path.join(tempfile.gettempdir(), "torch_profile_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_dag_cached(file_path: str, dtype: str) -> Optional["TraceDAG"]:
    """Load DAG from cache if available and fresh."""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(file_path)
        cache_path = os.path.join(cache_dir, f"dag_{cache_key}_{dtype}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                print(f"Loaded DAG from cache for {os.path.basename(file_path)}")
                return cached_data
    except Exception as e:
        print(f"Warning: Failed to load DAG from cache: {e}")
    return None


def save_dag_cached(dag: "TraceDAG", file_path: str, dtype: str):
    """Save DAG to cache."""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(file_path)
        cache_path = os.path.join(cache_dir, f"dag_{cache_key}_{dtype}.pkl")

        with open(cache_path, "wb") as f:
            pickle.dump(dag, f)
        print(f"Saved DAG to cache for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Warning: Failed to save DAG to cache: {e}")


def process_single_trace(args):
    """Process a single trace file and return the DAG. Used for parallel processing."""
    from .json_profile import JsonProfile

    trace_id, input_file, dtype, use_cache = args

    try:
        # Try to load from cache first if caching is enabled
        if use_cache:
            cached_dag = load_dag_cached(input_file, dtype)
            if cached_dag is not None:
                trace_name = os.path.basename(input_file).replace(".json", "")
                # Check if cached DAG has performance statistics
                has_perf_stats = False
                for node in cached_dag.nodes.values():
                    if node.node_type == "kernel" and (
                        node.achieved_flops_list
                        or node.achieved_bandwidth_list
                        or node.bound_type_list
                    ):
                        has_perf_stats = True
                        break

                if has_perf_stats:
                    return trace_id, cached_dag, trace_name
                else:
                    print(
                        f"  Cached DAG for {trace_name} lacks performance statistics, reprocessing..."
                    )

        profile = JsonProfile(input_file, dtype=dtype)

        if not profile._is_trace_augmented():
            print(f"  Augmenting trace {trace_id + 1} to add performance statistics...")
            profile.augment_trace()

        has_performance_data = False
        for event in profile.events:
            if event.get("cat") == "kernel" and "args" in event:
                if "kernel_flop" in event["args"] or "kernel_num_gb" in event["args"]:
                    has_performance_data = True
                    break

        if not has_performance_data:
            print(
                f"  Warning: Trace {trace_id + 1} still lacks performance data after augmentation. This may be due to missing shape/type information in the original trace."
            )

        dag = profile.build_trace_dag()

        # Save to cache if caching is enabled (even if no performance stats, to avoid reprocessing)
        if use_cache:
            save_dag_cached(dag, input_file, dtype)

        trace_name = os.path.basename(input_file).replace(".json", "")

        return trace_id, dag, trace_name
    except Exception as e:
        print(f"Error processing trace {trace_id}: {e}")
        raise


def create_multi_trace_visualization(
    input_files: List[str],
    output_file: str,
    dtype: str,
    use_cache: bool = True,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    format: str = "png",
    height: Optional[int] = None,
    color_mode: str = "time",
    baseline_profile: Optional["JsonProfile"] = None,
    compact: bool = True,
) -> "MultiTraceDAG":
    """Create a multi-trace DAG visualization from multiple JSON trace files."""
    from .dag_nodes import MultiTraceDAG
    from .json_profile import JsonProfile

    multi_dag = MultiTraceDAG()

    if use_parallel and len(input_files) > 1:
        return create_multi_trace_visualization_parallel(
            input_files,
            output_file,
            dtype,
            use_cache,
            max_workers,
            format,
            height,
            color_mode,
            baseline_profile,
        )

    # Original single-threaded implementation for single trace or when parallel is disabled
    # Each DAG created separately
    for trace_id, input_file in enumerate(input_files):
        print(f"Processing trace {trace_id + 1}/{len(input_files)}: {input_file}")

        if use_cache:
            cached_dag = load_dag_cached(input_file, dtype)
            if cached_dag is not None:
                trace_name = os.path.basename(input_file).replace(".json", "")
                multi_dag.add_trace_dag(trace_id, cached_dag, trace_name)
                print(
                    f"  Added {len(cached_dag.nodes)} nodes and {len(cached_dag.edges)} edges from cache"
                )
                continue

        profile = JsonProfile(input_file, dtype=dtype)

        if not profile._is_trace_augmented():
            print(f"  Augmenting trace {trace_id + 1}...")
            profile.augment_trace()

        dag = profile.build_trace_dag()

        if use_cache:
            save_dag_cached(dag, input_file, dtype)

        trace_name = os.path.basename(input_file).replace(".json", "")

        multi_dag.add_trace_dag(trace_id, dag, trace_name)

        print(
            f"  Added {len(dag.nodes)} nodes and {len(dag.edges)} edges from trace {trace_id + 1}"
        )

    # Apply height filtering
    if height is not None and height >= 0:
        print(f"Applying height filter: {height}")
        original_node_count = len(multi_dag.nodes)
        multi_dag = multi_dag.filter_by_height(height)
        print(
            f"Filtered multi-trace DAG from {original_node_count} to {len(multi_dag.nodes)} nodes"
        )

    multi_dag.assign_colors()

    multi_dag.calculate_kernel_time_gradients()

    visualize_multi_trace_dag(
        multi_dag, output_file, format, color_mode, baseline_profile, compact
    )

    return multi_dag


def create_multi_trace_visualization_parallel(
    input_files: List[str],
    output_file: str,
    dtype: str,
    use_cache: bool = True,
    max_workers: Optional[int] = None,
    format: str = "png",
    height: Optional[int] = None,
    color_mode: str = "time",
    baseline_profile: Optional["JsonProfile"] = None,
) -> "MultiTraceDAG":
    """Create a multi-trace DAG visualization using parallel processing."""
    from .dag_nodes import MultiTraceDAG

    multi_dag = MultiTraceDAG()

    if max_workers is None:
        max_workers = min(len(input_files), mp.cpu_count())

    print(
        f"Processing {len(input_files)} traces in parallel using {max_workers} workers..."
    )

    trace_args = [
        (i, input_file, dtype, use_cache) for i, input_file in enumerate(input_files)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_trace = {
            executor.submit(process_single_trace, args): args for args in trace_args
        }

        for future in as_completed(future_to_trace):
            try:
                trace_id, dag, trace_name = future.result()
                multi_dag.add_trace_dag(trace_id, dag, trace_name)
                print(
                    f"  Completed trace {trace_id + 1}/{len(input_files)}: {len(dag.nodes)} nodes, {len(dag.edges)} edges"
                )
            except Exception as exc:
                args = future_to_trace[future]
                print(f"Trace {args[0]} generated an exception: {exc}")

    if height is not None and height >= 0:
        print(f"Applying height filter: {height}")
        original_node_count = len(multi_dag.nodes)
        multi_dag = multi_dag.filter_by_height(height)
        print(
            f"Filtered multi-trace DAG from {original_node_count} to {len(multi_dag.nodes)} nodes"
        )

    multi_dag.assign_colors()
    multi_dag.calculate_kernel_time_gradients()

    visualize_multi_trace_dag(
        multi_dag, output_file, format, color_mode, baseline_profile
    )

    return multi_dag


def _find_connected_kernels_multi_trace(
    multi_dag: "MultiTraceDAG", op_node_name: str, trace_id: int
) -> list[str]:
    """
    Find all kernel nodes that are reachable from the given operation node in a specific trace.
    Uses BFS to traverse the DAG and collect all connected kernels.
    """
    visited = OrderedSet()
    kernel_nodes = []
    queue = [op_node_name]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for parent, child, edge_trace_id in multi_dag.edges:
            if parent == current and child not in visited and edge_trace_id == trace_id:
                if multi_dag.nodes[child].node_type == "kernel":
                    kernel_nodes.append(child)
                else:
                    queue.append(child)

    return kernel_nodes


def visualize_multi_trace_dag(
    multi_dag: "MultiTraceDAG",
    output_path: str = "multi_trace_dag.png",
    format: str = "png",
    color_mode: str = "time",
    baseline_profile: Optional["JsonProfile"] = None,
    compact: bool = True,
) -> None:
    """Create a PNG visualization of the multi-trace DAG with composite nodes and colored edges."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Install graphviz.")
        return

    # Performance statistics should already be computed by the existing infrastructure
    # in JsonProfile._compute_stats() and compute_stats_chunk()

    if color_mode != "time":
        enhanced_gradients = multi_dag.calculate_kernel_colors(
            color_mode, baseline_profile
        )
        multi_dag.trace_kernel_gradients = enhanced_gradients

    op_kernel_runtimes = {}
    for node_name, multi_node in multi_dag.nodes.items():
        if multi_node.node_type == "op":
            for trace_id in multi_node.present_in_traces:
                connected_kernels = _find_connected_kernels_multi_trace(
                    multi_dag, node_name, trace_id
                )
                total_runtime = 0.0
                for kernel_name in connected_kernels:
                    if (
                        kernel_name in multi_dag.nodes
                        and trace_id in multi_dag.nodes[kernel_name].trace_instances
                    ):
                        kernel_node = multi_dag.nodes[kernel_name].trace_instances[
                            trace_id
                        ]
                        kernel_runtime = sum(
                            dur for dur, _ in kernel_node.kernel_instances
                        )
                        total_runtime += kernel_runtime
                op_kernel_runtimes[(node_name, trace_id)] = total_runtime

    try:
        import graphviz

        dot = graphviz.Digraph(comment="Multi-Trace DAG")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box")

        safe_names = {}

        for i, (node_name, multi_node) in enumerate(multi_dag.nodes.items()):
            safe_name = f"node_{i}"
            safe_names[node_name] = safe_name

            label = _create_composite_node_label(
                multi_node, multi_dag, op_kernel_runtimes, compact
            )

            if multi_node.node_type == "kernel":
                dot.node(
                    safe_name, label, style="filled", fillcolor="white", shape="record"
                )
            else:
                dot.node(
                    safe_name, label, style="filled", fillcolor="white", shape="Mrecord"
                )

        edge_groups = {}
        for parent, child, trace_id in multi_dag.edges:
            key = (parent, child)
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(trace_id)

        for (parent, child), trace_ids in edge_groups.items():
            if parent in safe_names and child in safe_names:
                parent_safe = safe_names[parent]
                child_safe = safe_names[child]

                for i, trace_id in enumerate(trace_ids):
                    color = multi_dag.trace_colors[trace_id]

                    if len(trace_ids) == 1:
                        dot.edge(parent_safe, child_safe, color=color, penwidth="2")
                    else:
                        dot.edge(
                            parent_safe,
                            child_safe,
                            color=color,
                            penwidth="2",
                            constraint="true" if i == 0 else "false",
                        )

        _add_trace_legend(dot, multi_dag, color_mode, baseline_profile)

        base_path = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
        dot.render(base_path, format=format, cleanup=True)
        print(f"Multi-trace DAG visualization saved to {output_path}")

    except Exception as e:
        print(f"Graphviz multi-trace visualization failed: {e}")
        print("Multi-trace visualization requires graphviz. Please install graphviz.")


def _create_composite_node_label(
    multi_node: "MultiTraceDAGNode",
    multi_dag: "MultiTraceDAG",
    op_kernel_runtimes: dict = None,
    compact: bool = True,
) -> str:
    """Create a composite node label that shows data from each trace."""
    if op_kernel_runtimes is None:
        op_kernel_runtimes = {}

    sorted_trace_ids = sorted(multi_node.present_in_traces)

    safe_name = _safe_html_wrap(multi_node.name, 40, compact)

    num_traces = len(sorted_trace_ids)
    header_row = f'<TR><TD COLSPAN="{num_traces}"><B>{safe_name}</B></TD></TR>'

    data_cells = []
    for trace_id in sorted_trace_ids:
        if trace_id in multi_node.trace_instances:
            node = multi_node.trace_instances[trace_id]
            trace_name = multi_dag.trace_names[trace_id]

            if multi_node.node_type == "kernel":
                bg_color = multi_dag.trace_kernel_gradients.get(trace_id, {}).get(
                    multi_node.name, "white"
                )
                instance_count = len(node.kernel_instances)
                total_duration = sum(dur for dur, _ in node.kernel_instances)
                avg_duration = (
                    total_duration / instance_count if instance_count > 0 else 0
                )

                cell_content = f"{trace_name}<BR/>{instance_count} inst<BR/>{total_duration:.1f}μs total<BR/>{avg_duration:.1f}μs avg"

                if node.achieved_flops_list:
                    flops_min = min(node.achieved_flops_list)
                    flops_max = max(node.achieved_flops_list)
                    flops_avg = sum(node.achieved_flops_list) / len(
                        node.achieved_flops_list
                    )
                    if flops_max > 0.0:
                        cell_content += f"<BR/>FLOPS %: {flops_avg:.1f} (min={flops_min:.1f}, max={flops_max:.1f})"

                if node.achieved_bandwidth_list:
                    bw_min = min(node.achieved_bandwidth_list)
                    bw_max = max(node.achieved_bandwidth_list)
                    bw_avg = sum(node.achieved_bandwidth_list) / len(
                        node.achieved_bandwidth_list
                    )
                    if bw_max > 0.0:
                        cell_content += f"<BR/>BW %: {bw_avg:.1f} (min={bw_min:.1f}, max={bw_max:.1f})"

                if node.bound_type_list:
                    compute_count = node.bound_type_list.count("compute")
                    memory_count = node.bound_type_list.count("memory")
                    total_bound = compute_count + memory_count

                    if total_bound > 0:
                        if compute_count > memory_count:
                            cell_content += (
                                f"<BR/>Compute Bound ({compute_count}/{total_bound})"
                            )
                        elif memory_count > compute_count:
                            cell_content += (
                                f"<BR/>Memory Bound ({memory_count}/{total_bound})"
                            )
                        else:
                            cell_content += f"<BR/>Mixed Bound (C:{compute_count}, M:{memory_count})"
                    else:
                        cell_content += "<BR/>Bound: Unknown"

                data_cells.append(f'<TD BGCOLOR="{bg_color}">{cell_content}</TD>')
            else:
                bg_color = "#ffffff"
                instance_count = getattr(node, "instance_count", 0)

                kernel_runtime = op_kernel_runtimes.get(
                    (multi_node.name, trace_id), 0.0
                )
                if kernel_runtime > 0.0:
                    data_cells.append(
                        f'<TD BGCOLOR="{bg_color}">{trace_name}<BR/>{instance_count} instances<BR/>Kernel time: {kernel_runtime:.2f}μs</TD>'
                    )
                else:
                    data_cells.append(
                        f'<TD BGCOLOR="{bg_color}">{trace_name}<BR/>{instance_count} instances</TD>'
                    )
        else:
            bg_color = "#FFFFFF"
            trace_name = multi_dag.trace_names.get(trace_id, f"Trace {trace_id}")
            if multi_node.node_type == "kernel":
                data_cells.append(
                    f'<TD BGCOLOR="{bg_color}"><I>{trace_name}: Not present</I></TD>'
                )
            else:
                data_cells.append(
                    f'<TD BGCOLOR="{bg_color}"><I>{trace_name}: Not present</I></TD>'
                )

    data_row = f"<TR>{''.join(data_cells)}</TR>"

    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">{header_row}{data_row}</TABLE>>'


def _safe_html_wrap(text: str, max_width: int, compact: bool = False) -> str:
    """Safely wrap text for HTML table context by escaping first, then wrapping with HTML breaks."""
    escaped_text = text.replace("&", "&amp;")
    escaped_text = escaped_text.replace("<", "&lt;")
    escaped_text = escaped_text.replace(">", "&gt;")
    escaped_text = escaped_text.replace('"', "&quot;")
    escaped_text = escaped_text.replace("'", "&#39;")
    escaped_text = escaped_text.replace("[", "(")
    escaped_text = escaped_text.replace("]", ")")
    escaped_text = escaped_text.replace("{", "(")
    escaped_text = escaped_text.replace("}", ")")

    if compact:
        return escaped_text

    if len(escaped_text) <= max_width:
        return escaped_text

    lines = []
    current_line = ""
    i = 0

    while i < len(escaped_text):
        if escaped_text[i] == "&":
            entity_end = i + 1
            while entity_end < len(escaped_text) and escaped_text[entity_end] != ";":
                entity_end += 1
            if entity_end < len(escaped_text):
                entity_end += 1

            entity = escaped_text[i:entity_end]

            if len(current_line) + len(entity) > max_width and current_line:
                lines.append(current_line)
                current_line = entity
            else:
                current_line += entity

            i = entity_end
        else:
            if len(current_line) + 1 > max_width and current_line:
                lines.append(current_line)
                current_line = escaped_text[i]
            else:
                current_line += escaped_text[i]
            i += 1

    if current_line:
        lines.append(current_line)

    return "<BR/>".join(lines)


def _wrap_text(text: str, max_width: int) -> str:
    """Wrap text to fit within max_width characters per line, preserving word boundaries when possible."""
    if len(text) <= max_width:
        return text

    words = text.split()
    if not words:
        return text

    lines = []
    current_line = words[0]

    for word in words[1:]:
        if len(current_line) + 1 + len(word) <= max_width:
            current_line += " " + word
        else:
            if len(word) > max_width:
                lines.append(current_line)
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                current_line = word
            else:
                lines.append(current_line)
                current_line = word

    if current_line:
        lines.append(current_line)

    return "\\n".join(lines)


def _add_trace_legend(
    dot: "graphviz.Digraph",
    multi_dag: "MultiTraceDAG",
    color_mode: str = "time",
    baseline_profile: Optional["JsonProfile"] = None,
) -> None:
    """Add a legend showing trace colors and color mode information positioned at upper left."""
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend", style="filled", fillcolor="lightgray")
        legend.attr("node", shape="plaintext")
        legend.attr(rank="source")
        legend.attr(pos="0,0!")

        legend_rows = []

        legend_rows.append('<TR><TD COLSPAN="2"><B>Traces</B></TD></TR>')
        for trace_id in sorted(multi_dag.trace_names.keys()):
            trace_name = multi_dag.trace_names[trace_id]
            color = multi_dag.trace_colors[trace_id]
            legend_rows.append(
                f'<TR><TD BGCOLOR="{color}">{trace_name}</TD><TD>Edge Color</TD></TR>'
            )

        color_legend_text = multi_dag.get_color_legend_text(color_mode)
        if color_legend_text:
            legend_rows.append('<TR><TD COLSPAN="2"><B>Kernel Coloring</B></TD></TR>')
            lines = color_legend_text.replace("Legend\\n", "").split("\\n")
            for line in lines:
                if line.strip():
                    legend_rows.append(f'<TR><TD COLSPAN="2">{line}</TD></TR>')

        if legend_rows:
            legend_table = "".join(legend_rows)
            legend.node(
                "legend",
                f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">{legend_table}</TABLE>>',
                rank="source",
            )


def _tint_color(base_color: str, tint: str, intensity: float) -> str:
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


def calculate_kernel_colors(
    color_mode: str,
    base_time_gradients: Dict[str, str],
    kernel_nodes: Dict[str, Any],
    baseline_profile: Optional["JsonProfile"] = None,
) -> Dict[str, str]:
    """
    Calculate colors for kernel nodes based on the specified color mode.

    Args:
        color_mode: One of "time", "diff", "mem-utilization", "compute-utilization", "roofline"
        base_time_gradients: Dictionary mapping kernel names to base gradient colors
        kernel_nodes: Dictionary of kernel nodes containing performance statistics
        baseline_profile: Optional baseline profile for diff mode

    Returns:
        Dictionary mapping kernel names to color strings
    """
    if color_mode not in [
        "time",
        "diff",
        "mem-utilization",
        "compute-utilization",
        "roofline",
    ]:
        return {}

    if color_mode == "time":
        return base_time_gradients

    kernel_colors = {}

    if color_mode == "diff":
        if baseline_profile is None:
            print("Warning: diff coloring requested but no baseline profile provided")
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
                kernel_colors[name] = _tint_color(base_color, "red", 0.8)
            else:
                diff_ratio = (current_duration - baseline_duration) / baseline_duration
                if diff_ratio > 0.1:
                    tint_intensity = min(1.0, diff_ratio)
                    kernel_colors[name] = _tint_color(base_color, "red", tint_intensity)
                elif diff_ratio < -0.1:
                    tint_intensity = min(1.0, abs(diff_ratio))
                    kernel_colors[name] = _tint_color(
                        base_color, "blue", tint_intensity
                    )
                else:
                    kernel_colors[name] = base_color

    elif color_mode == "mem-utilization":
        # Modify base gradients based on memory bandwidth utilization
        for name, base_color in base_time_gradients.items():
            if name not in kernel_nodes:
                continue

            node = kernel_nodes[name]
            if (
                hasattr(node, "achieved_bandwidth_list")
                and node.achieved_bandwidth_list
            ):
                avg_utilization = sum(node.achieved_bandwidth_list) / len(
                    node.achieved_bandwidth_list
                )
                utilization_intensity = min(1.0, avg_utilization / 100.0)
                kernel_colors[name] = _tint_color(
                    base_color, "green", utilization_intensity
                )
            else:
                kernel_colors[name] = base_color

    elif color_mode == "compute-utilization":
        for name, base_color in base_time_gradients.items():
            if name not in kernel_nodes:
                continue

            node = kernel_nodes[name]
            if hasattr(node, "achieved_flops_list") and node.achieved_flops_list:
                avg_utilization = sum(node.achieved_flops_list) / len(
                    node.achieved_flops_list
                )
                utilization_intensity = min(1.0, avg_utilization / 100.0)
                kernel_colors[name] = _tint_color(
                    base_color, "purple", utilization_intensity
                )
            else:
                kernel_colors[name] = base_color

    elif color_mode == "roofline":
        for name, base_color in base_time_gradients.items():
            if name not in kernel_nodes:
                continue

            node = kernel_nodes[name]
            if (
                hasattr(node, "bound_type_list")
                and node.bound_type_list
                and hasattr(node, "achieved_flops_list")
                and node.achieved_flops_list
                and hasattr(node, "achieved_bandwidth_list")
                and node.achieved_bandwidth_list
            ):
                # Calculate the average roofline score
                bound_types = node.bound_type_list
                flops_utils = node.achieved_flops_list
                bw_utils = node.achieved_bandwidth_list

                total_score = 0.0
                valid_instances = 0

                for i in range(min(len(bound_types), len(flops_utils), len(bw_utils))):
                    bound_type = bound_types[i]
                    if bound_type == "compute":
                        total_score += flops_utils[i]
                        valid_instances += 1
                    elif bound_type == "memory":
                        total_score += bw_utils[i]
                        valid_instances += 1

                if valid_instances > 0:
                    avg_score = total_score / valid_instances
                    utilization_intensity = 1.0 - min(1.0, avg_score / 100.0)
                    kernel_colors[name] = _tint_color(
                        base_color, "red", utilization_intensity
                    )
                else:
                    kernel_colors[name] = base_color
            else:
                kernel_colors[name] = base_color

    return kernel_colors


def get_color_legend_text(color_mode: str) -> Optional[str]:
    """Get legend text for the specified color mode."""
    if color_mode == "diff":
        return "Legend\\nRed: Slower than baseline\\nBlue: Faster than baseline\\nWhite: Same as baseline"
    elif color_mode == "mem-utilization":
        return (
            "Legend\\nGreen: Memory bandwidth utilization\\nDarker = lower utilization"
        )
    elif color_mode == "compute-utilization":
        return "Legend\\nPurple: Compute utilization\\nDarker = lower utilization"
    elif color_mode == "roofline":
        return "Legend\\nRoofline Analysis\\nDark Red: Low utilization (worse)\\nLight Yellow: High utilization (better)\\nCompute-bound: Uses compute %\\nMemory-bound: Uses memory %"
    return None


class ParseException(RuntimeError):
    pass
