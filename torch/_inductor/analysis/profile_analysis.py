import logging
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor.analysis.device_info import DeviceSpec
from torch._inductor.analysis.json_profile import JsonProfile
from torch._inductor.analysis.utils import (
    create_multi_trace_visualization,
    get_cache_dir,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry


log = logging.getLogger(__name__)


ATEN_PREFIX = "aten::"


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


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


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float  # us
    achieved_flops: float
    achieved_bandwidth: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]


@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: Optional[DeviceSpec]
    stats: KernelNameMap

    def __repr__(self) -> str:
        return f"Device({self.name}, {self.index}): {self.info}"


DeviceMap = dict[int, Device]
Table = tuple[list[str], dict[str, list[str]]]


class ParseException(RuntimeError):
    pass


def main() -> None:
    """
    Main function for the profile analysis script.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=5,
        metavar=(
            "input_file1",
            "name1",
            "input_file2",
            "name2",
            "dtype",
        ),
        help="Two json traces to compare with, specified as <file1> <name1> <file2> <name2> <dtype>",
    )
    parser.add_argument(
        "--name_limit",
        type=int,
        help="the maximum name size in the final report",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        nargs=3,
        metavar=("input_file", "output_file", "dtype"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=2,
        metavar=("input_file", "dtype"),
        help="Run analysis on a single trace, specified as <file> <dtype>",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        metavar=("input_files", "output_file"),
        help="Combine multiple profiles into a single profile by merging trace events. Specify as <input_file1> \
<input_file2> [input_file3 ...] <output_file>. The last argument is the output file, all preceding arguments are \
input files to combine.",
    )
    parser.add_argument(
        "--visualize",
        nargs="+",
        metavar=("input_file", "args"),
        help="Create a DAG visualization of multiple traces showing operation flow from ops to kernels. \
Specify as <input_file1> [input_file2 ...] <dtype> <output_file>. At least 3 arguments required (1 input, dtype, output)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output format for visualization (default: png)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for trace analysis and DAG building (default: True for multiple traces)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing and use single-threaded mode",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker processes for parallel processing (default: number of CPU cores)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable DAG caching to disk (default: caching enabled)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the DAG cache directory before processing",
    )
    parser.add_argument(
        "--color",
        choices=["time", "diff", "mem-utilization", "compute-utilization", "roofline"],
        default="time",
        help="Coloring mode for kernel nodes: 'time' colors by kernel runtime percentage (default), 'diff' colors by duration difference between profiles, 'mem-utilization' colors by memory bandwidth utilization %, 'compute-utilization' colors by compute utilization %, 'roofline' colors by roofline analysis (darker = lower utilization)",
    )
    parser.add_argument(
        "--diff-baseline",
        help="Path to baseline profile for diff coloring. Required when --color=diff is used.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Limit the height of the visualization to only show this many levels of non-kernel nodes above kernels. For example, height=1 shows only direct parents of kernels. Default: no height limit.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Make kernel names compact without line wrapping.",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        metavar=("input_file", "n", "output_prefix"),
        help="Split a JSON profile into n equal parts by number of events. Specify as <input_file> <n> <output_prefix>",
    )
    args = parser.parse_args()

    compact_mode = args.compact

    # Validate color/diff-baseline arguments
    if args.color == "diff" and not args.diff_baseline:
        print("Error: --diff-baseline is required when --color=diff is used")
        return
    if args.diff_baseline and args.color != "diff":
        print(
            "Warning: --diff-baseline specified but --color is not 'diff'. Ignoring --diff-baseline."
        )

    # Handle cache clearing
    if args.clear_cache:
        cache_dir = get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared cache directory: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")

    # Determine parallelization settings
    use_parallel = (
        not args.no_parallel
    )  # Default is True unless --no-parallel is specified
    if args.parallel:
        use_parallel = True

    use_cache = not args.no_cache  # Default is True unless --no-cache is specified

    if args.diff:
        p1 = JsonProfile(args.diff[0], args.diff[1], dtype=args.diff[4])
        p1.augment_trace()
        p2 = JsonProfile(args.diff[2], args.diff[3], dtype=args.diff[4])
        p2.augment_trace()
        if args.name_limit:
            print(p1.report(p2, name_limit=args.name_limit))
        else:
            print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(
            args.analysis[0],
            dtype=args.analysis[1],
        )
        p1.augment_trace()
        if args.name_limit:
            print(p1.report(name_limit=args.name_limit))
        else:
            print(p1.report())
    if args.augment_trace:
        p = JsonProfile(args.augment_trace[0], dtype=args.augment_trace[2])
        p.augment_trace()
        p.dump(args.augment_trace[1])
    if args.combine:
        input_files = args.combine[:-1]  # All arguments except the last one
        output_file = args.combine[-1]  # Last argument is the output file

        if len(input_files) < 2:
            print("Error: At least 2 input files are required for combining")
            return

        # Load the first profile
        combined = JsonProfile(input_files[0], dtype=None)

        # Iteratively combine with all other profiles
        for input_file in input_files[1:]:
            profile = JsonProfile(input_file, dtype=None)
            combined = combined.combine_with(profile)

        combined.dump(output_file)
        print(f"Successfully combined {', '.join(input_files)} into {output_file}")
    if args.visualize:
        if len(args.visualize) < 3:
            print(
                "Error: --visualize requires at least 3 arguments: <input_file1> <dtype> <output_file>"
            )
            return

        input_files = args.visualize[:-2]  # All but last 2 arguments
        dtype = args.visualize[-2]  # Second to last argument
        output_file = args.visualize[-1]  # Last argument

        print(
            f"Creating multi-trace DAG visualization from {len(input_files)} traces..."
        )
        print(f"Using parallel processing: {use_parallel}")
        print(f"Using caching: {use_cache}")
        print(f"Output format: {args.format}")
        if args.max_workers:
            print(f"Max workers: {args.max_workers}")

        if len(input_files) == 1:
            # Single trace visualization (backward compatibility)
            profile = JsonProfile(input_files[0], dtype=dtype)

            # Check if trace needs augmentation and augment if needed
            if not profile._is_trace_augmented():
                print("Augmenting trace to add performance statistics...")
                profile.augment_trace()
                print("Trace augmentation completed.")

            # Handle baseline profile for diff coloring
            baseline_profile = None
            if args.color == "diff" and args.diff_baseline:
                baseline_profile = JsonProfile(args.diff_baseline, dtype=dtype)
                # Also check and augment baseline profile if needed
                if not baseline_profile._is_trace_augmented():
                    print("Augmenting baseline trace to add performance statistics...")
                    baseline_profile.augment_trace()
                    print("Baseline trace augmentation completed.")

            dag = profile.create_trace_dag_visualization(
                output_file,
                format=args.format,
                color_mode=args.color,
                baseline_profile=baseline_profile,
                height=args.height,
                compact=compact_mode,
            )
            print(f"DAG visualization completed and saved to {output_file}")
            print(
                f"Found {len(dag.nodes)} nodes and {len(dag.edges)} edges in the trace DAG"
            )
        else:
            # Handle baseline profile for diff coloring
            baseline_profile = None
            if args.color == "diff" and args.diff_baseline:
                baseline_profile = JsonProfile(args.diff_baseline, dtype=dtype)
                # Check and augment baseline profile if needed
                if not baseline_profile._is_trace_augmented():
                    print("Augmenting baseline trace to add performance statistics...")
                    baseline_profile.augment_trace()
                    print("Baseline trace augmentation completed.")

            multi_dag = create_multi_trace_visualization(
                input_files,
                output_file,
                dtype,
                use_cache=use_cache,
                use_parallel=use_parallel,
                max_workers=args.max_workers,
                format=args.format,
                height=args.height,
                color_mode=args.color,
                baseline_profile=baseline_profile,
                compact=compact_mode,
            )
            print(f"Multi-trace DAG visualization completed and saved to {output_file}")
            print(
                f"Combined {len(input_files)} traces with {len(multi_dag.nodes)} unique nodes"
            )

    if args.split:
        input_file = args.split[0]
        n = int(args.split[1])
        output_prefix = args.split[2]

        if n <= 0:
            print("Error: n must be a positive integer")
            return

        print(
            f"Splitting profile {input_file} into {n} parts with prefix {output_prefix}..."
        )
        split_profile(input_file, n, output_prefix)
        print(f"Successfully split profile into {n} parts")


def split_profile(input_file: str, n: int, output_prefix: str) -> None:
    """
    Split a JSON profile into n equal parts by number of events.

    The function sorts events by timestamp, splits them into n equal parts,
    and cleans up dangling event ID references.
    """
    import copy
    import json

    # Load the original profile
    with open(input_file, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    if not events:
        print("Warning: No trace events found in the profile")
        return

    # Sort events by timestamp to maintain temporal ordering
    events.sort(key=lambda e: e.get("ts", 0))

    # Calculate the number of events per split
    events_per_split = len(events) // n
    remainder = len(events) % n

    print(f"Total events: {len(events)}")
    print(
        f"Events per split: {events_per_split} (with {remainder} extra events distributed)"
    )

    start_idx = 0
    for i in range(n):
        # Calculate end index for this split
        current_split_size = events_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size

        # Extract events for this split
        split_events = events[start_idx:end_idx]

        # Build a mapping of event ID -> event for this split
        eid_to_event = {}
        split_event_ids = set()

        # First pass: collect all event IDs that exist in this split
        for event in split_events:
            if "id" in event:
                event_id = event["id"]
                eid_to_event[event_id] = event
                split_event_ids.add(event_id)

            # Also collect External IDs which are references to other events
            if "args" in event and "External id" in event["args"]:
                ext_id = event["args"]["External id"]
                split_event_ids.add(ext_id)

        # Second pass: clean up events, removing those with dangling references
        cleaned_events = []
        removed_count = 0

        for event in split_events:
            should_keep = True
            cleaned_event = copy.deepcopy(event)

            # Check if this event references external IDs not in this split
            if "args" in cleaned_event and "External id" in cleaned_event["args"]:
                ext_id = cleaned_event["args"]["External id"]
                # If the external ID is not in our split's event IDs, remove this event
                if ext_id not in eid_to_event:
                    should_keep = False
                    removed_count += 1

            # Also check for other potential ID references (pid, tid, etc. are usually fine)
            # but we might want to validate other fields that could reference events

            if should_keep:
                cleaned_events.append(cleaned_event)

        # Create the split profile data
        split_data = copy.deepcopy(data)
        split_data["traceEvents"] = cleaned_events

        # Save the split profile
        output_file = f"{output_prefix}_{i+1}_of_{n}.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)

        print(
            f"Created split {i+1}/{n}: {output_file} with {len(cleaned_events)} events"
        )
        if removed_count > 0:
            print(
                f"  Removed {removed_count} events with dangling external ID references"
            )

        start_idx = end_idx


if __name__ == "__main__":
    main()
