import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from tabulate import tabulate

import torch
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry
import logging
from logging import info
logging.basicConfig(level=logging.DEBUG)


PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"
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
adapters_map: dict[str, Any] = {}


def parse_list(lst: str) -> list[int]:
    lst = lst.replace("[", "").replace("]", "")
    substrings = lst.split(",")
    return [int(substring.strip()) for substring in substrings]


def zip_dicts(dict1: dict[Any, Any], dict2: dict[Any, Any], default: Any = None):
    """
    Zip two dictionaries together, indicating missing keys.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        default (Any):

    Yields:
        tuple: A tuple containing the key, the value from dict1 (or None if missing), and the value from dict2 (or None if missing).
    """
    # Find the union of all keys
    all_keys = OrderedSet(dict1.keys()) | OrderedSet(dict2.keys())

    # Iterate over all keys
    for key in all_keys:
        # Get the values from both dictionaries, or None if missing
        value1 = dict1.get(key, default)
        value2 = dict2.get(key, default)

        yield key, value1, value2


def register_adapter(aten: Union[str, list[str]]):  # type: ignore[no-untyped-def]
    def decorator(func):  # type: ignore[no-untyped-def]
        global _adapters_map

        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = func(*args, **kwargs)
            return result

        if isinstance(aten, str):
            adapters_map[aten] = wrapper
        else:
            for at in aten:
                adapters_map[at] = wrapper
        return wrapper

    return decorator


@register_adapter(["convolution", "_convolution", "cudnn_convolution"])
def conv_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    if len(tmp) == 4:
        transposed = False

    transposed = bool(tmp[6])
    tmp[6] = transposed

    kwargs = {}
    if not transposed:
        # calculate output shape if not transposed.
        def conv_out_dims(x, kernel, stride):
            return (x - kernel) // stride + 1

        stride = parse_list(concrete[3])
        inp = shapes[0]
        w = shapes[1]
        out_x_y = [
            conv_out_dims(*args) for args in zip(inp[2:], w[2:], stride)
        ]
        out = [inp[0], w[0]] + out_x_y  # we only need the xy values
        kwargs["out_val"] = out

    return tuple(tmp[:-1]), kwargs


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


def _calculate_flops(event: dict[str, Any]) -> int:
    name = event["name"]
    if name.startswith("aten::"):
        op_name = name[len("aten::") :]
        op_obj = getattr(torch.ops.aten, op_name)
        if op_obj not in flop_registry:
            return 0

        flop_function = flop_registry[op_obj]

        input_shapes = event["args"]["Input Dims"]
        concrete = event["args"]["Concrete Inputs"]
        if op_name in adapters_map:
            args, kwargs = adapters_map[op_name](input_shapes, concrete)
        else:
            args, kwargs = default_adapter(input_shapes, concrete)
        return flop_function(*args, **kwargs)
    elif "kernel_flop" in event["args"]:
        return event["args"]["kernel_flop"]
    else:
        info(f"Can't calculate flops for kernel: {name}")
        return 0


def _estimate_gb(event: dict[str, Any]) -> float:
    """
    This estimate isn't the best because it doesn't know if two input buffers are the same buffer, leading to an
    overestimate of the real achieved bandwidth.
    """
    if "Input Type" not in event["args"] or "Input Dims" not in event["args"]:
        return 0
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, tipe in sizes_and_types:
        if not hasattr(torch, tipe):
            isize = 0
        else:
            isize = getattr(torch, tipe).itemsize
        bw += isize * math.prod(flatten(size))
    return bw / 1e9


def _augment_trace_helper(data: dict[str, Any]) -> dict[str, Any]:
    # compute a mapping from exteral ids to non kernels, which contain the information we need to estimate flops etc
    extern_mapping = defaultdict(list)
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

    for event in data["traceEvents"]:
        if "cat" not in event:
            continue
        if event["cat"] == "kernel":
            if "args" not in event:
                raise ParseException(f"kernel has no args: {event}")

            external_op = extern_mapping[event["args"]["External id"]][0]
            external_op["args"]["kernel_flop"] = _calculate_flops(external_op)
            external_op["args"]["kernel_num_gb"] = _estimate_gb(external_op)
            event["args"]["kernel_flop"] = external_op["args"]["kernel_flop"]
            event["args"]["kernel_num_gb"] = external_op["args"]["kernel_num_gb"]
    return data


_dtype_map = {
    "float": torch.float,
    "int": torch.int,
    "long": torch.long,
    "long int": torch.long,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class DeviceInfo:
    tflops: dict[torch.dtype, float]
    dram_bw_gbs: float

    @staticmethod
    def get_device_info() -> tuple[dict[torch.dtype, int], float]:
        """
        This is the info that populates DeviceInfo, but it needs to be run on each device separately.
        For new hardware, run this function and then add the information to `_device_mapping`
        """
        # TODO int would probably be good to support
        floats = [torch.float, torch.bfloat16, torch.float16]
        return {
            dtype: get_device_tflops(dtype) for dtype in floats
        }, get_gpu_dram_gbps()


_device_mapping: dict[str, DeviceInfo] = {
    "NVIDIA H100": DeviceInfo(
        tflops={
            torch.float32: 0.033454080000000004,
            torch.bfloat16: 0.5352652800000001,
            torch.float16: 0.5352652800000001,
        },
        dram_bw_gbs=2446.848,
    )
}


def lookup_device_info(name: str) -> "DeviceInfo":
    """
    problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceInfo.get_device_info() and add it to _device_mapping.
    """
    if name not in _device_mapping:
        raise RuntimeError(
            f"Unsupported device in profile: {name}, consider contributing to _device_mapping."
        )
    return _device_mapping[name]


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float
    achieved_flops: float
    achieved_bandwidth: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]


@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: DeviceInfo
    stats: KernelNameMap

    def __repr__(self):
        return f"Device({self.name}, {self.index})"


DeviceMap = dict[int, Device]
Table = Tuple[List[str], Dict[str, List[str]]]


class JsonProfile:
    """operations on json perfetto traces"""

    _devices: DeviceMap

    def __init__(
        self,
        path: str,
        nruns: int,
        benchmark_name: Optional[str] = None,
    ):
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.nruns = nruns
        self.benchmark_name = benchmark_name
        self._create_devices()

    def convert_dtype(self, event) -> torch.dtype:
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
                return torch.float

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

    def _create_devices(self):
        self._devices = {
            dev["id"]: Device(
                dev["name"],
                dev["id"],
                lookup_device_info(dev["name"]),
                defaultdict(OrderedSet),
            )
            for dev in self.data["deviceProperties"]
        }

    def calculate_flops(self, event: dict[str, Any]) -> int:
        return _calculate_flops(event)

    def estimate_gb(self, event: dict[str, Any]) -> float:
        """
        This estimate isn't the best because it doesn't know if two input buffers are the same buffer, leading to an
        overestimate of the real achieved bandwidth.
        """
        return _estimate_gb(event)

    def augment_trace(self) -> None:
        self.data = _augment_trace_helper(self.data)

    def _compute_stats(self) -> None:
        """populates the name -> stats map"""
        for event in self.events:
            if "cat" not in event or "args" not in event or event["cat"] != "kernel":
                continue
            dev = self._devices[event["args"]["device"]]
            dur = event["dur"]
            if "kernel_flop" in event["args"]:
                assert dur != 0
                # 1000ms/s * flop / ms
                op_flops = 1e3 * event["args"]["kernel_flop"] / dur
                if op_flops == 0:
                    achieved_flops = 0
                else:
                    dtype = self.convert_dtype(event)
                    achieved_flops = op_flops / (1e12 * dev.info.tflops[dtype])
            else:
                op_flops = 0
                achieved_flops = 0

            if "kernel_num_gb" in event["args"]:
                assert dur != 0
                # 1000ms/s * gb / ms = gb/s
                op_gbps = 1e3 * event["args"]["kernel_num_gb"] / dur
                achieved_bandwidth = op_gbps / dev.info.dram_bw_gbs
            else:
                op_gbps = 0
                achieved_bandwidth = 0

            dev.stats[event["name"]].add(
                KernelStats(
                    flops=op_flops,
                    bw=op_gbps,
                    latency=dur,
                    achieved_bandwidth=achieved_bandwidth,
                    achieved_flops=achieved_flops,
                )
            )

    def _create_single_table(self, dev: Device) -> Table:
        """Create a table with the devices mapped to indices."""
        headers = [
            "Kernel Name",
            "Kernel Count",
            "FLOPS",
            "bw gbps",
            "Dur (ms)",
            "Achieved FLOPS %",
            "Achieved Bandwidth %",
        ]
        rows: dict[str, list[str]] = {}

        for kernel_name, stats_set in dev.stats.items():
            ker_count = 0
            flops = 0
            flops_count = 0
            achieved_flops = 0
            bw = 0
            bw_count = 0
            achieved_bandwidth = 0
            latency = 0
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
                str(flops / flops_count if flops_count != 0 else 0),
                str(bw / bw_count if bw_count != 0 else 0),
                str(latency / ker_count if ker_count != 0 else 0),
                str(achieved_flops / flops_count if flops_count != 0 else 0),
                str(achieved_bandwidth / bw_count if bw_count != 0 else 0),
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
        new_rows = {}

        for key, row1, row2 in zip_dicts(table1[1], table2[1], default=(["Empty"] * 5)):
            new_rows[key] = row1 + row2
        return new_headers, new_rows

    def report(
        self, other: Optional["JsonProfile"] = None, name_limit: int = 40
    ) -> str:
        def create_ret(table_headers, table_rows):
            table_flattened = [
                [kernel_name[:name_limit], *kernel_vals]
                for kernel_name, kernel_vals in table_rows.items()
            ]
            return tabulate(table_flattened, headers=table_headers)

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
                self_tables, other_tables, default=None
            ):
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
        # print(tabulate(table, headers=headers, tablefmt="grid"))

    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(self.data, f)


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList | dict[str, Any],
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)), (
            f"{list(all_events.keys())}"
        )

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


class ParseException(RuntimeError):
    pass


def flatten(lst: Sequence[Union[int, Sequence[int]]]) -> Sequence[int]:
    """Flatten a nested list of integers."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=6,
        metavar=("input_file1", "nruns1", "name1", "input_file2", "nruns2", "name2"),
        help="Two json traces to compare with, specified as <file1> <nruns1> <name1> <file2> <nruns2> <name2>",
    )
    parser.add_argument(
        "--name_limit",
        type=int,
        help="the maximum name size in the final report",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        type=str,
        nargs=2,
        metavar=("input_file", "output_file"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=3,
        metavar=("input_file", "nruns", "name"),
        help="Run analysis on a single trace, specified as <file> <nruns> <name>",
    )
    args = parser.parse_args()

    if args.diff:
        # todo add name to diff
        p1 = JsonProfile(args.diff[0], int(args.diff[1]), args.diff[2])
        p1.augment_trace()
        p2 = JsonProfile(args.diff[3], int(args.diff[4]), args.diff[5])
        p2.augment_trace()
        if args.name_limit:
            print(p1.report(p2, name_limit=args.name_limit))
        else:
            print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(args.analysis[0], args.analysis[1], args.analysis[2])
        p1.augment_trace()
        if args.name_limit:
            print(p1.report(name_limit=args.name_limit))
        else:
            print(p1.report())
    if args.augment_trace:
        p = JsonProfile(args.augment_trace[0], 1)
        p.augment_trace()
        p.dump(args.augment_trace[1])


if __name__ == "__main__":
    main()
