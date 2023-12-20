import hashlib
import io
import json
import logging
import operator
import re
from functools import reduce
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class AutotuningLogParser:
    """
    Parser which can be used to analyze the autotuning results
    written by DebugContext.log_autotuning_results(...)

    requires pandas to be installed
    """

    def __init__(
        self, autotuning_result_json_list_path_or_fd: Union[str, io.TextIOWrapper]
    ):
        self.autotuning_result_json_list_path_or_fd = (
            autotuning_result_json_list_path_or_fd
        )

    def get_records(self):
        return self.load_records(self.autotuning_result_json_list_path_or_fd)

    def get_dataframe(self):
        import pandas  # type: ignore[import-untyped]

        res = pandas.DataFrame.from_records(
            self.load_records(self.autotuning_result_json_list_path_or_fd)
        )
        if "Bias_shape" not in res.columns:
            res["Bias_shape"] = None
        return res

    def get_analysis(self):
        df = self.get_dataframe()
        return df.groupby(
            [
                "problem_hash",
                "BATCHSIZE",
                "M",
                "N",
                "K",
                "A_shape",
                "B_shape",
                "Bias_shape",
                "tile_shape",
                "backend",
                "kernel_schedule",
            ],
            dropna=False,
            as_index=False,
        ).benchmark_result.min()

    def get_detail_analysis(self):
        df = self.get_dataframe()
        return df.loc[
            df.groupby(
                [
                    "problem_hash",
                    "BATCHSIZE",
                    "M",
                    "N",
                    "K",
                    "A_shape",
                    "B_shape",
                    "Bias_shape",
                    "tile_shape",
                    "backend",
                    "kernel_schedule",
                ],
                dropna=False,
            ).benchmark_result.idxmin()
        ]

    def get_best_analysis(self):
        df = self.get_dataframe()
        return df.loc[
            df.groupby(
                [
                    "BATCHSIZE",
                    "M",
                    "N",
                    "K",
                    "A_shape",
                    "B_shape",
                    "Bias_shape",
                ],
                dropna=False,
            ).benchmark_result.idxmin()
        ]

    @staticmethod
    def strhash(s):
        return hashlib.md5(s.encode()).hexdigest()

    @staticmethod
    def dim_info(size, stride):
        if stride == 0:
            return f"*{size}"
        if stride == 1:
            return f"!{size}"
        return str(size)

    @staticmethod
    def parse_dtype(layout_str):
        match: Optional[re.Match[str]] = re.search(
            r"Layout\('cuda', torch\.([^, ]+),", layout_str
        )
        assert match is not None, f"Failed to parse dtype: {layout_str}"
        return match.group(1)

    @classmethod
    def parse_record(cls, rec: Dict[str, Any]):
        result = dict()
        result["backend"] = rec["backend"]
        result["name"] = rec.get("name", "ATen")
        inputs = json.dumps(rec["input_nodes"])
        result["problem_hash"] = cls.strhash(inputs)
        result["kernel_schedule"] = rec.get("kernel_schedule", "")
        result["tile_shape"] = rec.get("tile_shape", "[]")
        result["benchmark_result"] = rec["benchmark_result"]
        result["device"] = rec.get("device", "unknown")
        result["cuda_device_name"] = rec["cuda_device_name"]
        # result['keys'] = ",".join(rec.keys())
        for name, tinfo in cls.parse_inputs(rec["input_nodes"], rec["backend"]).items():
            if isinstance(tinfo, dict):
                for key, value in tinfo.items():
                    result[f"{name}_{key}"] = value
            elif tinfo is not None:
                result[name] = tinfo
        return result

    @staticmethod
    def load_raw_records(path_or_fd: Union[str, io.TextIOWrapper]):
        if isinstance(path_or_fd, str):
            with open(path_or_fd, encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)
        else:
            if hasattr(path_or_fd, "seek"):
                path_or_fd.seek(0)
            for line in path_or_fd:
                yield json.loads(line)

    @classmethod
    def load_records(
        cls,
        path_or_fd: Union[str, io.TextIOWrapper, List[Union[str, io.TextIOWrapper]]],
    ):
        if isinstance(path_or_fd, list):
            for pf in path_or_fd:
                yield from cls.load_records(pf)
        else:
            for record in cls.load_raw_records(path_or_fd):
                yield cls.parse_record(record)

    @staticmethod
    def load_raw_grepped_records(path, pattern):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if pattern in line:
                    yield json.loads(line)

    @classmethod
    def parse_layout(cls, layout_str):
        match: Optional[re.Match[str]] = re.search(
            r"size=\[([0-9]+), ([0-9]+)(, ([0-9]+))?\], stride=\[([0-9]+), ([0-9]+)(, ([0-9]+))?\]",
            layout_str,
        )
        assert match is not None, f"Failed to parse layout: {layout_str}"
        vals = match.groups()
        res: Dict[str, Any] = {}
        if vals[2] is None:
            assert vals[6] is None
            res = {
                "size": (1, int(vals[0]), int(vals[1])),
                "stride": (0, int(vals[4]), int(vals[5])),
            }
        else:
            res = {
                "size": (int(vals[0]), int(vals[1]), int(vals[3])),
                "stride": (int(vals[4]), int(vals[5]), int(vals[7])),
            }

        if res["stride"][-1] == 1:
            res["type"] = "row_major"
        else:
            res["type"] = "col_major"
        res["dtype"] = cls.parse_dtype(layout_str)
        res["shape"] = tuple(
            cls.dim_info(size, stride)
            for size, stride in zip(res["size"], res["stride"])
        )
        return res

    @classmethod
    def parse_inputs(cls, input_nodes, backend):
        if len(input_nodes) == 2:
            a_layout = cls.parse_layout(input_nodes[0]["layout"])
            b_layout = cls.parse_layout(input_nodes[1]["layout"])
            c_layout = None
        else:
            if backend == "CUDA" and input_nodes[0]["name"] != "addmm":
                a_layout = cls.parse_layout(input_nodes[0]["layout"])
                b_layout = cls.parse_layout(input_nodes[1]["layout"])
                c_layout = cls.parse_layout(input_nodes[2]["layout"])
            else:
                a_layout = cls.parse_layout(input_nodes[1]["layout"])
                b_layout = cls.parse_layout(input_nodes[2]["layout"])
                c_layout = cls.parse_layout(input_nodes[0]["layout"])
        assert (
            a_layout["size"][-1] == b_layout["size"][-2]
        ), f"Incompatible input layouts. {a_layout=}, {b_layout=}, {input_nodes[0]=}, {input_nodes[1]=}"
        problem_shape = (
            a_layout["size"][-2],
            b_layout["size"][-1],
            a_layout["size"][-1],
        )
        if len(a_layout["size"]) > 2:
            batch_size = prod(list(a_layout["size"][0:-2]))
        else:
            batch_size = 1
        M, N, K = problem_shape
        return {
            "problem_shape_MNK": problem_shape,
            "A": a_layout,
            "B": b_layout,
            "Bias": c_layout,
            "BATCHSIZE": batch_size,
            "M": M,
            "N": N,
            "K": K,
        }
