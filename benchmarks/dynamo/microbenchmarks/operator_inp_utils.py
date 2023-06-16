import functools
import logging
import math
import os
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, Generator, Iterable, Tuple

import torch
from torch.testing import make_tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

log = logging.getLogger(__name__)

OP_INP_DIRECTORY = os.path.join(os.path.dirname(__file__), "operator_inp_logs")

TIMM_DIR = os.path.join(OP_INP_DIRECTORY, "timm_train")
HF_DIR = os.path.join(OP_INP_DIRECTORY, "hf_train")
TORCHBENCH_DIR = os.path.join(OP_INP_DIRECTORY, "torchbench_train")

aten = torch.ops.aten
tensor_type = torch._C.TensorType.get()

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}


def truncate_inp(arg):
    if arg in dtype_abbrs:
        return dtype_abbrs[arg]
    elif isinstance(arg, torch.device):
        return arg.type
    else:
        return arg


# Serialize Function Call
class FuncCallWrapper:
    def __init__(self, call, *args, **kwargs):
        self.call = call
        self.args = tree_map(truncate_inp, args)
        self.kwargs = tree_map(truncate_inp, kwargs) if kwargs is not None else {}

    def __repr__(self):
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = "".join(
            [f", {str(key)}={value}" for key, value in self.kwargs.items()]
        )
        out = f"{self.call}({args}{kwargs})".strip('"')
        # f strings introduce quotations we dont want
        for key in dtype_abbrs_parsing:
            out = out.replace(f"'{key}'", key)
        return out


def serialize_sparse_tensor(e):
    if isinstance(e, torch._subclasses.FakeTensor):
        return FuncCallWrapper("ST", list(e.shape), e.dtype, e.layout, e.is_coalesced())
    else:
        return FuncCallWrapper(
            "ST", list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()
        )


def deserialize_sparse_tensor(size, dtype, layout, is_coalesced, nnz=None):
    raise NotImplementedError()


def deserialize_tensor(size, dtype, stride=None):
    if stride is not None:
        out = torch.empty_strided(size, stride, dtype=dtype)
    else:
        out = torch.empty(size, dtype=dtype)
    try:
        out.copy_(make_tensor(size, dtype=dtype, device="cpu"))
    except Exception as e:
        print(e)
        return out
    return out


def serialize_tensor(e):
    if not e.is_contiguous():
        return FuncCallWrapper("T", list(e.shape), e.dtype, stride=e.stride())
    else:
        return FuncCallWrapper("T", list(e.shape), e.dtype)


def serialize_torch_args(e):
    if isinstance(e, torch.Tensor):
        if e.is_sparse:
            return serialize_sparse_tensor(e)
        return serialize_tensor(e)
    else:
        return truncate_inp(e)


def contains_tensor(elems):
    for elem in tree_flatten(elems)[0]:
        if isinstance(elem, torch.Tensor):
            return True
    return False


def skip_args(elems):
    for i in tree_flatten(elems)[0]:
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False


def contains_tensor_types(type):
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema

    # skip constructors
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True
    if "_like" in op.name():
        return True

    # allow in place writes
    if schema.is_mutable:
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]

    # skip aliasing unless there are multiple outputs
    if len(tensor_outputs) != 1:
        return False

    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(
                tensor_outputs[0].alias_info.after_set
            ):
                return True

    return False


class OperatorInputsMode(TorchDispatchMode):
    def __init__(self, func_db=None):
        self.func_db = defaultdict(Counter) if func_db is None else func_db

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        arg_meta, kwarg_meta = tree_map(serialize_torch_args, (args, kwargs))

        out = func_overload(*args, **kwargs)

        inps = (args, kwargs)
        if contains_tensor(inps) and not skip_args(inps) and contains_tensor(out):
            serialized_str = repr((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][serialized_str] += 1

        return out

    def log_to_file(self, output_filename, *, skip_non_compute_operators=True):
        sorted_operators = sorted(self.func_db.keys())
        with open(output_filename, "w") as f:
            for operator in sorted_operators:
                if skip_non_compute_operators and non_compute_operator(eval(operator)):
                    continue
                f.write(f"Operator: {operator}\n")
                operator_inputs = self.func_db[operator]
                for inps, count in operator_inputs.items():
                    f.write(f"cnt: {count}, ")
                    # repr will add quotation marks around the dtype strings
                    for dtype_abbr in dtype_abbrs.values():
                        inps = inps.replace("'" + dtype_abbr + "'", dtype_abbr)
                    f.write(inps)
                    f.write("\n")


def map_to_device(e, device):
    if isinstance(e, torch.Tensor):
        return e.to(device)
    elif isinstance(e, torch.device):
        return device
    elif isinstance(e, str):
        if e == "cuda" or e == "cpu":
            return device.type
    else:
        return e


def map_to_dtype(e, dtype):
    if isinstance(e, torch.Tensor) and e.is_floating_point():
        return e.to(dtype)
    elif isinstance(e, torch.dtype):
        return dtype
    else:
        return e


def deserialize_args(inps):
    inps = inps.strip().strip("'")
    global_vals = {
        **{
            "T": deserialize_tensor,
            "ST": deserialize_sparse_tensor,
            "th": torch,
            "inf": math.inf,
            "torch": torch,
        },
        **dtype_abbrs_parsing,
    }
    # f strings introduce quotations we dont want
    for key in dtype_abbrs_parsing:
        inps = inps.replace(f"'{key}'", key)
    return eval(inps.strip().strip("'").strip('"'), global_vals)


class OperatorInputsLoader:
    def __init__(self, json_file_path):
        self.operator_db = defaultdict(Counter)

        with open(json_file_path, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            op_line = lines[i].strip("\n")
            assert "Operator: " in op_line, op_line
            operator = op_line[len("Operator: ") :]
            operator = (
                operator if operator != "aten.sum.SymInt" else "aten.sum.dim_IntList"
            )
            op_inps = Counter()
            i += 1
            while i < len(lines) and "Operator: " not in lines[i]:
                line = lines[i]
                cnt = eval(line[len("cnt: ") : line.find(",")])
                inps = line[line.find(",") + 2 :].strip("'")
                op_inps[inps] += cnt
                i += 1
            self.operator_db[operator] = op_inps

    def get_inputs_for_operator(
        self, operator, dtype=None, device="cuda"
    ) -> Generator[Tuple[Iterable[Any], Dict[str, Any]], None, None]:
        assert (
            str(operator) in self.operator_db
        ), f"Could not find {operator}, must provide overload"

        if "embedding" in str(operator):
            log.warning("Embedding inputs NYI, input data cannot be randomized")
            yield
            return

        # line[1] represents number of times these inputs occured, ignored for now
        for line in self.operator_db[str(operator)].items():
            inps = line[0]

            args, kwargs = deserialize_args(inps)

            # Backwards require some inputs to be float16 and some to be float32
            # So we record on half and upcast to float when specified
            if dtype and dtype != torch.float16:
                to_dtype = partial(map_to_dtype, dtype=dtype)
                args, kwargs = tree_map(to_dtype, (args, kwargs))

            if device:
                to_device = partial(map_to_device, device=torch.device(device))
                args, kwargs = tree_map(to_device, (args, kwargs))

            yield args, kwargs

    def get_all_ops(self):
        for key in self.operator_db.keys():
            try:
                op = eval(key)
            except AttributeError as ae:
                log.warning("Evaluating an op name into an OpOverload: %s", ae)
                continue
            yield op

    def get_call_frequency(self, op):
        assert (
            str(op) in self.operator_db
        ), f"Could not find {op}, must provide overload"

        count = 0
        for _, counter in self.operator_db[str(op)].items():
            count += counter
        return count

    def merge(self, other):
        for operator, counter_dict in other.operator_db.items():
            for inps, cnt in counter_dict.items():
                self.operator_db[operator][inps] += cnt

    @staticmethod
    def get_timm_loader():
        return OperatorInputsLoader._load_directory(TIMM_DIR)

    @staticmethod
    def get_huggingface_loader():
        return OperatorInputsLoader._load_directory(HF_DIR)

    @staticmethod
    def get_torchbench_loader():
        return OperatorInputsLoader._load_directory(TORCHBENCH_DIR)

    @staticmethod
    def _load_directory(inp_dir):
        assert os.path.isdir(inp_dir), inp_dir
        union = None
        for inp in os.listdir(inp_dir):
            if inp[-4:] != ".txt":
                continue
            path = os.path.join(inp_dir, inp)
            if union is None:
                union = OperatorInputsLoader(path)
            else:
                union.merge(OperatorInputsLoader(path))
        return union
