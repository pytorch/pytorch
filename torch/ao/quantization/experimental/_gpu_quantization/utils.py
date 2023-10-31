import os
from typing import Dict, Optional

import torch
from torch.ao.quantization.experimental._gpu_quantization.quant_primitives import (
    safe_int_mm,
)
from torch.utils._python_dispatch import TorchDispatchMode

local_rank = int(os.environ.get("LOCAL_RANK", -1))
log_dir = f"{os.path.expanduser('~')}/cluster/tmp"
log_fname = f"{log_dir}/smoothquant_test_log_{local_rank}.txt"

__all__ = [
    "log_with_rank",
    "clear_logs",
    "compute_error",
    "apply_logging_hook",
    "get_model_size_in_bytes",
]


def log_with_rank(*args):
    # append
    #
    #   {thing_to_log}
    #
    # to {file}_{rank}.txt, for printing stuff from multiple GPUs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_fname, "a") as f:
        f.write(" ".join([str(s) for s in args]) + "\n")
    if local_rank == 0:
        print(*args)


def clear_logs():
    if os.path.isfile(log_fname):
        os.remove(log_fname)


# basic SQNR
def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# logger for fqn + op + shape
# note: not safe for any kind of multithreading
_cur_fqn: Optional[str] = None


def _get_logging_hook(fqn):
    def forward_hook(module, input):
        global _cur_fqn
        _cur_fqn = fqn

    return forward_hook


def apply_logging_hook(model):
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(_get_logging_hook(name))


# collections.defaultdict printing is weird with lambdas, so hand writing for now
fqn_to_op_to_shape_to_count: Dict[
    Optional[str], Dict[Optional[str], Dict[Optional[str], int]]
] = {}


class LoggingTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        global _cur_fqn
        op_name: str = f"{func.__module__}.{func.__name__}"
        shape_str = ""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape_str += str(list(arg.shape)) + ", "
        if shape_str != "":
            shape_str = shape_str[:-2]

        if _cur_fqn not in fqn_to_op_to_shape_to_count:
            fqn_to_op_to_shape_to_count[_cur_fqn] = {}
        if op_name not in fqn_to_op_to_shape_to_count[_cur_fqn]:
            fqn_to_op_to_shape_to_count[_cur_fqn][op_name] = {}
        if shape_str not in fqn_to_op_to_shape_to_count[_cur_fqn][op_name]:
            fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] = 0
        fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] += 1

        return rs


# https://discuss.pytorch.org/t/finding-model-size/130275
def get_model_size_in_bytes(model):
    s = 0
    for p in model.parameters():
        s += p.nelement() * p.element_size()
    for b in model.buffers():
        s += b.nelement() * b.element_size()
    return s
