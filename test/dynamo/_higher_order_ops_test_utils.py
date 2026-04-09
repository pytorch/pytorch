# Owner(s): ["module: dynamo"]
# flake8: noqa: F401

import enum
import functools
import pprint
import re
import unittest
import warnings
from copy import deepcopy

import functorch.experimental.control_flow as control_flow
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    check_dynamic_shape_capture,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    empty_line_normalizer,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    munge_exc,
    parametrize,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)


def count_ops(gm, args, freq, op):
    actual = [node.target for node in gm.graph.nodes].count(op)
    if actual != freq:
        raise AssertionError(f"expected={freq}, actual={actual}")
    return gm


class Obj:
    pass


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        return self.existing * x


global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


# Checks that a dict matches a dict with "regex keys". That is,
# the keys are regex expressions.
def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(
                        f"Single key regex mapped to multiple keys. Please improve your "
                        f"regex. Got: regex='{regex_key}' "
                        f"keys='{regex_key_to_actual_key[regex_key]}',"
                        f"'{key}'"
                    )
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    for regex_key in regex_keys:
        if regex_key not in regex_key_to_actual_key:
            raise AssertionError(
                f"Got regex '{regex_key}' but could not match any key in dict with "
                f"keys {dct.keys()}"
            )
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    self.assertEqual(dct, new_dct)


def default_args_generator(seed_value):
    flat_args, args_spec = pytree.tree_flatten(seed_value)
    for i in range(3):
        new_flat_arg = []
        for val in flat_args:
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i
            elif isinstance(val, int):
                new_val = val + 1 * i
            elif isinstance(val, float):
                new_val = val + 0.1 * i
            elif isinstance(val, enum.Enum):
                new_val = val
            else:
                raise AssertionError("unexpected arg type")

            new_flat_arg.append(new_val)
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        yield new_args
