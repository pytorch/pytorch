# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import dataclasses
import io
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from re import escape

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._dynamo.test_case import TestCase
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._subclasses import FakeTensorMode
from torch.export import Dim, dynamic_dim, export, unflatten
from torch.export._trace import (
    _export,
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import onlyCPU, onlyCUDA, instantiate_device_type_tests, ops
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase as TorchTestCase,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    find_library_location,
)
from torch.testing._internal.hop_exportability_db import hop_export_opinfo_db

try:
    from . import testing
except ImportError:
    import testing

from torch.export import export

hop_tests = []

for _, val in hop_export_opinfo_db.items():
    hop_tests.extend(val)


class TestHOP(TestCase):
    def test_all_hops_have_op_info(self):
        from torch._ops import _higher_order_ops
        print(_higher_order_ops.keys())

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_aot_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return op.op(x)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            ep = export(model, (inp.input,))
            self.assertTrue(torch.allclose(model(inp.input), ep.module()(inp.input)))

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_pre_dispatch_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return op.op(x)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            ep = _export(model, (inp.input,), pre_dispatch=True)
            self.assertTrue(torch.allclose(model(inp.input), ep.module()(inp.input)))

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_retrace_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return op.op(x)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            ep = _export(model, (inp.input,), pre_dispatch=True)
            ep = ep.run_decompositions()
            self.assertTrue(torch.allclose(model(inp.input), ep.module()(inp.input)))

instantiate_device_type_tests(TestHOP, globals())

if __name__ == '__main__':
    run_tests()
