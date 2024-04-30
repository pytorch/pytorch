# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import io
import unittest

import torch
import torch._dynamo as torchdynamo
import torch.utils._pytree as pytree
from torch._dynamo.test_case import TestCase
from torch.export import export, load, save
from torch.export._trace import _export
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    TestCase as TorchTestCase,
)
from torch.testing._internal.hop_db import (
    hop_db,
    hop_that_doesnt_have_opinfo_test_allowlist,
)

hop_tests = []

for op_info in hop_db:
    op_info_hop_name = op_info.name
    if op_info_hop_name in hop_that_doesnt_have_opinfo_test_allowlist:
        continue
    hop_tests.append(op_info)


class TestHOPGeneric(TestCase):
    def test_all_hops_have_op_info(self):
        from torch._ops import _higher_order_ops

        hops_that_have_op_info = set([k.name for k in hop_db])
        all_hops = _higher_order_ops.keys()

        missing_ops = []

        for op in all_hops:
            if (
                op not in hops_that_have_op_info
                and op not in hop_that_doesnt_have_opinfo_test_allowlist
            ):
                missing_ops.append(op)

        self.assertTrue(len(missing_ops) == 0, f"Missing op info for {missing_ops}")


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestHOP(TestCase):
    def _compare(self, eager_model, export, args, kwargs):
        eager_args = copy.deepcopy(args)
        eager_kwargs = copy.deepcopy(kwargs)
        export_args = copy.deepcopy(args)
        export_kwargs = copy.deepcopy(kwargs)

        flat_orig_outputs = pytree.tree_leaves(eager_model(*eager_args, **eager_kwargs))
        flat_loaded_outputs = pytree.tree_leaves(
            export.module()(*export_args, **export_kwargs)
        )

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertEqual(type(orig), type(loaded))
            self.assertEqual(orig, loaded)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_aot_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = export(model, args, kwargs)
            self._compare(model, ep, args, kwargs)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_pre_dispatch_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            self._compare(model, ep, args, kwargs)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_retrace_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            self._compare(model, ep, args, kwargs)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    def test_serialize_export(self, device, dtype, op):
        class Foo(torch.nn.Module):
            def forward(self, *args):
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            ep = load(buffer)
            if "while_loop" in str(op):
                # while_loop's arguments are cast into list after deserailize
                # but while_loop expects it to still be tuple
                with self.assertRaisesRegex(
                    RuntimeError, "carried_inputs must be a tuple"
                ):
                    self._compare(model, ep, args, kwargs)
            else:
                self._compare(model, ep, args, kwargs)


instantiate_device_type_tests(TestHOP, globals())

if __name__ == "__main__":
    run_tests()
