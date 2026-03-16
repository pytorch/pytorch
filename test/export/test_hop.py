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
from torch.testing._internal.common_methods_invocations import xfail
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)
from torch.testing._internal.opinfo.core import DecorateInfo


hop_tests = []

for op_info in hop_db:
    op_info_hop_name = op_info.name
    if op_info_hop_name in FIXME_hop_that_doesnt_have_opinfo_test_allowlist:
        continue
    hop_tests.append(op_info)


def skipHopOps(test_case_name, base_test_name, to_skip):
    for entry in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = entry
        matching = [
            o for o in hop_tests
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        if not matching:
            raise AssertionError(f"Couldn't find OpInfo for {entry}")
        for op in matching:
            decorators = list(op.decorators)
            if expected_failure:
                decorators.append(DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name, base_test_name,
                    device_type=device_type, dtypes=dtypes,
                ))
            else:
                decorators.append(DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name, base_test_name,
                    device_type=device_type, dtypes=dtypes,
                ))
            op.decorators = tuple(decorators)

    def wrapped(fn):
        return fn

    return wrapped


_hop_export_skips = {
    xfail("invoke_quant", "simple"),
    xfail("flex_attention", "simple"),
    xfail("flex_attention_backward", "simple"),
    xfail("local_map_hop", "simple"),
}


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

    @skipHopOps("TestHOP", "test_aot_export", _hop_export_skips)
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
            ep = export(model, args, kwargs, strict=True)
            self._compare(model, ep, args, kwargs)
        # With PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1, a memory leak occurs during
        # strict-mode export. We need to manually reset the cache of backends.
        # Specifically, `cached_backends.clear()` is required.
        # Upon examining the items in `cached_backends`,
        # we notice that under strict-mode export, there exists
        # the `dynamo_normalization_capturing_compiler`, which must be
        # cleared to avoid memory leaks. An educated guess is that
        # the `dynamo_normalization_capturing_compiler` references input tensors
        # on CUDA devices and fails to free them.
        torchdynamo._reset_guarded_backend_cache()

    @skipHopOps("TestHOP", "test_pre_dispatch_export", _hop_export_skips)
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
        torchdynamo._reset_guarded_backend_cache()

    @skipHopOps("TestHOP", "test_retrace_export", _hop_export_skips)
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
        torchdynamo._reset_guarded_backend_cache()

    @skipHopOps("TestHOP", "test_serialize_export", _hop_export_skips)
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
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()


instantiate_device_type_tests(TestHOP, globals())

if __name__ == "__main__":
    run_tests()
