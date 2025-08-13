# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_methods_invocations import op_db, xfail, skipOps
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
import itertools
from torch.utils import _pytree as pytree
import contextlib

# following are failing with regular torch.export.export
export_failures = {
    xfail("allclose"),
    xfail("combinations"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("equal"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("sparse.sampled_addmm"),
    xfail("tensor_split"),
}

# following are failing fake export on cuda device 
fake_export_failures = {
    xfail("geqrf"),
    xfail("histogram"),
    xfail("masked.amax"),
    xfail("masked.amin"),
    xfail("masked.argmax"),
    xfail("masked.argmin"),
    xfail("masked.logaddexp"),
    xfail("masked.logsumexp"),
    xfail("masked.mean"),
    xfail("masked.prod"),
    xfail("masked.std"),
    xfail("masked.sum"),
    xfail("masked.var"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.scaled_dot_product_attention"),
    xfail("to_sparse"),
}

fake_decomposition_failures = {
}

def _test_export_helper(self, device, dtype, op, fake_export_on_cuda):
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    # Limit to first 100 inputs so tests don't take too long
    for sample_input in itertools.islice(sample_inputs_itr, 100):
        args = tuple([sample_input.input] + list(sample_input.args))
        kwargs = sample_input.kwargs

        # hack to skip non-tensor in args, as export doesn't support it
        if any(not isinstance(arg, torch.Tensor) for arg in args):
            continue

        if fake_export_on_cuda:
            mode = FakeTensorMode(allow_non_fake_inputs=True)
        else:
            mode = contextlib.nullcontext()

        with mode:
            if fake_export_on_cuda:
                args, kwargs = pytree.tree_map_only(torch.Tensor, lambda x: x.to("cuda:0"), (args, kwargs))

            class Module(torch.nn.Module):
                def forward(self, *args):
                    return op.op(*args, **kwargs)
            m = Module()

            ep = torch.export.export(m, args)

            for node in ep.graph.nodes:
                if node.op == "call_function":
                    fake_tensor = node.meta.get("val", None)
                    if isinstance(fake_tensor, FakeTensor):
                        self.assertEqual(fake_tensor.device, torch.device("cuda:0"))

            # ep = ep.run_decompositions()


class TestExportOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestExportOpInfo', 'test_export', export_failures | fake_export_failures)
    def test_export(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op, fake_export_on_cuda=True)


only_for = ("cpu")
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
