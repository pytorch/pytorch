# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa

import itertools

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils import _pytree as pytree


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
    xfail("to_sparse"),
    # cannot xfail as it is passing for cpu-only build
    skip("nn.functional.grid_sample"),
    skip("nn.functional.conv2d"),
    skip("nn.functional.scaled_dot_product_attention"),
}

fake_decomposition_failures = {
    xfail("linalg.matrix_rank"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("repeat_interleave"),
    xfail("take"),
}


def _test_export_helper(self, dtype, op):
    sample_inputs_itr = op.sample_inputs("cpu", dtype, requires_grad=False)

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    converter = mode.fake_tensor_converter
    # intentionally avoid cuda:0 to flush out some bugs
    target_device = "cuda:1"

    def to_fake_device(x):
        x = converter.from_real_tensor(mode, x)
        x.fake_device = torch.device(target_device)
        return x

    # Limit to first 100 inputs so tests don't take too long
    for sample_input in itertools.islice(sample_inputs_itr, 100):
        args = tuple([sample_input.input] + list(sample_input.args))
        kwargs = sample_input.kwargs

        # hack to skip non-tensor in args, as export doesn't support it
        if any(not isinstance(arg, torch.Tensor) for arg in args):
            continue

        if "device" in kwargs:
            kwargs["device"] = target_device

        with mode:
            args, kwargs = pytree.tree_map_only(
                torch.Tensor, to_fake_device, (args, kwargs)
            )

            class Module(torch.nn.Module):
                def forward(self, *args):
                    return op.op(*args, **kwargs)

            m = Module()

            ep = torch.export.export(m, args)

            for node in ep.graph.nodes:
                if node.op == "call_function":
                    fake_tensor = node.meta.get("val", None)
                    if isinstance(fake_tensor, FakeTensor):
                        self.assertEqual(
                            fake_tensor.device, torch.device(target_device)
                        )


class TestExportOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestExportOpInfo", "test_fake_export", export_failures | fake_export_failures
    )
    def test_fake_export(self, device, dtype, op):
        _test_export_helper(self, dtype, op)


only_for = "cpu"
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for=only_for)


if __name__ == "__main__":
    run_tests()
