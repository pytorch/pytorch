# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa

import itertools
import subprocess
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    onlyCUDA,
    op_db,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    skipIfRocm,
    TestCase,
)
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
    xfail("nn.functional.grid_sample"),
    xfail("to_sparse"),
    # following are failing due to OptionalDeviceGuard
    xfail("__getitem__"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("nonzero"),
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
    target_device = "cuda:0"

    def to_fake_device(x):
        return x.to(target_device)

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
    @unittest.skipIf(IS_FBCODE, "tests broken with unexpected successes internally")
    def test_fake_export(self, device, dtype, op):
        _test_export_helper(self, dtype, op)


instantiate_device_type_tests(TestExportOpInfo, globals(), only_for="cpu")


selected_ops = {
    "__getitem__",
    "nn.functional.batch_norm",
    "nn.functional.conv2d",
    "nn.functional.instance_norm",
    "nn.functional.multi_margin_loss",
    "nn.functional.scaled_dot_product_attention",
    "nonzero",
}
selected_op_db = [op for op in op_db if op.name in selected_ops]


class TestExportOnFakeCuda(TestCase):
    # In CI, this test runs on a CUDA machine with cuda build
    # We set CUDA_VISIBLE_DEVICES="" to simulate a CPU machine with cuda build
    # Running this on all ops in op_db is too slow, so we only run on a selected subset
    @onlyCUDA
    @ops(selected_op_db, allowed_dtypes=(torch.float,))
    def test_fake_export(self, device, dtype, op):
        test_script = f"""\
import torch
import itertools
from torch.testing._internal.common_methods_invocations import op_db
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils import _pytree as pytree

ops = [op for op in op_db if op.name == "{op.name}"]
assert len(ops) > 0

for op in ops:
    sample_inputs_itr = op.sample_inputs("cpu", torch.float, requires_grad=False)

    mode = FakeTensorMode(allow_non_fake_inputs=True)

    target_device = "cuda:0"

    def to_fake_device(x):
        return x.to(target_device)

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
                        assert fake_tensor.device == torch.device(target_device)
"""
        r = (
            (
                subprocess.check_output(
                    [sys.executable, "-c", test_script],
                    env={"CUDA_VISIBLE_DEVICES": ""},
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")

    @unittest.skipIf(not torch.backends.cuda.is_built(), "requires CUDA build")
    def test_preserve_original_behavior(self):
        test_script = f"""\
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

def cuda_calls_behavior_unchanged():
    exception_count = 0

    try:
        cpu_x = torch.randn(2)
        cuda_x = cpu_x.to("cuda")
    except Exception as e:
        exception_count += 1

    try:
        torch.randn(2, device="cuda")
    except Exception as e:
        exception_count += 1

    try:
        torch.cuda.get_device_capability()
    except Exception as e:
        exception_count += 1

    try:
        torch.cuda.set_device(1)
    except Exception as e:
        exception_count += 1

    try:
        torch.cuda.current_device()
    except Exception as e:
        exception_count += 1

    assert torch.cuda.is_available() == False
    assert torch.cuda.device_count() == 0
    assert exception_count == 5

cuda_calls_behavior_unchanged()

cpu_x = torch.randn(2)
with FakeTensorMode(allow_non_fake_inputs=True) as mode:
    cuda_x = mode.from_tensor(cpu_x)
    cuda_x.fake_device = torch.device("cuda")
    cuda_y = cuda_x + cuda_x
    assert cuda_y.device.type == "cuda"

# should fail again after exiting the fake mode, with the identical error message
cuda_calls_behavior_unchanged()
"""
        r = (
            (
                subprocess.check_output(
                    [sys.executable, "-c", test_script],
                    env={"CUDA_VISIBLE_DEVICES": ""},
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")


instantiate_device_type_tests(TestExportOnFakeCuda, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
