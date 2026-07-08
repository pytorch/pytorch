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
    onlyOn,
    ops,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_WINDOWS,
    run_tests,
    TestCase,
)
from torch.utils import _pytree as pytree


# following are failing with regular torch.export.export
export_failures = {
    xfail("allclose"),
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


def _test_export_helper(self, device, dtype, op):
    sample_inputs_itr = op.sample_inputs("cpu", dtype, requires_grad=False)

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    target_device = f"{device}:0"

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
    @skipOps(export_failures | fake_export_failures)
    @unittest.skipIf(IS_FBCODE, "tests broken with unexpected successes internally")
    def test_fake_export(self, device, dtype, op):
        _test_export_helper(self, device, dtype, op)


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
    # In CI, this test runs on a CUDA/XPU machine with corresponding build
    # We set CUDA_VISIBLE_DEVICES="" or ZE_AFFINITY_MASK="" to simulate a CPU machine
    # Running this on all ops in op_db is too slow, so we only run on a selected subset
    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(
        IS_WINDOWS,
        'Subprocess with env vars set imports op_db which triggers '
        "get_device_capability(); 0 devices raises Invalid device id on Windows.",
    )
    @ops(selected_op_db, allowed_dtypes=(torch.float,))
    def test_fake_export(self, device, dtype, op):
        device_type = torch.accelerator.current_accelerator().type
        env_var_name = {
            "cuda": "CUDA_VISIBLE_DEVICES",
            "xpu": "ZE_AFFINITY_MASK",
        }[device_type]
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

    target_device = "{device_type}:0"

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
                    env={f"{env_var_name}": ""},
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(
        IS_WINDOWS,
        "Failing on Windows, device_count() changes from 0 to 1 ",
    )
    def test_preserve_original_behavior(self):
        device_type = torch.accelerator.current_accelerator().type
        env_var_name = {
            "cuda": "CUDA_VISIBLE_DEVICES",
            "xpu": "ZE_AFFINITY_MASK",
        }[device_type]
        test_script = f"""\
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

def cuda_calls_behavior_unchanged():
    exception_count = 0

    try:
        cpu_x = torch.randn(2)
        cuda_x = cpu_x.to("{device_type}")
    except Exception as e:
        exception_count += 1

    try:
        torch.randn(2, device="{device_type}")
    except Exception as e:
        exception_count += 1

    try:
        torch.accelerator.get_device_capability()
    except Exception as e:
        exception_count += 1

    try:
        torch.accelerator.set_device_index(1)
    except Exception as e:
        exception_count += 1

    try:
        torch.accelerator.current_device_index()
    except Exception as e:
        exception_count += 1

    assert torch.accelerator.is_available() == False
    assert torch.accelerator.device_count() == 0
    assert exception_count == 5

cuda_calls_behavior_unchanged()

cpu_x = torch.randn(2)
with FakeTensorMode(allow_non_fake_inputs=True) as mode:
    cuda_x = mode.from_tensor(cpu_x)
    cuda_x.fake_device = torch.device("{device_type}")
    cuda_y = cuda_x + cuda_x
    assert cuda_y.device.type == "{device_type}"

# should fail again after exiting the fake mode, with the identical error message
cuda_calls_behavior_unchanged()
"""
        r = (
            (
                subprocess.check_output(
                    [sys.executable, "-c", test_script],
                    env={f"{env_var_name}": ""},
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")


instantiate_device_type_tests(
    TestExportOnFakeCuda, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
