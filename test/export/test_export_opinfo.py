# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa

import itertools
import os
import subprocess
import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
    ops,
    skipOps,
    skipXPUIf,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    HardwareClassification,
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
    xfail("tensor_split"),
}

# following are failing fake export on gpu device
fake_export_failures = {
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
}

# following are failing fake export with no gpu device available
fake_export_no_gpu_failures = {
    xfail("geqrf"),
    xfail("sparse.sampled_addmm"),
    xfail("to_sparse"),
    xfail("__getitem__"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("nonzero"),
}

fake_export_failures_cuda = fake_export_failures.copy()
fake_export_failures_xpu = fake_export_failures.copy()

if not torch.backends.cuda.is_built():
    fake_export_failures_cuda |= fake_export_no_gpu_failures

if not torch.xpu._is_compiled():
    fake_export_failures_xpu |= fake_export_no_gpu_failures
    fake_export_failures_xpu |= {
        # https://github.com/intel/torch-xpu-ops/issues/5256
        xfail("nn.functional.max_unpool2d"),
        xfail("nn.functional.max_unpool2d", "grad"),
        # https://github.com/intel/torch-xpu-ops/issues/5277
        xfail("nn.functional.scaled_dot_product_attention"),
    }

if torch.xpu.is_available():
    # https://github.com/intel/torch-xpu-ops/issues/5283
    fake_export_failures_xpu -= {xfail("histogram")}

fake_decomposition_failures = {
    xfail("linalg.matrix_rank"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("repeat_interleave"),
    xfail("take"),
}


def _test_export_helper(self, target_device, dtype, op):
    sample_inputs_itr = op.sample_inputs("cpu", dtype, requires_grad=False)

    mode = FakeTensorMode(allow_non_fake_inputs=True)

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
    hw_classification = HardwareClassification.CPU

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps(export_failures | fake_export_failures_cuda)
    @unittest.skipIf(IS_FBCODE, "tests broken with unexpected successes internally")
    def test_fake_export_cuda(self, dtype, op):
        target_device = "cuda:0"
        _test_export_helper(self, target_device, dtype, op)

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps(export_failures | fake_export_failures_xpu)
    @unittest.skipIf(IS_FBCODE, "tests broken with unexpected successes internally")
    def test_fake_export_xpu(self, dtype, op):
        target_device = "xpu:0"
        _test_export_helper(self, target_device, dtype, op)


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


def _get_env_by_device(device):
    env = {}
    device_type = torch.device(device).type
    if device_type == "cuda":
        env = {"CUDA_VISIBLE_DEVICES": ""}
    elif device_type == "xpu":
        env = os.environ.copy()
        env["ONEAPI_DEVICE_SELECTOR"] = "*:cpu"
    return env


class TestExportOnFakeDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    # In CI, this test runs on a GPU machine with gpu build
    # We set device-specific env variable to simulate a CPU machine with gpu build
    # Running this on all ops in op_db is too slow, so we only run on a selected subset
    @onlyAccelerator
    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/5157")
    @unittest.skipIf(
        IS_WINDOWS,
        "Subprocess with simulated CPU machine imports op_db which triggers "
        "get_device_capability(); 0 devices raises Invalid device id on Windows.",
    )
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

    target_device = "{device}"

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
                    env=_get_env_by_device(device),
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")

    @onlyAccelerator
    @unittest.skipIf(
        IS_WINDOWS,
        "Failing on Windows, device_count() changes from 0 to 1 ",
    )
    def test_preserve_original_behavior(self, device):
        test_script = f"""\
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

def accelerator_calls_behavior_unchanged():
    exception_count = 0

    try:
        cpu_x = torch.randn(2)
        accelerator_x = cpu_x.to("{device}")
    except Exception as e:
        exception_count += 1

    try:
        torch.randn(2, device="{device}")
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
        torch.get_device_module("{device}").current_device()
    except Exception as e:
        exception_count += 1

    assert torch.accelerator.is_available() == False
    assert torch.accelerator.device_count() == 0
    assert exception_count == 5

accelerator_calls_behavior_unchanged()

cpu_x = torch.randn(2)
with FakeTensorMode(allow_non_fake_inputs=True) as mode:
    accelerator_x = mode.from_tensor(cpu_x)
    accelerator_x.fake_device = torch.device("{device}")
    accelerator_y = accelerator_x + accelerator_x
    assert accelerator_y.device.type == torch.device("{device}").type

# should fail again after exiting the fake mode, with the identical error message
accelerator_calls_behavior_unchanged()
"""
        r = (
            (
                subprocess.check_output(
                    [sys.executable, "-c", test_script],
                    env=_get_env_by_device(device),
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")


instantiate_device_type_tests(
    TestExportOnFakeDevice, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
