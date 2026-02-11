# Owner(s): ["oncall: distributed"]

import gc
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp._common_utils import _get_param_to_fqns
from torch.distributed.utils import _apply_to_tensors, _replace_by_prefix
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TEST_HPU,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

if TEST_HPU:
    list_device = "hpu"
elif TEST_XPU:
    list_device = "xpu"
else:
    list_device = "cuda"


class TestUtils(TestCase):
    @parametrize(
        "device_list",
        [
            ["cpu"],
            [list_device],
            subtest(["cpu", list_device], name=f"cpu_{list_device}"),
        ],
    )
    @skip_if_lt_x_gpu(1)
    def test_apply_to_tensors(self, device_list):
        expected = 0

        def get_a_tensor():
            """Return a random tensor on random device."""
            dev = random.choice(device_list)
            shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
            t = torch.rand(shape).to(dev)
            nonlocal expected
            expected += t.numel()
            return t

        @dataclass
        class NonFrozenDataClass:
            some_key: str
            some_float: float
            some_tensor: list[torch.Tensor]

        @dataclass(frozen=True)
        class FrozenDataClass:
            some_key: str
            some_float: float
            some_tensor: list[torch.Tensor]

        # create a mixed bag of data.
        data = [1, "str"]
        data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
        data.insert(0, {"x", get_a_tensor(), get_a_tensor()})
        data.append(([1], get_a_tensor(), (1), [get_a_tensor()], {1, 2}))
        data.append(
            {"non_frozen_ds": NonFrozenDataClass("some_key", 1.0, [get_a_tensor()])}
        )
        data.append({"frozen_ds": FrozenDataClass("some_key", 1.0, [get_a_tensor()])})
        od = OrderedDict()
        od["k"] = "value"
        data.append(od)

        total = 0

        def fn(t):
            nonlocal total
            total += t.numel()
            return t

        new_data = _apply_to_tensors(fn, data)
        self.assertEqual(total, expected)
        for i, v in enumerate(data):
            self.assertEqual(type(new_data[i]), type(v))

    @skip_if_lt_x_gpu(1)
    def test_replace_by_prefix(self):
        state_dict = {
            "layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "layer.b": torch.tensor(3),
        }
        original_state_dict = state_dict.copy()
        _replace_by_prefix(state_dict, "layer.", "module.layer.")
        expected = {
            "module.layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "module.layer.b": torch.tensor(3),
        }
        # Note: comparing dicts with tensors directly doesn't work well, so we compare keys
        if set(state_dict.keys()) != set(expected.keys()):
            raise AssertionError(
                f"Expected keys {set(expected.keys())}, got {set(state_dict.keys())}"
            )
        _replace_by_prefix(state_dict, "module.layer.", "layer.")
        if set(state_dict.keys()) != set(original_state_dict.keys()):
            raise AssertionError(
                f"Expected keys {set(original_state_dict.keys())}, got {set(state_dict.keys())}"
            )

    @skip_if_lt_x_gpu(1)
    def test_packed_sequence(self):
        """Test to ensure RNN packed sequences are modified correctly."""
        rnn = nn.RNN(5, 5)

        x = torch.rand((5, 1, 5), dtype=torch.float)
        seq_length = torch.tensor([4], dtype=torch.int)

        def fill_fn(x):
            x.fill_(0)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_length)
        x, _ = rnn(x)
        x = _apply_to_tensors(fill_fn, x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        self.assertEqual(torch.sum(x), 0)

    def test_get_param_to_fqns_scales_linearly(self):
        """Regression test for https://github.com/pytorch/pytorch/issues/168329.

        _apply_to_modules had O(N_submodules * N_params) complexity when
        filter_fqns was large, causing multi-minute hangs on MoE + LoRA models.

        Instead of asserting on absolute elapsed time (which varies across
        machines), we measure the scaling ratio: increasing model size by 8x
        should scale ~8x for O(N) but ~64x for O(N^2). We use the minimum
        elapsed time over multiple iterations to reduce noise from GC pauses
        and scheduling jitter.
        """

        class Expert(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.up = nn.Linear(dim, dim, bias=False)
                self.down = nn.Linear(dim, dim, bias=False)

        class MoELayer(nn.Module):
            def __init__(self, dim, n_experts):
                super().__init__()
                self.experts = nn.ModuleList([Expert(dim) for _ in range(n_experts)])

        def make_model(n_layers):
            return nn.Sequential(*[MoELayer(16, 128) for _ in range(n_layers)])

        num_iters = 5
        warmup = make_model(1)
        _get_param_to_fqns(warmup)
        model_n = make_model(2)
        model_8n = make_model(16)
        gc.collect()
        gc.disable()
        try:
            elapsed_n = float("inf")
            for _ in range(num_iters):
                t0 = time.process_time()
                result_n = _get_param_to_fqns(model_n)  # noqa: F841
                elapsed_n = min(elapsed_n, time.process_time() - t0)
            elapsed_8n = float("inf")
            for _ in range(num_iters):
                t0 = time.process_time()
                result_8n = _get_param_to_fqns(model_8n)  # noqa: F841
                elapsed_8n = min(elapsed_8n, time.process_time() - t0)
        finally:
            gc.enable()
        ratio = elapsed_8n / elapsed_n
        self.assertLess(
            ratio,
            25.0,
            f"_get_param_to_fqns scaling ratio {ratio:.2f}x when 8x-ing "
            f"model size (elapsed_n={elapsed_n:.4f}s, elapsed_8n={elapsed_8n:.4f}s), "
            f"expected <25x for O(N) but got ~64x indicating O(N^2) "
            f"(see https://github.com/pytorch/pytorch/issues/168329)",
        )


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestUtils, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
