# Owner(s): ["oncall: distributed"]

from collections import OrderedDict
import random
import sys
import unittest

import torch
from torch import distributed as dist
from torch.distributed.fsdp.utils import (
    _apply_to_tensors,
    _replace_by_prefix,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
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


class TestUtils(TestCase):
    @parametrize(
        "devices", [["cpu"], ["cuda"], subtest(["cpu", "cuda"], name="cpu_cuda")]
    )
    def test_apply_to_tensors(self, devices):
        if "cuda" in devices and (
            not torch.cuda.is_available() or torch.cuda.device_count() < 1
        ):
            raise unittest.SkipTest("Skipped due to lack of GPU")

        expected = 0

        def get_a_tensor():
            """Return a random tensor on random device."""
            dev = random.choice(devices)
            shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
            t = torch.rand(shape).to(dev)
            nonlocal expected
            expected += t.numel()
            return t

        # create a mixed bag of data.
        data = [1, "str"]
        data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
        data.insert(0, set(["x", get_a_tensor(), get_a_tensor()]))
        data.append(([1], get_a_tensor(), (1), [get_a_tensor()], set((1, 2))))
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

    def test_replace_by_prefix(self):
        state_dict = {
            "layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "layer.b": torch.tensor(3),
        }
        original_state_dict = state_dict.copy()
        _replace_by_prefix(state_dict, "layer.", "module.layer.")
        assert state_dict == {
            "module.layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "module.layer.b": torch.tensor(3),
        }
        _replace_by_prefix(state_dict, "module.layer.", "layer.")
        assert state_dict == original_state_dict


instantiate_parametrized_tests(TestUtils)

if __name__ == "__main__":
    run_tests()
