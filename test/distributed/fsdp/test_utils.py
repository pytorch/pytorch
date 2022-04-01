# Owner(s): ["oncall: distributed"]

import random
import sys
import unittest

import torch
from torch import distributed as dist
from torch.distributed.fsdp.utils import (
    _apply_to_tensors,
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
        od = dict()
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


instantiate_parametrized_tests(TestUtils)

if __name__ == "__main__":
    run_tests()
