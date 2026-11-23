# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import device_supports_fp64
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_CPU, RUN_GPU


class TestVarStdReductionDedup(TestCase):
    @config.patch(var_std_reduction_dedup=True)
    @parametrize("correction", [0, 1])
    @parametrize("strided", [False, True])
    def test_narrowing(self, device, correction, strided):
        if not device_supports_fp64(torch.device(device)):
            self.skipTest("requires float64")

        def fn(x):
            return x.var(dim=-1, correction=correction), x.float().var(
                dim=-1, correction=correction
            )

        x = torch.tensor(
            [[2**30 + i for i in range(4)]], dtype=torch.float64, device=device
        )
        if strided:
            x = x.repeat_interleave(2, dim=-1)[:, ::2]
        expected = (
            torch.tensor([5 / (4 - correction)], dtype=torch.float64, device=device),
            torch.zeros(1, dtype=torch.float32, device=device),
        )
        self.assertEqual(fn(x), expected)
        counters.clear()
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), expected)
        self.assertEqual(counters["inductor"]["var_std_reduction_dedup"], 0)


if RUN_CPU or RUN_GPU:
    variance_test_devices = (["cpu"] if RUN_CPU else []) + (
        [GPU_TYPE] if RUN_GPU else []
    )
    instantiate_device_type_tests(
        TestVarStdReductionDedup,
        globals(),
        only_for=variance_test_devices,
        allow_xpu=True,
    )


if __name__ == "__main__":
    if RUN_CPU or RUN_GPU:
        run_tests(needs="filelock")
