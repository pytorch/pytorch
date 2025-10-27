# Owner(s): ["oncall: profiler"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._C._profiler import _ExperimentalConfig


class XpuScopeProfilerTest(TestCase):
    def test_scope_profiler(self):
        if not torch.xpu.is_available():
            pytest.skip("XPU not available")

        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
            experimental_config=_ExperimentalConfig(
                profiler_metrics=[
                    "GpuTime",
                    "GpuCoreClocks",
                    "AvgGpuCoreFrequencyMHz",
                    "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
                    "XVE_ACTIVE",
                    "XVE_STALL",
                ],
                profiler_measure_per_kernel=True,
                # custom_profiler_config="XPUPTI_PROFILER_MAX_SCOPES=1234",
            ),
        ) as p:
            r1 = torch.matmul(a, b)
            r2 = torch.add(r1, 1.0)
            result = torch.abs(r2)

        print(result.max().to("cpu"))

        p.export_chrome_trace("asdf.json")

        print(p.key_averages().table())


if __name__ == "__main__":
    run_tests()
