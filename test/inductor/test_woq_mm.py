# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config
import torch.utils.checkpoint
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CPU


class TestWoqMMPatternRewriterTemplate(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return [clone(x) for x in inputs]

    def _check_common(
        self,
        mod,
        args1=None,
        contains=True,
        atol=0.001,
        has_pattern=True,
        override_check_equal=False,
        x_dtype=torch.bfloat16,
        w_dtype=torch.int8,
        rtol=0.07,
    ):
        if args1 is None:
            x_shape = (1, 1, 256)
            w_shape = (12, 256)
            s_shape = 12
            args1 = [
                torch.randn(x_shape, device=self.device, dtype=x_dtype),
                torch.randint(-128, 127, w_shape, device=self.device, dtype=w_dtype),
                torch.randn(s_shape, device=self.device, dtype=x_dtype),
            ]
        else:
            args1 = list(args1)
        args2 = self._clone_inputs(args1)

        torch.manual_seed(1234)
        result1 = mod(*args1)

        counters.clear()
        torch.manual_seed(1234)
        result2, source_code = run_and_get_code(
            torch.compile(mod, fullgraph=True),
            *args2,
        )
        source_code = "\n".join(source_code)
        if has_pattern:
            self.assertGreaterEqual(counters["inductor"]["woq_mm"], 1)
        if contains:
            self.assertIn(
                "aten._weight_int8pack_mm",
                source_code,
            )

        if override_check_equal:
            self.assertEqual(result1, result2, atol=atol, rtol=rtol)

    @skipIfRocm
    def _test_woq_mm_rewriter_1(self):
        def mod(
            x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.linear(x, weight.to(dtype=x.dtype)) * scales

        self._check_common(mod, override_check_equal=True)


if HAS_CPU:

    class WoqMMPatternRewriterCpuTests(TestWoqMMPatternRewriterTemplate):
        device = "cpu"
        test_woq_mm_rewriter_1_cpu = (
            TestWoqMMPatternRewriterTemplate._test_woq_mm_rewriter_1
        )


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
