# Owner(s): ["module: inductor"]
import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.triton_utils import requires_cuda


class CondTests(TestCase):
    @requires_cuda
    def test_cond(self):
        def cond_f(pred, x, y):
            def true_fn(x, y):
                return torch.cat([x - 3, y * 3], dim=1) * 3.14

            def false_fn(x, y):
                return torch.cat([x + 3, y / 3], dim=1) / 2.17

            z = torch.cat([x, y], dim=0)

            return torch.cond(pred, true_fn, false_fn, [z * 2, z / 2])[0]

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_f = torch.compile(backend=cnt, fullgraph=True)(cond_f)

        x = torch.randn(2, 3, 3).cuda()
        y = torch.randn(4, 3, 3).cuda()

        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)

        for pred_value in [True, False]:
            pred = torch.tensor(pred_value).cuda()

            result = cond_f(pred, x, y)
            result_compiled = compiled_f(pred, x, y)

            self.assertEqual(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
