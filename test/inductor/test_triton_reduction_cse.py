# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


class TritonReductionCSETest(TestCase):
    @requires_gpu_and_triton
    def test_argmax_cache_key_includes_logical_index(self):
        def fn(x):
            return x.argmax(), x.t().argmax()

        x = torch.tensor(
            [[0.0, 3.0], [2.0, 1.0]],
            device=GPU_TYPE,
        )
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)

        self.assertEqual(compiled(x), fn(x))


if __name__ == "__main__":
    run_tests()
