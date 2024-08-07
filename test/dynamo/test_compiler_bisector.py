# Owner(s): ["module: dynamo"]

import unittest
from contextlib import contextmanager

import torch
import torch._prims_common as utils
import torch.nn.functional as F
from torch import nn
from torch._decomp import decomposition_table
from torch._dynamo.test_case import TestCase
from torch._inductor import config
from torch._inductor.bisect_helper import BisectionManager
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch.testing._internal.inductor_utils import HAS_CUDA


aten = torch.ops.aten

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


@requires_cuda
class TestCompilerBisector(TestCase):
    def test_bad_decomp(self):
        @out_wrapper()
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("self",),
            type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )
        def bad_exp_decomp(self, rate=1, generator=None):
            assert generator is None
            torch._check(
                not utils.is_complex_dtype(self.dtype)
                and not utils.is_integer_dtype(self.dtype)
                and not utils.is_boolean_dtype(self.dtype),
                lambda: f"Exponential distribution is a continuous probability distribution. \
                dtype must be a floating point but you specified {self.dtype}",
            )
            torch._check(
                rate > 0.0,
                lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
            )
            # on most hardwares dont need to subtract 1, but rng is device dependent,
            # so make it always bad for determinism
            return ((-1) / rate * torch.log1p(-torch.rand_like(self))) / 0.0

        @contextmanager
        def patch_exp_decomp():
            curr_decomp = decomposition_table[aten.exponential.default]
            decomposition_table[aten.exponential.default] = bad_exp_decomp
            try:
                yield

            finally:
                decomposition_table[aten.exponential.default] = curr_decomp

        class GumbelVectorQuantizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_groups = 32
                self.num_vars = 320

                self.weight_proj = nn.Linear(256, self.num_groups * self.num_vars)
                self.temperature = 2

                self.weight_proj.weight.data.normal_(mean=0.0, std=1)
                self.weight_proj.bias.data.zero_()

            def forward(self, hidden_states: torch.Tensor):
                batch_size, sequence_length, hidden_size = hidden_states.shape

                hidden_states = self.weight_proj(hidden_states)
                hidden_states = hidden_states.view(
                    batch_size * sequence_length * self.num_groups, -1
                )

                codevector_probs = F.gumbel_softmax(
                    hidden_states.float(), tau=self.temperature, hard=True
                ).type_as(hidden_states)
                return codevector_probs

        def test_fn():
            torch._dynamo.reset()
            with patch_exp_decomp():
                vq = GumbelVectorQuantizer().cuda()
                vq_compiled = torch.compile(vq)

                x = torch.randn(4, 400, 256).cuda()
                seed = torch.tensor(
                    [131, 86, 34, 149, 41, 131, 17, 0, 76, 0, 0, 0, 0, 0, 0, 0],
                    dtype=torch.uint8,
                )
                s = torch.cuda.set_rng_state(seed)
                with torch._dynamo.utils.preserve_rng_state():
                    out = vq(x)
                out_compiled = vq_compiled(x)

            return not out_compiled.isnan().any()

        out = BisectionManager.do_bisect(test_fn)
        self.assertEqual(out.backend, "aot_eager_decomp_partition")
        self.assertEqual(out.subsystem, "decomposition")
        self.assertEqual(out.bisect_number, 4)
        self.assertTrue("aten.exponential" in out.debug_info)

    def test_bad_lowering(self):
        def test_fn():
            torch._dynamo.reset()
            with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):

                def my_func(x):
                    return (x * -1).relu()

                inp = torch.rand([100], device="cuda")

                return torch.allclose(torch.compile(my_func)(inp), my_func(inp))

        out = BisectionManager.do_bisect(test_fn)
        self.assertEqual(out.backend, "inductor")
        self.assertEqual(out.subsystem, "lowerings")
        self.assertEqual(out.bisect_number, 1)
        self.assertTrue("relu" in out.debug_info)

    def test_eager_backend(self):
        # should indicate problem with first backend
        def test_fn():
            return False

        out = BisectionManager.do_bisect(test_fn)
        self.assertEqual(out.backend, "eager")
        self.assertEqual(out.subsystem, None)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
