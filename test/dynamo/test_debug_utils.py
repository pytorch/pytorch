# Owner(s): ["module: dynamo"]

import unittest

from contextlib import contextmanager

import torch

import torch._prims_common as utils
import torch.nn.functional as F

from functorch import make_fx
from torch import nn
from torch._decomp import decomposition_table
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import aot_graph_input_parser
from torch._dynamo.test_case import TestCase
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch.testing._internal.inductor_utils import HAS_CUDA

aten = torch.ops.aten

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestDebugUtils(TestCase):
    def test_cast_model_to_fp64_dtype_args(self):
        # Test that dtype arguments are converted to fp64

        def fn(x):
            return (
                torch.ops.prims.convert_element_type(x, torch.float16),
                x.to(torch.float16),
                torch.full(x.shape, 2, dtype=torch.float32, device=x.device),
                x.new_empty(x.shape),
            )

        x = torch.randn(32, device="cpu")
        decomps = torch._decomp.core_aten_decompositions()
        fx = make_fx(fn, decomposition_table=decomps)(x)

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

        fp64_model, fp64_examples = debug_utils.cast_to_fp64(fx, (x,))
        self.assertEqual(fp64_examples, (x.to(torch.float64),))

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float64)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float64);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,  # NOQA: B950
        )

    @requires_cuda
    def test_aot_graph_parser(self):
        from torch import device

        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[1001]",
            primals_3: "f32[1001, 64]",
            primals_4: "f32[4190]",
            primals_5: "f32[4190]",
            primals_6: "f32[1739, 4190]",
            primals_48: "f32[6144, 4191]",
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0
            lift_fresh_copy: "i64[4190]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant0
            )
            _tensor_constant0 = None
            index: "f32[6144, 4190]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy]
            )
            lift_fresh_copy = None

            _tensor_constant1: "i64[6]" = self._tensor_constant1
            lift_fresh_copy_1: "i64[6]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant1
            )
            _tensor_constant1 = None
            index_1: "f32[6144, 6]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy_1]
            )
            primals_48 = lift_fresh_copy_1 = None
            permute: "f32[6, 1001]" = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm: "f32[6144, 1001]" = torch.ops.aten.addmm.default(
                primals_2, index_1, permute
            )
            primals_2 = permute = None
            amax: "f32[6144, 1]" = torch.ops.aten.amax.default(addmm, [-1], True)
            sub: "f32[6144, 1001]" = torch.ops.aten.sub.Tensor(addmm, amax)
            exp: "f32[6144, 1001]" = torch.ops.aten.exp.default(sub)
            sub = None
            sum_1: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
            div: "f32[6144, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1)
            exp = None

            full_default: "i32[6144, 1001]" = torch.ops.aten.full.default(
                [6144, 1001],
                1,
                dtype=torch.int32,
                layout=torch.strided,
                device=device(type="cuda", index=0),
                pin_memory=False,
            )

            iota: "i32[1001]" = torch.ops.prims.iota.default(
                1001,
                start=0,
                step=1,
                dtype=torch.int32,
                device=device(type="cuda"),
                requires_grad=False,
            )

            mul: "i32[6144, 1001]" = torch.ops.aten.mul.Tensor(full_default, iota)
            full_default = iota = None

            iota_1: "i32[6144]" = torch.ops.prims.iota.default(
                6144,
                start=0,
                step=1001,
                dtype=torch.int32,
                device=device(type="cuda", index=0),
                requires_grad=False,
            )
            view: "i32[6150144]" = torch.ops.aten.reshape.default(mul, [-1])
            mul = None
            view_1: "f32[6150144]" = torch.ops.aten.reshape.default(div, [-1])
            div = None
            _embedding_bag = torch.ops.aten._embedding_bag.default(
                primals_3, view, iota_1, False, 0, False, view_1
            )

            return _embedding_bag

        kwargs = aot_graph_input_parser(forward, device="cuda")
        # runs successfully
        forward(**kwargs)

    @requires_cuda
    def test_sym_aot_graph_parser(self):
        def forward(
            self,
            primals_1: "f32[1001, 6]",  # noqa: F821
            primals_2: "f32[s0]",  # noqa: F821
            primals_3: "Sym(s0)",  # noqa: F821,
            primals_4: "f32[s1]",  # noqa: F821,
            primals_5: "Sym(s1)",  # noqa: F821,
        ):
            _tensor_constant0: "i64[4190]" = self._tensor_constant0

        kwargs = aot_graph_input_parser(
            forward, device="cuda", sym_shapes={"s0": 10}, default_sym_shape=5
        )

        self.assertEqual(list(kwargs["primals_2"].shape), [10])
        self.assertEqual(kwargs["primals_3"], 10)

        self.assertEqual(list(kwargs["primals_4"].shape), [5])
        self.assertEqual(kwargs["primals_5"], 5)

    def test_compiler_bisector(self):
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
            return -1 / rate * torch.log1p(-torch.rand_like(self))

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

        from torch._inductor.bisect_helper import BisectionManager

        out = BisectionManager.do_bisect(test_fn)
        self.assertEqual(out, (["aot_eager_decomp_partition", "decomposition"], 4))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
