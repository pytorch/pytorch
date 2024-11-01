# Owner(s): ["module: inductor"]

from functools import partial

import torch
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import ops
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


# These tests check issues for lowerings that aren't in the main pytorch repo
class TestCustomLowering(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_inductor_ops = torch.library.Library(  # noqa: TOR901
            "test_inductor_ops", "DEF"
        )
        cls.device_list = ["Meta", "CUDA", "XPU"]
        for device in cls.device_list:
            setattr(
                cls,
                "impl_" + device.lower(),
                torch.library.Library(  # noqa: TOR901
                    "test_inductor_ops", "IMPL", device
                ),
            )
        cls._register_jagged_to_padded_dense()
        cls._register_asm_op()

    @classmethod
    def tearDown(cls):
        super().tearDownClass()

    @classmethod
    def _register_jagged_to_padded_dense(cls):
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define(
            "jagged_to_padded_dense(Tensor input, Tensor offsets, SymInt max_seq_len, Scalar pad_value) -> Tensor"
        )

        def j2pd_meta(inp, offsets, max_seq_len, pad_value):
            return torch.empty(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                device=inp.device,
                dtype=inp.dtype,
            )

        def j2pd_gpu(inp, offsets, max_seq_len, pad_value):
            res = torch.full(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                pad_value,
                device=inp.device,
                dtype=inp.dtype,
            )
            for b in range(offsets.shape[0] - 1):
                for r in range(offsets[b + 1] - offsets[b]):
                    res[b][r] = inp[offsets[b] + r]
            return res

        def j2pd_lowering(inp, offsets, max_seq_len, pad_value):
            offsets_loader = offsets.make_loader()
            inp_loader = inp.make_loader()
            jagged_len = inp.get_size()[0]
            offsets_dtype = offsets.get_dtype()

            def inner_fn(index):
                batch_idx, seq_idx, emb_idx = index

                begin_idx = ops.indirect_indexing(
                    offsets_loader([batch_idx]),
                    jagged_len + 1,
                )
                end_idx = offsets_loader([batch_idx + 1])
                jagged_idx = begin_idx + seq_idx

                return ops.masked(
                    ops.lt(
                        ops.index_expr(jagged_idx, offsets_dtype),
                        end_idx,
                    ),
                    lambda: inp_loader([jagged_idx, emb_idx]),
                    pad_value,
                )

            return Pointwise.create(
                device=inp.get_device(),
                dtype=inp.get_dtype(),
                inner_fn=inner_fn,
                ranges=[offsets.get_size()[0] - 1, max_seq_len, inp.get_size()[1]],
            )

        register_lowering(
            torch.ops.test_inductor_ops.jagged_to_padded_dense, type_promotion_kind=None
        )(j2pd_lowering)

        cls.impl_meta.impl("jagged_to_padded_dense", j2pd_meta)
        cls.impl_cuda.impl("jagged_to_padded_dense", j2pd_gpu)
        cls.impl_xpu.impl("jagged_to_padded_dense", j2pd_gpu)

    @classmethod
    def _register_asm_op(cls):
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define("tanh_approx(Tensor input) -> Tensor")

        def tanh_approx_meta(inp):
            return torch.tanh(inp)

        cls.impl_meta.impl("tanh_approx", tanh_approx_meta)

        def tanh_approx_lowering(inp):
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(
            torch.ops.test_inductor_ops.tanh_approx, type_promotion_kind=None
        )(tanh_approx_lowering)

        cls.test_inductor_ops.define("add_custom(Tensor a, Tensor b) -> Tensor")

        def add_custom(a, b):
            return a + b

        cls.impl_meta.impl("add_custom", add_custom)

        def add_custom_lowering(a, b):
            fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
            return make_pointwise(fn)(a, b)

        register_lowering(
            torch.ops.test_inductor_ops.add_custom, type_promotion_kind=None
        )(add_custom_lowering)

    @requires_gpu()
    def test_jagged_to_padded_dense_sanity_cuda(self):
        def fn(inp, offsets, max_seq_len):
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((9, 96), device=GPU_TYPE)
        offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device=GPU_TYPE)
        max_seq_len = 4

        res = fn(inp, offsets, max_seq_len)
        self.assertEqual(inp[0], res[0][0])
        self.assertEqual(inp[1], res[0][1])
        self.assertEqual(inp[2], res[1][0])
        self.assertEqual(inp[3], res[1][1])
        self.assertEqual(inp[5], res[2][0])
        self.assertEqual(inp[8], res[2][3])

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    def test_jagged_to_padded_dense_zero_size(self):
        # Previously, the masking was being completely stripped for the
        # masked load of the input value. That would lead to an IMA
        # because cuda was trying to read index 0 of a zero-size tensor.
        def fn(inp, offsets, max_seq_len):
            inp = torch.bmm(inp, torch.ones((1, 96, 1), device=GPU_TYPE)).view((0, 1))
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((1, 0, 96), device=GPU_TYPE)
        offsets = torch.zeros(1025, device=GPU_TYPE, dtype=torch.int32)
        max_seq_len = 20

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu
    def test_tanh_approx(self):
        def fn(inp):
            return torch.ops.test_inductor_ops.tanh_approx(inp)

        inp = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        a = torch.tanh(inp)
        b = fn_opt(inp)
        self.assertEqual(a, b)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu
    def test_multi_inp_asm(self):
        def fn(a, b):
            return torch.ops.test_inductor_ops.add_custom(a, b)

        a = torch.randn(32, device=GPU_TYPE)
        b = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        out1 = a + b
        out2 = fn_opt(a, b)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
