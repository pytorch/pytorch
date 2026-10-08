# Owner(s): ["module: inductor"]

from functools import partial
from unittest import mock, skipIf

import torch
from torch._inductor import config
from torch._inductor.codegen.common import (
    BackendFeature,
    custom_backend_codegen_configs,
    custom_backend_passes,
    device_codegens,
    has_backend_feature,
    init_backend_registration,
    register_backend_for_device,
)
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import make_fallback, make_pointwise, register_lowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import ops
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)
from torch.utils._ordered_set import OrderedSet


# These tests check issues for lowerings that aren't in the main pytorch repo
class TestCustomLowering(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_inductor_ops = torch.library.Library(  # noqa: SCOPED_LIBRARY
            "test_inductor_ops", "DEF"
        )
        cls.device_list = ["Meta", "CUDA", "XPU"]
        for device in cls.device_list:
            setattr(
                cls,
                "impl_" + device.lower(),
                torch.library.Library(  # noqa: SCOPED_LIBRARY
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
            if torch.version.hip:
                # ROCm GCN assembly
                fn = partial(
                    ops.inline_asm_elementwise,
                    asm="v_add_f32 $0, $1, $2",
                    constraints="=v, v, v",
                )
            else:
                fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
            return make_pointwise(fn)(a, b)

        register_lowering(
            torch.ops.test_inductor_ops.add_custom, type_promotion_kind=None
        )(add_custom_lowering)

    def test_register_lowering_custom_dict(self):
        custom_lowering_dict = {}

        from torch._inductor.lowering import register_lowering

        @torch.library.custom_op("helion_test::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        @register_lowering(
            torch.ops.helion_test.foo, lowering_dict=custom_lowering_dict
        )
        def foo_lowering(x):
            return x

        if torch.ops.helion_test.foo not in custom_lowering_dict:
            raise AssertionError
        if torch.ops.helion_test.foo in torch._inductor.lowering.lowerings:
            raise AssertionError

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
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
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
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
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
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
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_reused_inline_asm_realized(self):
        def fn(inp):
            y = torch.ops.test_inductor_ops.tanh_approx(inp)
            return y.sum(dim=0), y.sum(dim=1)

        inp = torch.randn(32, 64, device=GPU_TYPE)
        expected = (torch.tanh(inp).sum(dim=0), torch.tanh(inp).sum(dim=1))
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), inp)

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual("\n".join(code).count("tanh.approx.f32"), 1)

    @requires_gpu()
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_multi_inp_asm(self):
        def fn(a, b):
            return torch.ops.test_inductor_ops.add_custom(a, b)

        a = torch.randn(32, device=GPU_TYPE)
        b = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        out1 = a + b
        out2 = fn_opt(a, b)
        self.assertEqual(out1, out2)

    @config.patch(joint_graph_constant_folding=False)
    def test_constant_creation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(1)

        make_fallback(torch.ops.aten.lift_fresh_copy.default)
        self.assertTrue(
            torch.allclose(torch.compile(M())(torch.ones(3)), torch.ones(3) + 1)
        )


class TestJaggedAtenLowerings(InductorTestCase):
    """Fused vs. fallback behavior of the jagged <-> padded dense lowerings."""

    def test_jagged_to_padded_dense_fused_or_fallback(self, device):
        values = torch.randn(10, 5, device=device)
        offsets = torch.tensor([0, 1, 3, 8, 10], device=device, dtype=torch.int64)
        max_length = offsets.diff().max().item()
        padding_value = 1.3

        def fn(values, offsets, max_length):
            return torch.ops.aten._jagged_to_padded_dense_forward(
                values, [offsets], [max_length], padding_value
            )

        # The fused kernel uses indirect_indexing + masked, so only backends
        # declaring BackendFeature.INDIRECT_INDEXING fuse; the rest fall back.
        expected = fn(values, offsets, max_length)
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), values, offsets, max_length
        )
        self.assertEqual(actual, expected)
        if has_backend_feature(torch.device(device), BackendFeature.INDIRECT_INDEXING):
            self.assertNotIn(
                "torch.ops.aten._jagged_to_padded_dense_forward", "".join(code)
            )
        else:
            self.assertIn(
                "torch.ops.aten._jagged_to_padded_dense_forward", "".join(code)
            )

    def test_padded_dense_to_jagged_fused_or_fallback(self, device):
        values = torch.randn(10, 5, device=device)
        offsets = torch.tensor([0, 1, 3, 8, 10], device=device, dtype=torch.int64)
        max_length = offsets.diff().max().item()
        total_L = values.shape[0]
        padded = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [max_length], 1.3
        )

        def fn(padded, offsets, total_L):
            return torch.ops.aten._padded_dense_to_jagged_forward(
                padded, [offsets], total_L
            )

        # get_inverse_offsets emits ops.bucketize and the fused gather uses
        # indirect_indexing, so only backends declaring both BUCKETIZE and
        # INDIRECT_INDEXING take the fused path; the rest fall back.
        expected = fn(padded, offsets, total_L)
        actual, code = run_and_get_code(
            torch.compile(fn, fullgraph=True), padded, offsets, total_L
        )
        self.assertEqual(actual, expected)
        has_bucketize = has_backend_feature(
            torch.device(device), BackendFeature.BUCKETIZE
        )
        has_indirect_indexing = has_backend_feature(
            torch.device(device), BackendFeature.INDIRECT_INDEXING
        )
        fused = has_bucketize and has_indirect_indexing
        if fused:
            self.assertNotIn(
                "torch.ops.aten._padded_dense_to_jagged_forward", "".join(code)
            )
        else:
            self.assertIn(
                "torch.ops.aten._padded_dense_to_jagged_forward", "".join(code)
            )


class TestJaggedOutOfTreeBackend(InductorTestCase):
    """An out-of-tree backend opts into (or out of) the fused jagged lowerings
    by declaring the feature on its registered scheduling."""

    def test_lowering_falls_back_when_scheduling_drops_feature(self):
        # Same op fuses on the default CPU scheduling (which declares
        # INDIRECT_INDEXING); registering a scheduling without it must make
        # the lowering pick the ATen fallback. This is the mechanism an
        # out-of-tree backend controls via register_backend_for_device.
        init_backend_registration()
        orig = device_codegens["cpu"]

        class NoIndirectIndexing(CppScheduling):
            backend_features = OrderedSet(
                [
                    BackendFeature.INPLACE_BUFFERS,
                    BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
                ]
            )

        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 1, 3, 8, 10], dtype=torch.int64)
        max_length = offsets.diff().max().item()

        def fn(values, offsets, max_length):
            return torch.ops.aten._jagged_to_padded_dense_forward(
                values, [offsets], [max_length], 1.3
            )

        with (
            mock.patch.dict(device_codegens),
            mock.patch.dict(custom_backend_codegen_configs),
            mock.patch.dict(custom_backend_passes),
        ):
            register_backend_for_device(
                "cpu",
                NoIndirectIndexing,
                orig.wrapper_codegen,
                orig.cpp_wrapper_codegen,
                orig.fx_wrapper_codegen,
            )
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True), values, offsets, max_length
            )
        self.assertEqual(actual, fn(values, offsets, max_length))
        self.assertIn("torch.ops.aten._jagged_to_padded_dense_forward", "".join(code))


instantiate_device_type_tests(
    TestJaggedAtenLowerings,
    globals(),
    only_for=("cpu", "cuda", "xpu"),
    allow_xpu=True,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
