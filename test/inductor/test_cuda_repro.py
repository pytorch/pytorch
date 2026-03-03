# Owner(s): ["module: inductor"]
# ruff: noqa: F841

import copy
import functools
import gc
import math
import os
import sys
import unittest

import torch
import torch._dynamo.config as dynamo_config
import torch.backends.cuda
import torch.nn.functional as F
from torch import nn
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import (
    run_and_get_code,
    run_and_get_graph_lowering,
    run_fw_bw_and_get_code,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM80OrLater,
    SM90OrLater,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    MI350_ARCH,
    parametrize,
    skipIfRocm,
    skipIfRocmArch,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    xfailIfROCm,
)
from torch.testing._internal.inductor_utils import IS_BIG_GPU


if TEST_WITH_ROCM:
    config.force_layout_optimization = 1
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import skipCUDAIf


try:
    try:
        import triton  # @manual
        from triton import language as tl  # @manual
    except ImportError:
        raise unittest.SkipTest("requires triton")  # noqa: B904

    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase
ToTuple = test_torchinductor.ToTuple
check_model_cuda = test_torchinductor.check_model_cuda
aten = torch.ops.aten


@instantiate_parametrized_tests
class CudaReproTests(TestCase):
    device = "cuda"
    common = check_model_cuda

    def test_mm_out_dtype_compile(self):
        a = torch.randn(1, 3, device="cuda", dtype=torch.float16)
        b = torch.randn(3, 2, device="cuda", dtype=torch.float16)

        def fn(x, y):
            return torch.mm(x, y, out_dtype=torch.float32)

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        result = compiled(a, b)
        expected = fn(a, b)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result, expected)

    def test_index_put_issue(self):
        def forward(
            self,
            arg76_1,
            expand_default,
            full_like_default,
            _to_copy_default_67,
            zeros,
        ):
            sum_sym_int_19 = torch.ops.aten.sum(_to_copy_default_67, [0], True)
            view_default_57 = torch.ops.aten.view.default(sum_sym_int_19, [512, 768])
            where_self = torch.ops.aten.where.self(
                expand_default, view_default_57, full_like_default
            )
            clone_default_12 = torch.ops.aten.clone.default(zeros)
            index_put__default = torch.ops.aten.index_put_.default(
                clone_default_12, [arg76_1], where_self, True
            )
            return (index_put__default,)

        inps = [
            (torch.Size([512]), torch.int64),
            (torch.Size([512, 768]), torch.bool),
            (torch.Size([512, 768]), torch.float16),
            (torch.Size([4, 512, 768]), torch.float16),
            (torch.Size([512, 768]), torch.float16),
        ]
        inps = [torch.zeros(())] + [
            torch.ones(shape, dtype=dtype, device="cuda") for (shape, dtype) in inps
        ]
        mod = make_fx(forward)(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    def test_view_replay_padding_issue_163328(self):
        class ReproModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_points_out = 120
                self.lc_num = 2
                input_channels = 16
                self.linear_main = nn.Linear(input_channels, self.num_points_out * 2)
                self.linear_lc = nn.Linear(input_channels, self.num_points_out * 2)

            def forward(self, x: torch.Tensor):
                bs, num_lat, num_lon, channels = x.shape
                index = num_lat - self.lc_num

                main_x = x[:, :index].reshape(bs * index * num_lon, channels)
                lc_x = x[:, index:].reshape(bs * self.lc_num * num_lon, channels)

                refline = self.linear_main(main_x).reshape(bs, index, num_lon, -1)
                lc_refline = self.linear_lc(lc_x).reshape(bs, self.lc_num, num_lon, -1)

                base = torch.cat([refline, lc_refline], dim=1).contiguous()
                out0 = base.reshape(bs, num_lat, num_lon, self.num_points_out, 2)
                out1 = base.reshape(bs, num_lat * num_lon, self.num_points_out * 2)
                return {"ten0": out0, "ten1": out1}

        torch.manual_seed(0)
        model = ReproModule().cuda()
        inputs = torch.randn(36, 9, 7, 16, device="cuda", requires_grad=True)

        eager_out = model(inputs)
        compiled_model = torch.compile(
            copy.deepcopy(model),
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=True,
        )
        compiled_out = compiled_model(inputs)

        self.assertEqual(compiled_out["ten0"], eager_out["ten0"])
        self.assertEqual(compiled_out["ten1"], eager_out["ten1"])

    def test_effn_attn_bias_padding(self):
        batch_size, num_heads, seq_len, head_dim = 2, 32, 512, 128

        def fn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            input_tensor: torch.Tensor,  # This will be our starting point
        ):
            # Input tensor should be [2, 1, 8192, 1] with appropriate strides
            bias = torch.ops.aten.expand(
                input_tensor, [2, 32, seq_len, seq_len]
            )  # Expands with stride pattern [65536, 0, 8, 0]

            return torch.ops.aten._scaled_dot_product_efficient_attention(
                query,
                key,
                value,
                bias,
                compute_log_sumexp=True,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

        input_tensor = torch.rand([2, 1, seq_len, 1], device="cuda")

        out, code = run_and_get_code(torch.compile(fn), query, key, value, input_tensor)

        input_tensor2 = torch.rand([2, 32, seq_len, seq_len], device="cuda").copy_(
            input_tensor
        )
        # even though the last dim is broadcasted, needs stride 1 for alignment
        # but dim 1 stride can be 0
        FileCheck().check("buf0").check("(262144, 0, 512, 1").run(code[0])

        # dont check rng state
        self.assertEqual(out[:2], fn(query, key, value, input_tensor2)[:2])

    # Fails on ROCm MI350
    # Mismatched elements: 23 / 33062912 (0.0%)
    # Greatest absolute difference: 0.07861328125 at index (14, 13, 1008, 36) (up to 1e-05 allowed)
    # Greatest relative difference: 2.90625 at index (14, 13, 1008, 36) (up to 0.016 allowed)
    @skipIfRocmArch(MI350_ARCH)
    def test_effn_attn_bias_padding_misaligned(self):
        seqlen_start = 1008

        for offset in range(-1, 2):
            seqlen = seqlen_start + offset
            torch._dynamo.reset()

            bsz = 32
            q = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16, device="cuda")
            mask = torch.ones([bsz, 1, seqlen, seqlen], dtype=torch.bool, device="cuda")
            inputs = [q, k, v, mask]

            def f(q, k, v, mask):
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, dropout_p=0.0
                    )

            f_compiled = torch.compile(f)

            out, code = run_and_get_code(f_compiled, *inputs)
            # padded bias should have an expanded dim
            FileCheck().check("buf0 =").check_same(", 0, ").run(code[0])
            # single fused padded kernel
            FileCheck().check_count("empty_strided_cuda(", 1, exactly=True).check(
                "return"
            ).run(code[0])

            self.assertEqual(out, f(*inputs))

    def test_input_channels_last(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
            ToTuple(),
        ).cuda()
        inp = torch.randn([2, 3, 16, 16]).to(memory_format=torch.channels_last).cuda()

        self.common(
            m,
            (inp,),
            check_lowp=False,
        )

        @torch.compile()
        def foo(m, inp):
            return m(inp)

        self.assertTrue(foo(m, inp)[0].is_contiguous(memory_format=torch.channels_last))

    # https://github.com/pytorch/torchdynamo/issues/1681#issuecomment-1283433527
    def test_unspec_inputs_interop(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                unsqueeze = torch.ops.aten.unsqueeze.default(x, 4)
                permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3])
                add = torch.ops.aten.add.Tensor(y, 1)
                return [permute, add]

        inps = [
            rand_strided((12, 3, 512, 64), (64, 196608, 768, 1), torch.float32, "cuda"),
            rand_strided((), (), torch.int64, "cpu"),
        ]
        mod = make_fx(Repro().to(device="cuda"))(*inps)
        compiled = compile_fx_inner(mod, inps)
        compiled(inps)

    @unittest.skipIf(
        IS_FBCODE, "RuntimeError: Triton Error [CUDA]: invalid device context"
    )
    def test_backward_context(self):
        def fn(x):
            return x * 3

        x = torch.randn(4, device="cuda", requires_grad=True)
        gO = torch.rand_like(x)
        opt_fn = torch.compile(fn)
        out = opt_fn(x)
        out.backward(gO)

    @config.patch(fallback_random=True)
    def test_dtype_factory_issue(self):
        def forward():
            randn = torch.ops.aten.randn.default(
                [12, 64, 1, 64],
                dtype=torch.float32,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(randn, -1)
            return (unsqueeze_default_2,)

        mod = make_fx(forward)()
        compiled = compile_fx_inner(mod, ())
        if compiled([])[0].device.type != "cuda":
            raise AssertionError(
                f"Expected device type 'cuda', got {compiled([])[0].device.type!r}"
            )

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_no_device_idx_repro_cudagraphs(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self):
                full = torch.ops.aten.full.default(
                    [8, 512],
                    1,
                    dtype=torch.float32,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                full_1 = torch.ops.aten.full.default(
                    [8, 512],
                    0,
                    dtype=torch.int64,
                    layout=torch.strided,
                    device=torch.device(type="cuda", index=0),
                    pin_memory=False,
                )
                return (full_1, full)

        self.common(Repro(), ())

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(
        automatic_dynamic_shapes=True,
        assume_static_by_default=False,
    )
    def test_dynamic_to_static_cudagraphs(self):
        for b in [False, True]:
            with config.patch({"triton.cudagraph_trees": b}):

                @torch.compile(backend="inductor")
                def fn(x, y):
                    r = x + y
                    return r, r.size(0)

                inputs = (
                    torch.randn((5, 5), device="cuda"),
                    torch.randn((5, 5), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 5)))

                inputs = (
                    torch.randn((6, 6), device="cuda"),
                    torch.randn((6, 6), device="cuda"),
                )
                self.assertTrue(same(fn(*inputs), (inputs[0] + inputs[1], 6)))

    def _test_split_reduction_impl(self, x):
        def max(x):
            return torch.max(x)

        max_c = torch.compile(max)

        out, code = run_and_get_code(max_c, x)
        self.assertEqual(out, max(x))

        if DO_PERF_TEST:
            ms_c = benchmarker.benchmark_gpu(lambda: max_c(x))
            ms_eager = benchmarker.benchmark_gpu(lambda: max(x))
            print(f"compile {ms_c=:.03f}, eager {ms_eager=:.03f}")

    def test_split_reduction_transposed(self):
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
        x = x.t().contiguous().t()

        self._test_split_reduction_impl(x)

    def test_split_reduction_channels_last(self):
        x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
        x = x.reshape([256, 256, 256, 2]).to(memory_format=torch.channels_last)

        self._test_split_reduction_impl(x)

    @config.patch({"emulate_precision_casts": True})
    def test_bool_emulate_low_precision(self):
        from torch import device

        inf = float("inf")

        def forward():
            full_1 = torch.ops.aten.full.default(
                [6, 6],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=device(type="cpu"),
                pin_memory=False,
            )
            device_put_3 = torch.ops.prims.device_put.default(
                full_1, device(type="cuda", index=0)
            )
            full_1 = None

            convert_element_type_40 = torch.ops.prims.convert_element_type.default(
                device_put_3, torch.bool
            )
            device_put_3 = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_40, 1)
            convert_element_type_40 = None
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            unsqueeze_4 = None
            expand = torch.ops.aten.expand.default(unsqueeze_5, [-1, 256, -1, 256])
            unsqueeze_5 = None
            clone = torch.ops.aten.clone.default(
                expand, memory_format=torch.contiguous_format
            )
            expand = None
            view_15 = torch.ops.aten.reshape.default(clone, [1536, 1536])
            clone = None
            scalar_tensor = torch.ops.aten.scalar_tensor.default(
                -inf, dtype=torch.float16, device=device(type="cuda", index=0)
            )
            scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
                0.0,
                dtype=torch.float16,
                layout=torch.strided,
                device=device(type="cuda", index=0),
            )
            where = torch.ops.aten.where.self(view_15, scalar_tensor_1, scalar_tensor)
            view_15 = scalar_tensor_1 = scalar_tensor = None
            return where

        from torch._inductor import config

        config.emulate_precision_casts = True
        self.assertEqual(torch.compile(forward)(), forward())

    @config.patch({"emulate_precision_casts": True})
    def test_emulate_low_precision(self):
        def foo(x):
            return torch.nn.functional.gelu(x) * 10.0

        inp = torch.rand([32], device="cuda", requires_grad=True, dtype=torch.bfloat16)
        out, codes = run_fw_bw_and_get_code(lambda: torch.compile(foo)(inp))

        # fwd, backward
        for code in codes:
            f = FileCheck()
            # in eager, there are two down casts
            for _ in range(2):
                f.check(".to(tl.bfloat16)").check_next(".to(tl.float32)")
            f.run(code)

        self.assertEqual(foo(inp), out)

    # TODO: Abstract this out, test more extensively
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic_shapes(self):
        torch._dynamo.reset()  # Needed since everywhere else uses "inductor"

        def f(x):
            return x.cos().view(x.shape).sin()

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        f2 = torch.compile(f, backend=cnts)

        f2(torch.randn(32))

        inp = torch.randn(16)
        real_out = f(inp)
        compiled_out = f2(inp)

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(real_out, compiled_out)
        torch._dynamo.reset()

    @config.patch({"triton.cudagraphs": True, "size_asserts": False})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_expanded_inputs_cudagraphs_no_size_asserts(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        inputs = (
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
            rand_strided((5, 5, 5, 5), (0, 5, 0, 1), device="cuda"),
        )
        self.assertTrue(same(fn(*inputs), inputs[0] + inputs[1]))

    @config.patch({"triton.cudagraph_trees": False})
    @config.patch({"triton.cudagraphs": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_inplace_updates_cudagraphs(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 20, requires_grad=True)
                )

            def forward(self, x):
                x = torch.matmul(x, self.weight1)
                return x

        from copy import deepcopy

        model = Repro().cuda()
        model_ref = deepcopy(model)
        model_opt = torch.compile(model, backend="inductor")

        input = torch.randn(10, 10, device="cuda", requires_grad=True)

        for _ in range(2):
            output_ref = model_ref(input)
            output_res = model_opt(input)
            output_ref.sum().backward()
            output_res.sum().backward()
            for p_ref, p_res in zip(model_ref.parameters(), model_opt.parameters()):
                self.assertEqual(p_ref.grad, p_res.grad)
            with torch.no_grad():
                for param in model_ref.parameters():
                    param.add_(1.0)
                for param in model_opt.parameters():
                    param.add_(1.0)

    # https://github.com/pytorch/torchdynamo/issues/1850
    def test_inductor_output_aliases_intermediate(self):
        def foo(x):
            out = x + x
            return out.t()

        foo_opt = torch.compile(foo, backend="inductor")

        inpt = torch.randn(10, 10, device="cuda", requires_grad=True)
        # TODO: this is broken, fix later
        # out = foo_opt(inpt)
        # out.add_(2)

        out_ref = foo(inpt)
        out_ref.add_(2)
        # self.assertEqual(out_ref, out)

    def test_accuracy_issue1(self):
        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=768, out_features=2, bias=True
                )

            def forward(self, start_positions: torch.Tensor, x: torch.Tensor):
                linear = self.linear(x)
                split = linear.split(1, dim=-1)
                getitem = split[0]
                squeeze = getitem.squeeze(-1)
                clamp = start_positions.clamp(0, 128)
                cross_entropy = torch.nn.functional.cross_entropy(
                    squeeze, clamp, None, None, 128, None, "mean", 0.0
                )
                return cross_entropy

        mod = Repro().cuda()
        opt_mod = torch.compile(mod, backend="inductor")
        mod.eval()
        opt_mod.eval()

        args = [
            ((1,), (1,), torch.int64, "cuda", False),
            ((1, 128, 768), (98304, 768, 1), torch.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        with torch.cuda.amp.autocast(enabled=False):
            if not same_two_models(mod, opt_mod, args):
                raise AssertionError("Dynamo failed")

    @config.patch(allow_buffer_reuse=False)
    def test_issue103461(self):
        def forward(add_1):
            var_mean = torch.ops.aten.var_mean.correction(
                add_1, [2], correction=0, keepdim=True
            )
            getitem_1 = var_mean[1]
            return getitem_1

        x = torch.randn(1, 8, 768, device="cuda")
        correct = forward(x)
        actual = torch.compile(forward, fullgraph=True)(x)
        self.assertEqual(actual, correct)

    def test_full_copy(self):
        def forward(x):
            full_10 = torch.ops.aten.full.default(
                [204, 204, 28],
                0,
                dtype=torch.float64,
                layout=torch.strided,
                device="cuda",
                pin_memory=False,
            )
            return x + full_10.to("cpu")

        o = torch.randn([204, 204, 28], dtype=torch.float64)
        correct = forward(o)
        actual = torch.compile(forward, fullgraph=True)(o)
        self.assertEqual(actual, correct)

    def test_autotune_inplace_kernel(self):
        """
        This UT tests autotune on an inplace kernel. The autotune should not contaminate
        the input buffers when tuning with multiple configs. For more details, refer to
        https://github.com/triton-lang/triton/issues/781
        https://github.com/pytorch/torchdynamo/issues/1670
        """
        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
        from torch._inductor.runtime.hints import AttrsDescriptorWrapper, HeuristicType
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.utils import triton_version_uses_attrs_dict

        def autotune(configs, meta):
            def decorator(fn):
                if triton_version_uses_attrs_dict():
                    # Newer versions of Triton puts constexpr in signature
                    # Ref: https://github.com/pytorch/pytorch/pull/145051
                    meta["signature"]["XBLOCK"] = "constexpr"

                return CachingAutotuner(
                    # force autotune by setting save_cache_hook to False
                    fn,
                    triton_meta=meta,
                    configs=configs,
                    save_cache_hook=False,
                    mutated_arg_names=["in_out_ptr0"],
                    reset_to_zero_arg_names=[],
                    optimize_mem=True,
                    heuristic_type=HeuristicType.POINTWISE,
                    inductor_meta={"grid_type": "Grid1D"},
                )

            return decorator

        @autotune(
            configs=[
                triton.Config({"XBLOCK": 1}),
                triton.Config({"XBLOCK": 2}),
            ],
            meta={
                "signature": {
                    "in_out_ptr0": "*fp32",
                    "in_ptr0": "*fp32",
                    "xnumel": "i32",
                },
                "device": DeviceProperties.create(torch.device("cuda")),
                "configs": [
                    AttrsDescriptorWrapper(divisible_by_16=(0, 1), equal_to_1=())
                ],
                "constants": {},
            },
        )
        @triton.jit
        def kernel(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * XBLOCK
            offsets = block_start + tl.arange(0, XBLOCK)
            mask = offsets < xnumel
            x = tl.load(in_out_ptr0 + offsets, mask=mask, other=0.0)
            y = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
            output = x + y
            tl.store(in_out_ptr0 + offsets, output, mask=mask)

        xnumel = 384
        in0 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
        inout1 = rand_strided((xnumel,), (1,), device="cuda", dtype=torch.float32)
        inout2 = inout1.clone()

        stream0 = get_cuda_stream(0)
        kernel.run(inout1, in0, xnumel, stream=stream0)
        kernel.run(inout2, in0, xnumel, stream=stream0)

        if not same(inout1, inout2, tol=0.001, equal_nan=True):
            raise AssertionError("failed autotune with inplace kernel")

    def test_sort_stride_issue(self):
        # This minified testcase comes from detectron2_maskrcnn_r_50_fpn
        # There was a false error from our size_assert code
        @torch.compile(fullgraph=True)
        def forward(pred_objectness_logits_3_: torch.Tensor):
            sort_3 = pred_objectness_logits_3_.sort(descending=True, dim=1)
            getitem_12 = sort_3[0]
            return getitem_12

        args = [((1, 100), (0, 1), torch.float16, "cuda", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        result = forward(*args)
        if not same(result, torch.sort(args[0], descending=True, dim=1)[0]):
            raise AssertionError

    def test_scalar_triton_index(self):
        # The indirect indexing via a scalar like below used to lead to
        # bad triton code that made triton segfault when compiling.
        # See https://github.com/pytorch/torchdynamo/issues/1515
        def fn(a):
            zero = torch.zeros((16,), device=a.device, dtype=torch.int64)
            return (a[zero],)

        a = torch.randn((8,), dtype=torch.float32, device="cuda")

        fn_optimized = torch.compile(fn, backend="inductor")
        if not same(fn(a), fn_optimized(a)):
            raise AssertionError

    def test_indirect_indexing_dense_mask(self):
        def fn(x, y):
            ne = torch.ops.aten.ne.Scalar(x, 1)
            sum_1 = torch.ops.aten.sum.dim_IntList(ne, [1])
            sub = torch.ops.aten.sub.Tensor(sum_1, 1)
            unsqueeze = torch.ops.aten.unsqueeze.default(sub, -1)
            gather = torch.ops.aten.gather.default(x, 1, unsqueeze)
            squeeze = torch.ops.aten.squeeze.default(gather)
            out = torch.ops.aten.multiply(y, squeeze)
            return (out,)

        a = torch.zeros((1, 128), dtype=torch.int64, device="cuda")
        b = torch.zeros((1, 128), dtype=torch.int64, device="cuda")

        fn_optimized = torch.compile(fn, backend="inductor")
        if not same(fn(a, b), fn_optimized(a, b)):
            raise AssertionError

    def test_simplify_dims(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (torch.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],))

    @config.patch(permute_fusion=True)
    def test_permute_fusion(self):
        class Repro(torch.nn.Module):
            def forward(self, view, reshape_2):
                permute = view.permute(0, 2, 1)
                view = None
                reshape = torch.reshape(permute, (-1, 642))
                bmm = torch.bmm(permute, reshape_2)
                return (bmm,)

        args = [
            ((1024, 642, 160), (102720, 160, 1), torch.float32, "cuda", True),
            ((1024, 642, 20), (12840, 20, 1), torch.float32, "cuda", True),
        ]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        mod = Repro()
        opt_mod = torch.compile(mod, backend="inductor")

        ref = mod(*args)
        res = opt_mod(*args)
        self.assertTrue(same(ref, res))

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_add_alpha_autotune(self):
        def fn(x, y):
            aten.add_.Tensor(x, y, alpha=0.55)
            return (x,)

        x1 = torch.zeros(2, 3, 4, 10, device="cuda")
        x2 = torch.zeros(2, 3, 4, 10, device="cuda")
        x3 = torch.zeros(2, 3, 4, 10, device="cuda")
        y = torch.randn(2, 3, 4, 10, device="cuda").to(
            memory_format=torch.channels_last
        )
        fn_fx = make_fx(fn)(x1, y)
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        fn(x2, y)
        fn_compiled([x3, y])
        if not same(x2, x3):
            raise AssertionError

    @config.patch({"triton.autotune_pointwise": True})
    def test_inplace_buffer_autotune(self):
        def foo(x, y, z):
            a = x @ y
            return a.unsqueeze(0).unsqueeze(0) + z

        x = torch.zeros(5, 5, device="cuda")
        y = torch.zeros(5, 5, device="cuda")
        z = torch.zeros(1, 1, 5, 5, device="cuda").to(memory_format=torch.channels_last)
        self.common(
            foo,
            (x, y, z),
            check_lowp=False,
        )

    def test_memory_history_inductor(self):
        def called_inside_compile(x, w, b):
            a = x @ w + b
            return torch.sigmoid(a)

        @torch.compile
        def fn(x, w, b):
            x = called_inside_compile(x, w, b)
            return called_inside_compile(x, w, b)

        w = torch.rand(3, 3, device="cuda")
        b = torch.rand(3, device="cuda")
        x = torch.rand(3, device="cuda")
        try:
            torch.cuda.memory.empty_cache()
            torch.cuda.memory._record_memory_history(True)
            r = fn(x, w, b)
        finally:
            torch.cuda.memory._record_memory_history(False)
        snapshot = str(torch.cuda.memory._snapshot())
        self.assertTrue("called_inside_compile" in snapshot)

    def test_negative_arange_dynamic_shapes(self):
        # Repro from alibi relative encodings
        def sign(x):
            return (x > 0) - (x < 0)

        class Repro(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                nheads = 16
                start = math.log2(0.5)
                end = math.log2(1 / (2**8))

                self.scales = nn.Buffer(
                    2
                    ** torch.arange(
                        start,
                        end + 1e-6 * sign(end - start),
                        (end - start) / (nheads - 1),
                    ).view(1, nheads, 1, 1),
                )
                self.emb = nn.Embedding(1024, 256)
                self.dec_layer = nn.TransformerDecoderLayer(
                    256, 16, 512, batch_first=True, norm_first=True
                )
                self.head = nn.Linear(256, 1024)

            def forward(self, enc_out: torch.Tensor, dec_in: torch.Tensor):
                padmask = dec_in == 0
                dec_mask = padmask.unsqueeze(-1) == padmask.unsqueeze(-2)
                dec_mask = dec_mask.to(dtype=torch.float32)
                dec_mask = dec_mask.tril(diagonal=0).cuda()

                q_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                k_pos = torch.arange(dec_in.size(1), dtype=torch.long, device="cuda")
                rel_pos = k_pos[None, :] - q_pos[:, None]
                values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)
                dec_bias = values * self.scales
                dec_bias.tril_(diagonal=0)

                dec_mask = dec_mask + dec_bias[0]
                out = self.emb(dec_in)
                out = self.dec_layer(out, enc_out, tgt_mask=dec_mask)
                return self.head(out)

        mod = Repro().cuda()
        opt_mod = torch.compile(mod, backend="inductor", dynamic=True)
        mod.eval()
        opt_mod.eval()

        enc_out = torch.rand(1, 512, 256).cuda()
        dec_inputs = [
            torch.randint(0, 512, (1, i + 1), dtype=torch.long).cuda() for i in range(8)
        ]

        for dec_inp in dec_inputs:
            if not same_two_models(mod, opt_mod, [enc_out, dec_inp], only_fwd=True):
                raise AssertionError("Inductor with dynamic shapes failed")

    def test_issue97695_1input(self):
        def fn(arg3_1, relu, permute_1):
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        if not same(ref, correct):
            raise AssertionError

        ref = torch.compile(fn, fullgraph=True)(*args)
        if not same(ref, correct):
            raise AssertionError

    def test_issue_103924(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temperature = 1
                self.layer = torch.nn.Softmax(dim=1)

            def forward(self, x):
                n_samples, _ = x.shape
                y = 1.0 * torch.ones(n_samples, dtype=x.dtype, device=x.device)
                inp = x / y[..., None]
                return self.layer(inp)

        x = torch.rand([4, 4], device="cuda")
        m = MyModule()
        opt_m = torch.compile(backend="inductor")(m)
        self.assertEqual(opt_m(x), m(x))

    def test_issue97695_2input(self):
        def fn(arg3_1, arg3_2, relu, permute_1):
            addmm_1 = torch.ops.aten.addmm.default(arg3_1, relu, permute_1)
            addmm_2 = torch.ops.aten.addmm.default(arg3_2, relu, permute_1)
            cat_2 = torch.ops.aten.cat.default([addmm_1, addmm_2], 1)
            return (cat_2,)

        args = [
            ((96,), (1,), torch.float32, "cuda"),
            ((96,), (1,), torch.float32, "cuda"),
            ((10, 256), (256, 1), torch.float32, "cuda"),
            ((256, 96), (1, 256), torch.float32, "cuda"),
        ]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
        correct = fn(*args)

        ref = torch.compile(fn, fullgraph=True)(*args)
        if not same(ref, correct):
            raise AssertionError

    def test_scatter_index_not_wrapped(self):
        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=self.device)
        index = torch.tensor([0, 1, 0, 1, 2, 0], device=self.device)
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        compiled_sr = torch.compile(torch.scatter_reduce)

        input_orig = input.clone()
        out, code = run_and_get_code(compiled_sr, input, 0, index, src, "sum")
        # tmp0 - not wrapping of negative numbers
        FileCheck().check("tl.device_assert(((0 <= tmp0) & (tmp0 < 4))").check_next(
            "atomic_add"
        ).run(code[0])
        self.assertEqual(
            out, torch.scatter_reduce(input_orig.clone(), 0, index, src, "sum")
        )

    def test_normalize_norm_leq_one(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(x, dim=-1)

        inp = torch.tensor([[3.799999, 0.0, 0.0]], device="cuda", dtype=torch.float32)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out = compiled(inp)
        norm = out.norm(dim=-1)
        self.assertTrue(
            torch.all(norm <= 1.0), f"expected norm <= 1.0 but got {norm.item()}"
        )

    def test_libdevice_routing(self):
        def foo(x):
            return x.exp()

        inp = torch.ones(64, device="cuda").to(torch.float64)

        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        inp = inp.to(torch.float)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check_not("tl_math.exp").check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

        def foo(x):
            return x.sigmoid()

        inp = torch.ones(64, device="cuda").to(torch.float64)
        out, code = run_and_get_code(torch.compile(foo), inp)
        FileCheck().check("libdevice.exp").run(code[0])
        self.assertEqual(foo(inp), out)

    def test_uint_view_copy(self):
        @torch.compile
        def view_copy(target, source):
            assert target.dtype == torch.bfloat16  # noqa: S101
            assert source.dtype == torch.uint16  # noqa: S101
            target.view(torch.uint16).copy_(source)

        target = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
        source = torch.full_like(target, 4, dtype=torch.uint16)

        out = target.view(torch.uint16).copy_(source).clone()
        view_copy(target, source)
        self.assertEqual(out, target.view(torch.uint16))

    def test_embedding_var_mean(self):
        def forward(arg0_1):
            full = torch.ops.aten.full.default(
                [1, 2048],
                1,
                dtype=torch.float32,
                layout=torch.strided,
                device=torch.device(type="cuda", index=0),
                pin_memory=False,
            )
            convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                full, torch.int64
            )
            cumsum = torch.ops.aten.cumsum.default(convert_element_type_1, 1)
            mul = torch.ops.aten.mul.Tensor(cumsum, convert_element_type_1)
            sub_1 = torch.ops.aten.sub.Tensor(mul, 1)
            slice_5 = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807)
            slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807)
            add_2 = torch.ops.aten.add.Tensor(slice_6, 2)
            embedding_1 = torch.ops.aten.embedding.default(arg0_1, add_2)
            var_mean = torch.ops.aten.var_mean.correction(
                embedding_1, [2], correction=0, keepdim=True
            )
            return [var_mean[0], var_mean[1], add_2]

        emb = torch.randn([2050, 768], device="cuda")
        gm = make_fx(forward)(emb)
        opt = torch._inductor.compile_fx.compile_fx_inner(gm, [emb])
        opt([emb])
        torch.cuda.synchronize()

    def test_deterministic_algorithms(self):
        N = 10000

        @torch.compile
        def fn(idx, values):
            x = torch.zeros(1, device="cuda")
            x[idx] += values
            return x

        idx = torch.zeros(N, dtype=torch.int64, device="cuda")
        values = torch.randn(N, device="cuda")

        r0 = fn(idx, values)
        with DeterministicGuard(True):
            r1 = fn(idx, values)
            for _ in range(10):
                rn = fn(idx, values)
                self.assertEqual(r1, rn, atol=0, rtol=0)

    # https://github.com/pytorch/pytorch/issues/96406
    def test_linear_cpu_input(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, data):
                data = data.to("cuda")
                return self.linear(data)

        mod = Model().cuda().eval()
        with torch.no_grad():
            self.common(mod, (torch.randn(4, 4),))

    @config.patch({"fallback_random": True, "triton.cudagraphs": True})
    def test_xlnet_lm_stride_repro(self):
        class Repro(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dropout = nn.Dropout(p=0.1, inplace=False)

            def forward(self, x):
                y = torch._C._nn.gelu(x)
                return self.dropout(y)

        mod = Repro()
        x = torch.randn((512, 1, 4096), requires_grad=True, device="cuda")
        y = torch.compile(mod)(x)
        # Inductor claims the output layout of gelu's saved variable for
        # backwards will be (4096, 4096, 1) but in actuality it is (4096,
        # 2097152, 1).  Fortunately this doesn't actually matter in practice.
        y.sum().backward()

    def test_lookup_seed_backward(self):
        @torch.compile(fullgraph=True)
        def forward(inductor_seeds, mul_4, view_15):
            inductor_lookup_seed_2 = torch.ops.prims.inductor_lookup_seed.default(
                inductor_seeds, 2
            )
            inductor_random_2 = torch.ops.prims.inductor_random.default(
                [2, 512, 768], inductor_lookup_seed_2, "rand"
            )
            gt_2 = torch.ops.aten.gt.Scalar(inductor_random_2, 0.1)
            mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_15)
            mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112)
            add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4)
            var_mean_1 = torch.ops.aten.var_mean.correction(
                add_5, [2], correction=0, keepdim=True
            )
            getitem_3 = var_mean_1[1]
            sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3)
            return (sub_3,)

        buf0 = torch.zeros((37,), dtype=torch.int64, device="cuda")
        buf1 = torch.zeros((2, 512, 768), device="cuda")
        buf2 = torch.zeros((2, 512, 768), device="cuda")
        forward(buf0, buf1, buf2)

    def test_issue100806(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 30)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = torch.cat((x, x), dim=1)
                x = x.view(-1, 2, 30)
                x = x[:, 1, :]
                x = self.relu(x)
                return x

        device = "cuda"
        batch_size = 2
        x = torch.randn(batch_size, 10).to(device)
        func = Model().to(device)

        with torch.no_grad():
            func.train(False)
            jit_func = torch.compile(func)

            res1 = func(x)
            res2 = jit_func(x)
            self.assertEqual(res1, res2)

    def test_issue103481(self):
        def fn(x, y):
            # NOTE: 6 dimensions is important! does not fail for 5 dimensions
            mean = torch.mean(x, [2, 3, 4, 5], keepdim=True)
            add = mean + y
            return add

        x = torch.rand(4, 4, 4, 4, 4, 4, device="cuda")
        y = torch.rand((), device="cuda")
        expect = fn(x, y)

        opt_fn = torch.compile(fn)
        actual = opt_fn(x, y)

        self.assertEqual(expect, actual)

    @config.patch({"triton.dense_indexing": True})
    @dynamo_config.patch(automatic_dynamic_shapes=True)
    def test_bucketize_dynamic_dense(self):
        """
        Make sure that ops.bucketize() can handle dense_indexing, which previously
        caused issues due to incorrect handling of the size of offsets.
        """

        def fn(values, offsets):
            return torch.bucketize(values, offsets)

        values = torch.rand((64, 64), device="cuda")
        offsets = torch.tensor([0.05, 0.1, 0.5, 0.8, 0.85, 0.95], device="cuda")

        expect = fn(values, offsets)

        opt_fn = torch.compile(fn, dynamic=True)
        actual = opt_fn(values, offsets)

        self.assertEqual(expect, actual)

    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
            "compile_threads": 1,
        }
    )
    def test_bucketize_epilogue(self):
        """
        See https://github.com/pytorch/pytorch/issues/148764.
        Make sure that when torch.bucketize appears as an epilogue, the codegen is valid.

        Note: during autotuning, there's also the option to _not_ do the fusion.
        So if you run the test with standard configs, the fused kernel would fail during
        autotuning, and another non-fused kernel would be selected (and Inductor would
        throw some errors, but the test would pass)

        So we set disallow_failing_autotune_kernels_TESTING_ONLY=True to prevent the
        autotuner from catching failures. And set compile_threads=1 so that compile
        failures aren't caught by the asyn runner infra.
        """

        def fn(x: torch.Tensor, y: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
            z = torch.mm(x, y)
            return torch.bucketize(z, buckets)

        buckets = torch.arange(-100, 100, 10, device="cuda")
        x = torch.randn(64, 64, device="cuda").clamp(-99, 99)
        y = torch.randn(64, 64, device="cuda").clamp(-99, 99)

        opt_fn = torch.compile(fn, mode="max-autotune")

        expected = fn(x, y, buckets)
        actual = opt_fn(x, y, buckets)

        self.assertEqual(expected, actual)

    def test_float64_constants(self):
        def fn():
            # NOTE: tensors of all the same value are constant folded, so we
            # need a tensor with two distinct values
            a = torch.tensor([1 / 10, 2 / 10], dtype=torch.float64, device="cuda")
            return a * 2e50

        cfn = torch.compile(fn)
        expect = fn()
        actual = cfn()
        self.assertEqual(expect, actual, atol=0, rtol=0)

    def test_issue104759(self):
        def fn(arg7_1, add_1, permute_2, select_scatter, slice_8):
            slice_scatter_4 = torch.ops.aten.slice_scatter.default(
                permute_2, select_scatter, 0, 1, 9223372036854775807
            )
            permute_3 = torch.ops.aten.permute.default(slice_scatter_4, [1, 3, 0, 2, 4])
            view_6 = torch.ops.aten.view.default(permute_3, [1, 1000, 48])
            view_7 = torch.ops.aten.view.default(view_6, [1000, 48])
            view_8 = torch.ops.aten.view.default(view_7, [1, 1000, 48])
            view_9 = torch.ops.aten.view.default(view_8, [1, 1000, 3, 4, 4])
            permute_4 = torch.ops.aten.permute.default(view_9, [2, 0, 3, 1, 4])
            slice_7 = torch.ops.aten.slice.Tensor(permute_4, 0, 1, 9223372036854775807)
            slice_scatter_5 = torch.ops.aten.slice_scatter.default(
                slice_8, slice_7, 4, 0, 9223372036854775807
            )
            slice_scatter_6 = torch.ops.aten.slice_scatter.default(
                arg7_1, slice_scatter_5, 3, 0, 1000
            )
            mul_8 = torch.ops.aten.mul.Scalar(add_1, 0.7071067811865476)
            slice_9 = torch.ops.aten.slice.Tensor(slice_scatter_6, 3, 0, 1000)
            slice_10 = torch.ops.aten.slice.Tensor(slice_9, 4, 0, 9223372036854775807)
            select_2 = torch.ops.aten.select.int(slice_10, 0, 0)
            permute_5 = torch.ops.aten.permute.default(select_2, [0, 1, 3, 2])
            mul_9 = torch.ops.aten.mul.Scalar(permute_5, 0.7071067811865476)
            expand = torch.ops.aten.expand.default(mul_8, [1, 4, 1000, 4])
            view_10 = torch.ops.aten.view.default(expand, [4, 1000, 4])
            expand_1 = torch.ops.aten.expand.default(mul_9, [1, 4, 4, 1000])
            view_11 = torch.ops.aten.view.default(expand_1, [4, 4, 1000])
            bmm = torch.ops.aten.bmm.default(view_10, view_11)
            return (bmm,)

        args = []
        args.append(torch.randn((2, 1, 4, 1200, 4), dtype=torch.float16, device="cuda"))
        args.append(
            rand_strided(
                (1, 4, 1000, 4), (16000, 4, 16, 1), dtype=torch.float16, device="cuda"
            )
        )
        args.append(
            rand_strided(
                (3, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (16, 48000, 4, 48, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )
        args.append(
            rand_strided(
                (2, 1, 4, 1000, 4),
                (19200, 19200, 4800, 4, 1),
                dtype=torch.float16,
                device="cuda",
            )
        )

        correct = fn(*args)
        mod = make_fx(fn, tracing_mode="real")(*args)
        compiled = compile_fx_inner(mod, args)
        ref = compiled(list(args))
        if not same(ref, correct):
            raise AssertionError

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_inplace_cudagraph(self):
        def fn(x, y, z):
            x = torch.zeros_like(x)
            return x.index_put_([y], z, True)

        x = torch.zeros((512, 512), device="cuda", dtype=torch.bool)
        y = torch.zeros((512,), device="cuda", dtype=torch.int64)
        z = torch.ones((512, 512), device="cuda", dtype=torch.bool)

        opt_fn = torch.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @config.patch({"triton.cudagraphs": True})
    @config.patch({"fx_graph_cache": True})
    def test_index_put_cudagraph(self):
        for _ in range(2):

            def fn(x, y, z):
                x = torch.zeros_like(x)
                return x.index_put([y], z, True)

            x = torch.zeros((512, 512), device="cuda", dtype=torch.bool)
            y = torch.zeros((512,), device="cuda", dtype=torch.int64)
            z = torch.ones((512, 512), device="cuda", dtype=torch.bool)

            opt_fn = torch.compile(fn, backend="inductor")

            ref = fn(x, y, z)

            # run it twice to test cuda graph issue
            res = opt_fn(x, y, z)
            res = opt_fn(x, y, z)

            self.assertEqual(ref, res)
            torch._dynamo.reset()
            gc.collect()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported"
    )
    def test_flash_attention_dynamic(self):
        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                self.q = nn.Linear(1024, 1024)
                self.k = nn.Linear(1024, 1024)
                self.v = nn.Linear(1024, 1024)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
                values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)

                attn = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                )

                return attn

        cnts = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        model = Model().cuda().half()
        model = torch.compile(model, backend=cnts, dynamic=True)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            input1 = torch.rand(5, 512, 1024, device="cuda", dtype=torch.float16)
            input2 = torch.rand(5, 513, 1024, device="cuda", dtype=torch.float16)
            input3 = torch.rand(5, 514, 1024, device="cuda", dtype=torch.float16)

            out1 = model(input1)
            out2 = model(input2)
            out3 = model(input3)

        self.assertEqual(cnts.frame_count, 2)

    @config.patch({"triton.cudagraphs": True})
    def test_index_put_no_fallback_cudagraph(self):
        def fn(x, y, z):
            x = torch.zeros_like(x)
            return x.index_put([y], z, True)

        x = torch.zeros((512, 512), device="cuda", dtype=torch.int32)
        y = torch.zeros((512,), device="cuda", dtype=torch.int64)
        z = torch.ones((512, 512), device="cuda", dtype=torch.int32)

        opt_fn = torch.compile(fn, backend="inductor")

        ref = fn(x, y, z)

        # run it twice to test cuda graph issue
        res = opt_fn(x, y, z)
        res = opt_fn(x, y, z)

        self.assertEqual(ref, res)

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_norm_rounding(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        x = torch.rand(1000, device="cuda", dtype=torch.bfloat16)
        scalar = torch.rand([], device="cuda", dtype=torch.float32)

        def fn(inp, scale):
            y = inp.norm()
            return y, y + scale

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

        expected = fn(x, scalar)
        actual = opt_fn(x, scalar)

        self.assertEqual(expected, actual)

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_min_pow_chain(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = torch.rand(
                [383, 55, 2, 3],
                dtype=torch.float16,
                device="cuda",
                requires_grad=True,
            )
            arg1 = torch.rand(
                [383, 55], dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            arg2 = torch.rand(
                [383, 55], dtype=torch.float32, device="cuda", requires_grad=True
            )
            arg3 = torch.rand(
                [383, 55], dtype=torch.float32, device="cuda", requires_grad=True
            )

            def fn(a0, a1, a2, a3):
                t1 = a0.min(dim=2).values
                t2 = t1.sum(dim=2)
                t6 = ((((a1) - a2) - a3) - a3) - a3
                t7 = t6 + t2
                t8 = torch.pow(torch.pow(torch.pow(torch.pow(t2, t7), t7), t7), t7)
                return t7, t8

            opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3)
            compiled_args = [
                arg0.clone().detach().requires_grad_(True),
                arg1.clone().detach().requires_grad_(True),
                arg2.clone().detach().requires_grad_(True),
                arg3.clone().detach().requires_grad_(True),
            ]
            compiled_out = opt_fn(*compiled_args)

            for eager_tensor, compiled_tensor in zip(eager_out, compiled_out):
                torch.testing.assert_close(
                    eager_tensor,
                    compiled_tensor,
                    rtol=1e-3,
                    atol=1e-3,
                )

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_emulate_precision_casts_mean_ratio_chain(self):
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        with dynamo_config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            arg0 = torch.rand(
                [125070], dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            arg1 = torch.rand(
                [1895, 3, 11], dtype=torch.float16, device="cuda", requires_grad=True
            )
            arg2 = torch.rand(
                [1895, 3, 11], dtype=torch.float32, device="cuda", requires_grad=True
            )
            arg3 = torch.rand(
                [1895, 3, 11], dtype=torch.float32, device="cuda", requires_grad=True
            )
            arg4 = torch.rand(
                [1895, 3, 11], dtype=torch.float32, device="cuda", requires_grad=True
            )
            arg5 = torch.rand(
                [5, 379, 165], dtype=torch.float32, device="cuda", requires_grad=True
            )

            def fn(a0, a1, a2, a3, a4, a5):
                t2 = a0.view(379, 165, 2).mean(dim=2)
                t7 = ((((a1) - a2) - a3) - a2) - a4
                t8 = t7.view(379, 165)
                t11 = torch.nn.functional.relu(a5).mean(dim=0)
                t12 = t2 - t11
                t13 = (((t2) / t8) / t11) / t12
                return t13

            opt_fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)

            eager_out = fn(arg0, arg1, arg2, arg3, arg4, arg5)
            compiled_args = [
                tensor.clone().detach().requires_grad_(True)
                for tensor in (arg0, arg1, arg2, arg3, arg4, arg5)
            ]
            compiled_out = opt_fn(*compiled_args)

            torch.testing.assert_close(
                eager_out,
                compiled_out,
                rtol=5e-2,
                atol=1e-1,
            )

    @torch._inductor.config.patch(emulate_precision_casts=True)
    def test_dont_inplace_disjoint_accesses(self):
        # TODO - would not need mms if we could annotate donated buffer..
        def forward(  # noqa: F821, F722
            arg0_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg1_1: "bf16[8, 4096, 2048][8388608, 2048, 1]cuda:0",  # noqa: F821, F722
            arg2_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg3_1: "bf16[2048, 2048][2048, 1]cuda:0",  # noqa: F821, F722
            arg4_1: "bf16[2048][1]cuda:0",  # noqa: F821, F722
            arg5_1: "bf16[2048][1]cuda:0",  # noqa: F821, F722
            arg6_1: "f32[4096, 128][128, 1]cuda:0",  # noqa: F821, F722
            arg7_1: "f32[4096, 128][128, 1]cuda:0",  # noqa: F821, F722
        ):
            permute = torch.ops.aten.permute.default(arg0_1, [1, 0])
            arg0_1 = None
            view = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            mm = torch.ops.aten.mm.default(view, permute)
            view = permute = None
            view_1 = torch.ops.aten.view.default(mm, [8, 4096, 2048])
            mm = None
            permute_1 = torch.ops.aten.permute.default(arg2_1, [1, 0])
            arg2_1 = None
            view_2 = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            mm_1 = torch.ops.aten.mm.default(view_2, permute_1)
            view_2 = permute_1 = None
            view_3 = torch.ops.aten.view.default(mm_1, [8, 4096, 2048])
            mm_1 = None
            permute_2 = torch.ops.aten.permute.default(arg3_1, [1, 0])
            arg3_1 = None
            view_4 = torch.ops.aten.view.default(arg1_1, [32768, 2048])
            arg1_1 = None
            mm_2 = torch.ops.aten.mm.default(view_4, permute_2)
            view_4 = permute_2 = None
            view_5 = torch.ops.aten.view.default(mm_2, [8, 4096, 2048])
            mm_2 = None
            convert_element_type_6 = torch.ops.prims.convert_element_type.default(
                view_1, torch.float32
            )
            view_1 = None
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
            mean = torch.ops.aten.mean.dim(pow_1, [-1], True)
            pow_1 = None
            add = torch.ops.aten.add.Tensor(mean, 1e-06)
            mean = None
            rsqrt = torch.ops.aten.rsqrt.default(add)
            add = None
            mul = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt)
            convert_element_type_6 = rsqrt = None
            convert_element_type_7 = torch.ops.prims.convert_element_type.default(
                arg4_1, torch.float32
            )
            arg4_1 = None
            mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_7, mul)
            convert_element_type_7 = mul = None
            convert_element_type_8 = torch.ops.prims.convert_element_type.default(
                mul_1, torch.bfloat16
            )
            mul_1 = None
            convert_element_type_9 = torch.ops.prims.convert_element_type.default(
                view_3, torch.float32
            )
            view_3 = None
            pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 2)
            mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True)
            pow_2 = None
            add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-06)
            mean_1 = None
            rsqrt_1 = torch.ops.aten.rsqrt.default(add_1)
            add_1 = None
            mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_9, rsqrt_1)
            convert_element_type_9 = rsqrt_1 = None
            convert_element_type_10 = torch.ops.prims.convert_element_type.default(
                arg5_1, torch.float32
            )
            arg5_1 = None
            mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_2)
            convert_element_type_10 = mul_2 = None
            convert_element_type_11 = torch.ops.prims.convert_element_type.default(
                mul_3, torch.bfloat16
            )
            mul_3 = None
            view_6 = torch.ops.aten.view.default(
                convert_element_type_8, [8, 4096, -1, 128]
            )
            convert_element_type_8 = None
            view_7 = torch.ops.aten.view.default(
                convert_element_type_11, [8, 4096, -1, 128]
            )
            convert_element_type_11 = None
            view_8 = torch.ops.aten.view.default(view_5, [8, 4096, -1, 128])
            view_5 = None
            convert_element_type_12 = torch.ops.prims.convert_element_type.default(
                view_6, torch.float32
            )
            view_6 = None
            convert_element_type_13 = torch.ops.prims.convert_element_type.default(
                view_7, torch.float32
            )
            view_7 = None
            unsqueeze = torch.ops.aten.unsqueeze.default(arg6_1, 0)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze = None
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
            unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2)
            unsqueeze_2 = None
            mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_12, unsqueeze_3)
            unsqueeze_3 = None
            view_9 = torch.ops.aten.view.default(
                convert_element_type_12, [8, 4096, 16, 2, 64]
            )
            convert_element_type_12 = None
            unbind = torch.ops.aten.unbind.int(view_9, -2)
            view_9 = None
            getitem = unbind[0]
            getitem_1 = unbind[1]
            unbind = None
            neg = torch.ops.aten.neg.default(getitem_1)
            getitem_1 = None
            cat = torch.ops.aten.cat.default([neg, getitem], -1)
            neg = getitem = None
            mul_5 = torch.ops.aten.mul.Tensor(cat, unsqueeze_1)
            cat = unsqueeze_1 = None
            add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5)
            mul_4 = mul_5 = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg6_1, 0)
            arg6_1 = None
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2)
            unsqueeze_4 = None
            unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg7_1, 0)
            arg7_1 = None
            unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2)
            unsqueeze_6 = None
            mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_7)
            unsqueeze_7 = None
            view_10 = torch.ops.aten.view.default(
                convert_element_type_13, [8, 4096, 16, 2, 64]
            )
            convert_element_type_13 = None
            unbind_1 = torch.ops.aten.unbind.int(view_10, -2)
            view_10 = None
            getitem_2 = unbind_1[0]
            getitem_3 = unbind_1[1]
            unbind_1 = None
            neg_1 = torch.ops.aten.neg.default(getitem_3)
            getitem_3 = None
            cat_1 = torch.ops.aten.cat.default([neg_1, getitem_2], -1)
            neg_1 = getitem_2 = None
            mul_7 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_5)
            cat_1 = unsqueeze_5 = None
            add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7)
            mul_6 = mul_7 = None
            convert_element_type_14 = torch.ops.prims.convert_element_type.default(
                add_2, torch.bfloat16
            )
            add_2 = None
            convert_element_type_15 = torch.ops.prims.convert_element_type.default(
                add_3, torch.bfloat16
            )
            add_3 = None
            permute_3 = torch.ops.aten.permute.default(
                convert_element_type_14, [0, 2, 1, 3]
            )
            convert_element_type_14 = None
            permute_4 = torch.ops.aten.permute.default(
                convert_element_type_15, [0, 2, 1, 3]
            )
            convert_element_type_15 = None
            permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3])
            view_8 = None
            return (permute_3, permute_4, permute_5)

        from torch._dynamo.debug_utils import aot_graph_input_parser

        kwargs = aot_graph_input_parser(forward)
        out, code = run_and_get_code(torch.compile(forward), **kwargs)
        # ignore tiny values.. prior to this fix absolute error was ~28
        self.assertEqual(forward(**kwargs), out, atol=0.01, rtol=2)
        FileCheck().check_not("in_out").run(code[0])

    # https://github.com/pytorch/pytorch/issues/104937
    def test_linear_with_zero_infeature_size(self):
        m = nn.Linear(in_features=0, out_features=0, bias=True).to("cuda")
        x = torch.rand(1, 1, 0, device="cuda")
        expect = m(x)
        opt_fn = torch.compile(m)
        actual = opt_fn(x)
        self.assertEqual(expect, actual)

    @config.patch(fallback_random=True)
    def test_multi_output_layout_fallback(self):
        mod = nn.RReLU(lower=3.2350976, upper=8.4220314, inplace=True)
        inp = torch.rand([4, 4]).cuda()
        m = torch.compile(mod)

        with freeze_rng_state():
            o1 = m(inp.clone())

        o2 = mod(inp.clone())

        self.assertEqual(o1, o2)

    def test_sorted_masks(self):
        @torch.compile()
        def foo(x, y):
            return (x + y).sum(dim=1)

        x = torch.rand([255, 255], device="cuda")
        y = torch.rand([255, 255], device="cuda")

        _, code = run_and_get_code(foo, x, y)
        FileCheck().check("tl.load").check_same("r0_mask").check_same("xmask").run(
            code[0]
        )

    def test_cat_int8_one_kernel(self):
        @torch.compile()
        def cat(inps):
            return torch.cat(inps) + 1

        for dtype in [torch.uint8, torch.int8]:
            inps = [
                torch.empty([256, 256], dtype=dtype, device="cuda") for _ in range(4)
            ]

            out, code = run_and_get_code(cat, inps)
            self.assertEqual(torch.cat(inps) + 1, out)
            FileCheck().check_not("aten.cat.default(").check_count(
                ".run(", 1, exactly=True
            ).run(code[0])

    @config.patch("triton.use_block_ptr", True)
    def test_selecsls42b_misaligned_address(self):
        # https://github.com/triton-lang/triton/issues/2836

        @torch.compile(fullgraph=True)
        def fn(arg207_1, arg208_1, convert_element_type_40, expand, full, mul_3):
            div = torch.ops.aten.div.Scalar(expand, 16)
            where = torch.ops.aten.where.self(arg207_1, full, div)
            convert_element_type_43 = torch.ops.prims.convert_element_type.default(
                where, torch.float32
            )
            sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_43, [0, 2, 3])
            sub = torch.ops.aten.sub.Tensor(convert_element_type_40, arg208_1)
            mul = torch.ops.aten.mul.Tensor(convert_element_type_43, sub)
            sum_3 = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3])
            mul_1 = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
            unsqueeze = torch.ops.aten.unsqueeze.default(mul_1, 0)
            unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
            unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3)
            mul_2 = torch.ops.aten.mul.Tensor(sum_3, 0.0078125)
            mul_4 = torch.ops.aten.mul.Tensor(mul_2, mul_3)
            unsqueeze_3 = torch.ops.aten.unsqueeze.default(mul_4, 0)
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2)
            unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
            mul_6 = torch.ops.aten.mul.Tensor(sub, unsqueeze_5)
            sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_43, mul_6)
            sub_2 = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2)
            return (sub_2,)

        args = [
            torch.randn((8, 1024, 4, 4), device="cuda") > 0,  # torch.bool tensor
            torch.randn((1, 1024, 1, 1), device="cuda"),
            torch.randn((8, 1024, 4, 4), device="cuda"),
            torch.randn((8, 1024, 1, 1), dtype=torch.float16, device="cuda").expand(
                (8, 1024, 4, 4)
            ),
            torch.randn((), device="cuda"),
            torch.randn((1024,), device="cuda"),
        ]
        fn(*args)
        torch.cuda.synchronize()  # shake out Triton Error [CUDA]: misaligned address

    def test_mutated_aligned_tensor(self):
        t = torch.rand(4096, device="cuda", dtype=torch.float16)

        def foo(x):
            return x.add_(1)

        foo_c = torch.compile(dynamic=False)(foo)

        t_orig = t.clone()

        # First invocation, assume alignment, second invocation,
        # copy to alignment and then mutate after fn invocation
        self.assertEqual(foo_c(t[:-1]), foo(t_orig[:-1]))
        self.assertEqual(t, t_orig)

        self.assertEqual(foo_c(t[1:]), foo(t_orig[1:]))
        self.assertEqual(t, t_orig)

    def test_non_commutative_scan_op(self):
        from torch._higher_order_ops.associative_scan import associative_scan

        a = torch.randn(1024, 8192, dtype=torch.float64, device="cuda")
        b = torch.randn(1024, 8192, dtype=torch.float64, device="cuda")

        def baseline(v, u):
            A = []
            A.append(b[:, 0])
            for i in range(1, v.shape[1]):
                A.append(a[:, i] * A[i - 1] + b[:, i])
            return torch.stack(A, dim=1)

        def combine_fn(i, j):
            ia, ib = i
            ja, jb = j
            return ia * ja, ib * ja + jb

        @torch.compile
        def compiled_scan(a, b):
            return associative_scan(combine_fn, (a, b), dim=-1)[1]

        out1 = baseline(a, b)
        out2 = compiled_scan(a, b)
        self.assertEqual(out1, out2)

    def test_dynamic_persistent_reductions(self):
        @torch.compile(dynamic=True)
        def inner_reduce(x):
            assert x.shape[1] <= 1024  # noqa: S101
            return x.sum(1)

        a = torch.randn(50, 600, device="cuda")
        out, code = run_and_get_code(inner_reduce, a)
        self.assertEqual(inner_reduce(a), out)
        self.assertTrue("for roffset" not in code)

        @torch.compile(dynamic=True)
        def outer_reduce(x):
            assert x.shape[0] <= 64  # noqa: S101
            return x.sum(0)

        out, code = run_and_get_code(outer_reduce, a)
        self.assertEqual(outer_reduce(a), out)
        self.assertTrue("for roffset" not in code)

    def test_scaled_dot_product_efficient_attention_backward(self):
        from torch import nn, Tensor

        class SelfAttention(nn.Module):
            def __init__(
                self,
                num_attention_heads: int = 12,
                hidden_size: int = 768,
                attention_probs_dropout_prob: float = 0.1,
            ):
                super().__init__()

                self.num_attention_heads = num_attention_heads
                self.attention_head_size = hidden_size // num_attention_heads

                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

                self.dropout_prob = attention_probs_dropout_prob

            def transpose_for_scores(self, x: Tensor) -> Tensor:
                new_x_shape = x.size()[:-1] + (
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                return x.view(new_x_shape).permute(0, 2, 1, 3)

            def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                query_layer = self.transpose_for_scores(self.query(hidden_states))
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=False,
                )
                return attn_output

        device = torch.device("cuda")
        num_attention_heads = 8
        hidden_size = 512
        attention_probs_dropout_prob = 0.0
        model = SelfAttention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        ).to(device)

        model = torch.compile(model)

        # runs without failure
        batch_size = 8
        length = 1
        inputs_embeds = torch.randn(batch_size, length, hidden_size, device=device)
        attention_mask = torch.ones(batch_size, 1, length, length, device=device)
        attn_output = model(hidden_states=inputs_embeds, attention_mask=attention_mask)[
            0
        ]
        loss = attn_output.mean()
        loss.backward()

    def test_non_contiguous_unaligned_input_indices(self):
        from torch._inductor.compile_fx import remove_unaligned_input_idxs

        inputs = [torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda")[1:]]
        idxs = remove_unaligned_input_idxs(inputs, [1])
        self.assertEqual(idxs, [])

        inputs = [
            torch.ones(2, 2, device="cuda"),
            torch.ones(2, 2, device="cuda"),
            torch.ones(2, 2, device="cuda")[1:],
        ]
        idxs = remove_unaligned_input_idxs(inputs, [0, 2])
        self.assertEqual(idxs, [0])

    @config.patch("triton.cudagraphs", True)
    def test_unused_cpu_input_cudagraphs(self):
        def fn(x, y):
            return x.sin().sin().sin().sin().cos() + 1

        fx_graph = torch.fx.symbolic_trace(fn)
        inp = [torch.randn(64, device="cuda"), torch.randn(64, device="cpu")]
        compiled_fn, (graph,) = run_and_get_graph_lowering(
            torch._inductor.compile, fx_graph, inp
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})
        self.assertEqual(compiled_fn(*inp), fn(*inp))

    def test_epilogue_fusion_with_view(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.linear = torch.nn.Linear(262144, 100)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.relu(self.linear(x))

        m = ToyModel().to(device="cuda:0")
        input_tensor = torch.randn(32, 3, 64, 64).to(device="cuda:0")
        from torch._inductor.utils import fresh_cache

        with fresh_cache():
            cm = torch.compile(m, mode="max-autotune")
            out = cm(input_tensor)
            out2 = m(input_tensor)
            self.assertEqual(out, out2, atol=1e-3, rtol=1e-3)

    @config.patch("triton.cudagraphs", True)
    def test_cpu_index(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x[torch.arange(32)]

        result, (graph,) = run_and_get_graph_lowering(
            fn, torch.randn(64, device="cuda")
        )
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

        inp = torch.randn(64, device="cuda", requires_grad=True)
        result, (graph,) = run_and_get_graph_lowering(fn, inp)
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

        result, (graph,) = run_and_get_graph_lowering(lambda: result.sum().backward())
        self.assertEqual(graph.disable_cudagraphs_reason, None)
        self.assertEqual(graph.device_types, {"cuda"})

    @unittest.skipIf(IS_FBCODE, "Not runnable in fbcode")
    def test_triton_interpret(self):
        import subprocess

        script = """
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch

@torch.compile()
def foo(x):
    return x + 1

# somehow gives different results.. still, check that it doesn't error
foo(torch.rand([256], device="cuda"))
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_reflection_pad_loop_order(self):
        def fn(x, y):
            a = torch.nn.functional.pad(x, (5, 5, 5, 5), mode="reflect")
            b = torch.nn.functional.pad(y, (5, 5, 5, 5), mode="reflect")
            return a + b

        cfn = torch.compile(fn)
        a = torch.rand((10, 10, 10), device="cuda")
        b = torch.rand((10, 10, 10), device="cuda")
        expect = fn(a, b)
        actual, code = run_and_get_code(cfn, a, b)
        self.assertEqual(expect, actual)

        # Expect the code iterates in contiguous order, and is not tiled
        lines = code[0].split("\n")
        start = lines.index("@triton.jit")
        kernel_code = "\n".join(lines[start : start + 14])
        self.assertExpectedInline(
            kernel_code,
            """\
@triton.jit
def triton_poi_fused_add_reflection_pad2d_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x1 = ((xindex // 20) % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (99 + ((-1)*tl_math.abs((-9) + tl_math.abs((-5) + x0))) + ((-10)*tl_math.abs((-9) + tl_math.abs((-5) + x1))) + 100*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)""",  # noqa: B950
        )

    @skipCUDAIf(not SM80OrLater, "uses bfloat16 which requires SM >= 80")
    def test_int64_index_intermediate(self):
        def foo(inp):
            view_23 = torch.ops.aten.view.default(inp, [-1, 8192, 8192])
            split_1 = torch.ops.aten.split.Tensor(view_23, 1024, 1)
            view_23 = None
            getitem_17 = split_1[0]
            getitem_18 = split_1[1]
            getitem_19 = split_1[2]
            getitem_20 = split_1[3]
            getitem_21 = split_1[4]
            getitem_22 = split_1[5]
            getitem_23 = split_1[6]
            getitem_24 = split_1[7]
            split_1 = None
            cat_1 = torch.ops.aten.cat.default(
                [
                    getitem_17,
                    getitem_18,
                    getitem_19,
                    getitem_20,
                    getitem_21,
                    getitem_22,
                    getitem_23,
                    getitem_24,
                ]
            )
            getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = (
                getitem_22
            ) = getitem_23 = getitem_24 = None
            return cat_1

        for mark_dynamic in [False, True]:
            inp = torch.rand((65536, 8192), dtype=torch.bfloat16, device="cuda")
            if mark_dynamic:
                torch._dynamo.mark_dynamic(inp, 0)
            foo_c = torch.compile(foo)
            torch.testing.assert_allclose(foo(inp), foo_c(inp))

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_float8_e8m0fnu(self):
        device = "cuda"
        dtype = torch.float8_e8m0fnu
        hp_dtype = torch.float32  # and torch.bfloat16

        def foo(x0):
            x1 = x0.to(dtype)
            x2 = x1.to(hp_dtype)
            return x2

        x0 = torch.randn(16, 16, device=device, dtype=hp_dtype)
        foo_c = torch.compile(foo, backend="inductor", fullgraph=True)

        with torch.no_grad():
            y_c = foo_c(x0)

        self.assertEqual(foo(x0), y_c)

        dtype = torch.float8_e8m0fnu

        def foo(x0):
            x1 = x0 + 1
            x2 = x1.view(dtype).view([16 * 16])
            return x2

        x0 = torch.randint(0, 255, (16, 16), device=device, dtype=torch.uint8)
        foo_c = torch.compile(foo, backend="inductor", fullgraph=True)

        with torch.no_grad():
            result, code = run_and_get_code(foo_c, x0)

        FileCheck().check("call").check_not("torch.ops.aten.reshape.default(").run(
            code[0]
        )
        self.assertEqual(foo(x0), result)

    @unittest.skipIf(
        not config.is_fbcode(),
        "bfloat16 atomic add is only supported in fbcode today #97016",
    )
    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    def test_atomic_add_bfloat16(self):
        def f(x, y):
            return torch.index_select(x, 0, y)

        x = torch.randn(
            2000, 384, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        y = torch.ones(713268, dtype=torch.int64, device="cuda")
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: torch.compile(f)(x, y))
        fc = FileCheck()
        fc.check("tl.atomic_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    def test_red_dtype_mismatch(self):
        for per in (True, False):
            torch._dynamo.reset()
            if not per:
                torch._inductor.config.triton.persistent_reductions = False

            def f(arg0_1, arg1_1):
                embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
                view = torch.ops.aten.view.default(embedding, [64, 3072])
                unsqueeze = torch.ops.aten.unsqueeze.default(view, 0)
                expand = torch.ops.aten.expand.default(unsqueeze, [576, -1, -1])
                view_1 = torch.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
                permute = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                view_2 = torch.ops.aten.view.default(clone, [2, 18432, 3072])
                iota = torch.ops.prims.iota.default(
                    36,
                    start=0,
                    step=1,
                    dtype=torch.int64,
                    device="cuda",
                    requires_grad=False,
                )
                view_3 = torch.ops.aten.view.default(iota, [1, 36])
                max_1 = torch.ops.aten.max.default(view_3)
                return (max_1,)

            x = torch.ones(1, 64, device="cuda", dtype=torch.int64)
            y = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16)
            out = f(x, y)
            self.assertEqual(torch.compile(f)(x, y), out)

    @skipCUDAIf(
        not SM90OrLater, "uses bfloat16 atomic add instrs which requires SM >= 90"
    )
    @unittest.skipIf(
        config.is_fbcode(),
        "bfloat16 atomic add is supported in fbcode, so we won't fallback",
    )
    def test_index_add_fallback(self):
        def f(x, y):
            return torch.index_select(x, 0, y)

        x = torch.randn(
            2000, 384, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        y = torch.ones(713268, dtype=torch.int64, device="cuda")
        x_ref = x.clone().detach().requires_grad_(True)
        y_ref = y.clone().detach()

        out, (_, bw_code) = run_fw_bw_and_get_code(lambda: torch.compile(f)(x, y))
        fc = FileCheck()
        fc.check("aten.index_add")
        fc.run(bw_code)

        self.assertEqual(f(x_ref, y_ref), out)

    @requires_multigpu()
    def test_not_initializing_wrong_device(self):
        device_stats = torch.cuda.memory_stats("cuda:0")

        @torch.compile()
        def foo(x, y):
            return x @ y

        x = torch.rand([256, 256], device="cuda:1", requires_grad=True)
        y = torch.rand([256, 256], device="cuda:1", requires_grad=True)

        foo(x, y).sum().backward()

        device_stats2 = torch.cuda.memory_stats("cuda:0")
        self.assertTrue(
            device_stats2["active.all.peak"] <= device_stats["active.all.peak"]
        )

    @config.patch(
        {
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
        }
    )
    def test_3d_tiling(self):
        full_size, view_size, num_block_pointers, num_tiles = (
            (5, 5, 5, 5, 5),
            (3, 3, 5, 3, 5),
            1,
            2,
        )
        GPU_TYPE = "cuda"

        def get_input() -> torch.Tensor:
            device = torch.device(GPU_TYPE)
            full = torch.randn(full_size).to(device)
            return torch.as_strided(full, view_size, full.stride())

        a, b = get_input(), get_input()

        opt_fn = torch.compile(functools.partial(torch.add))
        result, (code,) = run_and_get_code(opt_fn, a, b)
        self.assertEqual(result, a + b)
        self.assertIn("znumel", code)

    @unittest.skipIf(config.is_fbcode(), "Dependence on functorch.einops")
    def test_repeated_masked_load(self):
        counters.clear()

        target_size = (8, 2)
        mem_eff_temporal_upsampling_interp_chunks = 2
        from functorch.einops import rearrange

        x = torch.randn(1, 8, 12, 12, 4, dtype=torch.float16, device="cuda")
        x = x.permute(0, 1, 4, 2, 3)  # make non-contiguous
        x = rearrange(x, "b c t h w -> b c t (h w)")

        def interpolate_chunked(x):
            chunks = x.chunk(chunks=mem_eff_temporal_upsampling_interp_chunks, dim=1)
            r = []
            for t in chunks:
                r.append(
                    torch.nn.functional.interpolate(
                        t.float(), size=target_size, mode="nearest"
                    ).to(t.dtype)
                )
            return torch.cat(r, dim=1)

        out_eager = interpolate_chunked(x)
        out_compiled = torch.compile(interpolate_chunked)(x)

        self.assertEqual(out_eager, out_compiled)

        unique_graphs = counters["stats"].get("unique_graphs", None)
        self.assertIsNotNone(
            unique_graphs,
            "Expected Dynamo to record unique_graphs counter",
        )
        self.assertEqual(
            unique_graphs,
            1,
            "Repeated masked loads should compile to a single stable graph",
        )

    def test_max_autotune_nograd(self):
        """
        https://github.com/pytorch/pytorch/issues/155688
        Smallest repro for max-autotune not working with no_grad
        Before adding __int__ function to torch.utils._sympy.functions.Identity,
        running the max_autotune mode would raise an error:
        TypeError: Expected a number but got Identity
        """

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.linear_layers = nn.ModuleList(
                    [
                        nn.Linear(4, 1, bias=True),
                        nn.Linear(5, 1, bias=True),
                        nn.Linear(6, 1, bias=True),
                        nn.Linear(7, 1, bias=True),
                        nn.Linear(8, 1, bias=True),
                    ]
                )

            def forward(self, x):
                for layer in self.linear_layers:
                    x2 = layer(x)
                    x2 = F.relu(x2)
                    x = torch.cat((x, x2), dim=1)

                return x

        model = ToyModel().to("cuda")
        input_tensor = torch.randn((2, 4)).to("cuda")

        compile_default = torch.compile(model, mode="default")
        compile_max_autotune = torch.compile(model, mode="max-autotune")

        with torch.no_grad():
            default_output = compile_default(input_tensor)
            max_autotune_output = compile_max_autotune(input_tensor)

        self.assertEqual(default_output, max_autotune_output)

    def test_adaptive_avg_pool3d_issue_157248(self):
        """Test for GitHub issue #157248: Conv2d-unsqueeze-AdaptiveAvgPool3d produces incorrect results"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.adaptive_pool = torch.nn.AdaptiveAvgPool3d((4, 4, 4))

            def forward(self, x):
                x = self.conv(x)
                # This specific unsqueeze position was problematic due to zero strides
                x = x.unsqueeze(1)
                x = self.adaptive_pool(x)
                return x

        model = Model().cuda()
        model.eval()
        test_cases = [
            (1, 3, 8, 8),
            (2, 3, 16, 16),
            (1, 3, 32, 32),
            (1, 3, 15, 15),
            (2, 3, 13, 13),
        ]

        for batch, channels, h, w in test_cases:
            with self.subTest(input_shape=(batch, channels, h, w)):
                input_tensor = torch.randn(batch, channels, h, w, device="cuda")

                # Test eager mode
                with torch.no_grad():
                    eager_output = model(input_tensor)

                # Test compiled mode with inductor
                compiled_model = torch.compile(model, backend="inductor")
                with torch.no_grad():
                    compiled_output = compiled_model(input_tensor)

                # They should be identical (or very close)
                self.assertTrue(
                    torch.allclose(eager_output, compiled_output, rtol=1e-5, atol=1e-5),
                    f"Results differ for input shape {(batch, channels, h, w)}. "
                    f"Max diff: {torch.max(torch.abs(eager_output - compiled_output)):.6f}",
                )

    @parametrize(
        "quantiles_shape,quantiles_strides,batch_size",
        [
            ((100, 10), (10, 1), 16),  # Contiguous C-order
            ((100, 10), (1, 100), 16),  # Transposed/F-order
            ((80, 12), (1, 80), 16),  # Transposed different size
            ((50, 20), (1, 50), 16),  # Transposed medium
            ((200, 8), (1, 200), 16),  # Transposed large x small
            ((25, 40), (1, 25), 16),  # Transposed small x large
            ((20, 5, 8), (40, 1, 5), 16),  # 3D case with mixed strides
            ((20, 5, 8), (1, 20, 100), 16),  # 3D case different stride order
        ],
    )
    def test_searchsorted_stride_permutations(
        self, quantiles_shape, quantiles_strides, batch_size
    ):
        class Foo(torch.nn.Module):
            def __init__(self, quantiles: torch.Tensor) -> None:
                super().__init__()
                if quantiles.shape[0] <= 0:
                    raise AssertionError(
                        f"Expected quantiles.shape[0] > 0, got {quantiles.shape[0]}"
                    )
                quantiles = quantiles.T
                self.q = torch.nn.Parameter(quantiles, requires_grad=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.searchsorted(self.q, x.T).T

        torch.manual_seed(42)

        # Create contiguous tensor first
        numel = 1
        for dim in quantiles_shape:
            numel *= dim
        data = torch.randn(numel, dtype=torch.float32, device="cuda")

        # Create tensor with specified shape and strides
        quantiles = torch.as_strided(
            data, size=quantiles_shape, stride=quantiles_strides
        )

        quantiles = torch.sort(quantiles, dim=0)[0]

        x_shape = (batch_size,) + quantiles_shape[1:]
        x = torch.randn(*x_shape, dtype=torch.float32, device="cuda")

        foo = Foo(quantiles)
        foo_compiled = torch.compile(Foo(quantiles), fullgraph=True)

        # Test eager vs compiled
        with torch.no_grad():
            eager = foo(x)
            compiled = foo_compiled(x)

        self.assertEqual(eager, compiled)

    def test_identity_load(self):
        device = "cuda"

        def f(x, y):
            y2 = torch.cat(
                [
                    x[:, 1:],
                    y[:, None] + 32 * 2048,
                ],
                dim=1,
            )

            x2 = x[:, 1:, None]
            y3 = y2[:, -1:, None]

            return (
                torch.cat([x2, y3], dim=1)
                + torch.arange(-2048, 0, device=device)[None, None, :]
            ).reshape(1, 32 * 2048)

        # This succeeds
        eager_out = f(
            torch.zeros(1, 32, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        # This crashes
        compile_out, code = run_and_get_code(
            torch.compile(f),
            torch.zeros(1, 32, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        # make sure the identity is maintained
        FileCheck().check("(1 + ((31)").run(code[0])

        self.assertEqual(eager_out, compile_out)

    def test_qwen2_7b_sdpa_input_alignment_requires_recompile(self):
        # SDPA constraints ensures inputs have alignment (8).
        device = "cuda"

        def forward(q_proj, k_proj, attn_mask):
            scale = 0.08838834764831845  # 1/sqrt(128)

            B = attn_mask.size(0)
            S = attn_mask.size(3)
            D = 128
            d_model = q_proj.size(1)

            query_states = q_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hq, S, D]
            q = query_states.contiguous()

            Hkv = k_proj.size(1) // D
            Hq = query_states.size(1)

            nrepeats = Hq // Hkv
            key_states = k_proj.view(B, S, -1, D).transpose(1, 2)  # [B, Hkv, S, D]
            kv_repeated = key_states[:, :, None, :].expand(B, Hkv, nrepeats, S, D)
            kv_repeated = kv_repeated.contiguous()
            k = kv_repeated.reshape(B, Hq, S, D)
            v = k.clone()  # value tensor

            inf = torch.scalar_tensor(
                float("-inf"), dtype=torch.bfloat16, device=device
            )
            zero = torch.scalar_tensor(0.0, dtype=torch.bfloat16, device=device)
            where = torch.where(condition=attn_mask, input=zero, other=inf)
            pad_amount = 8 - (S % 8)
            padded = torch.nn.functional.pad(
                where, (0, pad_amount), value=0.0
            )  # pad last-dim
            sliced = padded[..., :S]  # back to [B,1,S,S]
            attn_bias = sliced.expand(B, Hq, S, S)

            sdpa_out, logsumexp, seed, offset = (
                torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q,
                    k,
                    v,
                    attn_bias,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=scale,
                    compute_log_sumexp=True,
                )
            )

            zeros = torch.zeros(B, S, d_model, device=device, dtype=torch.bfloat16)
            zeros = zeros.reshape(B, S, Hq, D)
            grad_out = zeros.permute(0, 2, 1, 3)

            out = (
                torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                    grad_out,
                    q,
                    k,
                    v,
                    attn_bias,
                    sdpa_out,
                    logsumexp,
                    seed,
                    offset,
                    dropout_p=0.0,
                    scale=scale,
                    grad_input_mask=[True, True, True, False],
                )
            )
            return out

        B = 2
        S = 6144
        D = 128
        Hq = 28
        Hkv = 4

        example_inputs = (
            torch.randn((B * S, Hq * D), dtype=torch.bfloat16, device=device),  # q_proj
            torch.randn(
                (B * S, Hkv * D), dtype=torch.bfloat16, device=device
            ),  # k_proj
            torch.zeros((B, 1, S, S), dtype=torch.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        compiled = torch.compile(forward, dynamic=True)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

        # run once more with seqlen that isn't divisible by 8
        S = 6102
        example_inputs = (
            torch.randn((S * B, Hq * D), dtype=torch.bfloat16, device=device),  # q_proj
            torch.randn(
                (S * B, Hkv * D), dtype=torch.bfloat16, device=device
            ),  # k_proj
            torch.zeros((B, 1, S, S), dtype=torch.bool, device=device),  # attn_mask
        )
        correct = forward(*example_inputs)
        actual = compiled(*example_inputs)
        self.assertEqual(actual, correct)

    @config.patch({"eager_numerics.division_rounding": True})
    def test_truediv_emulate_division_rounding(self):
        from decimal import Decimal

        y, x = 7.0, 11.0

        @torch.compile
        def compiled_divide(x, y):
            return x / y

        for y_dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for x_dtype in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ]:
                y_ten = torch.tensor([y], dtype=y_dtype, device="cuda")
                x_ten = torch.tensor([x], dtype=x_dtype, device="cuda")

                torch._dynamo.reset()
                compiled_div = Decimal(compiled_divide(x_ten, y_ten).item())
                eager_div = Decimal((x_ten / y_ten).item())

                self.assertEqual(eager_div, compiled_div)

    @config.patch({"eager_numerics.division_rounding": False})
    @xfailIfROCm
    def test_truediv_base_not_bitwise_equivalent(self):
        from decimal import Decimal

        y, x = 7.0, 11.0

        y_ten = torch.tensor([y], dtype=torch.float32, device="cuda")
        x_ten = torch.tensor([x], dtype=torch.float32, device="cuda")

        compile_out, code = run_and_get_code(
            torch.compile(lambda x, y: x / y),
            x_ten,
            y_ten,
        )
        compiled_div = Decimal(compile_out.item())
        eager_div = Decimal((x_ten / y_ten).item())

        self.assertNotEqual(eager_div, compiled_div)
        self.assertTrue("div_rn" not in code)

    @config.patch({"eager_numerics.disable_ftz": True})
    def test_disabling_ftz_yields_subnormals(self):
        from decimal import Decimal

        x = -127.0
        x_ten = torch.tensor([x], dtype=torch.float32, device="cuda")

        def fn(x):
            return 2.0**x

        compile_out = torch.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertTrue(compile_decimal > Decimal(0))

    @skipIfRocm(msg="ROCm preserves subnormals by default")
    @config.patch({"eager_numerics.disable_ftz": False})
    def test_not_disabling_ftz_yields_zero(self):
        from decimal import Decimal

        x = -128.0
        x_ten = torch.tensor([x], dtype=torch.float32, device="cuda")

        def fn(x):
            return 2.0**x

        compile_out = torch.compile(fn)(x_ten)
        compile_decimal = Decimal(compile_out.item())

        self.assertEqual(compile_decimal, Decimal(0))

    @config.patch(
        {"triton.use_block_ptr": True, "triton.codegen_upcast_to_fp32": False}
    )
    def test_float16_reduction_with_int_output(self):
        @torch.compile
        def fn(input: torch.Tensor) -> torch.Tensor:
            return torch.argmax(input, dim=0)

        input = torch.randn(20, 20, device="cuda", dtype=torch.float16)
        _, code = run_and_get_code(fn, input)
        # There should not be any conversions to float16 in this code, since the input
        # is already float16 and the output is int64.
        self.assertNotIn(".to(tl.float16)", code[0])

    @config.patch("eager_numerics.division_rounding", True)
    def test_reciprocal_precision_rounding(self):
        # Test that reciprocal matches eager when division_rounding is enabled.
        # This requires OpDecompositions.reciprocal to use float32 constant so
        # that div_rn can be applied (the dtype check requires both operands float32).
        def fn(x):
            return torch.reciprocal(x)

        x = torch.randn(1000, device="cuda", dtype=torch.float32) + 0.1
        self.common(fn, [x])


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON

    if HAS_CUDA_AND_TRITON and not TEST_WITH_ASAN:
        run_tests(needs="filelock")
