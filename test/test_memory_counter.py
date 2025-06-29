# Owner(s): ["module: unknown"]
# ruff: noqa: F841
import functools
import unittest

import torch
import torch.nn.functional as F
import torch.utils.memory_counter
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


try:
    from torchvision import models as torchvision_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

HAS_CUDA = torch.cuda.is_available()


def MemoryCounterMode(*args, **kwargs):
    return torch.utils.memory_counter.MemoryCounterMode(*args, **kwargs, display=False)


def get_total_memory(mode):
    return str(sum(v for _, v in mode.memory_counts["Global"].items()))


def T(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)


@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO, "torchdynamo doesn't work with __torch_dispatch__ right now"
)
class TestMemoryCounter(TestCase):
    def test_memory_counter_variety(self):
        mod = torch.nn.Linear(9, 10)
        with MemoryCounterMode() as mode:
            torch.mm(T(4, 5), T(5, 6))
            torch.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            torch.matmul(T(5, 6), T(6, 7))
            torch.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            mod(T(8, 9))

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_op(self):
        with MemoryCounterMode() as mode:
            torch.mm(T(4, 5), T(5, 6))
        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

        with mode:
            torch.bmm(T(3, 4, 5), T(3, 5, 6))
        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

        with mode:
            torch.addmm(T(4, 6), T(4, 5), T(5, 6))
            torch.addmm(T(4, 1), T(4, 5), T(5, 6))
            torch.addmm(T(6), T(4, 5), T(5, 6))

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

        with mode:
            torch.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

        with mode:
            torch.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

        with mode:
            torch.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_backward(self):
        with MemoryCounterMode() as mode:
            a = T(4, 5, requires_grad=True)
            a = torch.mm(a, T(5, 6))
            a = a.unsqueeze(0).expand(7, 4, 6)
            a = torch.bmm(a, T(7, 6, 7))
            a.sum().backward()

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_backward_reset(self):
        with MemoryCounterMode() as mode:
            a = T(4, 5, requires_grad=True)
            a.mm(a.t()).sum().backward()
            a.mm(a.t()).sum().backward()

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_torchscript(self):
        def foo(x):
            return torch.mm(x, x)

        with MemoryCounterMode() as mode:
            foo(T(5, 5))
        unscripted_memory = get_total_memory(mode)
        ts_foo = torch.jit.script(foo)
        with mode:
            ts_foo(T(5, 5))
        self.assertEqual(unscripted_memory, get_total_memory(mode))

    def test_autograd_op(self):
        class _CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input: torch.Tensor) -> torch.Tensor:
                return torch.mm(input, input)

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                return torch.mm(grad_output, grad_output) + torch.mm(
                    grad_output, grad_output
                )

        a = T(5, 5, requires_grad=True)
        with MemoryCounterMode() as mode:
            a = _CustomOp.apply(a)
            a.sum().backward()

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_conv_backwards_as_decomposition(self):
        # [conv backwards decomposition as conv forwards]

        class onlyConvs(torch.autograd.Function):
            @staticmethod
            def forward(inp, weight, transposed):
                if not transposed:
                    return F.conv1d(inp, weight)
                else:
                    return F.conv_transpose1d(inp, weight)

            @staticmethod
            def setup_context(ctx, inputs, output):
                inp, weight, transposed = inputs
                ctx.save_for_backward(inp, weight)
                ctx.transposed = transposed

            @staticmethod
            def backward(ctx, grad_out):
                inp, weight = ctx.saved_tensors
                if not ctx.transposed:
                    grad_inp = F.conv_transpose1d(grad_out, weight)
                    grad_weight = F.conv1d(inp, grad_out)
                    return grad_inp, grad_weight, None
                else:
                    grad_inp = F.conv1d(grad_out, weight)
                    grad_weight = F.conv1d(
                        grad_out.transpose(1, 0), inp.transpose(1, 0)
                    )
                    return grad_inp, grad_weight.transpose(1, 0), None

        from torch.func import grad

        x = torch.randn(2, 3, 16, dtype=torch.float64)
        weight = torch.randn(3, 4, 4, dtype=torch.float64)

        def boring_conv(x, weight, transposed):
            if not transposed:
                return F.conv1d(x, weight).pow(2).sum()
            else:
                return F.conv_transpose1d(x, weight).pow(2).sum()

        def only_convs(x, weight, transposed):
            return onlyConvs.apply(x, weight, transposed).pow(2).sum()

        boring_grads = grad(boring_conv, argnums=(0, 1))(x, weight, True)
        fun_grads = grad(only_convs, argnums=(0, 1))(x, weight, True)

        self.assertEqual(boring_grads, fun_grads)

    def test_convs(self):
        def assert_equivalence(f, expected_forward=None):
            with MemoryCounterMode() as mode:
                f()
            conv_forward_memory = mode.get_memory_counts()["Global"][
                torch.ops.aten.convolution
            ]
            conv_backward_memory = mode.get_memory_counts()["Global"][
                torch.ops.aten.convolution_backward
            ]

            # TODO
            self.assertEqual(conv_forward_memory, 0)
            self.assertEqual(conv_backward_memory, 0)
            if expected_forward is not None:
                self.assertEqual(conv_forward_memory, 0)

        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 2, 2, requires_grad=True)
        assert_equivalence(lambda: F.conv_transpose2d(x, weight).sum().backward(), 0)

        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 1, 1, requires_grad=True)
        assert_equivalence(lambda: F.conv2d(x, weight).sum().backward(), 0)

        for in_channels, out_channels, groups in [
            (1, 1, 1),
            (1, 3, 1),
            (3, 1, 1),
            (3, 7, 1),
            (2, 4, 2),
            (4, 2, 2),
        ]:
            x = torch.rand(1, in_channels, 4, 4, requires_grad=True)
            weight = torch.randn(out_channels, in_channels, 2, 2, requires_grad=True)
            assert_equivalence(lambda: F.conv2d(x, weight).sum().backward())
            transposed_weight = torch.randn(
                in_channels, out_channels, 2, 2, requires_grad=True
            )
            assert_equivalence(
                lambda: F.conv_transpose2d(x, transposed_weight).sum().backward()
            )

    @skipIfNoTorchVision
    def test_module(self):
        resnet18 = torchvision_models.resnet18()
        with MemoryCounterMode(resnet18) as mode:
            a = T(1, 3, 224, 224, requires_grad=True)
            resnet18(a).sum().backward()

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")
        layer1_conv_memory = mode.memory_counts["ResNet.layer1"][
            torch.ops.aten.convolution
        ]
        layer1_conv_back_memory = mode.memory_counts["ResNet.layer1"][
            torch.ops.aten.convolution_backward
        ]
        # TODO
        self.assertExpectedInline(str(layer1_conv_memory), """0""")
        self.assertExpectedInline(str(layer1_conv_back_memory), """0""")

    def test_conv_transpose_loop(self):
        x = torch.rand(1, 4, 30, 2)
        model = torch.nn.ConvTranspose2d(4, 8, (2, 2), stride=2)

        with MemoryCounterMode() as mode:
            for i in range(50):
                out = model(x)
                out.sum().backward()
        # TODO
        self.assertExpectedInline(str(mode.get_total_memory()), """0""")

    def test_custom(self):
        mode = MemoryCounterMode(
            custom_mapping={torch.ops.aten.add: lambda *args, out_shape: 5}
        )
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_memory(mode), """5""")

        def count(*args, out_val):
            return out_val.numel()

        count._get_raw = True

        mode = MemoryCounterMode(custom_mapping={torch.ops.aten.add: count})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_memory(mode), """20""")

    def test_noop(self):
        with MemoryCounterMode() as mode:
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
        or not PLATFORM_SUPPORTS_CUDNN_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_sdpa(self):
        batch_size = 4
        n_heads = 8
        seq_len_q = 128
        seq_len_k = 256
        head_dim = 64
        head_dim_v = 64
        dtype = torch.float16

        torch.manual_seed(0)

        def get_memory(
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
            backend,
            with_backward=False,
        ):
            query = torch.randn(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.randn(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.randn(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim_v,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )

            if backend == "math":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "flash":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "mem_efficient":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                )
            elif backend == "cudnn":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=True,
                )

            mode = MemoryCounterMode()
            with backend, mode:
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0, is_causal=True
                )
                if with_backward:
                    out.sum().backward()
            return int(get_total_memory(mode))

        # Sets seq_len_q == seq_len_k and dim_q == dim_v
        run_uniform_memory = functools.partial(
            get_memory,
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_q,
            head_dim,
            head_dim,
            dtype,
        )

        memory = [
            run_uniform_memory(backend, with_backward=False)
            for backend in ["math", "flash", "mem_efficient", "cudnn"]
        ]
        memory_fw_math, memory_fw_flash, memory_fw_efficient, memory_fw_cudnn = memory
        # TODO
        self.assertEqual(memory_fw_math, 0)
        self.assertEqual(memory_fw_flash, 0)
        self.assertEqual(memory_fw_efficient, 0)
        self.assertEqual(memory_fw_cudnn, 0)

        memory = [
            run_uniform_memory(backend, with_backward=True)
            for backend in ["math", "flash", "mem_efficient", "cudnn"]
        ]
        (
            memory_fw_bw_math,
            memory_fw_bw_flash,
            memory_fw_bw_efficient,
            memory_fw_bw_cudnn,
        ) = memory
        # TODO
        self.assertEqual(memory_fw_bw_math, 0)
        self.assertEqual(memory_fw_bw_flash, 0)
        self.assertEqual(memory_fw_bw_efficient, 0)
        self.assertEqual(memory_fw_bw_cudnn, 0)

        run_nonuniform_memory = functools.partial(
            get_memory,
            batch_size,
            n_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
        )
        # Flash does not support non-uniform attention, i.e. seq_len_q != seq_len_k or dim_q != dim_v"
        non_uniform_backends = ["math", "mem_efficient"]
        memory = [
            run_nonuniform_memory(backend, with_backward=False)
            for backend in non_uniform_backends
        ]
        memory_fw_math, memory_fw_efficient = memory
        # TODO
        self.assertEqual(memory_fw_math, 0)
        self.assertEqual(memory_fw_efficient, 0)

        memory = [
            run_nonuniform_memory(backend, with_backward=True)
            for backend in non_uniform_backends
        ]
        memory_fw_bw_math, memory_fw_bw_efficient = memory
        # TODO
        self.assertEqual(memory_fw_bw_math, 0)
        self.assertEqual(memory_fw_bw_efficient, 0)

    @skipIfRocm  # Nested tensor
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_sdpa_nested_tensor(self):
        def get_memory(q, k, v, backend, with_backward=False):
            mode = MemoryCounterMode()

            if backend == "math":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "flash":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                    enable_cudnn=False,
                )
            elif backend == "mem_efficient":
                backend = torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                )

            with backend, mode:
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0, is_causal=True
                )
                if with_backward:
                    if out.is_nested:
                        out.values().sum().backward()
                    else:
                        out.sum().backward()

            return int(get_total_memory(mode))

        def get_nested_inputs(
            batch_size,
            n_heads,
            max_seq_len_q,
            max_seq_len_k,
            head_dim,
            head_dim_v,
            dtype,
        ):
            q_lengths = torch.tensor(
                [
                    max_seq_len_q // 4,
                    max_seq_len_q // 4 * 2,
                    max_seq_len_q // 4 * 3,
                    max_seq_len_q // 4 * 4,
                ]
            )
            k_lengths = torch.tensor(
                [
                    max_seq_len_k // 4,
                    max_seq_len_k // 4 * 2,
                    max_seq_len_k // 4 * 3,
                    max_seq_len_k // 4 * 4,
                ]
            )
            q_offsets, k_offsets = (
                torch.cat((torch.tensor([0]), torch.cumsum(lengths, dim=0))).cuda()
                for lengths in (q_lengths, k_lengths)
            )
            q_values = torch.randn(
                q_offsets[-1],
                head_dim * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )
            k_values = torch.randn(
                k_offsets[-1],
                head_dim * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )
            v_values = torch.randn(
                k_offsets[-1],
                head_dim_v * n_heads,
                dtype=dtype,
                requires_grad=True,
                device="cuda",
            )

            q = torch.nested.nested_tensor_from_jagged(q_values, q_offsets)
            k = torch.nested.nested_tensor_from_jagged(k_values, k_offsets)
            v = torch.nested.nested_tensor_from_jagged(v_values, k_offsets)

            q = q.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, n_heads, head_dim_v).transpose(1, 2)

            return q, k, v

        def get_dense_memory(q, k, v, backend, with_backward=False):
            def split_tensor(x):
                return (
                    y.unsqueeze(0).transpose(1, 2).detach().requires_grad_(True)
                    for y in x.transpose(1, 2).unbind(0)
                )

            q_tensors = split_tensor(q)
            k_tensors = split_tensor(k)
            v_tensors = split_tensor(v)

            memory = 0
            for q_i, k_i, v_i in zip(q_tensors, k_tensors, v_tensors):
                memory += get_memory(
                    q_i, k_i, v_i, backend=backend, with_backward=with_backward
                )

            return memory

        uniform_config = {
            "batch_size": 4,
            "n_heads": 8,
            "max_seq_len_q": 128,
            "max_seq_len_k": 128,
            "head_dim": 64,
            "head_dim_v": 64,
            "dtype": torch.float16,
        }

        # max_seq_len_q != max_seq_len_k doesn't work for flash attention with dense tensors.
        differing_config = {
            "batch_size": 4,
            "n_heads": 8,
            "max_seq_len_q": 128,
            "max_seq_len_k": 256,
            "head_dim": 64,
            "head_dim_v": 64,
            "dtype": torch.float16,
        }

        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=False,
            ),
            get_memory(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=False,
            ),
        )
        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=False,
            ),
            get_memory(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=False,
            ),
        )
        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=False,
            ),
            get_memory(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=False,
            ),
        )

        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=True,
            ),
            get_memory(
                *get_nested_inputs(**uniform_config),
                backend="flash",
                with_backward=True,
            ),
        )
        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=True,
            ),
            get_memory(
                *get_nested_inputs(**uniform_config),
                backend="mem_efficient",
                with_backward=True,
            ),
        )
        self.assertEqual(
            get_dense_memory(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=True,
            ),
            get_memory(
                *get_nested_inputs(**differing_config),
                backend="mem_efficient",
                with_backward=True,
            ),
        )

    @skipIfRocm  # Nested tensor
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_nested_attention_fake_tensors(self):
        x = torch.randn(123, 4, 16, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor([0, 30, 60, 90, 123], device="cuda")
        max_seqlen = 40
        with FakeTensorMode() as fake_mode:
            fake_x = fake_mode.from_tensor(x)
            fake_offsets = fake_mode.from_tensor(offsets)

            with MemoryCounterMode() as fake_memory_counter_mode:
                torch.ops.aten._flash_attention_forward(
                    fake_x,
                    fake_x,
                    fake_x,
                    fake_offsets,
                    fake_offsets,
                    max_seqlen,
                    max_seqlen,
                    0.0,
                    False,
                    False,
                )

        dense_x = torch.randn(
            4, 40, 4, 16, dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)

        with MemoryCounterMode() as real_memory_counter_mode:
            torch.ops.aten._flash_attention_forward(
                dense_x,
                dense_x,
                dense_x,
                None,
                None,
                max_seqlen,
                max_seqlen,
                0.0,
                False,
                False,
            )

        self.assertEqual(
            int(get_total_memory(fake_memory_counter_mode)),
            int(get_total_memory(real_memory_counter_mode)),
        )

    def test_addmm_out(self):
        def f(x):
            y = torch.zeros(10, 10)
            return torch.mm(x, x, out=y)

        with MemoryCounterMode() as mode:
            f(torch.randn(10, 10))

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")

    def test_hook_registration(self):
        model = torch.nn.Linear(100, 100)
        x = torch.randn(3, 100)

        with MemoryCounterMode() as mode:
            self.assertEqual(len(torch.nn.modules.module._global_forward_pre_hooks), 1)
            self.assertEqual(len(torch.nn.modules.module._global_forward_hooks), 1)
            model(x).sum().backward()

        self.assertEqual(len(torch.nn.modules.module._global_forward_pre_hooks), 0)
        self.assertEqual(len(torch.nn.modules.module._global_forward_hooks), 0)

    def test_pytrees(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                return {"a": torch.mm(x, x)}

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = Foo()

            def forward(self, x):
                return self.b(self.a(x))

        mod = Mod()
        with MemoryCounterMode() as mode:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
        # TODO
        self.assertExpectedInline(
            (mode.memory_counts["Mod"][torch.ops.aten.mm]), """0"""
        )

        class Mod2(torch.nn.Module):
            def forward(self, x):
                return (torch.mm(x, x),)

        mod = Mod2()
        with MemoryCounterMode() as mode:
            mod(torch.randn(10, 10, requires_grad=True))[0].sum().backward()
        # Placeholder function returns 0
        self.assertExpectedInline(
            (mode.memory_counts["Mod2"][torch.ops.aten.mm]), """0"""
        )

    def test_warning(self):
        mod = torch.nn.Linear(2, 2)
        with self.assertWarnsRegex(UserWarning, "not needed"):
            MemoryCounterMode(mod)

    def test_custom_op(self):
        from torch.utils.memory_counter import (
            MemoryCounterMode,
            register_memory_formula,
        )

        @torch.library.custom_op("mylib::foo", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        called = 0

        with self.assertRaisesRegex(
            ValueError, "expected each target to be OpOverloadPacket"
        ):
            register_memory_formula(torch.ops.mylib.foo.default)(lambda x: x)

        @register_memory_formula(torch.ops.mylib.foo)
        def formula(*args, **kwargs):
            nonlocal called
            called += 1
            return 9001

        x = torch.randn(3)
        with MemoryCounterMode(display=False) as mode:
            y = foo(x)

        self.assertEqual(called, 1)
        self.assertExpectedInline(get_total_memory(mode), """9001""")

    @skipIfNoTorchVision
    def test_inference_mode(self):
        def get_memory(model):
            with MemoryCounterMode(model) as mode:
                a = T(1, 3, 224, 224)
                model(a).sum()
            return mode

        resnet18 = torchvision_models.resnet18()

        mode_standard = get_memory(resnet18)

        with torch.inference_mode():
            mode_inference = get_memory(resnet18)

        self.assertEqual(
            get_total_memory(mode_standard), get_total_memory(mode_inference)
        )

        layer1_conv_memory_standard = mode_standard.memory_counts["ResNet.layer1"][
            torch.ops.aten.convolution
        ]
        layer1_conv_memory_inference = mode_inference.memory_counts["ResNet.layer1"][
            torch.ops.aten.convolution
        ]
        self.assertEqual(layer1_conv_memory_standard, layer1_conv_memory_inference)

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm(self):
        unittest.assertFalse(True, "Make sure this is running in CI")
        dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
        with MemoryCounterMode() as mode:
            torch._scaled_mm(
                torch.randn((3 * 16, 5 * 16), device="cuda").to(dtype),
                torch.randn((7 * 16, 5 * 16), device="cuda").to(dtype).t(),
                scale_a=torch.ones((), device="cuda"),
                scale_b=torch.ones((), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        # TODO
        self.assertExpectedInline(get_total_memory(mode), """0""")


if __name__ == "__main__":
    run_tests()
