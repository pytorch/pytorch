# Owner(s): ["oncall: pt2"]
import random
import unittest
from math import prod

import torch
import torch._functorch.config as config
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils._triton import has_triton
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula


if has_triton():
    # note: if we only import triton in the test, the test fails:
    # def relu_kernel_(inp_ptr, out_ptr, sz, BLOCK_SIZE: tl.constexpr):
    # NameError('tl is not defined')
    import triton
    import triton.language as tl


def compile_with_ac(f, memory_budget):
    return torch.compile(f, backend="aot_eager_decomp_partition")


def get_act_mem(f):
    out = f()
    out.backward()
    start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    out = f()
    cur_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
    act_mem = (cur_mem - start_mem) / (1024 * 1024)
    out.backward()
    return act_mem


def get_bw_flops(f):
    # Normalized so that a 512 square matmul returns 1
    f().backward()
    out = f()
    with FlopCounterMode(display=False) as mode:
        out.backward()
    return mode.get_total_flops() / (512**3 * 2)


def create_pair(B_I, O):
    # results in B_I * O memory, requires B_I * B_I * O flops
    # arithmetic intensity of B_I
    x = torch.randn(B_I * 512, B_I * 512, requires_grad=True)
    w = torch.randn(B_I * 512, O * 512, requires_grad=True)
    return x, w


def get_mem_and_flops(f, memory_budget=None):
    # Returns megabytes rounded to 1 decimal point and FLOPs
    # Note that each value of size (512, 512, torch.float32) is 1 MiB
    torch._dynamo.reset()
    with config.patch(activation_memory_budget=memory_budget):
        if memory_budget is not None:
            f = torch.compile(f, backend="aot_eager_decomp_partition")

        # We round this to nearest 10th of a megabyte.
        return round(get_act_mem(f), 1), get_bw_flops(f)


class MemoryBudgetTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.set_default_device("cuda")

    def test_rematerializes_cheap(self):
        def f(x, w):
            x = x.cos()
            x = torch.mm(x, w)
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        w = torch.randn(512, 512, requires_grad=True)

        def call():
            return f(x, w)

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 1.0)
        mem_10, flops_10 = get_mem_and_flops(call, memory_budget=1.0)
        # Recomputing `.cos()` is not free here.
        self.assertEqual(mem_10, 1.0)
        self.assertEqual(eager_flops, flops_10)
        mem_5, flops_5 = get_mem_and_flops(call, memory_budget=0.5)
        # We can just recompute `x.cos()` here to only depend on the inputs
        self.assertEqual(mem_5, 0.0)
        self.assertEqual(flops_5, eager_flops)

    def test_matmul_even_chain(self):
        def f(x, ws):
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        def call():
            return f(x, ws)

        eager_mem, eager_flops = get_mem_and_flops(call)
        for budget in range(0, 11):
            mem, flops = get_mem_and_flops(call, memory_budget=budget / 10)
            if budget <= 5:
                # We start saving the matmuls
                self.assertEqual(mem, budget)
                self.assertEqual(flops, eager_flops + (5 - budget))
            elif budget < 10:
                # We're only recomputing the `cos` operations
                self.assertEqual(mem, 5.0)
                self.assertEqual(flops, eager_flops)
            elif budget == 10:
                self.assertEqual(mem, 10.0)
                self.assertEqual(flops, eager_flops)

    def test_matmul_uneven_chain(self):
        # This function is constructed so that we are saving one input of size
        # [512, in_dim] for each w
        # In addition, every matmul has a same ratio of compute to "memory
        # saved", so this test is essentially testing our knapsack solving

        def f(x, ws):
            xs = [torch.mm(x, w).cos() for w in ws]
            return sum(x.sum() for x in xs)

        x = torch.randn(512, 512, requires_grad=True)

        def make_weights(w_shapes):
            ws = []
            for idx, dim in enumerate(w_shapes):
                ws.append(torch.randn(512, dim * 512, requires_grad=True))
            return ws

        def make_weights_chain(w_shapes):
            ws = []
            for idx, _ in enumerate(w_shapes):
                old_dim = 512 if idx == 0 else w_shapes[idx - 1] * 512
                new_dim = w_shapes[idx] * 512
                ws.append(torch.randn(old_dim, new_dim, requires_grad=True))
            return ws

        weight_configs = [
            (
                [11, 3, 4, 2],
                [
                    18,  # 11 + 4 + 3
                    17,  # 11 + 4 + 2
                    16,  # 11 + 3 + 2
                    15,  # 11 + 4
                    14,  # 11 + 3
                    13,  # 11 + 2
                    11,  # 11 + 2
                    7,  # 4 + 3
                    6,  # 4 + 2
                    5,  # 3 + 2
                ],
            ),
            (
                [3, 5, 11, 17, 14],
                [
                    42,  # 17 + 14 + 9
                    30,  # 11 + 15 + 5
                    19,  # 11 + 5 + 3
                    8,  # 5 + 3
                    3,  # 3
                ],
            ),
        ]
        random.seed(0)
        random_arr = [random.randint(0, 50) for _ in range(10)]
        exact_sums = []
        for i in range(10):
            random.shuffle(random_arr)
            exact_sums.append(sum(random_arr[:i]))
        weight_configs.append((random_arr, exact_sums))

        for weight_shapes, exact_solves in weight_configs:
            ws = make_weights(weight_shapes)

            def call():
                return f(x, ws)

            eager_mem, eager_flops = get_mem_and_flops(call)
            total_mem = sum(weight_shapes)
            self.assertEqual(eager_mem, sum(weight_shapes))
            for mem_achieved in exact_solves:
                mem, _ = get_mem_and_flops(call, memory_budget=mem_achieved / total_mem)
                self.assertEqual(mem, mem_achieved)

    # needs CUDA, but this test file all needs CUDA.
    @unittest.skipIf(not has_triton(), "test needs triton")
    def test_custom_triton_kernel(self):
        @triton.jit
        def relu_kernel_(inp_ptr, out_ptr, sz, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
            msk = block < sz
            inp = tl.load(inp_ptr + block, mask=msk)
            relu = tl.where(inp < 0, 0, inp)
            tl.store(out_ptr + block, relu, mask=msk)

        @torch._library.triton_op("testac::triton_relu", mutates_args=())
        def triton_relu(x: torch.Tensor) -> torch.Tensor:
            y = torch.empty_like(x)
            sz = y.numel()
            BLOCK_SIZE = 256
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            torch._library.capture_triton(relu_kernel_)[grid](x, y, sz, BLOCK_SIZE)
            return y

        @torch._library.triton_op("testac::triton_relu_backward", mutates_args=())
        def triton_relu_backward(grad_out: torch.Tensor) -> torch.Tensor:
            grad_x = torch.empty_like(grad_out)
            sz = grad_out.numel()
            BLOCK_SIZE = 256
            grid = (triton.cdiv(sz, BLOCK_SIZE),)
            # I know this is wrong, but whatever..
            torch._library.capture_triton(relu_kernel_)[grid](
                grad_out, grad_x, sz, BLOCK_SIZE
            )
            return grad_x

        def _triton_relu_backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
            return triton_relu_backward(grad_out)

        def _triton_relu_setup_context(ctx, inputs, output):
            pass

        triton_relu.register_autograd(
            _triton_relu_backward,
            setup_context=_triton_relu_setup_context,
        )

        @register_flop_formula(
            [torch.ops.testac.triton_relu, torch.ops.testac.triton_relu_backward]
        )
        def triton_relu_flops(inp_shape, *args, **kwargs):
            return prod(inp_shape)

        def f(x, ws):
            x = torch.ops.testac.triton_relu(x)
            for w in ws:
                x = torch.ops.testac.triton_relu(torch.mm(x, w))
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True, device="cuda")
        ws = [
            torch.randn(512, 512, requires_grad=True, device="cuda") for _ in range(5)
        ]

        def call():
            return f(x, ws)

        expected = call()
        for budget in range(0, 11):
            memory_budget = budget / 10
            torch._dynamo.reset()
            with config.patch(activation_memory_budget=memory_budget):
                if memory_budget is not None:
                    f_compile = torch.compile(
                        call, backend="aot_eager_decomp_partition"
                    )

                self.assertEqual(expected, f_compile())

    def test_prioritize_cheaper_matmul(self):
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        x1, w1 = create_pair(1, 4)
        x2, w2 = create_pair(2, 2)

        def call():
            return f([x1, x2], [w1, w2])

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 8)
        self.assertEqual(eager_flops, 24)
        comp_mem, comp_flops = get_mem_and_flops(call, memory_budget=0.5)
        self.assertEqual(comp_mem, 4)
        # We are recomputing x1 @ w1 here!
        self.assertEqual(comp_flops, eager_flops + 4)

    @config.patch(activation_memory_budget_runtime_estimator="profile")
    def test_profile(self):
        def f(x, ws):
            x = x.cos()
            for w in ws:
                x = torch.mm(x, w).cos()
            return x.sum()

        x = torch.randn(512, 512, requires_grad=True)
        ws = [torch.randn(512, 512, requires_grad=True) for _ in range(5)]

        def call():
            return f(x, ws)

        eager_mem, eager_flops = get_mem_and_flops(call)
        mem, flops = get_mem_and_flops(call, memory_budget=0.2)
        # We start saving the matmuls
        self.assertEqual(mem, 2)
        self.assertEqual(flops, eager_flops + 3)

    def test_prioritize_cheaper_matmul2(self):
        def f(xs, ws):
            xs = [torch.mm(x, w).cos() for x, w in zip(xs, ws)]
            return sum(x.sum() for x in xs)

        data = [(4, 4), (6, 2), (2, 6)]
        xs, ws = zip(*[create_pair(a, b) for a, b in data])

        def call():
            return f(xs, ws)

        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 40)
        self.assertEqual(eager_flops, 320)
        mem, flops = get_mem_and_flops(call, memory_budget=28 / eager_mem)
        # Save w1 and w2
        self.assertEqual(mem, 28)
        # We're recomputing w3 (the cheap one!)
        self.assertEqual(flops - eager_flops, 2 * 2 * 6)
        mem, flops = get_mem_and_flops(call, memory_budget=16 / eager_mem)
        # Save w2. Note that even though saving w1 gets us closer to our memory
        # limit, w2 is actually *more* FLOPs than w1!
        self.assertEqual(mem, 12)
        self.assertEqual(flops - eager_flops, 2 * 2 * 6 + 4 * 4 * 4)

    def test_attention_vs_linear(self):
        def f(x, w):
            orig_shape = x.shape
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
            # I know this isn't technically right lol
            x = torch.nn.functional.scaled_dot_product_attention(
                x, x, x, is_causal=False
            ).reshape(*orig_shape)
            x = torch.mm(x, w)
            x = x.cos()
            return x.sum()

        def try_seq_length(S, D, expected_recompute):
            x = torch.randn(S * 512, D * 512, requires_grad=True)
            w = torch.randn(D * 512, D * 512, requires_grad=True)

            def call():
                return f(x, w)

            with FlopCounterMode(display=False) as mode:
                call()
            mm_flops = mode.get_flop_counts()["Global"][torch.ops.aten.mm]
            attn_flops = mode.get_total_flops() - mm_flops
            mm_flops /= 512**3 * 2
            attn_flops /= 512**3 * 2

            eager_mem, eager_flops = get_mem_and_flops(call)
            self.assertEqual(eager_mem, S * D * 2)

            mem, flops = get_mem_and_flops(
                call, memory_budget=0.6
            )  # Force it to recompute one of mm or attn
            self.assertEqual(mem, S * D)
            if expected_recompute == "attn":
                expected_flops = attn_flops
            else:
                expected_flops = mm_flops
            self.assertEqual(flops - eager_flops, expected_flops)

        # General behind this test is that if sequence length * 2 > D, then
        # attention is more expensive than the linear.
        try_seq_length(1, 1, "mm")
        try_seq_length(1, 3, "attn")
        try_seq_length(2, 2, "mm")
        try_seq_length(2, 1, "mm")
        try_seq_length(2, 5, "attn")
        try_seq_length(4, 7, "mm")
        try_seq_length(4, 9, "attn")


if __name__ == "__main__":
    # I'm using the cuda memory allocator to verify memory allocations
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests()
