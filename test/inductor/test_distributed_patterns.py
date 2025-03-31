# Owner(s): ["oncall: pt2"]
import dataclasses
import functools

import torch
import torch.distributed as dist
from torch import nn
from torch._dynamo import compiled_autograd
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import IS_MACOS, skipIfRocm, skipIfXpu
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, requires_gpu


# Fake distributed
WORLD_SIZE = 2


def init_fake_distributed(device="cpu"):
    @torch.no_grad
    def all_gather(t):
        return torch.cat([t] * WORLD_SIZE, 0)

    @torch.no_grad
    def reduce_scatter(t):
        # clone since reduce_scatter input and output should not be aliases.
        return t.narrow(0, 0, t.size(0) // WORLD_SIZE).clone()

    def fw_pre_hook(mod, inp):
        mod.unsharded_weight.untyped_storage().resize_(
            mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
        )
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
            mod.unsharded_weight
        ):
            torch.ops.fsdp.copy_(mod.unsharded_weight, all_gather(mod.sharded_weight))
        mod._parameters["weight"] = mod.unsharded_weight

    # Forward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather

    def fw_post_hook(mod, inp, out):
        mod._parameters["weight"] = mod.sharded_weight
        mod.unsharded_weight.untyped_storage().resize_(0)

    def bw_pre_hook(mod, gO):
        mod.unsharded_weight.untyped_storage().resize_(
            mod.unsharded_weight.nelement() * mod.unsharded_weight.element_size()
        )
        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(
            mod.unsharded_weight
        ):
            torch.ops.fsdp.copy_(mod.unsharded_weight, all_gather(mod.sharded_weight))
        mod._parameters["weight"] = mod.unsharded_weight

    # Backward:
    #   mod.sharded_weight = local_shard (always)
    #   Before:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather
    #   After:
    #     mod.weight = local_shard
    #     mod.unsharded_weight = zero-sized allgather

    def bw_post_hook(mod, gI, gO):
        grad = mod.weight.grad
        new_grad = reduce_scatter(grad)
        mod._parameters["weight"] = mod.sharded_weight
        mod.weight.grad = new_grad
        mod.unsharded_weight.untyped_storage().resize_(0)

    torch.manual_seed(1234)
    m = nn.Linear(20, 10, bias=False, device=device)

    # Mimics eager 1st iteration
    m.sharded_weight = nn.Parameter(reduce_scatter(m.weight))
    m.unsharded_weight = nn.Parameter(all_gather(m.sharded_weight))
    m.unsharded_weight.untyped_storage().resize_(0)

    m.register_full_backward_pre_hook(bw_pre_hook)
    m.register_full_backward_hook(bw_post_hook)
    m.register_forward_pre_hook(fw_pre_hook)
    m.register_forward_hook(fw_post_hook)
    return m, torch.rand(2, 20, requires_grad=True, device=device)


def init_module_bw_hooks(allow_eager):
    def bw_pre_hook(mod, gO):
        assert allow_eager or torch._dynamo.is_compiling()
        assert mod.weight.size() == (10, 10)
        mod.hook_count_pre.add_(1)
        return (torch.sin(gO[0] + 1.2),)

    def bw_post_hook(mod, gI, gO):
        assert allow_eager or torch._dynamo.is_compiling()
        assert mod.weight.size() == (10, 10)
        mod.hook_count_post.add_(1)
        return (torch.sin(gI[0] + 3.4),)

    torch.manual_seed(1234)
    m = nn.Linear(10, 10)
    m.hook_count_pre = torch.tensor(0)
    m.hook_count_post = torch.tensor(0)
    m.register_full_backward_pre_hook(bw_pre_hook)
    m.register_full_backward_hook(bw_post_hook)
    return m, torch.rand(2, 10, requires_grad=True)


def steps(m, inp):
    for _ in range(4):
        out = m(inp)
        out.sum().backward()
    return out


fw_graph = [None]
bw_graph = [None]


def aot_graph_capture_backend(gm, args):
    from functorch.compile import min_cut_rematerialization_partition
    from torch._functorch.aot_autograd import aot_module_simplified

    def fw_compiler(gm, _):
        fw_graph[0] = gm
        return gm

    def bw_compiler(gm, _):
        bw_graph[0] = gm
        return gm

    return aot_module_simplified(
        gm,
        args,
        fw_compiler,
        bw_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )


class DistributedPatternTests(TestCase):
    def test_intermediate_hook_with_closure(self):
        @dataclasses.dataclass
        class CustomObj:
            val: torch.Tensor

        def fn(x, obj):
            y = x.sin()
            closure_var = y + 1
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            z = y.sin()
            return z

        opt = torch.compile(fn, fullgraph=True)

        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_nested_closure(self):
        @dataclasses.dataclass
        class CustomObj:
            val: torch.Tensor

        def fn(x, obj):
            def run():
                y = x.sin()
                closure_var = y + 1
                y.register_hook(lambda grad: grad + obj.val + closure_var)
                z = y.sin()
                return z

            return run()

        opt = torch.compile(fn, fullgraph=True)

        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    @torch.no_grad()
    def _test_storage_resize_zero(self, device):
        @torch.compile(fullgraph=True)
        def fn(x):
            y = torch.sin(x)
            x.untyped_storage().resize_(0)
            return torch.cos(y)

        x = torch.randn(10, device=device)
        expected = torch.cos(torch.sin(x))
        y = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)

    def test_storage_resize_zero_cpu(self):
        self._test_storage_resize_zero("cpu")

    @skipIfRocm
    @requires_gpu()
    def test_storage_resize_zero_gpu(self):
        self._test_storage_resize_zero(GPU_TYPE)

    @torch.no_grad()
    def _test_storage_resize_nonzero(self, device):
        @torch.compile(fullgraph=True)
        def fn(x, out):
            y = torch.sin(x)
            assert out.untyped_storage().size() == 0
            out.untyped_storage().resize_(x.untyped_storage().size())
            out.copy_(y.cos())

        x = torch.randn(10, device=device)
        out = torch.randn(10, device=device)
        expected = torch.cos(torch.sin(x))
        out.untyped_storage().resize_(0)
        fn(x, out)
        self.assertEqual(out.untyped_storage().size(), x.untyped_storage().size())
        self.assertEqual(out, expected)

    def test_storage_resize_nonzero_cpu(self):
        self._test_storage_resize_nonzero("cpu")

    @skipIfRocm
    @requires_gpu()
    def test_storage_resize_nonzero_gpu(self):
        self._test_storage_resize_nonzero(GPU_TYPE)

    @torch.no_grad()
    def test_unsafe_set_version_counter1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(w, x):
            x = x.sin()
            v = w._version
            w.copy_(x + 1)
            torch._C._autograd._unsafe_set_version_counter((w,), (v,))
            return w, v

        for v in (3, 0, 1):
            w1 = torch.randn(16)
            for i in range(v):
                w1.fill_(i)  # bump w1._version
            self.assertEqual(w1._version, v)
            x1 = torch.randn(16)
            w2, v2 = fn(w1, x1)

            self.assertIs(w1, w2)
            self.assertEqual(w1, x1.sin() + 1)
            self.assertEqual(v2, v)
            self.assertEqual(w1._version, v)
            self.assertEqual(cnt.frame_count, 1)

    def test_unsafe_set_version_counter2(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            r = w.sin()
            with torch.no_grad():
                v = w._version
                w.copy_(x)
                torch._C._autograd._unsafe_set_version_counter((w,), (v,))
            return r

        w1 = torch.randn(1, requires_grad=True)
        x1 = torch.randn(1)
        expected_r1 = w1.detach().sin()

        r1 = fn(w1, x1)
        r1.backward()
        self.assertEqual(r1, expected_r1)
        self.assertEqual(w1, x1)
        self.assertEqual(w1.grad, x1.cos())

    @torch.no_grad()
    def test_unsafe_preserve_version_counter1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(w, x):
            x = x.sin()
            with torch.autograd._unsafe_preserve_version_counter(w):
                w.copy_(x + 1)
            return w

        w1 = torch.randn(16).fill_(0).fill_(1)
        x1 = torch.randn(16)
        v1 = w1._version
        w2 = fn(w1, x1)
        v2 = w1._version

        self.assertIs(w1, w2)
        self.assertEqual(w1, x1.sin() + 1)
        self.assertEqual(v1, v2)

    def test_unsafe_preserve_version_counter2(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def fn(w, x):
            r = w.sin()
            with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(w):
                w.copy_(x)
            return r

        w1 = torch.randn(1, requires_grad=True)
        x1 = torch.randn(1)
        expected_r1 = w1.detach().sin()

        r1 = fn(w1, x1)
        r1.backward()
        self.assertEqual(r1, expected_r1)
        self.assertEqual(w1, x1)
        self.assertEqual(w1.grad, x1.cos())

    def test_module_backward_hooks_eager(self):
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(False)
        fw_cnt = CompileCounter()
        bw_cnt = CompileCounter()
        with compiled_autograd._enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            m2 = torch.compile(m2, backend=fw_cnt, fullgraph=True)
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

        self.assertEqual(fw_cnt.frame_count, 1)
        self.assertEqual(fw_cnt.op_count, 5)
        self.assertEqual(bw_cnt.frame_count, 2)  # grad=None and grad!=None
        self.assertEqual(
            bw_cnt.op_count, 84
        )  # Number of ops in the Dynamo-produced graphs

    def test_module_backward_hooks_aot(self):
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(True)
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        with compiled_autograd._enable(lambda gm: gm):
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

    def test_module_backward_hooks_inductor(self):
        m1, inp1 = init_module_bw_hooks(True)
        out1 = steps(m1, inp1)

        m2, inp2 = init_module_bw_hooks(False)
        m2 = torch.compile(m2, fullgraph=True)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(m2, inp2)

        self.assertEqual(m1.hook_count_pre, m2.hook_count_pre)
        self.assertEqual(m1.hook_count_post, m2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(m1.weight.grad, m2.weight.grad)
        self.assertEqual(m1.bias.grad, m2.bias.grad)

    def test_module_backward_hooks_multi_layers(self):
        a1, inp1 = init_module_bw_hooks(True)
        b1, _ = init_module_bw_hooks(True)
        out1 = steps(torch.nn.Sequential(a1, b1), inp1)

        a2, inp2 = init_module_bw_hooks(False)
        b2, _ = init_module_bw_hooks(False)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(
                torch.compile(torch.nn.Sequential(a2, b2), fullgraph=True), inp2
            )

        self.assertEqual(a1.hook_count_pre, a2.hook_count_pre)
        self.assertEqual(a1.hook_count_post, a2.hook_count_post)
        self.assertEqual(b1.hook_count_pre, b2.hook_count_pre)
        self.assertEqual(b1.hook_count_post, b2.hook_count_post)
        self.assertEqual(out1, out2)
        self.assertEqual(inp1.grad, inp2.grad)
        self.assertEqual(a1.weight.grad, a2.weight.grad)
        self.assertEqual(a1.bias.grad, a2.bias.grad)
        self.assertEqual(b1.weight.grad, b2.weight.grad)
        self.assertEqual(b1.bias.grad, b2.bias.grad)

    # TODO(jansel): support bw hooks with graph break

    def _assert_same_grad(self, a, b):
        self.assertEqual(type(a), type(b))
        self.assertEqual(a, b)
        self.assertEqual(a.grad, b.grad)
        self.assertEqual(a.requires_grad, b.requires_grad)

    def test_nn_param_return1(self):
        def fn(x):
            p = torch.nn.Parameter(x)
            return p, p.sin()

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        r1.sum().backward()
        p2, r2 = opt(x2)
        r2.sum().backward()
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    def test_nn_param_return2(self):
        def fn(x):
            p = torch.nn.Parameter(x, requires_grad=False)
            return p, x + 1

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        p2, r2 = opt(x2)
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    def test_nn_param_return3(self):
        def fn(x):
            p = torch.nn.Parameter(x + 123)
            return p, p.sin()

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        r1.sum().backward()
        p2, r2 = opt(x2)
        r2.sum().backward()
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    def test_nn_param_return4(self):
        def fn(x):
            p = torch.nn.Parameter(x + 123, requires_grad=False)
            return p, x + 1

        opt = torch.compile(fn, fullgraph=True)
        x1 = torch.randn(16)
        x2 = x1.clone()

        p1, r1 = fn(x1)
        p2, r2 = opt(x2)
        self._assert_same_grad(r1, r2)
        self._assert_same_grad(p1, p2)

    @torch._functorch.config.patch(recompute_views=True)
    def test_fake_distributed_aot_eager(self):
        m1, inp1 = init_fake_distributed()
        out1 = steps(m1, inp1)

        m2, inp2 = init_fake_distributed()
        m2 = torch.compile(m2, backend="aot_eager", fullgraph=True)
        bw_cnt = CompileCounter()
        with compiled_autograd._enable(torch.compile(backend=bw_cnt, fullgraph=True)):
            out2 = steps(m2, inp2)

        self._assert_same_grad(m1.weight, m2.weight)
        self._assert_same_grad(inp1, inp2)
        self._assert_same_grad(out1, out2)
        # Recompile on grad==None/grad!=None
        self.assertEqual(bw_cnt.frame_count, 2)

    @skipIfRocm
    @skipIfXpu
    @requires_gpu()
    @torch._functorch.config.patch(recompute_views=True)
    def test_fake_distributed_inductor(self):
        m1, inp1 = init_fake_distributed(GPU_TYPE)
        out1 = steps(m1, inp1)

        m2, inp2 = init_fake_distributed(GPU_TYPE)
        m2 = torch.compile(m2, fullgraph=True)
        with compiled_autograd._enable(torch.compile(fullgraph=True)):
            out2 = steps(m2, inp2)

        self._assert_same_grad(m1.weight, m2.weight)
        self._assert_same_grad(inp1, inp2)
        self._assert_same_grad(out1, out2)


class FakeDistributedTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super(
            type(self), self
        ).setUp()  # use explicit params for compiled autograd test wrapping
        fake_store = FakeStore()
        dist.init_process_group(
            "fake", store=fake_store, rank=0, world_size=self.world_size
        )

    def tearDown(self):
        super(
            type(self), self
        ).tearDown()  # use explicit params for compiled autograd test wrapping
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def world_size(self) -> int:
        return 2

    @skipIfRocm
    @skipIfXpu
    @requires_gpu()
    @torch._functorch.config.patch(unsafe_allow_optimization_of_collectives=True)
    def test_partitioner_saves_primal_instead_of_allgathered_primal(self):
        import torch.distributed._functional_collectives as funcol

        def scale(
            t: torch.Tensor, amax_t: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            max_v = torch.finfo(torch.float8_e4m3fn).max
            scale_t = torch.clamp(amax_t.float(), min=1e-12) / max_v
            t_fp8 = (t / scale_t).to(torch.float8_e4m3fn)
            return t_fp8, scale_t

        def fp8_matmul(
            first: torch.Tensor,
            amax_first: torch.Tensor,
            second_t: torch.Tensor,
            amax_second_t: torch.Tensor,
            parallel: str,
        ) -> torch.Tensor:
            tp_group = torch.distributed.group.WORLD
            first_fp8, scale_first = scale(first, amax_first)
            second_t_fp8, scale_second_t = scale(second_t, amax_second_t)

            if parallel == "col":
                first_fp8 = funcol.all_gather_tensor(
                    first_fp8, gather_dim=0, group=tp_group
                )
                scale_first = funcol.all_gather_tensor(
                    scale_first, gather_dim=0, group=tp_group
                )

            res = torch._scaled_mm(
                first_fp8,
                second_t_fp8.t(),
                scale_a=scale_first,
                scale_b=scale_second_t.t(),
                out_dtype=torch.bfloat16,
            )

            if parallel == "row":
                res = funcol.reduce_scatter_tensor(
                    res, "sum", scatter_dim=0, group=tp_group
                )

            return res

        REVERSE = {"col": "row", "row": "col"}

        @torch.compiler.allow_in_graph
        class Fp8LinearFn(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: torch.autograd.function.FunctionCtx,
                a: torch.Tensor,
                b_t: torch.Tensor,
                parallel: str,
            ) -> torch.Tensor:
                amax_a = a.abs().amax(dim=-1, keepdim=True)
                amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
                out = fp8_matmul(a, amax_a, b_t, amax_b_t, parallel)
                ctx.save_for_backward(a, b_t, amax_b_t)
                ctx.parallel = parallel
                return out

            @staticmethod
            def backward(
                ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, None]:
                a: torch.Tensor
                b_t: torch.Tensor
                amax_b_t: torch.Tensor
                a, b_t, amax_b_t = ctx.saved_tensors
                parallel = REVERSE[ctx.parallel]

                # Workaround for https://github.com/pytorch/pytorch/issues/141881.
                b_t = b_t + grad_out[0, :, None]

                b = b_t.t().contiguous()
                amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
                amax_b = amax_b_t.t().amax(dim=-1, keepdim=True)
                amax_b = amax_b.repeat_interleave(
                    b.shape[0] // amax_b.shape[0], dim=0, output_size=b.shape[0]
                )
                grad_a = fp8_matmul(grad_out, amax_grad_out, b, amax_b, parallel)

                tp_group = torch.distributed.group.WORLD
                if parallel == "col":
                    grad_out = funcol.all_gather_tensor(
                        grad_out, gather_dim=0, group=tp_group
                    )
                if parallel == "row":
                    a = funcol.all_gather_tensor(a, gather_dim=0, group=tp_group)
                grad_b = grad_out.t() @ a

                return grad_a, grad_b, None

        @torch.compile(backend=aot_graph_capture_backend)
        def ffn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
            x = Fp8LinearFn.apply(x, w1, "col")
            x = torch.nn.functional.relu(x)
            x = Fp8LinearFn.apply(x, w2, "row")
            return x

        in_ = torch.randn(
            (3072 // WORLD_SIZE, 4096), device="cuda", dtype=torch.bfloat16
        )
        w1 = torch.randn(
            (8192 // WORLD_SIZE, 4096),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        w2 = torch.randn(
            (4096, 8192 // WORLD_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        out = ffn(in_, w1, w2)
        self.assertExpectedInline(
            str(fw_graph[0].code.strip()),
            """\
def forward(self, primals_1, primals_2, primals_3):
    abs_1 = torch.ops.aten.abs.default(primals_1)
    amax = torch.ops.aten.amax.default(abs_1, [-1], True);  abs_1 = None
    abs_2 = torch.ops.aten.abs.default(primals_2)
    amax_1 = torch.ops.aten.amax.default(abs_2, [-1], True);  abs_2 = None
    _to_copy = torch.ops.aten._to_copy.default(amax, dtype = torch.float32);  amax = None
    clamp = torch.ops.aten.clamp.default(_to_copy, 1e-12);  _to_copy = None
    div = torch.ops.aten.div.Tensor(clamp, 448.0);  clamp = None
    div_1 = torch.ops.aten.div.Tensor(primals_1, div)
    _to_copy_1 = torch.ops.aten._to_copy.default(div_1, dtype = torch.float8_e4m3fn);  div_1 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(amax_1, dtype = torch.float32);  amax_1 = None
    clamp_1 = torch.ops.aten.clamp.default(_to_copy_2, 1e-12);  _to_copy_2 = None
    div_2 = torch.ops.aten.div.Tensor(clamp_1, 448.0);  clamp_1 = None
    div_3 = torch.ops.aten.div.Tensor(primals_2, div_2);  primals_2 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(div_3, dtype = torch.float8_e4m3fn);  div_3 = None
    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(_to_copy_1, 2, '0');  _to_copy_1 = None
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(div, 2, '0');  div = None
    wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
    t = torch.ops.aten.t.default(_to_copy_3);  _to_copy_3 = None
    t_1 = torch.ops.aten.t.default(div_2);  div_2 = None
    _scaled_mm = torch.ops.aten._scaled_mm.default(wait_tensor, t, wait_tensor_1, t_1, None, None, torch.bfloat16);  wait_tensor = t = wait_tensor_1 = t_1 = None
    relu = torch.ops.aten.relu.default(_scaled_mm);  _scaled_mm = None
    abs_3 = torch.ops.aten.abs.default(relu)
    amax_2 = torch.ops.aten.amax.default(abs_3, [-1], True);  abs_3 = None
    abs_4 = torch.ops.aten.abs.default(primals_3)
    amax_3 = torch.ops.aten.amax.default(abs_4, [-1], True);  abs_4 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(amax_2, dtype = torch.float32);  amax_2 = None
    clamp_2 = torch.ops.aten.clamp.default(_to_copy_4, 1e-12);  _to_copy_4 = None
    div_4 = torch.ops.aten.div.Tensor(clamp_2, 448.0);  clamp_2 = None
    div_5 = torch.ops.aten.div.Tensor(relu, div_4)
    _to_copy_5 = torch.ops.aten._to_copy.default(div_5, dtype = torch.float8_e4m3fn);  div_5 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(amax_3, dtype = torch.float32)
    clamp_3 = torch.ops.aten.clamp.default(_to_copy_6, 1e-12);  _to_copy_6 = None
    div_6 = torch.ops.aten.div.Tensor(clamp_3, 448.0);  clamp_3 = None
    div_7 = torch.ops.aten.div.Tensor(primals_3, div_6)
    _to_copy_7 = torch.ops.aten._to_copy.default(div_7, dtype = torch.float8_e4m3fn);  div_7 = None
    t_2 = torch.ops.aten.t.default(_to_copy_7);  _to_copy_7 = None
    t_3 = torch.ops.aten.t.default(div_6);  div_6 = None
    _scaled_mm_1 = torch.ops.aten._scaled_mm.default(_to_copy_5, t_2, div_4, t_3, None, None, torch.bfloat16);  _to_copy_5 = t_2 = div_4 = t_3 = None
    reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(_scaled_mm_1, 'sum', 2, '0');  _scaled_mm_1 = None
    wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
    t_5 = torch.ops.aten.t.default(amax_3);  amax_3 = None
    amax_5 = torch.ops.aten.amax.default(t_5, [-1], True);  t_5 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(amax_5, 1);  amax_5 = None
    return (wait_tensor_2, primals_1, primals_3, relu, unsqueeze_1)"""
            "",
        )


if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
