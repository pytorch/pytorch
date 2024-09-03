# Owner(s): ["module: inductor"]

from typing import List

import torch
import torch._inductor.config as inductor_config
from functorch import make_fx
from torch import Tensor
from torch._dynamo.utils import counters
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.logging_utils import logs_to_string


aten = torch.ops.aten


const = torch.tensor(0.0)
device = GPU_TYPE


def num_reinplacing_failures():
    return counters["inductor"]["possibly_missed_reinplacing_opportunities"]


@torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
def sin(x: torch.Tensor, result: torch.Tensor) -> None:
    result.copy_(x.sin())


@torch.library.custom_op("_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"})
def sin_cos(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> None:
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())


if HAS_GPU:
    import triton
    import triton.language as tl

    @triton.jit
    def sin_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = tl.sin(x)
        tl.store(out_ptr + offsets, output, mask=mask)

    def sin_triton(x, out):
        n_elements = x.numel()
        sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

else:

    def sin_triton(x, out):
        return


@torch.library.custom_op("test_view::boo", mutates_args={"x"})
def boo(x: torch.Tensor) -> None:
    x.sin_()


class TestReinplacingPassCorrectness(InductorTestCase):
    def setUp(self):
        counters.clear()
        return super().setUp()

    def _test(self, f):
        nf = torch.compile(f)
        inp = (
            torch.randn(4, device=device),
            torch.ones(2, device=device, dtype=torch.int),
        )
        inp2 = (inp[0].clone(), inp[1].clone())
        self.assertEqual(f(*inp), nf(*inp2))
        self.assertEqual(inp, inp2)

    def test_dont_modify_live(self):
        def f(x, y):
            x = x.cos()
            x2 = x.index_put((y,), const)
            return x2, x

        self._test(f)

    def test_dont_modify_view_of_live(self):
        def f(x, y):
            x = x.cos()
            x2 = aten.alias(x)
            x2 = x2.index_put((y,), const)
            y = x2 + x.cos()
            return y

        self._test(f)

    def test_dont_modify_input(self):
        def f(x, y):
            return x.index_put((y,), const)

        self._test(f)

    def test_should_modify_inner(self):
        def f(x, y):
            x = x.cos()
            x = x.index_put((y,), const)
            return x

        self._test(f)

    def test_should_modify_input(self):
        def f(x, y):
            x = x.index_put_((y,), const)
            return x

        self._test(f)

    def test_counters_functionalize_old(self):
        counters.clear()

        def f(x):
            out = torch.empty_like(x)
            _, new_out = auto_functionalized(sin._opoverload, x=x, result=out)
            y = out * new_out
            return new_out, y

        x = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x)
        reinplace_inplaceable_ops_core(gm.graph)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)

    def test_counters_functionalize_v2(self):
        counters.clear()

        def f(x):
            out = torch.empty_like(x)
            _, new_out = auto_functionalized_v2(
                sin._opoverload,
                x=x,
                _result_base_index=0,
                _result_size=(3,),
                _result_stride=(1,),
                _result_storage_offset=0,
                _all_bases=[out],
            )
            y = out * new_out
            return new_out, y

        x = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x)
        reinplace_inplaceable_ops_core(gm.graph)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)

    def get_not_inplaced_count(self, graph):
        counter = 0
        auto_functionalized_found = False
        for node in graph.nodes:
            if (node.target == torch.ops.higher_order.auto_functionalized) or (
                node.target == torch.ops.higher_order.auto_functionalized_v2
            ):
                auto_functionalized_found = True
                counter += len(node.meta["only_clone_these_tensors"])
        assert auto_functionalized_found
        return counter

    def test_view_inplaced_functionalize_v2(self):
        def f(arg0_1):
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return ()

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

    # introduce a view another_view that is used `after` the copy
    def test_view_inplaced2_functionalize_v2(self):
        def f(arg0_1):
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

    # introduce a view another_view that is used `before` the copy
    def test_views_not_inplaced_functionalize_v2(self):
        def f(arg0_1):
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            use_another_view = another_view * 10
            copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return use_another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    # a view over input without copy node, inplace not allowed
    def test_views_not_inplaced2_functionalize_v2(self):
        def f(arg0_1):
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            return

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    # no copy nodes, view over local, with a use for another view
    def test_views_not_inplaced3_functionalize_v2(self):
        def f(arg0_1):
            a = torch.ones(10)
            another_view = a[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(),
                _x_stride=(),
                _x_storage_offset=0,
                _all_bases=[a],
            )
            getitem_1 = auto_functionalized[1]
            return another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    def test_multi_output_intermediate(self):
        for requires_grad in [False, True]:
            for enable_v2 in [False, True]:
                with inductor_config.patch(
                    {"enable_auto_functionalized_v2": enable_v2}
                ):
                    counters.clear()

                    def f(x):
                        out1 = torch.empty_like(x)
                        out2 = torch.empty_like(x)
                        sin_cos(x, out1, out2)
                        return out1, out2, x**2

                    x = torch.randn(3, device=device, requires_grad=requires_grad)
                    res1, res2, _ = torch.compile(f)(x)
                    self.assertEqual(res1, x.sin())
                    self.assertEqual(res2, x.cos())
                    self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_mutations(self):
        counters.clear()

        def f(x, out):
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        out = torch.randn(3, device=device)
        result = torch.compile(f)(x, out)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(result, out)
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_intermediate(self):
        counters.clear()

        def f(x):
            out = torch.empty_like(x)
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        result = torch.compile(f)(x)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_lists_functionalize_v2(self):
        with inductor_config.patch({"enable_auto_functionalized_v2": True}):

            @torch.library.custom_op("mylib::mutate_op", mutates_args={"y"})
            def mutate_op(y: List[Tensor]) -> None:
                y[0].add_(2)
                y[1].add_(3)

            @torch.compile(fullgraph=True, dynamic=False, backend="inductor")
            def f(b):
                mutate_op([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4], device=device)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(x1)
            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # We can inplace the base y. no clones emitted.
            self.assertEqual(num_reinplacing_failures(), 0)
            self.assertEqual(post_grad_graphs.count("aten.clone"), 0)

    def test_lists_old_functionalize(self):
        with inductor_config.patch({"enable_auto_functionalized_v2": False}):

            @torch.library.custom_op("mylib::mutate_op", mutates_args={"y"})
            def mutate_op(y: List[Tensor]) -> None:
                y[0].add_(2)
                y[1].add_(3)

            @torch.compile(fullgraph=True, dynamic=False, backend="inductor")
            def f(b):
                mutate_op([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4], device=device)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(x1)
            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # Can't reinplace on views yet (1 for the "entire list" failing to reinplace)
            self.assertEqual(num_reinplacing_failures(), 1)

            # Both list inputs failed to reinplace. So we should have emitted clones for them.
            self.assertEqual(post_grad_graphs.count("aten.clone"), 2)

    @parametrize(
        "factory_op",
        [
            subtest(torch.ones_like, name="ones_like"),
            subtest(torch.empty_like, name="empty_like"),
        ],
    )
    @parametrize(
        "sin_op",
        [
            subtest(sin, name="sin_op"),
            subtest(sin_triton, name="sin_triton"),
        ],
    )
    def test_partitioner_recomputes_factory(self, factory_op, sin_op):
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = factory_op(x)
                sin_op(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin_op(saved, out)
                return out

        @torch.compile(backend="inductor")
        def f(x):
            return MySin.apply(x)

        x = torch.randn(3, requires_grad=True, device=device)
        y = f(x)
        self.assertEqual(num_reinplacing_failures(), 0)


instantiate_parametrized_tests(TestReinplacingPassCorrectness)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests(needs="filelock")
