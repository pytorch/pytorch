# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import torch
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    num_profiled_runs,
    ProfilingMode,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from typing import List, Optional, Tuple

from torch.testing import FileCheck
from torch.testing._internal.jit_utils import (
    disable_autodiff_subgraph_inlining,
    JitTestCase,
)


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


@unittest.skipIf(
    GRAPH_EXECUTOR == ProfilingMode.SIMPLE, "Simple Executor doesn't support gradients"
)
class TestAutodiffSubgraphSlicing(JitTestCase):
    # TODO: It is better if we can test directly on graphs instead of the current
    # end-to-end fashion.
    def _perform_ad_subgraph_slicing(self, fn, *input_sizes):
        with disable_autodiff_subgraph_inlining():
            with enable_profiling_mode_for_profiling_tests():
                ge = torch.jit.script(fn)
                inputs = [torch.randn(size, requires_grad=True) for size in input_sizes]
                ge(*inputs, profile_and_replay=True)
                return ge.graph_for(*inputs)

    def assertGraphSize(self, graph, size):
        nodes = list(
            filter(
                lambda n: (
                    n.kind() != "prim::BailOut"
                    and n.kind() != "prim::BailoutTemplate"
                    and n.kind() != "prim::TypeCheck"
                    and n.kind() != "prim::RequiresGradCheck"
                ),
                graph.nodes(),
            )
        )
        self.assertEqual(len(list(nodes)), size)

    def test_chunk_constant_script_ad(self):
        @torch.jit.script
        def func(x):
            x1, x2 = torch.chunk(x, 2)
            return (x1, x2)

        input = torch.rand(6, 10).requires_grad_()
        with disable_autodiff_subgraph_inlining():
            with enable_profiling_mode_for_profiling_tests():
                func(input, profile_and_replay=True)
                FileCheck().check_not("prim::DifferentiableGraph").run(
                    func.graph_for(input)
                )

    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.PROFILING,
        "This threshold is only valid for Profiling Executor",
    )
    def test_diff_graph_inline_threshold(self):
        with enable_profiling_mode_for_profiling_tests():
            NUM_RUNS = 1
            with num_profiled_runs(NUM_RUNS):

                @torch.jit.script
                def foo(x):
                    #  two nodes should be fused
                    #  see https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/runtime/graph_executor_impl.h#L49
                    return torch.sigmoid(torch.sigmoid(x))

                @torch.jit.script
                def bar(x):
                    #  two nodes should NOT be fused
                    return torch.sigmoid(x)

                input = torch.rand([4, 4], requires_grad=True)
                foo(input)
                foo(input)

                bar(input)
                bar(input)

                self.assertGraphContainsExactly(
                    foo.graph_for(input), "prim::DifferentiableGraph", 1
                )
                self.assertGraphContainsExactly(
                    bar.graph_for(input), "prim::DifferentiableGraph", 0
                )

    def test_bias_as_module_attr(self):
        with enable_profiling_mode_for_profiling_tests():

            class M(torch.nn.Module):
                def __init__(self, has_bias):
                    super().__init__()
                    self.ll = torch.nn.Linear(10, 10, has_bias)

                def forward(self, x, y):
                    return self.ll(x + y) * x + y

            x = torch.rand(10, 10, requires_grad=True)
            no_bias = M(False)
            scripted_no_bias = torch.jit.script(no_bias)
            scripted_no_bias(x, x)
            scripted_no_bias(x, x)
            scripted_no_bias(x, x)
            has_bias = M(True)
            check_against_reference(
                self,
                scripted_no_bias,
                no_bias,
                lambda x: x,
                (
                    x,
                    x,
                ),
                check_types=False,
            )
            scripted_has_bias = torch.jit.script(has_bias)
            scripted_has_bias(x, x)
            scripted_has_bias(x, x)
            scripted_has_bias(x, x)
            check_against_reference(
                self,
                scripted_has_bias,
                has_bias,
                lambda x: x,
                (
                    x,
                    x,
                ),
                check_types=False,
            )

    def test_constructed_bias(self):
        with enable_profiling_mode_for_profiling_tests():

            def method1(x, weight, b1, b2):
                bias = b1 * b2
                return torch.nn.functional.linear(x, weight, bias)

            N = 10
            x = torch.rand(N, N, requires_grad=True)
            weight = torch.rand(N, N, requires_grad=True)
            b1 = torch.rand(N, N, requires_grad=True)
            b2 = torch.rand(N, N, requires_grad=True)
            scripted = self.checkScript(method1, (x, weight, b1, b2))
            # check_types requires last_graph on scripted to be set, so we just skip it
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, b1, b2),
                check_types=False,
            )

    def test_bias_as_arg(self):
        with enable_profiling_mode_for_profiling_tests():

            def method1(x, weight, bias: Optional[torch.Tensor]):
                return torch.nn.functional.linear(x, weight, bias).relu() + 2

            N = 10
            x = torch.rand(N, N, requires_grad=True)
            weight = torch.rand(N, N, requires_grad=True)
            bias = None
            scripted = self.checkScript(method1, (x, weight, bias))
            # check_types requires last_graph on scripted to be set, so we just skip it
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, bias),
                check_types=False,
            )
            bias = torch.rand(N, N, requires_grad=True)
            scripted = self.checkScript(method1, (x, weight, bias))
            # check_types requires last_graph on scripted to be set, so we just skip it
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, bias),
                check_types=False,
            )

    def test_requires_grad_for_tensor_list(self):
        with enable_profiling_mode_for_profiling_tests():
            # output & var_list[0] should have requires_grad set to True
            def func(
                input0: torch.Tensor, input1: torch.Tensor
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                var_list = [input0, input1]
                var = torch.cat(var_list)
                output = var + 1.0
                return output, var_list

            jit_f = torch.jit.script(func)
            input0 = torch.randn((2,), requires_grad=True)
            input1 = torch.randn((2,))
            output_ref = func(input0, input1)
            for _ in range(2):
                output = jit_f(input0, input1)
                assert output_ref[0].requires_grad == output[0].requires_grad
                assert output_ref[1][0].requires_grad == output[1][0].requires_grad
                assert output_ref[1][1].requires_grad == output[1][1].requires_grad

    @unittest.skip(
        "disable until we property handle tensor lists with undefined gradients"
    )
    def test_differentiable_graph_ops_requires_grad(self):
        x = torch.randn(8, 2, dtype=torch.float).requires_grad_()
        y = torch.randn(8, 2, dtype=torch.float)

        def t(x: torch.Tensor, y: torch.Tensor, flag: bool):
            o = x + 1.0
            o1 = torch.relu(o)
            o = y + 1.5
            o2 = torch.relu(o)
            o3 = o1 + o2

            if flag:
                o = o1 + 1.0
                oo1 = torch.relu(o)
                o = o2 + 2.5
                oo2 = torch.relu(o)
                oo3 = oo1 + oo2
            else:
                o = o1 * 1.0
                oo1 = torch.relu(o)
                o = o2 * 2.0
                oo2 = torch.relu(o)
                oo3 = oo1 + oo2

            return o1, o2, o3, oo1, oo2, oo3

        with enable_profiling_mode_for_profiling_tests():
            t_jit = torch.jit.script(t)
            jit_o = t_jit(x, y, False)
            jit_o = t_jit(x, y, False)
            o = t(x, y, False)

            FileCheck().check("prim::DifferentiableGraph").run(
                t_jit.graph_for(x, y, False)
            )
            # validate the differentiableGraphOps are marking proper requires_grad
            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.requires_grad, jit_oo.requires_grad)
                self.assertEqual(oo, jit_oo)
            # one more runs to trigger fusion
            jit_o = t_jit(x, y, False)
            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.dtype, jit_oo.dtype)
                self.assertEqual(oo.requires_grad, jit_oo.requires_grad)
                self.assertEqual(oo, jit_oo)

    @unittest.skipIf(
        GRAPH_EXECUTOR == ProfilingMode.PROFILING,
        "Simple Executor doesn't support gradients",
    )
    def test_prune_grad(self):
        @torch.jit.script
        def t(input, bias):
            return torch.nn.functional.relu(input + bias)

        input = torch.randn(2, 8, requires_grad=True)
        bias = torch.randn(8, requires_grad=False)  # bias does NOT require grad
        NUM_PROFILED_RUNS = 1
        with num_profiled_runs(NUM_PROFILED_RUNS):
            WARMUP = 3  # 2 runs to reach backward + 1 to optimize it
            for _ in range(WARMUP):
                o = t(input, bias)
                o.sum().backward()

            fwd_plan = list(t.get_debug_state().execution_plans.values())[0]
            bwd_graph = list(
                fwd_plan.code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
            tup = next(bwd_graph.outputs())
            self.assertEqual(len(list(tup.node().inputs())), 1)

    def test_simple_merge(self):
        # o --> o
        def fn(x, y, z):
            a = x * y
            b = a * z
            return b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_simple_no_merge(self):
        # o: autodiff supported. x: not autodiff supported.
        # o --> x
        def fn(x, y, z):
            a = x * y
            b = torch.zeros([abs(int(y))])
            return a, b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)
        g_str = str(graph)
        FileCheck().check("aten::Int").check("aten::zeros").check_not("aten::mul").run(
            g_str[0 : g_str.find("return")]
        )
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_does_not_merge_unrelated(self):
        # o  o
        def fn(w, x, y, z):
            a = x * y
            b = w * z
            return a, b

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_merges_without_cycles(self):
        # o --> o --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = a * y
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_merges_dense(self):
        #   o      o
        #   |\    /|
        #   | \  / |
        #   |  /\  |
        #   vv    vv
        #   o      o
        def fn(x, y):
            a, b = x.chunk(2)
            c, d = y.chunk(2)
            return a + c, b + d

        graph = self._perform_ad_subgraph_slicing(fn, 2, 2)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_does_not_create_cycles(self):
        # o --> x --> o
        # |           ^
        #  \_________/
        def fn(w, x, y):
            a = w * x
            b = torch.zeros(abs(int(a)))
            c = a * b
            return c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_merges_up(self):
        # o --> x     o
        # |           ^
        #  \_________/
        def fn(w, x, y, z):
            a = w * x
            b = torch.zeros(abs(int(y)))
            c = a * z
            return b, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)
        g_str = str(graph)
        FileCheck().check_not("aten::add").run(g_str[0 : g_str.find("return")])
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_merges_down(self):
        # o     x --> o
        # |           ^
        #  \_________/
        def fn(v, w, x, y):
            a = v * w
            b = torch.ones(int(y))
            c = b * a
            return a, c

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        # add moved down
        g_str = str(graph)
        FileCheck().check_not("aten::add").run(g_str[0 : g_str.find("return")])
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_respects_lexical_scoping(self):
        def fn(x, k):
            y = x * 1.1
            if bool(k):
                k = k + y
            z = y * k
            return z, k

        graph = self._perform_ad_subgraph_slicing(fn, 1, 1)
        # We should not have combined the two multiplications into
        # the same group; they should each be a separate DiffGraph
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 3)

    def test_merge_respects_aliasing(self):
        def fn(x, k, cond):
            y = x * 1.1
            y = y * k
            y = y * 2.2
            if bool(cond):
                z1 = y[0]
                z2 = y[1]
                z1.add_(3)
                out = z2 + k + 3.3
                out = out * out
                return out

        graph = self._perform_ad_subgraph_slicing(fn, [2, 2], [2, 2], 1)
        # z2 did did not get merged into the subgraph
        FileCheck().check("prim::If").check("aten::select").check_next(
            "aten::select"
        ).check_next("aten::add_").check("Differentiable").run(graph)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_aliased_outputs(self):
        with enable_profiling_mode_for_profiling_tests():
            # Case 1: aliasing between relu and t
            # is within a DifferentiableGraph. It should be valid
            # to merge both split_with_sizes in relu in one graph
            input_str = """
    graph(%a : Tensor):
        %b : Tensor = aten::relu(%a)
        %2 : Tensor = aten::t(%b)
        return (%2)
    """

            graph = torch._C.parse_ir(input_str)
            torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
            FileCheck().check("with prim::DifferentiableGraph").check(
                "aten::relu"
            ).check("aten::t").run(graph)

            # Case 2: aliasing between relu and split_with_sizes
            # are both outputs of a Diff graph. It should be invalid
            # to merge both split_with_sizes in relu in one graph
            # i.e. relu and split_with_sizes should be in different
            # differentiable graphs
            input_str = """
    graph(%a : Tensor):
        %b : Tensor = aten::relu(%a)
        %0 : int[] = prim::Constant[value=[2, 2, 1]]()
        %1 : int = prim::Constant[value=0]()
        %2 : Tensor[] = aten::split_with_sizes(%b, %0, %1)
        %3 : (Tensor[], Tensor[]) = prim::TupleConstruct(%b, %2)
        return (%3)
"""

            graph = torch._C.parse_ir(input_str)
            torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
            FileCheck().check("Tensor = prim::DifferentiableGraph").check(
                "with prim::DifferentiableGraph"
            ).check("Tensor = aten::relu").check_not("aten::split_with_sizes").run(
                graph
            )

            # Case 3: two aliased nodes in a graph.
            # Both `split_with_sizes` should be unfused
            input_str = """
    graph(%a : Tensor):
        %b : Tensor = aten::relu(%a)
        %s1 : int[] = prim::Constant[value=[2, 2, 1]]()
        %s2 : int[] = prim::Constant[value=[3, 1]]()
        %1 : int = prim::Constant[value=0]()
        %2 : Tensor[] = aten::split_with_sizes(%b, %s1, %1)
        %3 : Tensor[] = aten::split_with_sizes(%b, %s2, %1)
        %4 : (Tensor, Tensor[]) = prim::TupleConstruct(%b, %2, %3)
        return (%4)
"""

            graph = torch._C.parse_ir(input_str)
            torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
            FileCheck().check("Tensor = prim::DifferentiableGraph").check(
                "with prim::DifferentiableGraph"
            ).check("Tensor = aten::relu").check_not("aten::split_with_sizes").run(
                graph
            )

            # Case 4: the aliased output has a descendant
            # Both should be unfused. Note, %3 comes before %2
            # to test that we unfuse in the reverse topo order
            input_str = """
    graph(%a : Tensor):
        %b : Tensor = aten::relu(%a)
        %0 : int[] = prim::Constant[value=[2, 2, 1]]()
        %1 : int = prim::Constant[value=0]()
        %2 : Tensor = aten::t(%b)
        %3 : Tensor = aten::relu(%2)
        %4 : (Tensor, Tensor, Tensor[]) = prim::TupleConstruct(%b, %3, %2)
        return (%4)
"""

            graph = torch._C.parse_ir(input_str)
            torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
            FileCheck().check("Tensor = prim::DifferentiableGraph").check(
                "with prim::DifferentiableGraph"
            ).check("Tensor = aten::relu").check_not("aten::t").run(graph)

            # Case 5: multiple aliased groups
            # Both should be unfused. Note, %3 comes before %2
            # to test that we unfuse in the reverse topo order
            input_str = """
    graph(%a : Tensor):
        %b : Tensor = aten::relu(%a)
        %c : Tensor = aten::abs(%a)
        %0 : int[] = prim::Constant[value=[2, 2, 1]]()
        %1 : int = prim::Constant[value=0]()
        %d : Tensor = aten::t(%c)
        %2 : Tensor = aten::t(%b)
        %3 : Tensor = aten::relu(%2)
        %4 : (Tensor, Tensor, Tensor[]) = prim::TupleConstruct(%3, %2, %d, %b, %c, %b)
        return (%4)
"""

            graph = torch._C.parse_ir(input_str)
            torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
            FileCheck().check("Tensor = prim::DifferentiableGraph").check(
                "with prim::DifferentiableGraph"
            ).check("Tensor = aten::relu").check_not("aten::t").run(graph)

    def test_has_profiled_info_aliasing_outputs(self):
        # The expectation is that CallFunction will prevent the final profile node from
        # getting merged into the DifferentiableGraph, and that create_autodiff_subgraphs
        # will instead add this to the type for %4.
        ir = """
        graph(%a : Tensor):
            %1 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%a)
            %2 : Tensor = aten::relu(%1)
            %3 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%2)
            %4 : Tensor = aten::relu(%3)
            %5 : Tensor = prim::CallFunction(%4)
            %6 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%4)
            return (%6)
        """

        graph = torch._C.parse_ir(ir)
        torch._C._jit_pass_create_autodiff_subgraphs(graph)

        for n in graph.nodes():
            if n.kind() == "prim::DifferentiableGraph":
                diff_graph = n.g("Subgraph")

        outputs = list(diff_graph.outputs())
        self.assertEqual(1, len(outputs))
        output = outputs[0]
        self.assertEqual(False, output.requiresGrad())

        FileCheck().check("= prim::DifferentiableGraph").check(
            "with prim::DifferentiableGraph"
        ).check(" = aten::relu").check("requires_grad=0").check("aten::relu").run(graph)
