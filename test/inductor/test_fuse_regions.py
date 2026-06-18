# Owner(s): ["module: inductor"]

from operator import getitem
from types import SimpleNamespace

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


class TestFuseRegions(InductorTestCase):
    def test_apply_fuse_region_annotations_no_annotations_is_noop(self):
        from torch._inductor.fx_passes.fuse_regions import apply_fuse_region_annotations

        graph = Graph()
        x = graph.placeholder("x")
        sin = graph.call_function(torch.ops.aten.sin.default, (x,))
        graph.output(sin)
        gm = GraphModule(torch.nn.Module(), graph)
        before_nodes = list(gm.graph.nodes)

        apply_fuse_region_annotations(gm.graph)

        self.assertEqual(list(gm.graph.nodes), before_nodes)

    def test_apply_fuse_region_annotations_groups_contiguous_nodes(self):
        from torch._inductor.fx_passes.fuse_regions import (
            apply_fuse_region_annotations,
            FUSE_REGION,
        )

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        sin = graph.call_function(torch.ops.aten.sin.default, (relu,))
        cos = graph.call_function(torch.ops.aten.cos.default, (sin,))
        graph.output(cos)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(4)
        for node in (x, add, relu, sin, cos):
            node.meta["val"] = fake
        for node in (add, relu):
            node.meta["custom"] = {FUSE_REGION: "region_a"}
        for node in (sin, cos):
            node.meta["custom"] = {FUSE_REGION: "region_b"}

        apply_fuse_region_annotations(gm.graph)

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(
            [node.meta[FUSE_REGION] for node in region_nodes],
            [
                "region_a_fwd",
                "region_b_fwd",
            ],
        )

    def test_apply_fuse_region_annotations_is_idempotent(self):
        from torch._inductor.fx_passes.fuse_regions import (
            apply_fuse_region_annotations,
            FUSE_REGION,
        )

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        graph.output(relu)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(4)
        for node in (x, add, relu):
            node.meta["val"] = fake
        for node in (add, relu):
            node.meta["custom"] = {FUSE_REGION: "region_a"}

        apply_fuse_region_annotations(gm.graph)
        apply_fuse_region_annotations(gm.graph)

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(len(region_nodes), 1)
        self.assertEqual(region_nodes[0].meta[FUSE_REGION], "region_a_fwd")

    def test_apply_fuse_region_annotations_reads_compile_with_inductor_metadata(self):
        from torch._inductor.fx_passes.fuse_regions import (
            apply_fuse_region_annotations,
            FUSE_REGION,
        )

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        graph.output(relu)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(4)
        for node in (x, add, relu):
            node.meta["val"] = fake
        for node in (add, relu):
            node.meta["custom"] = {
                "compile_with_inductor": {FUSE_REGION: "compiled_region"}
            }

        apply_fuse_region_annotations(gm.graph)

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(len(region_nodes), 1)
        self.assertEqual(region_nodes[0].meta[FUSE_REGION], "compiled_region_fwd")

    def test_apply_fuse_region_annotations_splits_forward_backward(self):
        from torch._inductor.fx_passes.fuse_regions import (
            apply_fuse_region_annotations,
            FUSE_REGION,
        )

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        sin = graph.call_function(torch.ops.aten.sin.default, (relu,))
        cos = graph.call_function(torch.ops.aten.cos.default, (sin,))
        graph.output(cos)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(4)
        for node in (x, add, relu, sin, cos):
            node.meta["val"] = fake
        for node in (add, relu, sin, cos):
            node.meta["custom"] = {FUSE_REGION: "shared"}
        for node in (sin, cos):
            node.meta["autograd_backward"] = True

        apply_fuse_region_annotations(gm.graph)

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(
            [node.meta[FUSE_REGION] for node in region_nodes],
            [
                "shared_fwd",
                "shared_bwd",
            ],
        )

    def test_apply_fuse_region_annotations_rejects_invalid_annotation(self):
        from torch._inductor.fx_passes.fuse_regions import (
            apply_fuse_region_annotations,
            FUSE_REGION,
        )

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        graph.output(relu)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(4)
        for node in (x, add, relu):
            node.meta["val"] = fake
        for node in (add, relu):
            node.meta["custom"] = {FUSE_REGION: 1}

        with self.assertRaisesRegex(AssertionError, "expected custom fuse_region"):
            apply_fuse_region_annotations(gm.graph)

    def test_post_grad_passes_apply_fuse_region_annotations_by_default(self):
        from torch._inductor.fx_passes.fuse_regions import FUSE_REGION
        from torch._inductor.fx_passes.post_grad import post_grad_passes
        from torch._inductor.virtualized import V
        from torch._subclasses.fake_tensor import FakeTensorMode

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        sin = graph.call_function(torch.ops.aten.sin.default, (relu,))
        cos = graph.call_function(torch.ops.aten.cos.default, (sin,))
        graph.output(cos)
        gm = GraphModule(torch.nn.Module(), graph)

        fake_mode = FakeTensorMode()
        fake = fake_mode.from_tensor(torch.empty(4))
        for node in (x, add, relu, sin, cos):
            node.meta["val"] = fake
        for node in (add, relu):
            node.meta["custom"] = {FUSE_REGION: "region_a"}
        for node in (sin, cos):
            node.meta["custom"] = {FUSE_REGION: "region_b"}

        with config.patch(pattern_matcher=False), V.set_fake_mode(fake_mode):
            post_grad_passes(gm, is_inference=False)

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(
            [node.meta[FUSE_REGION] for node in region_nodes],
            [
                "region_a_fwd",
                "region_b_fwd",
            ],
        )

    def test_fuse_region_annotations_affect_fx_graph_cache_key(self):
        from torch._inductor.codecache import compiled_fx_graph_hash
        from torch._inductor.fx_passes.fuse_regions import FUSE_REGION

        def make_gm(annotate: bool):
            graph = Graph()
            x = graph.placeholder("x")
            add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
            relu = graph.call_function(torch.ops.aten.relu.default, (add,))
            graph.output(relu)
            gm = GraphModule(torch.nn.Module(), graph)
            if annotate:
                for node in (add, relu):
                    node.meta["custom"] = {FUSE_REGION: "region_a"}
            return gm

        example_inputs = [torch.empty(4)]
        unannotated_key, _ = compiled_fx_graph_hash(
            make_gm(False), example_inputs, {}, []
        )
        annotated_key, debug_lines = compiled_fx_graph_hash(
            make_gm(True), example_inputs, {}, []
        )

        self.assertNotEqual(unannotated_key, annotated_key)
        self.assertTrue(any("fuse_region_annotations" in line for line in debug_lines))

    def test_mark_fuse_region_preserves_region_id_single_output(self):
        from torch._inductor.fx_passes.fuse_regions import FUSE_REGION, mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        sin = graph.call_function(torch.ops.aten.sin.default, (x,))
        graph.output(sin)
        gm = GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = torch.empty(4)
        sin.meta["val"] = torch.empty(4)

        mark_fuse_region(gm.graph, [sin], fuse_region_id="shared")

        region_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        ]
        self.assertEqual(len(region_nodes), 1)
        self.assertEqual(region_nodes[0].meta[FUSE_REGION], "shared")

    def test_mark_fuse_region_rejects_invalid_region_id(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        sin = graph.call_function(torch.ops.aten.sin.default, (x,))
        graph.output(sin)
        gm = GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = torch.empty(4)
        sin.meta["val"] = torch.empty(4)

        with self.assertRaisesRegex(AssertionError, "None or str"):
            mark_fuse_region(gm.graph, [sin], fuse_region_id=1)

    def test_mark_fuse_region_rejects_graph_boundaries(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = torch.empty(4)

        with self.assertRaisesRegex(AssertionError, "exclude graph boundaries"):
            mark_fuse_region(gm.graph, [x])

    @config.patch(reorder_for_locality=False)
    @requires_gpu()
    def test_fuse_region_id_allows_separate_islands_to_fuse(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        def fn(x):
            a = x * 2
            outside = x + 3
            b = torch.relu(a)
            return b, outside

        def add_shared_regions(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node], fuse_region_id="shared")
            mark_fuse_region(graph, [relu_node], fuse_region_id="shared")
            return graph

        def add_unique_regions(graph):
            mul_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.mul.Tensor
            )[0]
            relu_node = graph.find_nodes(
                op="call_function", target=torch.ops.aten.relu.default
            )[0]
            mark_fuse_region(graph, [mul_node], fuse_region_id="mul")
            mark_fuse_region(graph, [relu_node], fuse_region_id="relu")
            return graph

        x = torch.rand([256, 256], device=GPU_TYPE)

        torch._dynamo.reset()
        with config.patch(post_grad_custom_post_pass=add_shared_regions):
            result, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, fn(x))
        self.assertIn("fused_mul_relu", code[0])

        torch._dynamo.reset()
        with config.patch(post_grad_custom_post_pass=add_unique_regions):
            result, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, fn(x))
        self.assertNotIn("fused_mul_relu", code[0])
        self.assertIn("fused_mul", code[0])
        self.assertIn("fused_relu", code[0])

    def test_non_adjacent_reuse_respects_fuse_region_config(self):
        from torch._inductor.codegen.wrapper import AllocateLine
        from torch._inductor.virtualized import V

        nodes = [
            SimpleNamespace(region="free"),
            SimpleNamespace(region="middle"),
            SimpleNamespace(region="alloc"),
        ]
        scheduler = SimpleNamespace(
            nodes=nodes,
            get_fuse_region=lambda node: node.region,
        )
        graph = SimpleNamespace(scheduler=scheduler)
        estimate_peak = SimpleNamespace(
            overall_peak_memory=1024,
            peak_between=lambda free_line, alloc_line: 0,
        )

        free_line = SimpleNamespace(scheduler_node_index=0)
        alloc_line = object.__new__(AllocateLine)
        alloc_line.comm_buffer = False
        alloc_line.scheduler_node_index = 2
        alloc_line.wrapper = SimpleNamespace(estimate_peak=estimate_peak)

        with V.set_graph_handler(graph):
            self.assertFalse(config.allow_buffer_reuse_across_fuse_regions)
            self.assertFalse(alloc_line.should_reuse_buffer(free_line, 128))

            with config.patch(allow_buffer_reuse_across_fuse_regions=False):
                self.assertFalse(alloc_line.should_reuse_buffer(free_line, 128))

            with config.patch(allow_buffer_reuse_across_fuse_regions=True):
                self.assertTrue(alloc_line.should_reuse_buffer(free_line, 128))

            nodes[2].region = "free"
            with config.patch(allow_buffer_reuse_across_fuse_regions=False):
                self.assertTrue(alloc_line.should_reuse_buffer(free_line, 128))

            nodes[1].region = "alloc"
            adjacent_free_line = SimpleNamespace(scheduler_node_index=1)
            with config.patch(allow_buffer_reuse_across_fuse_regions=False):
                self.assertTrue(alloc_line.should_reuse_buffer(adjacent_free_line, 128))

    def test_fuse_region_uses_invoke_subgraph_arg_extraction(self):
        from torch._inductor.fx_passes.fuse_regions import FUSE_REGION, mark_fuse_region
        from torch._inductor.fx_utils import _extract_subgraphs_and_args

        def make_region() -> GraphModule:
            graph = Graph()
            x = graph.placeholder("x")
            add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
            mul = graph.call_function(torch.ops.aten.mul.Tensor, (add, 2))
            graph.output(mul)
            return GraphModule({}, graph)

        region = make_region()
        add_node = region.graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )[0]
        mul_node = region.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        )[0]
        placeholder = region.graph.find_nodes(op="placeholder")[0]

        x = torch.empty(1)
        placeholder.meta["val"] = x
        add_node.meta["val"] = x
        mul_node.meta["val"] = x

        mark_fuse_region(region.graph, [add_node, mul_node], fuse_region_id="shared")
        node = next(
            node
            for node in region.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        )
        subgraph = getattr(region, node.args[0].target)
        extracted = list(
            _extract_subgraphs_and_args(
                node,
                {subgraph},
                subgraph,
                node.args[1],
                x,
            )
        )

        self.assertEqual(len(extracted), 1)
        self.assertIs(extracted[0][0], subgraph)
        self.assertIs(extracted[0][1][0], x)
        self.assertEqual(node.meta[FUSE_REGION], "shared")

    def test_fuse_region_flattens_tuple_boundary_inputs(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        bias = graph.placeholder("bias")
        layer_norm = graph.call_function(
            torch.ops.aten.native_layer_norm.default,
            (x, [4], weight, bias, 1e-5),
        )
        getitem_0 = graph.call_function(getitem, (layer_norm, 0))
        graph.output(getitem_0)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(2, 4)
        x.meta["val"] = fake
        weight.meta["val"] = torch.empty(4)
        bias.meta["val"] = torch.empty(4)
        layer_norm.meta["val"] = (fake, torch.empty(2, 1), torch.empty(2, 1))
        getitem_0.meta["val"] = fake

        mark_fuse_region(gm.graph, [getitem_0])

        invoke_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        )
        self.assertEqual(len(invoke_node.args), 5)
        self.assertTrue(
            all(
                node.op == "call_function" and node.target is getitem
                for node in invoke_node.args[2:]
            )
        )

        subgraph = getattr(gm, invoke_node.args[0].target)
        self.assertEqual(len(subgraph.graph.find_nodes(op="placeholder")), 3)

    def test_fuse_region_rejects_boundary_cycle(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2))
        add = graph.call_function(torch.ops.aten.add.Tensor, (mul, 3))
        relu = graph.call_function(torch.ops.aten.relu.default, (add,))
        graph.output(relu)
        gm = GraphModule(torch.nn.Module(), graph)

        fake = torch.empty(2)
        x.meta["val"] = fake
        mul.meta["val"] = fake
        add.meta["val"] = fake
        relu.meta["val"] = fake

        with self.assertRaisesRegex(AssertionError, "acyclic"):
            mark_fuse_region(gm.graph, [mul, relu])

    def test_fuse_region_allows_no_external_outputs(self):
        from torch._inductor.fx_passes.fuse_regions import mark_fuse_region

        graph = Graph()
        x = graph.placeholder("x")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        graph.output(x)
        gm = GraphModule(torch.nn.Module(), graph)

        x.meta["val"] = torch.empty(2, 2)
        add.meta["val"] = torch.empty(2, 2)

        region_node = mark_fuse_region(gm.graph, [add])
        gm.recompile()

        self.assertEqual(region_node.meta["val"], ())
        subgraph = getattr(gm, region_node.args[0].target)
        output_node = subgraph.graph.find_nodes(op="output")[0]
        self.assertEqual(output_node.args[0], ())
        self.assertEqual(gm(torch.ones(2, 2)), torch.ones(2, 2))


if __name__ == "__main__":
    if IS_LINUX:
        run_tests(needs="filelock")
