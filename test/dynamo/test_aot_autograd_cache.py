# Owner(s): ["module: dynamo"]

import pickle
from functools import partial

import torch
import torch._dynamo
import torch._dynamo.test_case

import torch._functorch._aot_autograd
from functorch.compile import (
    aot_module_simplified,
    default_decompositions,
    min_cut_rematerialization_partition,
)
from torch._functorch._aot_autograd.autograd_cache import (
    autograd_cache_hash,
    BypassAOTAutogradCache,
    deserialize_graph_module,
    serialize_graph_module,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._subclasses import FakeTensorMode


def _get_dynamo_output(fn, inputs):
    # Reset dynamo between runs
    torch._dynamo.reset()
    fx_graph = None

    def compiler(gm, inputs):
        nonlocal fx_graph
        fx_graph = gm
        return gm

    g = torch.compile(fn, backend=compiler, fullgraph=True)
    result = g(*inputs)
    return (result, fx_graph)


def get_post_autograd_graphs(fn, inps):
    def extract_graph(fx_g, _, graph_cell):
        graph_cell[0] = fx_g
        return fx_g

    (_, post_dynamo) = _get_dynamo_output(fn, inps)
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_module_simplified(
        post_dynamo,
        inps,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        partition_fn=min_cut_rematerialization_partition,
        decompositions=default_decompositions,
    )
    return fw_graph_cell[0], bw_graph_cell[0]


class AOTAutogradSerializationTests(torch._dynamo.test_case.TestCase):
    def test_basic_serialize(self, device="cpu"):
        def fn(x, y):
            return x.sin().sin() + y

        fw, _ = get_post_autograd_graphs(fn, [torch.randn(3), torch.randn(3)])
        fw_serialized = pickle.dumps(serialize_graph_module(fw))
        # Use a empty FakeTensormode for testing
        # The fake tensor mode used is the same one in autograd in all runs
        with FakeTensorMode():
            new_fw = deserialize_graph_module(pickle.loads(fw_serialized))
        self.assertEqual(len(fw.graph.nodes), len(new_fw.graph.nodes))
        self.assertEqual(
            fw.print_readable(print_output=False),
            new_fw.print_readable(print_output=False),
        )


class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):
    def default_config(self):
        return AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            inference_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
            dynamic_shapes=True,
            aot_autograd_arg_pos_to_source=None,
            is_export=False,
            no_tangents=False,
            enable_log=False,
        )

    def gen_cache_key(self, f, config, inputs=None):
        if inputs is None:
            inputs = [torch.randn(3)]
        _, fx_g = _get_dynamo_output(f, inputs)
        return autograd_cache_hash(fx_g, config)

    def test_basic_hash_key(self):
        def fn(x):
            return x.sin().cos()

        config = self.default_config()
        # Check hash is stable on multiple runs
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

    def test_identical_graphs_and_configs(self):
        def fn(x):
            return x.sin().cos()

        def fn2(x):
            y = x.sin()
            z = y.cos()
            return z

        # Make the id different, but otherwise identical
        config = self.default_config()
        config2 = self.default_config()
        config2.aot_id = 1

        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        self.assertEqual(c1, c2)

    def test_different_graphs(self):
        def fn(x):
            return x.cos().sin()

        def fn2(x):
            return x.sin().cos()

        config = self.default_config()
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn2, config)
        self.assertNotEqual(c1, c2)

    def test_different_configs(self):
        def fn(x):
            return x.cos().sin()

        config = self.default_config()
        config2 = self.default_config()
        config2.dynamic_shapes = False
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        self.assertNotEqual(c1, c2)

    def test_incompatible_function(self):
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(_, x):
                torch._dynamo.graph_break()
                return x.sin()

        def fn(x):
            return AllowInGraphFunc.apply(x)

        config = self.default_config()
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
