# Owner(s): ["module: dynamo"]

import copy
import pickle

import torch._inductor.test_case


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_source_graph_module_copies_are_isolated(self):
        # _src and the exec'd forward are shared between copies; everything else
        # nn.Module keeps on an instance is state, hook dicts included, and a
        # copy sharing it lets an update on one copy silently edit the other.
        # __reduce__ used to pickle the SHARED _src, whose body aliases the
        # original's parameter/buffer containers -- so mutating the original
        # after a deepcopy round-tripped the mutated tensors into the copy's
        # pickle even though live calls were isolated. __reduce__ now
        # snapshots the instance's own state.
        from torch._dynamo.precompile_context import (
            _EagerGraphSource,
            _SourceGraphModule,
        )

        src = _EagerGraphSource(
            code="def forward(self, x):\n    return x + self.b\n",
            import_block="",
            body={"_buffers": {"b": torch.ones(3)}},
        )
        original = _SourceGraphModule(src)
        dup = copy.deepcopy(original)
        dup.register_forward_hook(lambda *args: None)
        dup._non_persistent_buffers_set.add("b")
        self.assertEqual(len(original._forward_hooks), 0)
        self.assertEqual(original._non_persistent_buffers_set, set())
        self.assertIs(dup._src, original._src)

        x = torch.randn(3)
        original._buffers["b"].mul_(100)
        self.assertEqual(original(x), x + 100)
        self.assertEqual(dup(x), x + 1)  # live isolation
        self.assertEqual(pickle.loads(pickle.dumps(dup))(x), x + 1)
        # And an instance pickles its CURRENT parameters/buffers/submodules,
        # not its load-time ones (other nn.Module state still comes from _src).
        self.assertEqual(pickle.loads(pickle.dumps(original))(x), x + 100)

    def test_eager_artifact_round_trips_a_hop_graph_as_source(self):
        # GraphModule.__reduce__ re-traces the generated source at load; cond
        # rejects the Proxy and autocast enter/exit EXECUTE and leave no node.
        # The top level must travel as source, its HOP bodies as real Graphs.
        from torch._dynamo.precompile_context import (
            _SourceGraphModule,
            EagerCacheArtifact,
        )

        def fn(x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))
            return y + 1

        gms = []

        def backend(gm, example_inputs):
            gms.append(gm)
            return gm.forward

        x = torch.randn(3)
        torch.compile(fn, backend=backend, fullgraph=True)(x)
        (gm,) = gms
        artifact = EagerCacheArtifact(key="k", content=gm.forward)
        loaded = pickle.loads(pickle.dumps(artifact)).after_deserialization()
        module = loaded.__self__
        self.assertIsInstance(module, _SourceGraphModule)
        self.assertFalse(hasattr(module, "graph"))
        self.assertTrue(module._modules)
        for sub in module._modules.values():
            self.assertIsInstance(sub, torch.fx.GraphModule)
            self.assertGreater(len(sub.graph.nodes), 0)
        self.assertEqual(loaded(x), gm.forward(x))
        self.assertEqual(loaded(-x.abs()), gm.forward(-x.abs()))
        # Re-serializing a loaded artifact goes through _SourceGraphModule.__reduce__.
        reloaded = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(artifact))))
        self.assertEqual(reloaded.after_deserialization()(x), gm.forward(x))

    def test_take_artifact_removes_the_staged_backend(self):
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        PrecompileContext.record_artifact(EagerCacheArtifact(key="k", content=None))
        self.assertEqual(PrecompileContext.take_artifact("k").key, "k")
        self.assertIsNone(PrecompileContext.take_artifact("k"))
        self.assertIsNone(PrecompileContext.serialize_artifact_by_key("k"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
