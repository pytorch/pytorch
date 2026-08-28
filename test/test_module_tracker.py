# Owner(s): ["module: unknown"]

from copy import copy

import torch
from torch import nn
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils.checkpoint import checkpoint
from torch.utils.module_tracker import ModuleTracker


class TestModuleTracker(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_module_hierarchy(self):
        seen_fw = []
        seen_bw = []

        class Foo(nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = nn.ModuleDict({"nest": Foo()})
                self.c = nn.ModuleList([Foo()])

            def forward(self, x):
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModuleTracker() as tracker:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    def test_user_graph_module_hierarchy_with_dynamo(self):
        seen = []

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gm = torch.fx.symbolic_trace(Leaf())
                self.gm.register_forward_pre_hook(
                    lambda mod, args: seen.append(
                        (copy(tracker.parents), tracker.is_bw)
                    )
                )

            def forward(self, x):
                return self.gm(x).cos()

        mod = torch.compile(Outer(), backend="eager")

        with ModuleTracker() as tracker:
            mod(torch.randn(2, 2))

        self.assertEqual(seen, [({"Global", "Outer", "Outer.gm"}, False)])

    def test_compiled_user_graph_module_hierarchy(self):
        seen = []

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        gm = torch.fx.symbolic_trace(Leaf())
        gm.register_forward_pre_hook(
            lambda mod, args: seen.append((copy(tracker.parents), tracker.is_bw))
        )
        mod = torch.compile(gm, backend="eager")
        self.assertTrue(gm._is_torch_compile)
        self.assertFalse(torch._dynamo.utils.is_dynamo_runtime_module(gm))

        with ModuleTracker() as tracker:
            mod(torch.randn(2, 2))
            mod(torch.randn(2, 2))

        self.assertEqual(
            seen,
            [({"Global", "Leaf"}, False), ({"Global", "Leaf"}, False)],
        )

    @skipIfTorchDynamo("test itself calls torch.compile with a backend")
    def test_compiled_user_module_descendants_remain_visible(self):
        mod = nn.TransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
        )
        mod.torchdynamo_force_dynamic = False
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(2, 3, 4)
        compiled(x)

        seen = []
        with ModuleTracker() as tracker:
            handle = mod.linear1.register_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
            )
            try:
                compiled(x)
            finally:
                handle.remove()

        self.assertEqual(
            seen,
            [
                (
                    {
                        "Global",
                        "TransformerEncoderLayer",
                        "TransformerEncoderLayer.linear1",
                    },
                    False,
                )
            ],
        )

    @skipIfTorchDynamo("test itself calls torch.compile with a custom backend")
    def test_dynamo_runtime_graph_module_hierarchy(self):
        seen = []
        runtime_modules = []

        def backend(gm, example_inputs):
            runtime_gm = torch.fx.symbolic_trace(gm)
            runtime_gm.register_forward_pre_hook(
                lambda mod, args: seen.append((copy(tracker.parents), tracker.is_bw))
            )
            runtime_modules.append(runtime_gm)
            return runtime_gm

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        with torch._dynamo.config.patch(verify_correctness=True):
            mod = torch.compile(Leaf(), backend=backend)

            with ModuleTracker() as tracker:
                mod(torch.randn(2, 2))

        self.assertEqual(len(runtime_modules), 1)
        self.assertEqual(runtime_modules[0].meta, {})
        self.assertFalse(
            torch._dynamo.utils.is_dynamo_runtime_module(runtime_modules[0])
        )
        self.assertTrue(seen)
        for parents, is_bw in seen:
            self.assertEqual(parents, {"Global", "Leaf"})
            self.assertFalse(is_bw)

        seen.clear()
        with ModuleTracker() as tracker:
            runtime_modules[0](torch.randn(2, 2))
        self.assertEqual(seen, [({"Global", "GraphModule"}, False)])

    @skipIfTorchDynamo("test itself calls torch.compile with a backend")
    def test_nested_dynamo_runtime_graph_module_hierarchy(self):
        class CondMod(nn.Module):
            def forward(self, pred, x):
                return torch.cond(
                    pred,
                    lambda value: value.sin(),
                    lambda value: value.cos(),
                    (x,),
                )

        pred = torch.tensor(True)
        x = torch.randn(2, 2)
        mod = torch.compile(CondMod(), backend="eager", fullgraph=True)
        mod(pred, x)

        seen = []
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        id(module),
                        copy(tracker.parents),
                        tracker.is_bw,
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
                if isinstance(module, torch.fx.GraphModule)
                else None
            )
            try:
                mod(pred, x)
                mod(torch.tensor(False), x)
            finally:
                handle.remove()

        self.assertEqual(len(seen), 2)
        self.assertEqual(len({module_id for module_id, *_ in seen}), 2)
        for _, parents, is_bw, is_dynamo_runtime in seen:
            self.assertEqual(parents, {"Global", "CondMod"})
            self.assertFalse(is_bw)
            self.assertTrue(is_dynamo_runtime)

    def test_nested_dynamo_runtime_wrapper_hierarchy(self):
        from torch._dynamo.utils import (
            _DynamoRuntimeModule,
            dynamo_runtime_modules,
            get_dynamo_runtime_module_refs,
        )

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        class RuntimeWrapper(_DynamoRuntimeModule):
            def __init__(self) -> None:
                super().__init__()
                self.gm = torch.fx.symbolic_trace(Leaf())

            def forward(self, x):
                return self.gm(x)

            def _dynamo_runtime_module_values(self):
                return (self.gm,)

        runtime_wrapper = RuntimeWrapper()
        root = nn.Module()
        root.runtime_wrapper = runtime_wrapper
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_module("runtime_wrapper", (x,)))
        gm = torch.fx.GraphModule(root, graph)
        runtime_module_refs = get_dynamo_runtime_module_refs(gm)
        runtime_module_ids = {id(runtime_wrapper), id(runtime_wrapper.gm)}

        seen = []
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
                if id(module) in runtime_module_ids
                else None
            )
            try:
                with dynamo_runtime_modules(runtime_module_refs):
                    gm(torch.randn(2, 2))
            finally:
                handle.remove()

        self.assertEqual(seen, [({"Global"}, True), ({"Global"}, True)])

    def test_dynamo_runtime_wrapper_closure_hierarchy(self):
        from torch._dynamo.backends.debugging import eager_noexcept
        from torch._dynamo.backends.distributed import SubmodCompiler
        from torch._dynamo.utils import (
            dynamo_runtime_modules,
            get_dynamo_runtime_module_refs,
            is_dynamo_runtime_module,
        )

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        input_mod = torch.fx.symbolic_trace(Leaf())
        fake_mode = torch._subclasses.FakeTensorMode()
        compiler = SubmodCompiler(input_mod, eager_noexcept, fake_mode)
        wrapper, immediate_refs = compiler.compile_submod(
            input_mod,
            [fake_mode.from_tensor(torch.randn(2, 2))],
            {},
        )
        rediscovered_refs = get_dynamo_runtime_module_refs(wrapper)

        seen = []
        handle = input_mod.register_forward_pre_hook(
            lambda module, args: seen.append(
                (copy(tracker.parents), is_dynamo_runtime_module(module))
            )
        )
        try:
            with ModuleTracker() as tracker:
                with dynamo_runtime_modules(immediate_refs):
                    wrapper(torch.randn(2, 2))
                with dynamo_runtime_modules(rediscovered_refs):
                    wrapper(torch.randn(2, 2))
        finally:
            handle.remove()

        self.assertEqual(seen, [({"Global"}, True), ({"Global"}, True)])

    @skipIfTorchDynamo("test directly exercises Dynamo's DDP submodule compiler")
    def test_dynamo_ddp_submodule_compiler_hierarchy(self):
        from torch._dynamo.backends.distributed import SubmodCompiler

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        input_mod = torch.fx.symbolic_trace(Leaf())
        root = nn.Module()
        root.submod_0 = input_mod
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_module("submod_0", (x,)))
        split_gm = torch.fx.GraphModule(root, graph)
        fake_mode = torch._subclasses.FakeTensorMode()

        def backend(gm, args, **kwargs):
            gm(*args)
            return gm.forward

        compiler = SubmodCompiler(split_gm, backend, fake_mode)
        seen = []
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
                if isinstance(module, torch.fx.GraphModule)
                else None
            )
            try:
                compiler.run(fake_mode.from_tensor(torch.randn(2, 2)))
            finally:
                handle.remove()

        self.assertEqual(seen, [({"Global"}, True), ({"Global"}, True)])

    def test_dynamo_runtime_module_context_restores_nested_state(self):
        from torch._dynamo.utils import (
            dynamo_compiler_modules,
            dynamo_runtime_modules,
            get_dynamo_runtime_module_refs,
            is_dynamo_runtime_module,
        )

        outer = nn.Identity()
        inner = nn.Identity()
        outer_refs = get_dynamo_runtime_module_refs(outer)
        inner_refs = get_dynamo_runtime_module_refs(inner)

        with ModuleTracker():
            with dynamo_runtime_modules(outer_refs):
                self.assertTrue(is_dynamo_runtime_module(outer))
                self.assertFalse(is_dynamo_runtime_module(inner))
                with self.assertRaisesRegex(RuntimeError, "test exception"):
                    with dynamo_runtime_modules(inner_refs):
                        self.assertTrue(is_dynamo_runtime_module(outer))
                        self.assertTrue(is_dynamo_runtime_module(inner))
                        raise RuntimeError("test exception")
                self.assertTrue(is_dynamo_runtime_module(outer))
                self.assertFalse(is_dynamo_runtime_module(inner))

                with self.assertRaisesRegex(RuntimeError, "compiler exception"):
                    with dynamo_compiler_modules():
                        self.assertTrue(is_dynamo_runtime_module(outer))
                        self.assertTrue(is_dynamo_runtime_module(inner))
                        raise RuntimeError("compiler exception")
                self.assertTrue(is_dynamo_runtime_module(outer))
                self.assertFalse(is_dynamo_runtime_module(inner))

        self.assertFalse(is_dynamo_runtime_module(outer))
        self.assertFalse(is_dynamo_runtime_module(inner))

    @skipIfTorchDynamo("test itself calls torch.compile with a custom backend")
    def test_dynamo_backend_input_graph_module_hierarchy(self):
        seen = []

        def backend(gm, example_inputs):
            gm.register_forward_pre_hook(
                lambda mod, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(mod),
                    )
                )
            )
            gm(*example_inputs)
            return gm.forward

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        mod = torch.compile(Leaf(), backend=backend)
        with ModuleTracker() as tracker:
            mod(torch.randn(2, 2))

        self.assertTrue(seen)
        for parents, is_dynamo_runtime in seen:
            self.assertEqual(parents, {"Global", "Leaf"})
            self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself calls torch.compile with a backend")
    def test_dynamo_runtime_graph_module_closure(self):
        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        mod = torch.compile(Leaf(), backend="eager_noexcept")
        seen = []
        with torch._dynamo.config.patch(verify_correctness=True):
            with ModuleTracker() as tracker:
                handle = nn.modules.module.register_module_forward_pre_hook(
                    lambda module, args: seen.append(
                        (
                            copy(tracker.parents),
                            torch._dynamo.utils.is_dynamo_runtime_module(module),
                        )
                    )
                    if isinstance(module, torch.fx.GraphModule)
                    else None
                )
                try:
                    mod(torch.randn(2, 2))
                    seen.clear()
                    mod(torch.randn(2, 2))
                finally:
                    handle.remove()

        self.assertEqual(seen, [({"Global", "Leaf"}, True)])

    @skipIfTorchDynamo("test itself calls torch.compile with an AOT backend")
    def test_aot_ts_runtime_module_hierarchy(self):
        compiled = torch.compile(
            lambda x: x.sin().cos(), backend="aot_ts", fullgraph=True
        )
        seen = []
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        type(module).__name__,
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
                if "ScriptModule" in type(module).__name__
                else None
            )
            try:
                compiled(torch.randn(3, 3))
                compiled(torch.randn(3, 3))
            finally:
                handle.remove()

        self.assertTrue(seen)
        for module_name, parents, is_dynamo_runtime in seen:
            self.assertNotIn(module_name, parents)
            self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself calls torch.compile with an AOT backend")
    def test_aot_autograd_compiler_output_hierarchy(self):
        from torch._dynamo.backends.common import aot_autograd

        seen = []
        compiled_graphs = []

        def make_compiler(kind):
            def compiler(gm, example_inputs):
                compiled_graphs.append(kind)

                def record(module, args):
                    seen.append(
                        (
                            kind,
                            tracker.parents.copy(),
                            tracker.is_bw,
                            torch._dynamo.utils.is_dynamo_runtime_module(module),
                        )
                    )

                gm.register_forward_pre_hook(record)
                gm(*example_inputs)

                def run(args):
                    return gm(*args)

                run._boxed_call = True
                return run

            return compiler

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin().cos()

        backend = aot_autograd(
            fw_compiler=make_compiler("forward"),
            bw_compiler=make_compiler("backward"),
            inference_compiler=make_compiler("inference"),
        )
        compiled = torch.compile(Leaf(), backend=backend, fullgraph=True)
        with (
            torch._functorch.config.patch(force_non_lazy_backward_lowering=False),
            ModuleTracker() as tracker,
        ):
            output = compiled(torch.randn(3, requires_grad=True))
            self.assertEqual(compiled_graphs, ["forward"])
            output.sum().backward()

        self.assertEqual(compiled_graphs, ["forward", "backward"])
        torch._dynamo.reset()
        inference_compiled = torch.compile(Leaf(), backend=backend, fullgraph=True)
        with torch.no_grad(), ModuleTracker() as tracker:
            inference_compiled(torch.randn(3))

        self.assertEqual(compiled_graphs, ["forward", "backward", "inference"])
        self.assertTrue(
            any(kind == "forward" and not is_bw for kind, _, is_bw, _ in seen)
        )
        self.assertTrue(any(kind == "backward" and is_bw for kind, _, is_bw, _ in seen))
        self.assertTrue(
            any(kind == "inference" and not is_bw for kind, _, is_bw, _ in seen)
        )
        for _, parents, _, is_dynamo_runtime in seen:
            self.assertEqual(parents, {"Global", "Leaf"})
            self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself calls torch.compile with an AOT backend")
    def test_cached_regional_aot_runtime_module_hierarchy(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.utils import fresh_inductor_cache
        from torch.fx.passes.regional_inductor import regional_inductor

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin().cos()

        backend = aot_autograd(
            fw_compiler=regional_inductor,
            bw_compiler=regional_inductor,
            inference_compiler=regional_inductor,
        )
        mod = Leaf()
        x = torch.randn(3)

        def run():
            seen = []
            with torch.no_grad(), ModuleTracker() as tracker:
                handle = nn.modules.module.register_module_forward_pre_hook(
                    lambda module, args: seen.append(
                        (
                            copy(tracker.parents),
                            torch._dynamo.utils.is_dynamo_runtime_module(module),
                        )
                    )
                    if isinstance(module, torch.fx.GraphModule)
                    else None
                )
                try:
                    torch.compile(mod, backend=backend, fullgraph=True)(x)
                finally:
                    handle.remove()
            return seen

        with (
            fresh_inductor_cache(),
            torch._functorch.config.patch(
                bundled_autograd_cache=True,
                force_autograd_cache=True,
                strict_autograd_cache=True,
            ),
        ):
            AOTAutogradCache.clear()
            torch._dynamo.utils.counters.clear()
            first_seen = run()
            torch._dynamo.reset()
            cached_seen = run()

        self.assertEqual(
            torch._dynamo.utils.counters["aot_autograd"]["autograd_cache_hit"], 1
        )
        for seen in (first_seen, cached_seen):
            self.assertTrue(seen)
            for parents, is_dynamo_runtime in seen:
                self.assertEqual(parents, {"Global", "Leaf"})
                self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself calls torch.compile with an AOT backend")
    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_invoke_subgraph_runtime_module_hierarchy(self):
        compiled = torch.compile(
            lambda x: x.sin().cos(), backend="invoke_subgraph", fullgraph=True
        )
        seen = []
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda module, args: seen.append(
                    (
                        type(module).__name__,
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(module),
                    )
                )
                if isinstance(module, torch.fx.GraphModule)
                else None
            )
            try:
                compiled(torch.randn(3, 3))
                compiled(torch.randn(3, 3))
            finally:
                handle.remove()

        self.assertTrue(seen)
        for module_name, parents, is_dynamo_runtime in seen:
            self.assertNotIn(module_name, parents)
            self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself enables compiled autograd")
    def test_compiled_autograd_runtime_graph_module_hierarchy(self):
        seen = []

        def compiler(gm):
            gm.register_forward_pre_hook(
                lambda mod, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(mod),
                    )
                )
            )
            return gm

        x = torch.randn(3, requires_grad=True)
        with ModuleTracker() as tracker:
            with torch._dynamo.compiled_autograd._enable(compiler):
                x.sin().sum().backward()

        self.assertEqual(seen, [({"Global"}, True)])

    @skipIfTorchDynamo("test itself enables compiled autograd")
    def test_aot_compiled_autograd_runtime_module_hierarchy(self):
        from torch._dynamo.utils import GmWrapper

        seen = []
        x = torch.randn(3, requires_grad=True)
        with ModuleTracker() as tracker:
            handle = nn.modules.module.register_module_forward_pre_hook(
                lambda mod, args: seen.append(
                    (
                        copy(tracker.parents),
                        torch._dynamo.utils.is_dynamo_runtime_module(mod),
                    )
                )
                if isinstance(mod, GmWrapper)
                else None
            )
            try:
                with torch._dynamo.compiled_autograd._enable(
                    torch.compile(backend="aot_eager")
                ):
                    x.sin().sum().backward()
            finally:
                handle.remove()

        self.assertTrue(seen)
        for parents, is_dynamo_runtime in seen:
            self.assertEqual(parents, {"Global"})
            self.assertTrue(is_dynamo_runtime)

    @skipIfTorchDynamo("test itself calls torch.compile with a backend")
    def test_aot_eager_runtime_graph_module_hierarchy(self):
        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        for force_autograd_cache in (False, True):
            with self.subTest(force_autograd_cache=force_autograd_cache):
                torch._dynamo.reset()
                seen = []
                seen_bw = []
                with torch._functorch.config.patch(
                    force_autograd_cache=force_autograd_cache
                ):
                    mod = torch.compile(Leaf(), backend="aot_eager")
                    with ModuleTracker() as tracker:
                        handle = nn.modules.module.register_module_forward_pre_hook(
                            lambda module, args: seen.append(
                                (
                                    copy(tracker.parents),
                                    torch._dynamo.utils.is_dynamo_runtime_module(
                                        module
                                    ),
                                )
                            )
                            if isinstance(module, torch.fx.GraphModule)
                            else None
                        )
                        try:
                            output = mod(torch.randn(2, 2, requires_grad=True))
                            output.register_hook(
                                lambda grad: seen_bw.append(
                                    (copy(tracker.parents), tracker.is_bw)
                                )
                            )
                            output.sum().backward()
                        finally:
                            handle.remove()

                self.assertTrue(seen)
                for parents, is_dynamo_runtime in seen:
                    self.assertEqual(parents, {"Global", "Leaf"})
                    self.assertTrue(is_dynamo_runtime)
                self.assertEqual(seen_bw, [({"Global", "Leaf"}, True)])

    def test_restored_aot_eager_runtime_graph_module(self):
        from torch._dynamo.backends.debugging import AOTEagerOutputCode, boxed_nop

        x = torch.randn(2, 2)
        root = nn.Module()
        root.inner = nn.Identity()
        graph = torch.fx.Graph()
        graph_input = graph.placeholder("x")
        graph.output(graph.call_module("inner", (graph_input,)))
        gm = torch.fx.GraphModule(root, graph)
        with torch._functorch.config.patch(force_autograd_cache=True):
            output = boxed_nop(gm, [x])
        self.assertIsInstance(output, AOTEagerOutputCode)
        output.prepare_for_serialization()
        output.post_compile()

        seen = []
        handle = nn.modules.module.register_module_forward_pre_hook(
            lambda module, args: seen.append(
                (
                    torch._dynamo.utils.is_dynamo_runtime_module(output.gm),
                    torch._dynamo.utils.is_dynamo_runtime_module(module),
                )
            )
            if module is output.gm.inner
            else None
        )
        try:
            self.assertEqual(output([x]), x)
        finally:
            handle.remove()

        self.assertEqual(seen, [(True, False)])

    def test_user_exported_graph_module_hierarchy(self):
        seen = []

        class Leaf(nn.Module):
            def forward(self, x):
                return x.sin()

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gm = torch.export.export(
                    Leaf(), (torch.randn(2, 2),), strict=True
                ).module()
                self.gm.register_forward_pre_hook(
                    lambda mod, args: seen.append(
                        (copy(tracker.parents), tracker.is_bw)
                    )
                )

            def forward(self, x):
                return self.gm(x).cos()

        mod = Outer()
        self.assertIn("dynamo_compile_id", mod.gm.meta)
        self.assertNotIn("backend_id", mod.gm.meta)
        self.assertFalse(torch._dynamo.utils.is_dynamo_runtime_module(mod.gm))

        with ModuleTracker() as tracker:
            mod(torch.randn(2, 2))

        self.assertEqual(seen, [({"Global", "Outer", "Outer.gm"}, False)])

    @skipIfTorchDynamo("unexplained 3.13+ recursion error")
    def test_confused_hierarchy(self):
        class MyMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(2, 2)
                self.ran = False

            def forward(self, inp):
                if not self.ran:
                    self.ran = True
                    return self(inp)
                else:
                    self.ran = False
                    return self.inner(inp)

        mod = MyMod()
        inp = torch.rand(1, 2, requires_grad=True)

        # Should not fail
        with ModuleTracker():
            res = mod(inp)
            res.sum().backward()

        # Should not fail
        with ModuleTracker():
            res = checkpoint(lambda inp: mod(inp), inp)
            res.sum().backward()

    def test_bw_detection(self):
        mod = nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})


if __name__ == "__main__":
    run_tests()
