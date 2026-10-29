# Owner(s): ["module: dynamo"]

import collections
import contextlib
import dis
import functools
import inspect
import types
import unittest
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_package,
    has_triton_tensor_descriptor_host_tma,
)


def _filter_instructions(instructions, opname):
    return list(filter(lambda x: x.opname == opname, instructions))


@contextlib.contextmanager
def _force_tensor_descriptor_support():
    import torch.utils._triton as triton_utils

    with mock.patch.object(
        triton_utils, "_device_supports_tensor_descriptor", return_value=True
    ):
        triton_utils.has_triton_tensor_descriptor_host_tma.cache_clear()
        try:
            yield
        finally:
            triton_utils.has_triton_tensor_descriptor_host_tma.cache_clear()


@instantiate_parametrized_tests
class ReconstructTest(torch._dynamo.test_case.TestCase):
    @contextlib.contextmanager
    def register_bytecode_hook(self, fn):
        def hook(code, out_code):
            fn(list(dis.get_instructions(out_code)))
            return None

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def test_ConstDict_optimize_reconstruct(self):
        """
        Emit code to reconstruct only the key that changed
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct only d[40]
            self.assertEqual(build_map[0].argval, 1)

        def f(d, t):
            d[40] = t + 1

        t = torch.randn(3, 4)
        d = {1: t}
        d_opt = d.copy()
        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def _compile_and_capture_side_effects(self, fn, *args):
        """Compile fn and return side-effect metadata from bytecode hooks."""
        captured = {}

        def rewrite_hook(code, out_code):
            return out_code.replace(co_name=f"{out_code.co_name}_hooked")

        def inspect_hook(code, out_code):
            captured["refs"] = (
                torch._dynamo.convert_frame.get_compiled_code_side_effects(out_code)
            )
            captured["has_side_effects"] = (
                torch._dynamo.convert_frame.compiled_code_has_side_effects(out_code)
            )

        torch._dynamo.reset()
        rewrite_handle = torch._dynamo.convert_frame.register_bytecode_hook(
            rewrite_hook
        )
        inspect_handle = torch._dynamo.convert_frame.register_bytecode_hook(
            inspect_hook
        )
        try:
            torch.compile(fn, backend="eager", fullgraph=True)(*args)
        finally:
            inspect_handle.remove()
            rewrite_handle.remove()

        return captured

    def test_bytecode_hook_exposes_side_effect_refs(self):
        def mutating_fn(x, lst):
            lst.append(x + 1)
            return x * 2

        def pure_fn(x):
            return x * 2

        x = torch.randn(3)

        mutated = self._compile_and_capture_side_effects(mutating_fn, x, [])
        self.assertEqual(mutated["refs"], ("L['lst']",))
        self.assertTrue(mutated["has_side_effects"])

        pure = self._compile_and_capture_side_effects(pure_fn, x)
        self.assertEqual(pure["refs"], ())
        self.assertFalse(pure["has_side_effects"])

    def test_side_effect_refs_dict_mutation(self):
        def fn(x, d):
            d["result"] = x + 1
            return x * 2

        result = self._compile_and_capture_side_effects(fn, torch.randn(3), {})
        self.assertEqual(result["refs"], ("L['d']",))
        self.assertTrue(result["has_side_effects"])

    def test_side_effect_refs_tensor_in_container(self):
        # Relevant to cudagraphs: a compiled function computes tensors and
        # stores them into an external container as a side effect.
        def fn(x, outputs):
            y = x * 2
            z = x + 3
            outputs.append(y)
            outputs.append(z)
            return x

        result = self._compile_and_capture_side_effects(fn, torch.randn(4), [])
        self.assertEqual(result["refs"], ("L['outputs']",))
        self.assertTrue(result["has_side_effects"])

    def test_side_effect_refs_multiple_containers(self):
        def fn(x, lst, d):
            lst.append(x + 1)
            d["out"] = x * 2
            return x

        result = self._compile_and_capture_side_effects(fn, torch.randn(3), [], {})
        self.assertEqual(len(result["refs"]), 2)
        self.assertIn("L['lst']", result["refs"])
        self.assertIn("L['d']", result["refs"])
        self.assertTrue(result["has_side_effects"])

    def test_ConstDict_pop_reconstruct(self):
        """
        If something is pop'ed from the dict, we reconstruct everything
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 2)

        def f(d, t):
            d.pop(2)
            d[40] = t + 1

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_popitem_reconstruct(self):
        """
        If something is pop'ed from the dict, we reconstruct everything
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 1)

        def f(d, t):
            d.popitem()

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_popitem_reconstruct_graph_break(self):
        """
        If something is pop'ed from the dict, we reconstruct everything.
        Calling dict.popitem will graph break.
        """

        def f(d, t):
            d.popitem()

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        opt_f = torch.compile(backend="eager")(f)
        opt_f(d_opt, t)
        self.assertEqual(d, d_opt)

    def test_ConstDict_del_reconstruct(self):
        """
        If something is deleted from the dict, we reconstruct everything
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 2)

        def f(d, t):
            del d[2]
            d[40] = t + 1

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_get_reconstruct(self):
        """
        dict.get shouldn't affect anything
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            self.assertEqual(build_map[0].argval, 1)
            load_const = _filter_instructions(instructions, "LOAD_CONST")
            self.assertNotIn(123, load_const)

        def f(d, t):
            d[456] = d.get(456) + t

        t = torch.randn(3, 4)
        d = {123: t, 456: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_ConstDict_clear_reconstruct(self):
        """
        If dict.clear() is used, we reconstruct everything
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 1)

        def f(d, t):
            d.clear()
            d[3] = t + 3

        t = torch.randn(3, 4)
        d = {1: t, 2: t + 1}
        d_opt = d.copy()

        f(d, t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            opt_f(d_opt, t)
            self.assertEqual(d, d_opt)

    def test_create_dict_reconstruct(self):
        """
        If dict is created inside a function, everything needs to be reconstructed
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 1)
            # reconstruct everything
            self.assertEqual(build_map[0].argval, 2)

        def f(t):
            return {1: t, 2: t + 1}

        t = torch.randn(3, 4)
        d = f(t)

        with self.register_bytecode_hook(hook):
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            d_opt = opt_f(t)
            self.assertEqual(d, d_opt)

    @unittest.skipIf(
        IS_FBCODE, "capturing functional_call is not enabled by default in FB_CODE"
    )
    def test_functional_call_reconstruct(self):
        """
        PyTorch shouldn't codegen any key/value when functional_call is used
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            # don't reconstruct anything
            self.assertEqual(len(build_map), 0)

        m = torch.nn.Linear(3, 3)
        new_bias = torch.randn(3)
        new_weight = torch.randn(3, 3)

        def fn(new_weight, new_bias, x):
            return torch.func.functional_call(
                m, {"weight": new_weight, "bias": new_bias}, x
            )

        x = torch.randn(2, 3)
        expected = torch.nn.functional.linear(x, new_weight, new_bias)
        with self.register_bytecode_hook(hook):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            got = opt_fn(new_weight, new_bias, x)
            self.assertEqual(expected, got)

    @unittest.skipIf(
        IS_FBCODE, "capturing functional_call is not enabled by default in FB_CODE"
    )
    def test_functional_call_reconstruct_2(self):
        """
        PyTorch shouldn't codegen any key/value when functional_call is used
        """

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            # don't reconstruct anything
            self.assertEqual(len(build_map), 0)

        class DummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.ModuleDict(
                    {
                        "b": torch.nn.ModuleDict(
                            {
                                "c": torch.nn.ModuleDict(
                                    {
                                        "d": torch.nn.ModuleDict(
                                            {"e": torch.nn.Linear(10, 10, bias=False)}
                                        )
                                    }
                                )
                            }
                        )
                    }
                )

            def forward(self, x):
                return self.a.b.c.d.e(x)

        model = DummyModule()

        def fn(model, states, x):
            return torch.func.functional_call(model, states, x)

        x = torch.randn(2, 3)
        states = model.state_dict()
        x = torch.randn(10, 10)
        expected = fn(model, states, x)
        with self.register_bytecode_hook(hook):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            got = opt_fn(model, states, x)
            self.assertEqual(expected, got)

    def test_ordered_dict_no_reconstruct_without_mutation(self):
        """Sourced OrderedDict should not emit BUILD_MAP when not mutated."""

        def hook(instructions: list[dis.Instruction]):
            build_map = _filter_instructions(instructions, "BUILD_MAP")
            self.assertEqual(len(build_map), 0)

        def fn(od, x):
            return x + od["a"]

        od = collections.OrderedDict(a=1, b=2)
        with self.register_bytecode_hook(hook):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            got = opt_fn(od, torch.tensor(1.0))
            self.assertEqual(got, torch.tensor(2.0))

    def test_graph_break_in_wrapped_user_function(self):
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        @torch.compile(backend="eager")
        def gn(x):
            x = torch.no_grad()(fn)(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)

    def test_graph_break_in_wrapped_user_method(self):
        class Foo:
            def __init__(self):
                self.a = 1
                self.b = 2

            def fn(self, x):
                x = x + self.a
                torch._dynamo.graph_break()
                assert torch.compiler.is_compiling()  # noqa: S101
                assert not torch.is_grad_enabled()  # noqa: S101
                return x + self.b

        obj = Foo()

        @torch.compile(backend="eager")
        def gn(x):
            obj.fn = torch.no_grad()(obj.fn)
            x = obj.fn(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)

    def test_graph_break_in_wrapped_nested_function(self):
        @torch.compile(backend="eager")
        def gn(x):
            a = 1
            b = 2

            @torch.no_grad()
            def fn(x):
                x = x + a
                torch._dynamo.graph_break()
                assert torch.compiler.is_compiling()  # noqa: S101
                assert not torch.is_grad_enabled()  # noqa: S101
                return x + b

            x = fn(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)

    def test_graph_break_in_wrapped_skipped_function(self):
        from torch._dynamo import trace_rules
        from torch._dynamo.testing import _skipped_function_for_test_reconstruct
        from torch._dynamo.variables import SkipFunctionVariable

        self.assertIs(
            trace_rules.lookup(_skipped_function_for_test_reconstruct),
            SkipFunctionVariable,
        )

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        @torch.compile(backend="eager")
        def gn(x):
            x = torch.no_grad()(_skipped_function_for_test_reconstruct)(fn, x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    @unittest.skipIf(
        not has_triton_experimental_host_tma(),
        "Test requires triton.tools.experimental_descriptor API",
    )
    def test_tma_experimental_reconstruct(self):
        import triton

        def create_tma(tensor):
            tma = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
                tensor.data_ptr(),
                tensor.size(0),
                tensor.size(1),
                32,
                32,
                tensor.element_size(),
            )
            return tensor + 1, tma

        x = torch.randn(128, 128, device=GPU_TYPE)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        ref = create_tma(x)
        res = torch.compile(create_tma, backend=backend)(x)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(ref[1].desc, res[1].desc)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @_force_tensor_descriptor_support()
    def test_tma_stable_reconstruct(self):
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor API")

        import triton

        def create_tma(tensor):
            tma = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                tensor,
                [32, 32],
            )
            return tensor + 1, tma

        x = torch.randn(128, 128)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        ref = create_tma(x)
        res = torch.compile(create_tma, backend=backend)(x)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(ref, res)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @_force_tensor_descriptor_support()
    def test_tma_stable_reconstruct_inherited_classmethod(self):
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor API")

        import triton.tools.tensor_descriptor as descriptor_module

        class ClassMethodDescriptor(descriptor_module.TensorDescriptor):
            @classmethod
            def from_tensor(cls, tensor, block_shape):
                return cls(tensor, tensor.shape, tensor.stride(), block_shape)

        class InheritedClassMethodDescriptor(ClassMethodDescriptor):
            pass

        descriptor_class = InheritedClassMethodDescriptor

        def create_tma(tensor):
            descriptor = descriptor_class.from_tensor(tensor, [16])
            return tensor + 1, descriptor

        with self.assertRaisesRegex(
            Unsupported,
            "TMA descriptor class is not importable",
        ):
            torch.compile(create_tma, backend="eager", fullgraph=True)(torch.randn(16))

        torch._dynamo.reset()
        descriptor_class.__module__ = descriptor_module.__name__
        setattr(descriptor_module, descriptor_class.__name__, descriptor_class)
        try:
            x = torch.randn(16)
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            result, descriptor = torch.compile(
                create_tma, backend=backend, fullgraph=True
            )(x)

            self.assertEqual(len(backend.graphs), 1)
            self.assertEqual(result, x + 1)
            self.assertIsInstance(descriptor, descriptor_class)
            self.assertEqual(descriptor.block_shape, [16])
        finally:
            delattr(descriptor_module, descriptor_class.__name__)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @_force_tensor_descriptor_support()
    def test_tma_stable_sourceless_classmethod_lookup(self):
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor API")

        from triton.tools.tensor_descriptor import TensorDescriptor

        from torch._dynamo.variables.functions import UserMethodVariable
        from torch._dynamo.variables.user_defined import UserDefinedClassVariable

        class ClassMethodDescriptor(TensorDescriptor):
            @classmethod
            def from_tensor(cls, tensor, block_shape):
                return cls(tensor, tensor.shape, tensor.stride(), block_shape)

        descriptor_class_vt = UserDefinedClassVariable(ClassMethodDescriptor)
        factory_vt = descriptor_class_vt.resolve_cls_descriptor(
            None,
            "from_tensor",
            ClassMethodDescriptor.__dict__["from_tensor"],
            None,
        )
        self.assertIsInstance(factory_vt, UserMethodVariable)
        self.assertIs(factory_vt.im_self, descriptor_class_vt)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize(
        "call_style", ["padding", "rounding", "both", "positional", "keywords"]
    )
    def test_tma_stable_factory_explicit_defaults(self, call_style):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            def create_tma(tensor):
                factory = descriptor_module.TensorDescriptor.from_tensor
                if call_style == "padding":
                    descriptor = factory(tensor, [16], padding="zero")
                elif call_style == "rounding":
                    descriptor = factory(tensor, [16], round_f32_to_tf32=False)
                elif call_style == "both":
                    descriptor = factory(
                        tensor, [16], padding="zero", round_f32_to_tf32=False
                    )
                elif call_style == "positional":
                    descriptor = factory(tensor, [16], "zero", False)
                else:
                    descriptor = factory(
                        tensor=tensor,
                        block_shape=[16],
                        padding="zero",
                        round_f32_to_tf32=False,
                    )
                return tensor + 1, descriptor

            tensor = torch.randn(16)
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            result, descriptor = torch.compile(
                create_tma, backend=backend, fullgraph=True
            )(tensor)
            self.assertEqual(len(backend.graphs), 1)
            self.assertEqual(result, tensor + 1)
            self.assertIsInstance(descriptor, descriptor_module.TensorDescriptor)
            self.assertEqual(descriptor.base, tensor)
            self.assertEqual(descriptor.block_shape, [16])
            self.assertEqual(descriptor.padding, "zero")
            self.assertFalse(descriptor.round_f32_to_tf32)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("flag", [False, 0])
    @parametrize("structured", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_factory_default_type(self, flag, structured, fullgraph):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            default = (False,) if structured else False
            if structured:
                flag = (flag,)

            class FlagDescriptor(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape, flag=default):
                    if structured:
                        flag = flag[0]
                    padding = "zero" if flag is False else "nan"
                    return cls(
                        tensor, tensor.shape, tensor.stride(), block_shape, padding
                    )

            FlagDescriptor.__module__ = descriptor_module.__name__
            with mock.patch.object(
                descriptor_module, "FlagDescriptor", FlagDescriptor, create=True
            ):

                def create_tma(tensor):
                    descriptor = FlagDescriptor.from_tensor(tensor, [16], flag=flag)
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                eager_result, eager_descriptor = create_tma(tensor)
                compiled = torch.compile(
                    create_tma, backend="eager", fullgraph=fullgraph
                )
                if fullgraph and (structured or flag is not False):
                    with self.assertRaisesRegex(
                        Unsupported, "Unsupported TMA descriptor factory call"
                    ):
                        compiled(tensor)
                else:
                    result, descriptor = compiled(tensor)
                    self.assertEqual(result, eager_result)
                    self.assertEqual(descriptor.padding, eager_descriptor.padding)
                    if structured:
                        factory = FlagDescriptor.from_tensor.__func__
                        with mock.patch.object(factory, "__defaults__", ((0,),)):
                            self.assertEqual(
                                compiled(tensor)[1].padding,
                                create_tma(tensor)[1].padding,
                            )

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("mutation", ["default_order", "keyword_only"])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_factory_code_mutation(self, mutation, fullgraph):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class CodeDescriptor(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape, padding="zero", other="nan"):
                    return cls(
                        tensor, tensor.shape, tensor.stride(), block_shape, padding
                    )

            if mutation == "default_order":

                def replacement(cls, tensor, block_shape, other="zero", padding="nan"):
                    return cls(
                        tensor, tensor.shape, tensor.stride(), block_shape, padding
                    )

            else:

                def replacement(cls, *, tensor, block_shape, padding):
                    return cls(
                        tensor, tensor.shape, tensor.stride(), block_shape, padding
                    )

            CodeDescriptor.__module__ = descriptor_module.__name__
            with mock.patch.object(
                descriptor_module, "CodeDescriptor", CodeDescriptor, create=True
            ):

                def create_tma(tensor):
                    descriptor = CodeDescriptor.from_tensor(
                        tensor=tensor, block_shape=[16], padding="zero"
                    )
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(
                    create_tma, backend=counter, fullgraph=fullgraph
                )
                self.assertEqual(compiled(tensor)[1].padding, "zero")
                self.assertEqual(counter.frame_count, 1)
                factory = CodeDescriptor.from_tensor.__func__
                original_code = factory.__code__
                try:
                    factory.__code__ = replacement.__code__
                    self.assertEqual(create_tma(tensor)[1].padding, "zero")
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported, "Unsupported TMA descriptor factory call"
                        ):
                            compiled(tensor)
                    else:
                        result, descriptor = compiled(tensor)
                        self.assertEqual(result, tensor + 1)
                        self.assertEqual(descriptor.padding, "zero")
                finally:
                    factory.__code__ = original_code

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("default_kind", ["positional", "positional_length", "keyword"])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_factory_default_mutation(self, default_kind, fullgraph):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            if default_kind == "keyword":

                class KeywordDescriptor(descriptor_module.TensorDescriptor):
                    @classmethod
                    def from_tensor(cls, tensor, block_shape, *, padding="zero"):
                        return cls(
                            tensor,
                            tensor.shape,
                            tensor.stride(),
                            block_shape,
                            padding=padding,
                        )

                descriptor_class = KeywordDescriptor
                descriptor_class.__module__ = descriptor_module.__name__
                patcher = mock.patch.object(
                    descriptor_module,
                    "KeywordDescriptor",
                    descriptor_class,
                    create=True,
                )
                factory = descriptor_class.from_tensor.__func__
            else:
                descriptor_class = descriptor_module.TensorDescriptor
                factory = descriptor_class.from_tensor
                patcher = mock.patch.object(
                    factory, "__defaults__", factory.__defaults__
                )
            patcher.start()
            self.addCleanup(patcher.stop)

            def create_tma(tensor):
                descriptor = descriptor_class.from_tensor(tensor, [16], padding="zero")
                return tensor + 1, descriptor

            tensor = torch.randn(16)
            counter = torch._dynamo.testing.CompileCounter()
            compiled = torch.compile(create_tma, backend=counter, fullgraph=fullgraph)
            self.assertEqual(compiled(tensor)[1].padding, "zero")
            self.assertEqual(counter.frame_count, 1)

            if default_kind == "keyword":
                factory.__kwdefaults__["padding"] = "nan"
            elif default_kind == "positional_length":
                factory.__defaults__ = ("zero", "nan", False)
            else:
                factory.__defaults__ = ("nan", False)

            self.assertEqual(create_tma(tensor)[1].padding, "zero")
            if fullgraph:
                with self.assertRaisesRegex(
                    Unsupported, "Unsupported TMA descriptor factory call"
                ):
                    compiled(tensor)
            else:
                result, descriptor = compiled(tensor)
                self.assertEqual(result, tensor + 1)
                self.assertEqual(descriptor.padding, "zero")

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize(
        "signature_kind", ["keyword_only", "custom_signature", "reordered", "required"]
    )
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_factory_incompatible_reconstruction(
        self, signature_kind, fullgraph
    ):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class SignatureDescriptor(descriptor_module.TensorDescriptor):
                pass

            if signature_kind in ("keyword_only", "custom_signature"):

                @classmethod
                def from_tensor(cls, *, tensor, block_shape):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            elif signature_kind == "reordered":

                @classmethod
                def from_tensor(cls, block_shape, tensor):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            else:

                @classmethod
                def from_tensor(cls, tensor, block_shape, required):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            if signature_kind == "custom_signature":
                from_tensor.__func__.__signature__ = inspect.Signature(
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for name in ("cls", "tensor", "block_shape")
                )
            SignatureDescriptor.from_tensor = from_tensor
            SignatureDescriptor.__module__ = descriptor_module.__name__
            with mock.patch.object(
                descriptor_module,
                "SignatureDescriptor",
                SignatureDescriptor,
                create=True,
            ):

                def create_tma(tensor):
                    descriptor = SignatureDescriptor.from_tensor(
                        tensor=tensor, block_shape=[16]
                    )
                    if signature_kind == "required":
                        return tensor + 1
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                compiled = torch.compile(
                    create_tma, backend="eager", fullgraph=fullgraph
                )
                if fullgraph:
                    with self.assertRaisesRegex(
                        Unsupported, "Unsupported TMA descriptor factory call"
                    ):
                        compiled(tensor)
                elif signature_kind == "required":
                    with self.assertRaisesRegex(TypeError, "required"):
                        compiled(tensor)
                else:
                    result, descriptor = compiled(tensor)
                    self.assertEqual(result, tensor + 1)
                    self.assertIsInstance(descriptor, SignatureDescriptor)
                    self.assertEqual(descriptor.base, tensor)
                    self.assertEqual(descriptor.block_shape, [16])

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("fixed_inputs", [0, 1, 2])
    @parametrize("call_style", ["positional", "mixed", "keywords"])
    def test_tma_stable_factory_plain_forwarder(self, call_style, fixed_inputs):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            original = descriptor_module.TensorDescriptor.from_tensor

            if fixed_inputs == 0:

                def from_tensor(*args, **kwargs):
                    return original(*args, **kwargs)

            elif fixed_inputs == 1:

                def from_tensor(tensor, *args, **kwargs):
                    return original(tensor, *args, **kwargs)

            else:

                def from_tensor(tensor, block_shape, *args, **kwargs):
                    return original(tensor, block_shape, *args, **kwargs)

            with mock.patch.object(
                descriptor_module.TensorDescriptor,
                "from_tensor",
                staticmethod(from_tensor),
            ):

                def create_tma(tensor):
                    factory = descriptor_module.TensorDescriptor.from_tensor
                    if call_style == "positional":
                        descriptor = factory(tensor, [16])
                    elif call_style == "mixed":
                        descriptor = factory(tensor, block_shape=[16])
                    else:
                        descriptor = factory(tensor=tensor, block_shape=[16])
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                backend = torch._dynamo.testing.EagerAndRecordGraphs()
                result, descriptor = torch.compile(
                    create_tma, backend=backend, fullgraph=True
                )(tensor)
                self.assertEqual(len(backend.graphs), 1)
                self.assertEqual(result, tensor + 1)
                self.assertEqual(descriptor.base, tensor)
                self.assertEqual(descriptor.block_shape, [16])
                self.assertEqual(descriptor.padding, "zero")

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("wrapper_kind", ["wrapped", "signature"])
    @parametrize("explicit", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_factory_wrapped_defaults(
        self, wrapper_kind, explicit, fullgraph
    ):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            original = descriptor_module.TensorDescriptor.from_tensor
            if wrapper_kind == "wrapped":

                @functools.wraps(original)
                def from_tensor(tensor, block_shape, padding=None):
                    if padding is None:
                        return original(tensor, block_shape)
                    return original(tensor, block_shape, padding)

            else:

                def from_tensor(*args, **kwargs):
                    return original(*args, **kwargs)

                from_tensor.__signature__ = inspect.signature(original)

            with (
                mock.patch.object(
                    descriptor_module.TensorDescriptor,
                    "from_tensor",
                    staticmethod(from_tensor),
                ),
                mock.patch.object(original, "__defaults__", original.__defaults__),
            ):

                def create_tma(tensor):
                    factory = descriptor_module.TensorDescriptor.from_tensor
                    if explicit:
                        descriptor = factory(tensor, [16], padding="zero")
                    else:
                        descriptor = factory(tensor, [16])
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                compiled = torch.compile(
                    create_tma, backend="eager", fullgraph=fullgraph
                )
                if explicit and fullgraph:
                    with self.assertRaisesRegex(
                        Unsupported, "Unsupported TMA descriptor factory call"
                    ):
                        compiled(tensor)
                    return

                self.assertEqual(compiled(tensor)[1].padding, "zero")
                original.__defaults__ = ("nan", False)
                result, descriptor = compiled(tensor)
                self.assertEqual(result, tensor + 1)
                self.assertEqual(descriptor.padding, "zero" if explicit else "nan")
                self.assertEqual(descriptor.padding, create_tma(tensor)[1].padding)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("argument", ["padding", "rounding", "nonconstant"])
    def test_tma_stable_factory_nondefault_arguments(self, argument):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            def create_tma(tensor):
                factory = descriptor_module.TensorDescriptor.from_tensor
                if argument == "padding":
                    return factory(tensor, [16], padding="nan")
                if argument == "rounding":
                    return factory(tensor, [16], round_f32_to_tf32=True)
                return factory(tensor, [16], round_f32_to_tf32=tensor)

            with self.assertRaisesRegex(
                Unsupported, "Unsupported TMA descriptor factory call"
            ):
                torch.compile(create_tma, backend="eager", fullgraph=True)(
                    torch.randn(16)
                )

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    def test_tma_stable_factory_concrete_signature(self):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class ClassMethodDescriptor(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, base, block, *, padding="nan"):
                    return cls(base, base.shape, base.stride(), block, padding=padding)

            ClassMethodDescriptor.__module__ = descriptor_module.__name__
            with mock.patch.object(
                descriptor_module,
                "ClassMethodDescriptor",
                ClassMethodDescriptor,
                create=True,
            ):

                def create_tma(tensor):
                    descriptor = ClassMethodDescriptor.from_tensor(
                        base=tensor, block=[16], padding="nan"
                    )
                    return tensor + 1, descriptor

                tensor = torch.randn(16)
                result, descriptor = torch.compile(
                    create_tma, backend="eager", fullgraph=True
                )(tensor)
                self.assertEqual(result, tensor + 1)
                self.assertIsInstance(descriptor, ClassMethodDescriptor)
                self.assertEqual(descriptor.base, tensor)
                self.assertEqual(descriptor.block_shape, [16])
                self.assertEqual(descriptor.padding, "nan")

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize(
        "factory_kind",
        ["base", "classmethod", "inherited_classmethod", "inherited_staticmethod"],
    )
    @parametrize("as_argument", [False, True])
    def test_tma_stable_factory_python_type(self, factory_kind, as_argument):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class ClassMethodDescriptor(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            class InheritedClassMethodDescriptor(ClassMethodDescriptor):
                pass

            class InheritedStaticMethodDescriptor(descriptor_module.TensorDescriptor):
                pass

            descriptor_class = {
                "base": descriptor_module.TensorDescriptor,
                "classmethod": ClassMethodDescriptor,
                "inherited_classmethod": InheritedClassMethodDescriptor,
                "inherited_staticmethod": InheritedStaticMethodDescriptor,
            }[factory_kind]

            def inspect_factory(tensor, factory=None):
                if factory is None:
                    factory = descriptor_class.from_tensor
                return (
                    tensor + 1,
                    isinstance(factory, types.MethodType),
                    isinstance(factory, types.FunctionType),
                    type(factory),
                )

            x = torch.randn(16)
            args = (x, descriptor_class.from_tensor) if as_argument else (x,)
            compiled = torch.compile(inspect_factory, backend="eager", fullgraph=True)
            self.assertEqual(compiled(*args), inspect_factory(*args))

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    def test_tma_stable_factory_and_import_rebinding(self):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class FactoryA(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            class FactoryB(FactoryA):
                pass

            for descriptor_type in (FactoryA, FactoryB):
                descriptor_type.__module__ = descriptor_module.__name__

            with (
                mock.patch.object(descriptor_module, "FactoryA", FactoryA, create=True),
                mock.patch.object(descriptor_module, "FactoryB", FactoryB, create=True),
            ):
                factory = FactoryA.from_tensor

                def create_tma(tensor):
                    return tensor + 1, factory(tensor, [16])

                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(create_tma, backend=counter, fullgraph=True)
                x = torch.randn(16)

                self.assertIsInstance(compiled(x)[1], FactoryA)
                self.assertEqual(counter.frame_count, 1)
                factory = FactoryB.from_tensor
                self.assertIsInstance(compiled(x)[1], FactoryB)
                self.assertEqual(counter.frame_count, 2)

                def create_from_class(tensor):
                    return tensor + 1, FactoryA.from_tensor(tensor, [16])

                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(
                    create_from_class, backend=counter, fullgraph=True
                )

                self.assertIsInstance(compiled(x)[1], FactoryA)
                self.assertEqual(counter.frame_count, 1)
                descriptor_module.FactoryA = FactoryB
                with self.assertRaisesRegex(
                    Unsupported, "TMA descriptor class is not importable"
                ):
                    compiled(x)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("is_classmethod", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_same_code_factory_rebinding(self, is_classmethod, fullgraph):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class FactoryA(descriptor_module.TensorDescriptor):
                pass

            class FactoryB(FactoryA):
                pass

            def make_factory(descriptor_type):
                if is_classmethod:

                    def from_tensor(cls, tensor, block_shape):
                        return descriptor_type(
                            tensor, tensor.shape, tensor.stride(), block_shape
                        )

                else:

                    def from_tensor(tensor, block_shape):
                        return descriptor_type(
                            tensor, tensor.shape, tensor.stride(), block_shape
                        )

                from_tensor.__module__ = descriptor_module.__name__
                return from_tensor

            first = make_factory(FactoryA)
            second = make_factory(FactoryB)
            self.assertIs(first.__code__, second.__code__)
            factory_decorator = classmethod if is_classmethod else staticmethod
            for descriptor_type in (FactoryA, FactoryB):
                descriptor_type.__module__ = descriptor_module.__name__

            with (
                mock.patch.object(descriptor_module, "FactoryA", FactoryA, create=True),
                mock.patch.object(descriptor_module, "FactoryB", FactoryB, create=True),
                mock.patch.object(FactoryA, "from_tensor", factory_decorator(first)),
                mock.patch.object(FactoryB, "from_tensor", factory_decorator(second)),
            ):
                factory = FactoryA.from_tensor

                def create_tma(tensor):
                    return tensor + 1, factory(tensor, [16])

                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(
                    create_tma, backend=counter, fullgraph=fullgraph
                )
                x = torch.randn(16)
                self.assertIs(type(compiled(x)[1]), FactoryA)
                self.assertEqual(counter.frame_count, 1)

                factory = (
                    types.MethodType(second, FactoryA) if is_classmethod else second
                )
                self.assertIs(type(create_tma(x)[1]), FactoryB)
                if is_classmethod and fullgraph:
                    with self.assertRaisesRegex(
                        Unsupported, "captured TMA descriptor factory no longer matches"
                    ):
                        compiled(x)
                else:
                    result, descriptor = compiled(x)
                    self.assertEqual(result, x + 1)
                    self.assertIs(type(descriptor), FactoryB)
                    self.assertEqual(counter.frame_count, 2)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    def test_tma_stable_factory_owner_after_class_reuse(self):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class FactoryA(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            class FactoryB(FactoryA):
                pass

            for descriptor_type in (FactoryA, FactoryB):
                descriptor_type.__module__ = descriptor_module.__name__

            with (
                mock.patch.object(descriptor_module, "FactoryA", FactoryA, create=True),
                mock.patch.object(descriptor_module, "FactoryB", FactoryB, create=True),
                mock.patch.object(torch, "_tma_guard_owner", FactoryA, create=True),
            ):

                def create_tma(tensor, factory):
                    name = torch._tma_guard_owner.__name__
                    return tensor + 1, factory(tensor, [16]), name

                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(create_tma, backend=counter, fullgraph=True)
                x = torch.randn(16)
                self.assertIs(type(compiled(x, FactoryA.from_tensor)[1]), FactoryA)
                self.assertEqual(counter.frame_count, 1)
                self.assertIs(type(create_tma(x, FactoryB.from_tensor)[1]), FactoryB)
                result, descriptor, name = compiled(x, FactoryB.from_tensor)
                self.assertEqual(result, x + 1)
                self.assertIs(type(descriptor), FactoryB)
                self.assertEqual(name, "FactoryA")
                self.assertEqual(counter.frame_count, 2)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @parametrize("replace_owner", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_tma_stable_retained_factory_alias(self, replace_owner, fullgraph):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class FactoryA(descriptor_module.TensorDescriptor):
                @classmethod
                def from_tensor(cls, tensor, block_shape):
                    return cls(tensor, tensor.shape, tensor.stride(), block_shape)

            class FactoryB(FactoryA):
                pass

            for descriptor_type in (FactoryA, FactoryB):
                descriptor_type.__module__ = descriptor_module.__name__

            with (
                mock.patch.object(descriptor_module, "FactoryA", FactoryA, create=True),
                mock.patch.object(descriptor_module, "FactoryB", FactoryB, create=True),
            ):
                factory = FactoryA.from_tensor

                def create_tma(tensor):
                    return tensor + 1, factory(tensor, [16])

                counter = torch._dynamo.testing.CompileCounter()
                compiled = torch.compile(
                    create_tma, backend=counter, fullgraph=fullgraph
                )
                x = torch.randn(16)
                self.assertIs(type(compiled(x)[1]), FactoryA)
                self.assertEqual(counter.frame_count, 1)

                if replace_owner:
                    replacement = FactoryB.from_tensor
                else:

                    @classmethod
                    def replacement(cls, tensor, block_shape):
                        return FactoryB(
                            tensor, tensor.shape, tensor.stride(), block_shape
                        )

                with mock.patch.object(FactoryA, "from_tensor", replacement):
                    self.assertIs(type(create_tma(x)[1]), FactoryA)
                    if fullgraph:
                        with self.assertRaisesRegex(
                            Unsupported,
                            "captured TMA descriptor factory no longer matches",
                        ):
                            compiled(x)
                    else:
                        result, descriptor = compiled(x)
                        self.assertEqual(result, x + 1)
                        self.assertIs(type(descriptor), FactoryA)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    def test_tma_stable_inherited_staticmethod_owner(self):
        import triton.tools.tensor_descriptor as descriptor_module

        with _force_tensor_descriptor_support():
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires triton.tools.tensor_descriptor API")

            class InheritedDescriptor(descriptor_module.TensorDescriptor):
                pass

            def create_tma(tensor):
                return tensor + 1, InheritedDescriptor.from_tensor(tensor, [16])

            x = torch.randn(16)
            result, descriptor = torch.compile(
                create_tma, backend="eager", fullgraph=True
            )(x)

            self.assertEqual(result, x + 1)
            self.assertIsInstance(descriptor, descriptor_module.TensorDescriptor)
            self.assertNotIsInstance(descriptor, InheritedDescriptor)

    @unittest.skipIf(not has_triton_package(), "requires Triton")
    @_force_tensor_descriptor_support()
    def test_tma_stable_backend_subclass_reconstruct(self):
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires triton.tools.tensor_descriptor API")

        import triton.tools.tensor_descriptor as descriptor_module

        class BackendDescriptor(descriptor_module.TensorDescriptor):
            @staticmethod
            def from_tensor(tensor, block_shape):
                return BackendDescriptor(
                    tensor, tensor.shape, tensor.stride(), block_shape
                )

        BackendDescriptor.__module__ = descriptor_module.__name__
        with mock.patch.object(
            descriptor_module, "BackendDescriptor", BackendDescriptor, create=True
        ):

            def create_tma(tensor):
                descriptor = BackendDescriptor.from_tensor(tensor, [16])
                return tensor + 1, descriptor

            x = torch.randn(16)
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            result, descriptor = torch.compile(
                create_tma, backend=backend, fullgraph=True
            )(x)

            self.assertEqual(len(backend.graphs), 1)
            self.assertEqual(result, x + 1)
            self.assertIsInstance(descriptor, BackendDescriptor)
            self.assertEqual(descriptor.block_shape, [16])

    def test_self_referential_sourceful(self):
        l = []
        l.append((0, l))

        def fn(x, l):
            x = x + 1
            # self-referential object on the stack during a graph break
            print(l)
            return x + len(l)

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3)
        self.assertEqual(fn(inp, l), opt_fn(inp, l))

    def test_self_referential_sourceless(self):
        @torch.compile(backend="eager")
        def fn(x, construct_fn):
            l = construct_fn()

            x += 1
            print(l)
            x += 1
            # if reconstruction failed on the graph break, we should error here
            assert torch.compiler.is_compiling()  # noqa: S101
            return l

        @torch.compile(backend="eager", fullgraph=True)
        def fn2(x, construct_fn):
            l = construct_fn()
            x += 1
            return l

        def construct_list():
            l = []
            l.append(l)
            return l

        out = fn(torch.ones(3), construct_list)
        self.assertIs(out[0], out)
        out = fn2(torch.ones(3), construct_list)
        self.assertIs(out[0], out)

        def construct_deque():
            d = collections.deque()
            d.append(d)
            return d

        out = fn(torch.ones(3), construct_deque)
        self.assertIs(out[0], out)
        out = fn2(torch.ones(3), construct_deque)
        self.assertIs(out[0], out)

        def construct_dict():
            d = {}
            d[0] = d
            return d

        out = fn(torch.ones(3), construct_dict)
        self.assertIs(out[0], out)
        out = fn2(torch.ones(3), construct_dict)
        self.assertIs(out[0], out)

        def construct_ordereddict():
            d = collections.OrderedDict()
            d[0] = d
            return d

        out = fn(torch.ones(3), construct_ordereddict)
        self.assertIs(out[0], out)
        out = fn2(torch.ones(3), construct_ordereddict)
        self.assertIs(out[0], out)

        def construct_defaultdict():
            d = collections.defaultdict()
            d[0] = d
            return d

        out = fn(torch.ones(3), construct_defaultdict)
        self.assertIs(out[0], out)
        out = fn2(torch.ones(3), construct_defaultdict)
        self.assertIs(out[0], out)

    def test_self_referential_subclass_sourceless(self):
        # A container subclass built inside the graph is materialized and cached
        # before its contents are emitted, so the self-reference must resolve to
        # that object rather than to a fresh placeholder.
        class ListSub(list):
            pass

        class DequeSub(collections.deque):
            pass

        class DictSub(dict):
            pass

        class OrderedDictSub(collections.OrderedDict):
            pass

        def fn(x, construct_fn):
            obj = construct_fn()
            x += 1
            return obj

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        def construct_list_sub():
            obj = ListSub()
            obj.append(obj)
            return obj

        def construct_deque_sub():
            obj = DequeSub()
            obj.append(obj)
            return obj

        def construct_dict_sub():
            obj = DictSub()
            obj[0] = obj
            return obj

        def construct_ordered_dict_sub():
            obj = OrderedDictSub()
            obj[0] = obj
            return obj

        def construct_counter():
            obj = collections.Counter()
            obj[0] = obj
            return obj

        for construct_fn in (
            construct_list_sub,
            construct_deque_sub,
            construct_dict_sub,
            construct_ordered_dict_sub,
            construct_counter,
        ):
            out = fn(torch.ones(3), construct_fn)
            self.assertIs(out[0], out)
            out = opt_fn(torch.ones(3), construct_fn)
            self.assertIs(out[0], out)

    def test_non_self_referential_list_is_not_stored(self):
        # Non-self referential list should not be stored as a temporary variable.
        def fn(x):
            l = [1, 2, 3]
            return x, l

        def gn(x):
            l = [1, 2, 3]
            l.append(l)
            return x, l

        def hook(instructions: list[dis.Instruction]):
            from torch._dynamo.bytecode_transformation import create_dup_top

            dup_top_inst = create_dup_top().opname
            for i, inst in enumerate(instructions):
                if inst.opname == "BUILD_LIST" and i + 2 < len(instructions):
                    assert not (  # noqa: S101
                        instructions[i + 1].opname == dup_top_inst
                        and instructions[i + 2].opname == "STORE_FAST"
                    ), "found list stored as tmp"

        with self.register_bytecode_hook(hook):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            opt_fn(torch.ones(3))
            with self.assertRaisesRegex(AssertionError, "found list stored as tmp"):
                opt_gn = torch.compile(gn, backend="eager", fullgraph=True)
                opt_gn(torch.ones(3))

    def test_opaque_reference_as_python_constant(self):
        """TSOV.as_python_constant must succeed for reference-type opaque
        objects. Without this, __eq__ between two opaque objects graph breaks.
        """
        import torch._custom_class_base
        import torch._library.opaque_object

        class Config(torch._custom_class_base.CustomClassBase):
            def __init__(self, v):
                self.v = v

            def __bool__(self):
                return True

            def __eq__(self, other):
                return isinstance(other, Config) and self.v == other.v

            def __hash__(self):
                return hash(self.v)

        torch._library.opaque_object.register_custom_class(Config, typ="symbolic")

        cfg = Config(42)

        def fn(x, cfg):
            if cfg:
                return x + 1
            return x

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt(torch.ones(4), cfg)
        self.assertEqual(result, torch.ones(4) + 1)

    def test_call_once_guard_allows_super_delegation(self):
        """_add_call_once_guard must key on (id(self), id(original_method))
        so that super().as_python_constant() between VT subclasses is not
        mistaken for a self-referential call.
        """
        from torch._dynamo.variables.base import VariableTracker

        class _Parent(VariableTracker):
            def as_python_constant(self):
                return 42

        class _Child(_Parent):
            def as_python_constant(self):
                return super().as_python_constant()

        child = _Child()
        # With name-based keying, _Child and _Parent share the same key
        # (id(self), "as_python_constant"), causing a false
        # AsPythonConstantNotImplementedError("self-referential").
        self.assertEqual(child.as_python_constant(), 42)
        self.assertTrue(child.is_python_constant())

    @parametrize("name", ["list", "list_subclass", "deque", "deque_subclass"])
    def test_self_referential_sourceful_sequence_keeps_identity(self, name):
        # The self-reference on a container passed in from outside must resolve
        # to that object, not to the throwaway built for the mutation replay.
        class ListSub(list):
            pass

        class DequeSub(collections.deque):
            pass

        obj = {
            "list": lambda: [1],
            "list_subclass": lambda: ListSub([1]),
            "deque": lambda: collections.deque([1]),
            "deque_subclass": lambda: DequeSub([1]),
        }[name]()

        def fn(x, o):
            o.append(o)
            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3), obj)
        self.assertIs(obj[1], obj)

    @parametrize(
        "name", ["dict", "dict_subclass", "ordereddict", "ordereddict_subclass"]
    )
    def test_self_referential_sourceful_mapping_keeps_identity(self, name):
        class DictSub(dict):
            pass

        class OrderedDictSub(collections.OrderedDict):
            pass

        obj = {
            "dict": lambda: {"a": 1},
            "dict_subclass": lambda: DictSub(a=1),
            "ordereddict": lambda: collections.OrderedDict(a=1),
            "ordereddict_subclass": lambda: OrderedDictSub(a=1),
        }[name]()

        def fn(x, o):
            o["self"] = o
            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3), obj)
        self.assertIs(obj["self"], obj)

    @parametrize("name", ["list", "list_subclass", "deque", "deque_subclass"])
    def test_mutated_container_aliases_after_attr_store(self, name):
        # Storing the mutated container onto another object must store that same
        # object, not the throwaway built for the mutation replay.
        class ListSub(list):
            pass

        class DequeSub(collections.deque):
            pass

        class Holder:
            pass

        obj = {
            "list": lambda: [1],
            "list_subclass": lambda: ListSub([1]),
            "deque": lambda: collections.deque([1]),
            "deque_subclass": lambda: DequeSub([1]),
        }[name]()
        holder = Holder()

        def fn(x, o, h):
            o.append(2)
            h.obj = o
            return x + 1

        torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3), obj, holder)
        self.assertIs(holder.obj, obj)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
