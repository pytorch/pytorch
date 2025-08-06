# Owner(s): ["module: dynamo"]

import contextlib
import dis
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import IS_FBCODE
from torch.testing._internal.inductor_utils import requires_triton
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_tensor_descriptor_host_tma,
)


def _filter_instructions(instructions, opname):
    return list(filter(lambda x: x.opname == opname, instructions))


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

    def test_graph_break_in_wrapped_user_function(self):
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()
            assert not torch.is_grad_enabled()
            return x + 2

        @torch.compile(backend="eager")
        def gn(x):
            x = torch.no_grad()(fn)(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()
            assert torch.is_grad_enabled()
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
                assert torch.compiler.is_compiling()
                assert not torch.is_grad_enabled()
                return x + self.b

        obj = Foo()

        @torch.compile(backend="eager")
        def gn(x):
            obj.fn = torch.no_grad()(obj.fn)
            x = obj.fn(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()
            assert torch.is_grad_enabled()
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
                assert torch.compiler.is_compiling()
                assert not torch.is_grad_enabled()
                return x + b

            x = fn(x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()
            assert torch.is_grad_enabled()
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
            assert torch.compiler.is_compiling()
            assert not torch.is_grad_enabled()
            return x + 2

        @torch.compile(backend="eager")
        def gn(x):
            x = torch.no_grad()(_skipped_function_for_test_reconstruct)(fn, x)
            # reconstruction failure would cause a skipped frame
            assert torch.compiler.is_compiling()
            assert torch.is_grad_enabled()
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)

    @requires_triton()
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

        x = torch.randn(128, 128, device="cuda")

        ref = create_tma(x)
        res = torch.compile(create_tma, backend="eager")(x)
        self.assertEqual(ref[1].desc, res[1].desc)

    @requires_triton()
    @unittest.skipIf(
        not has_triton_tensor_descriptor_host_tma(),
        "Test requires triton.tools.tensor_descriptor API",
    )
    def test_tma_stable_reconstruct(self):
        import triton

        def create_tma(tensor):
            tma = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                tensor,
                [32, 32],
            )
            return tensor + 1, tma

        x = torch.randn(128, 128, device="cuda")

        ref = create_tma(x)
        res = torch.compile(create_tma, backend="eager")(x)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
