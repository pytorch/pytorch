# Owner(s): ["module: dynamo"]

import collections
import contextlib
import dis
import enum
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.utils.checkpoint
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import (
    has_triton_experimental_host_tma,
    has_triton_tensor_descriptor_host_tma,
)


_FROZEN_NUMBERS = frozenset({1, 2, 3})
_CLASS_HASH_EFFECTS = []


class _IgnoreClassOptionsMeta(type):
    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace)


def _filter_instructions(instructions, opname):
    return list(filter(lambda x: x.opname == opname, instructions))


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

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_module_membership_partial_dict_hash_effects(self):
        hash_calls = []

        class Key(enum.Enum):
            ITEM = 1

            def __hash__(self):
                hash_calls.append("hash")
                return 1

        modules = torch.nn.ModuleDict({"ok": torch.nn.Identity()})
        modules.torchdynamo_force_dynamic = False

        def fn(x):
            payload = {Key.ITEM: 1, "tensor": x}
            y = x + 1
            torch._dynamo.graph_break()
            try:
                _ = payload in modules
            except TypeError as error:
                message = str(error)
            else:
                message = "no error"
            return y + len(hash_calls), payload, message

        counter = torch._dynamo.testing.CompileCounter()
        executions = []

        def backend(gm, example_inputs):
            forward = counter(gm, example_inputs)

            def execute(*args):
                executions.append("forward")
                return forward(*args)

            return execute

        x = torch.arange(2, dtype=torch.float32)
        expected = fn(x)
        eager_hash_calls = list(hash_calls)
        hash_calls.clear()
        result = torch.compile(fn, backend=backend, fullgraph=False)(x)
        self.assertGreater(counter.op_count, 0)
        self.assertTrue(executions)
        self.assertEqual(result[2], "unhashable type: 'dict'")
        self.assertEqual(result[2], expected[2])
        self.assertIs(result[1]["tensor"], x)
        self.assertIs(expected[1]["tensor"], x)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(hash_calls, eager_hash_calls)

    @parametrize("use_bool", [False, True])
    def test_truth_testing_partial_constant_dict(self, use_bool):
        hash_calls = []

        class Key(enum.Enum):
            ITEM = 7

            def __hash__(self):
                hash_calls.append("hash")
                if len(hash_calls) == 2:
                    raise RuntimeError("unexpected native key rehash")
                return 7

        def fn(x, values):
            if use_bool:
                return x + int(bool(values))
            if values:
                return x + 1
            return x - 1

        x = torch.arange(2, dtype=torch.float32)
        values = {Key.ITEM: 19, "tensor": x}
        hash_calls.clear()
        expected = fn(x, values)
        counter = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=counter, fullgraph=True)(x, values)
        self.assertEqual(result, expected)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_unbound_setattr_partial_dict_name_hash_effects(self):
        hash_calls = []

        class Key(enum.Enum):
            ITEM = 7

            def __hash__(self):
                hash_calls.append("hash")
                return 7

        class Holder:
            pass

        def fn(x, holder, name):
            prefix = x + 1
            torch._dynamo.graph_break()
            try:
                object.__setattr__(holder, name, 17)
            except TypeError as exc:
                return prefix, str(exc)
            return x - 1, "accepted"

        x = torch.arange(2, dtype=torch.float32)
        holder = Holder()
        name = {Key.ITEM: 19, "tensor": x}
        hash_calls.clear()
        expected = fn(x, holder, name)
        self.assertEqual(expected, (x + 1, "attribute name must be string, not 'dict'"))
        self.assertEqual(hash_calls, [])
        counter = torch._dynamo.testing.CompileCounter()
        executions = []

        def backend(gm, inputs):
            forward = counter(gm, inputs)

            def execute(*args):
                result = forward(*args)
                executions.append(1)
                return result

            return execute

        result = torch.compile(fn, backend=backend, fullgraph=False)(x, holder, name)
        self.assertEqual(result, expected)
        self.assertEqual(holder.__dict__, {})
        self.assertIs(name["tensor"], x)
        self.assertEqual(counter.frame_count, 1)
        self.assertGreater(counter.op_count, 0)
        self.assertEqual(executions, [1])
        self.assertEqual(hash_calls, [])

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_unused_class_kwargs_hash_effect_order(self):
        hash_calls = _CLASS_HASH_EFFECTS
        hash_calls.clear()
        self.addCleanup(hash_calls.clear)

        class Key(enum.Enum):
            ITEM = 7

            def __hash__(self):
                _CLASS_HASH_EFFECTS.append("hash")
                return 7

        def fn(x, options):
            class Generated(metaclass=_IgnoreClassOptionsMeta, options=options):
                bias = len(_CLASS_HASH_EFFECTS)

            return x + Generated.bias

        x = torch.arange(2, dtype=torch.float32)
        options = {Key.ITEM: 19}
        hash_calls.clear()
        expected = fn(x, options)
        self.assertEqual(expected, fn(x.double(), options).float())
        self.assertEqual(hash_calls, [])
        counter = torch._dynamo.testing.CompileCounter()
        executions = []

        def backend(gm, inputs):
            forward = counter(gm, inputs)

            def execute(*args):
                result = forward(*args)
                executions.append(1)
                return result

            return execute

        result = torch.compile(fn, backend=backend, fullgraph=True)(x, options)
        self.assertEqual(counter.frame_count, 1)
        self.assertGreater(counter.op_count, 0)
        self.assertEqual(executions, [1])
        self.assertEqual(list(options.values()), [19])
        self.assertIs(next(iter(options)), Key.ITEM)
        self.assertEqual(result, expected)

    @parametrize("original", [False, True])
    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_checkpoint_partial_dict_hash_effects(self, original):
        hash_calls = []

        class Key(enum.Enum):
            ITEM = 7

            def __hash__(self):
                hash_calls.append("hash")
                return 7

        class State:
            def __init__(self, payload):
                self.payload = payload

        state = None

        def current_fn(y, payload):
            y = y + len(payload)

            def body(z):
                nonlocal state
                state = payload
                return z.sin()

            return torch.utils.checkpoint.checkpoint(body, y, use_reentrant=False) + 1

        def original_fn(y, holder):
            y = y + len(holder.payload)

            def body(z):
                holder.payload = 17
                return z.sin()

            return torch.utils.checkpoint.checkpoint(body, y, use_reentrant=False) + 1

        x = torch.arange(2, dtype=torch.float32, requires_grad=True)
        payload = {Key.ITEM: 19, "tensor": x}
        holder = State(payload)
        fn = original_fn if original else current_fn
        args = (x, holder) if original else (x, payload)
        hash_calls.clear()
        expected = fn(*args)
        expected_state = holder.payload if original else state
        self.assertEqual(hash_calls, [])
        state = None
        holder.payload = payload
        counter = torch._dynamo.testing.CompileCounter()
        executions = []

        def backend(gm, inputs):
            forward = counter(gm, inputs)

            def execute(*args):
                result = forward(*args)
                executions.append(1)
                return result

            return execute

        result = torch.compile(fn, backend=backend)(*args)
        self.assertEqual(result, expected)
        if original:
            self.assertEqual(holder.payload, expected_state)
        else:
            self.assertIs(state, expected_state)
        self.assertIs(payload["tensor"], x)
        self.assertGreater(counter.frame_count, 0)
        self.assertGreater(counter.op_count, 0)
        self.assertEqual(len(executions), counter.frame_count)
        # Preserve the API parent's hash effects for these two snapshot paths.
        self.assertEqual(hash_calls, ["hash"] * (1 if original else 3))

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

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
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

        x = torch.randn(128, 128, device=GPU_TYPE)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        ref = create_tma(x)
        res = torch.compile(create_tma, backend=backend)(x)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(ref, res)

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

    def test_return_python_enum_constant(self):
        class Choice(enum.Enum):
            A = 7

        def fn(x):
            member = Choice(7)
            return x + 1, member, member

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        first = compiled(x)
        second = compiled(-x)
        self.assertEqual(first, fn(x))
        self.assertEqual(second, fn(-x))
        self.assertIs(first[1], Choice.A)
        self.assertIs(first[1], first[2])
        self.assertIs(second[1], first[1])
        self.assertIs(second[1], second[2])
        self.assertEqual(counter.frame_count, 1)

    @parametrize("constructor_name", [False, True])
    def test_return_derived_frozenset(self, constructor_name):
        if constructor_name:

            def fn(x):
                frozenset
                result = _FROZEN_NUMBERS.difference({2})
                return x + 1, result, result

            self.assertIn("frozenset", fn.__code__.co_names)
        else:

            def fn(x):
                result = _FROZEN_NUMBERS.difference({2})
                return x + 1, result, result

            self.assertNotIn("frozenset", fn.__code__.co_names)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        first = compiled(x)
        second = compiled(-x)
        self.assertEqual(first, fn(x))
        self.assertEqual(second, fn(-x))
        self.assertIs(type(first[1]), frozenset)
        self.assertIs(first[1], first[2])
        self.assertIs(second[1], first[1])
        self.assertIs(second[1], second[2])
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 1)

    @parametrize("num_items", [5, 6])
    def test_return_colliding_frozenset_preserves_iteration(self, num_items):
        def fn(x, items):
            result = frozenset(items).difference(())
            return x + 1, result, result

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        items = tuple(range(0, num_items * 8, 8))
        first = compiled(x, items)
        second = compiled(-x, items)
        eager_first = fn(x, items)
        eager_second = fn(-x, items)
        self.assertEqual(first, eager_first)
        self.assertEqual(second, eager_second)
        self.assertIs(type(first[1]), frozenset)
        self.assertIs(first[1], first[2])
        self.assertIs(second[1], second[2])
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 1)
        self.assertEqual(tuple(first[1]), tuple(eager_first[1]))
        self.assertEqual(tuple(second[1]), tuple(eager_second[1]))

    def test_compile_frozenset_with_fake_id_declines(self):
        def fn(x):
            obj = [1, 2, 3]
            return x + 1, frozenset((id(obj),))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Reconstruction of FakeIdVariable"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))

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
