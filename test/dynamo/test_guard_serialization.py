# Owner(s): ["module: dynamo"]

import dataclasses
import importlib
import pickle
import sys
import types
import unittest
from collections.abc import Iterator
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, CompileId
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing
from torch.overrides import TorchFunctionMode
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils import _pytree as pytree


@dataclasses.dataclass
class _FrameState:
    f_locals: dict
    f_globals: dict
    f_code: types.CodeType
    f_builtins: dict


class GlobalModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


def global_func(x):
    return x + 1


class GlobalTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class SubclassWithMeta(torch.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.extra = extra

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, args)
        kwargs_a = pytree.tree_map_only(SubclassWithMeta, lambda x: x.a, kwargs)
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithMeta)
            return SubclassWithMeta(out_a, extra=args[0].extra)
        return out_a

    def __tensor_flatten__(self):
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert isinstance(meta, dict)
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithMeta(a, extra, outer_size, outer_stride)


class SubclassWithCustomMetadataGuard(torch.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.extra = extra

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithCustomMetadataGuard, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithCustomMetadataGuard)
            return SubclassWithCustomMetadataGuard(out_a, extra=args[0].extra)
        return out_a

    @classmethod
    def __metadata_guard__(cls, meta1, meta2):
        # Define custom metadata guard logic that only looks at "bar" to determine
        # metadata equivalence. This is more purposefully more lax than the default
        # guard behavior.
        return meta1["extra"]["bar"] == meta2["extra"]["bar"]

    def __tensor_flatten__(self):
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert isinstance(meta, dict)
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithCustomMetadataGuard(a, extra, outer_size, outer_stride)


class SubclassWithSubclassInnerTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, extra, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, a, extra, outer_size=None, outer_stride=None):
        self.a = a
        self.inner_sub = SubclassWithMeta(a + 1, extra=extra)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensor, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithSubclassInnerTensor)
            return SubclassWithSubclassInnerTensor(out_a, extra=args[0].inner_sub.extra)
        return out_a

    def __tensor_flatten__(self):
        return ["a", "inner_sub"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a = inner_tensors["a"]
        extra = inner_tensors["inner_sub"].extra
        if type(a) is torch.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return SubclassWithSubclassInnerTensor(a, extra, outer_size, outer_stride)


# defines a custom __eq__() / __hash__() to be registered as a pytree constant type
class CustomConstantType:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        # custom eq ignores b
        return self.a == other.a

    def __hash__(self):
        # custom hash ignores b
        return hash(self.a)


pytree.register_constant(CustomConstantType)


class TestGuardSerialization(torch._inductor.test_case.TestCase):
    def test_function_locals(self):
        def foo(x):
            return x + 1

        def fn(x, g):
            return g(x) + 1

        self._test_serialization("TENSOR_MATCH", fn, torch.randn(3), foo)

    def _tracefunc(self, frame, event, arg):
        if event != "call":
            return

        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=dict(frame.f_globals),
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args, **kwargs):
        # kwargs might contain a callable that generates kwargs
        kwarg_gen_fn = kwargs.get("_gen_fn", None)
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()

        self._frame_state = None
        sys.settrace(self._tracefunc)
        if isinstance(fn, torch.nn.Module):
            fn = fn.forward
        try:
            fn(*args, **kwargs)
        finally:
            sys.settrace(None)

        assert self._frame_state is not None

        # Set f_locals from regenerated kwargs to handle exhausted input iterators
        # NB: This is super janky and might cause unforeseen problems
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()
            for key in self._frame_state.f_locals.keys():
                if key in kwargs and isinstance(kwargs[key], Iterator):
                    self._frame_state.f_locals[key] = kwargs[key]

        def guard_filter_fn(guards):
            ret = [
                g.guard_type == guard_type or guard_type in g.derived_guard_types
                for g in guards
            ]
            self.assertTrue(any(ret))
            return ret

        ref_gm = None
        loaded_gm = None

        def transform(instructions: list, code_options: dict[str, object]):
            """
            The goal is here is not to reimplement dynamo, but just to have a
            simplified version to extract the state from symbolic convert.
            Should not work on all cases, but should work on simple functions
            in this test file.
            """
            nonlocal ref_gm
            nonlocal loaded_gm

            tracer = InstructionTranslator(
                instructions,
                self._frame_state.f_code,
                self._frame_state.f_locals,
                self._frame_state.f_globals,
                self._frame_state.f_builtins,
                fn.__closure__ or (),
                torch.overrides._get_current_function_mode_stack(),
                code_options,
                torch._dynamo.lookup_backend("eager"),
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
                package=None,
            )
            with (
                compile_context(CompileContext(CompileId(0, 0))),
                tracing(tracer.output.tracing_context),
                tracer.set_current_tx(),
                get_metrics_context(),
                dynamo_timed(""),
            ):
                tracer.run()

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    guards_serialization_mode="save",
                )
                ref_gm = check_fn_manager.guard_manager
                guards_state = check_fn_manager.guards_state
                self.assertIsNotNone(guards_state)
                guards_state = pickle.loads(guards_state)

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    guards_state.output_graph,
                    guards_serialization_mode="load",
                    shape_code_parts=guards_state.shape_code_parts,
                )
                loaded_gm = check_fn_manager.guard_manager

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            self._frame_state = None

        self.assertIsNotNone(ref_gm)
        self.assertIsNotNone(loaded_gm)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        self.assertIsInstance(inputs, dict)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))

    def test_tensor_match(self):
        def f(x: torch.Tensor):
            return x + 1

        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", f, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float32)}, True
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(3, dtype=torch.float32)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float64)}, False
        )
        self._test_check_fn(ref, loaded, {"x": None}, False)

    def test_not_present_in_generic_dict(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x + 1

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization(
            "NOT_PRESENT_IN_GENERIC_DICT", fn, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(ref, loaded, {"m": m}, True)

        m.forward = types.MethodType(lambda x: x + 2, m)
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_hasattr_serialization(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1

            def forward(self, x: torch.Tensor):
                if hasattr(self, "a"):
                    return x + self.a
                else:
                    return x + 2

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization("HASATTR", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        delattr(m, "a")
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_type_match(self):
        class LocalModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x + 1

        m = LocalModule()

        def fn(m, x):
            return m(x)

        with self.assertRaisesRegex(
            TypeError, "Please define the class at global scope"
        ):
            self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))

        m = GlobalModule()
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        self._test_check_fn(ref, loaded, {"m": GlobalModule()}, True)
        self._test_check_fn(ref, loaded, {"m": torch.nn.Module()}, False)

    def test_tensor_subclass_metadata_match(self):
        class LocalSubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, a, outer_size=None, outer_stride=None):
                if outer_size is None:
                    outer_size = a.size()
                if outer_stride is None:
                    outer_stride = a.stride()

                shape = outer_size
                kwargs = {}
                kwargs["strides"] = outer_stride
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

            def __init__(self, a, outer_size=None, outer_stride=None):
                self.a = a

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if kwargs is None:
                    kwargs = {}
                args_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, args)
                kwargs_a = pytree.tree_map_only(LocalSubclass, lambda x: x.a, kwargs)
                out_a = func(*args_a, **kwargs_a)
                if isinstance(out_a, torch.Tensor):
                    return LocalSubclass(out_a)
                return out_a

            def __tensor_flatten__(self):
                return ["a"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                assert meta is None
                a = inner_tensors["a"]
                if type(a) is torch.Tensor:
                    assert outer_size is not None
                    assert outer_stride is not None
                return LocalSubclass(a, outer_size, outer_stride)

        def fn(x):
            return x * 2

        # === example subclass defined locally (error) ===
        local_sub = LocalSubclass(torch.randn(3))
        with self.assertRaisesRegex(
            PackageError, "Please define the class at global scope"
        ):
            self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, local_sub)

        # === example subclass with None extra metadata ===
        from torch.testing._internal.two_tensor import TwoTensor

        tt = TwoTensor(torch.randn(3), torch.randn(3))
        ref, loaded = self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, tt)
        self._test_check_fn(ref, loaded, {"x": tt}, True)
        self._test_check_fn(ref, loaded, {"x": torch.ones_like(tt)}, True)

        # used below for convenience; returned func accepts some metadata and whether the
        # guard is expected to pass for the given subclass type
        def _get_meta_test_check_fn(ref, loaded, subclass_type):
            def _f(meta, expected, ref=ref, loaded=loaded, subclass_type=subclass_type):
                self._test_check_fn(
                    ref,
                    loaded,
                    {"x": subclass_type(torch.randn(3), extra=meta)},
                    expected,
                )

            return _f

        # === example subclass with extra metadata ===
        extra_meta = {
            "foo": 5,
            "bar": "hello",
        }
        sub = SubclassWithMeta(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(ref, loaded, SubclassWithMeta)
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with custom metadata guard logic ===
        sub = SubclassWithCustomMetadataGuard(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithCustomMetadataGuard
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"; custom logic says this is okay
        check_with_meta({"foo": 6, "bar": "hello"}, True)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

        # === example subclass with subclass inner tensor ===
        sub = SubclassWithSubclassInnerTensor(torch.randn(3), extra=extra_meta)
        ref, loaded = self._test_serialization(
            "TENSOR_SUBCLASS_METADATA_MATCH", fn, sub
        )
        self._test_check_fn(ref, loaded, {"x": sub}, True)
        check_with_meta = _get_meta_test_check_fn(
            ref, loaded, SubclassWithSubclassInnerTensor
        )
        check_with_meta(dict(extra_meta), True)
        # different "foo"
        check_with_meta({"foo": 6, "bar": "hello"}, False)
        # different "bar"
        check_with_meta({"foo": 5, "bar": "world"}, False)

    def test_equals_match(self):
        def fn(x, y):
            # CustomConstantType is registered as a pytree constant so this should
            # result in an EQUALS_MATCH guard.
            if x in y:
                return torch.zeros(3)
            return torch.ones(3)

        x = CustomConstantType(4, 5)
        y = [CustomConstantType(2, 3), CustomConstantType(4, 5)]
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        # custom __eq__ says that CustomConstantType(4, 5) == CustomConstantType(4, 9)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": CustomConstantType(4, 5),
                "y": [CustomConstantType(2, 3), CustomConstantType(4, 9)],
            },
            True,
        )
        self._test_check_fn(ref, loaded, {"x": x, "y": []}, False)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": x,
                "y": [CustomConstantType(2, 3), CustomConstantType(6, 7)],
            },
            False,
        )

    def test_constant_match(self):
        # === bool constant ===
        def fn(x, y):
            if y:
                return x + 1
            return x + 2

        x = torch.randn(3)
        y = True

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": True}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": True}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": False}, False)

        # === None constant ===
        def fn(x, y):
            if y is None:
                return x + 1
            return x + 2

        x = torch.randn(3)
        y = None

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": None}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": None}, True)
        # guard should fail for non-None y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 5}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": True}, False)

        # === int constant ===
        def fn(x, y):
            return x + y

        x = torch.randn(3)
        y = 5

        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, x, y)
        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 5}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4), "y": 5}, True)
        # guard should fail for different y value
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "y": 6}, False)

    def test_nn_module(self):
        def fn(m, x):
            return m(x)

        m = GlobalModule()
        x = torch.randn(3)

        # config setting controls whether the NN_MODULE guard is installed
        with patch("torch._dynamo.config.inline_inbuilt_nn_modules", False):
            # we don't support NN_MODULE because it adds an ID_MATCH guard, and we don't
            # support that in serialization
            with self.assertRaisesRegex(
                PackageError, "NN_MODULE guard cannot be serialized."
            ):
                self._test_serialization("NN_MODULE", fn, m, x)

    def test_function_match(self):
        def fn(x):
            # usage of this context manager installs a FUNCTION_MATCH guard
            with torch.no_grad():
                y = x * 2
            return y

        x = torch.randn(3)

        # we don't support FUNCTION_MATCH because it adds an ID_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "FUNCTION_MATCH guard cannot be serialized."
        ):
            self._test_serialization("FUNCTION_MATCH", fn, x)

    def test_closure_match(self):
        def fn(x):
            # usage of this global function installs a CLOSURE_MATCH guard
            return global_func(x)

        x = torch.randn(3)

        # we don't support CLOSURE_MATCH because it adds a FUNCTION_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLOSURE_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLOSURE_MATCH", fn, x)

    def test_sequence_length(self):
        # tuple input installs a SEQUENCE_LENGTH guard
        def fn(t, x):
            return t[1] + x

        t = tuple(torch.randn(3) for _ in range(3))
        x = torch.randn(3)

        ref, loaded = self._test_serialization("SEQUENCE_LENGTH", fn, t, x)
        self._test_check_fn(ref, loaded, {"x": x, "t": t}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": torch.randn(3),
                "t": tuple(torch.randn(3) for _ in range(3)),
            },
            True,
        )
        # different types in tuple of same length shouldn't fail SEQUENCE_LENGTH guard
        # (it should fail the separate TYPE_MATCH guard but that isn't tested here)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "t": (0, 1, 2)}, True)
        # different length tuple
        self._test_check_fn(
            ref,
            loaded,
            {
                "x": torch.randn(3),
                "t": tuple(torch.randn(3) for _ in range(4)),
            },
            False,
        )

    def test_tuple_iterator_len(self):
        def fn(t, x):
            if len(list(t)) > 2:
                return x * 2
            return x + 1

        tup = (1, 2, 3)
        x = torch.randn(3)

        # func to generate kwargs; useful for avoiding iterator exhaustion issues
        def _gen_kwargs(tup=tup, x=x):
            return {"t": iter(tup), "x": x}

        ref, loaded = self._test_serialization(
            "TUPLE_ITERATOR_LEN", fn, _gen_fn=_gen_kwargs
        )

        # same tuple
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": x}, True)
        self._test_check_fn(ref, loaded, {"t": iter(tup), "x": torch.randn(4)}, True)
        # same length tuple, different contents
        self._test_check_fn(ref, loaded, {"t": iter((3, 2, 1)), "x": x}, True)
        self._test_check_fn(
            ref, loaded, {"t": iter((3, 2, 1)), "x": torch.randn(4)}, True
        )
        # different tuple lengths
        self._test_check_fn(ref, loaded, {"t": iter((1, 2)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2)), "x": torch.randn(4)}, False
        )
        self._test_check_fn(ref, loaded, {"t": iter((1, 2, 3, 4)), "x": x}, False)
        self._test_check_fn(
            ref, loaded, {"t": iter((1, 2, 3, 4)), "x": torch.randn(4)}, False
        )

    def test_range_iterator_match(self):
        def fn(x, r):
            y = x
            for val in r:
                y = x + val
            return y

        x = torch.randn(3)

        def _gen_kwargs(x=x):
            return {"x": x, "r": iter(range(2, 15, 3))}

        ref, loaded = self._test_serialization(
            "RANGE_ITERATOR_MATCH", fn, _gen_fn=_gen_kwargs
        )

        # same range
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 15, 3))}, True
        )
        # equivalent even with different end
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 16, 3))}, True)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 16, 3))}, True
        )
        # different start
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(1, 15, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(1, 15, 3))}, False
        )
        # different end resulting in different values
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 18, 3))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 18, 3))}, False
        )
        # different step
        self._test_check_fn(ref, loaded, {"x": x, "r": iter(range(2, 15, 4))}, False)
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(4), "r": iter(range(2, 15, 4))}, False
        )

    def test_dict_version(self):
        def fn(x):
            return pytree.tree_leaves(x)[0] + 1

        with self.assertRaisesRegex(
            PackageError, "DICT_VERSION guard cannot be serialized."
        ):
            self._test_serialization("DICT_VERSION", fn, {"t": torch.randn(3)})

    def test_dict_contains(self):
        def fn(x):
            if x.__contains__("t"):
                return x["t"] + 1
            else:
                return torch.ones(3)

        ref, loaded = self._test_serialization(
            "DICT_CONTAINS", fn, {"t": torch.randn(3)}
        )

        self._test_check_fn(ref, loaded, {"x": {"t": torch.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)
        self._test_check_fn(
            ref, loaded, {"x": {"t": torch.randn(3), "d": torch.randn(3)}}, True
        )

    def test_bool_match(self):
        def fn(x, b):
            if b:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("BOOL_MATCH", fn, torch.randn(3), True)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, False)

    def test_none_match(self):
        def fn(x, b):
            if b is None:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("NONE_MATCH", fn, torch.randn(3), None)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, False)

    def test_id_match(self):
        def fn(x):
            return x + id(x)

        with self.assertRaisesRegex(
            PackageError, "ID_MATCH guard cannot be serialized."
        ):
            self._test_serialization("ID_MATCH", fn, torch.randn(3))

    def test_dispatch_key_set_match(self):
        def fn(x, dks):
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(3)
        dks = torch._C._dispatch_keys(x)
        ref, loaded = self._test_serialization("DISPATCH_KEY_SET_MATCH", fn, x, dks)

        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, True)

        x = torch.randn(3, device="meta")
        dks = torch._C._dispatch_keys(x)
        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, False)

    def test_name_match(self):
        def fn(x, y):
            return torch.cond(x, lambda x: y + 1, lambda x: y - 1, (y,))

        x = torch.tensor(True)
        y = torch.randn(3)
        ref, loaded = self._test_serialization("NAME_MATCH", fn, x, y)

        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)

        op = importlib.import_module("torch._higher_order_ops.cond").cond_op
        prev, op.__name__ = op.__name__, ""
        try:
            self._test_check_fn(ref, loaded, {"x": x, "y": y}, False)
        finally:
            op.__name__ = prev

    def test_dual_level(self):
        def fn(x):
            with torch.autograd.forward_ad.dual_level():
                return x + 1

        x = torch.randn(3)
        ref, loaded = self._test_serialization("DUAL_LEVEL", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch.autograd.forward_ad.dual_level():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_functorch_stack_match(self):
        # Test when functorch stack is empty.
        def fn(x):
            return torch.func.jvp(torch.sin, (x,), (x,))

        x = torch.randn(3, 4)
        ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)

        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch._functorch.vmap.vmap_increment_nesting(2, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        def fn(x):
            def g(x):
                return torch.vmap(torch.func.grad(torch.sin))(x)

            return torch.vmap(g)(x)

        x = torch.randn(4, 5)
        ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Test when there are more than 0 functorch layers.
        # Simulate the case where torch.compile is nested inside eager transforms.

        # Case 1: vmap
        def fn(x):
            return x.sum()

        ref = loaded = None

        def run(x):
            nonlocal ref, loaded
            # Turn off automatic dynamic shape to so that functionalization
            # doesn't produce extra SymInt to serialize.
            with torch._dynamo.config.patch(automatic_dynamic_shapes=False):
                ref, loaded = self._test_serialization("FUNCTORCH_STACK_MATCH", fn, x)
            return fn(x)

        torch.vmap(run)(x)

        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 2: grad
        x = torch.randn(3, 2)
        ref = loaded = None
        torch.func.grad(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, True)
            with torch._functorch.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 3: jvp + vmap
        x = torch.randn(3, 4)
        ref = loaded = None

        def fn(x):
            return torch.func.jvp(torch.sin, (x,), (x,))

        torch.func.jvp(torch.vmap(run), (x,), (x,))
        self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.jvp_increment_nesting():
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            with torch._functorch.eager_transforms.jvp_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 4: functionalize
        x = torch.randn(3, 2)
        ref = loaded = None
        torch.func.functionalize(run)(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)

        torch._C._functorch._func_increment_nesting(True)
        try:
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch._C._functorch._func_decrement_nesting()

        with torch._functorch.eager_transforms.jvp_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

        # Case 5: vmap + grad
        def fn(x):
            return x.sum()

        x = torch.randn(3, 2)
        ref = loaded = None
        torch.vmap(torch.func.grad(run))(x)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            with torch._functorch.eager_transforms.grad_increment_nesting():
                self._test_check_fn(ref, loaded, {"x": x}, True)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
                self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.vmap.vmap_increment_nesting(1, "error"):
            self._test_check_fn(ref, loaded, {"x": x}, False)

        with torch._functorch.eager_transforms.grad_increment_nesting():
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_duplicate_input(self):
        def fn(x, x_):
            return x + x_

        x = torch.randn(3, 2)
        with self.assertRaisesRegex(
            PackageError, "DUPLICATE_INPUT guard cannot be serialized"
        ):
            self._test_serialization("DUPLICATE_INPUT", fn, x, x)

    def test_weakref_alive(self):
        mod = torch.nn.Linear(10, 10, bias=False)
        for p in mod.parameters():
            p.grad = torch.rand_like(p)

        opt = torch.optim.SGD(mod.parameters(), lr=0.1)

        def fn():
            params = []
            opt._init_group(opt.param_groups[0], params, [], [])
            return params[0].sum()

        with self.assertRaisesRegex(
            PackageError, "WEAKREF_ALIVE guard cannot be serialized"
        ):
            with torch.set_grad_enabled(False):
                self._test_serialization("WEAKREF_ALIVE", fn)

    def test_mapping_keys_check(self):
        def fn(mp):
            return mp["a"] + 1

        mp = types.MappingProxyType({"a": torch.randn(3, 2), "b": torch.randn(3, 2)})
        ref, loaded = self._test_serialization("MAPPING_KEYS_CHECK", fn, mp)
        self._test_check_fn(ref, loaded, {"mp": mp}, True)
        self._test_check_fn(
            ref,
            loaded,
            {
                "mp": types.MappingProxyType(
                    {"b": torch.randn(3, 2), "a": torch.randn(3, 2)}
                )
            },
            False,
        )
        self._test_check_fn(
            ref, loaded, {"mp": types.MappingProxyType({"a": torch.randn(3, 2)})}, False
        )

    def test_dict_keys_match(self):
        def fn(x):
            ret = 1
            for k in x:
                ret += x[k]
            return ret

        x = {"a": torch.randn(3, 2), "b": torch.randn(3, 2)}
        ref, loaded = self._test_serialization("DICT_KEYS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(
            ref,
            loaded,
            {"x": {"b": torch.randn(3, 2), "a": torch.randn(3, 2)}},
            False,
        )
        self._test_check_fn(ref, loaded, {"x": {"a": torch.randn(3, 2)}}, False)

    @torch._dynamo.config.patch("skip_nnmodule_hook_guards", False)
    def test_empty_nn_module_hooks_dict(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x + 1

        m = Module()

        def fn(x):
            return m(x)

        x = torch.ones(2, dtype=torch.float32)
        ref, loaded = self._test_serialization("EMPTY_NN_MODULE_HOOKS_DICT", fn, x)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, True)

        h = m.register_forward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_forward_pre_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

        h = m.register_backward_hook(lambda *args, **kwargs: None)
        self._test_check_fn(ref, loaded, {"m": m, "x": x}, False)
        h.remove()

    def test_grad_mode(self):
        def fn(x):
            return x + 1

        x = torch.randn(3, 2)
        with torch.enable_grad():
            ref, loaded = self._test_serialization("GRAD_MODE", fn, x)
        with torch.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_deterministic_algorithms(self):
        def fn(x):
            return x + 1

        deterministic_restore = torch.are_deterministic_algorithms_enabled()
        try:
            x = torch.randn(3, 2)
            torch.use_deterministic_algorithms(True)
            ref, loaded = self._test_serialization("DETERMINISTIC_ALGORITHMS", fn, x)
            torch.use_deterministic_algorithms(False)
            self._test_check_fn(ref, loaded, {"x": x}, False)
            torch.use_deterministic_algorithms(True)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch.use_deterministic_algorithms(deterministic_restore)

    def test_torch_function_state(self):
        def fn(x):
            return x + 1

        x = torch.randn(3, 2)

        class LocalTorchFunctionMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with GlobalTorchFunctionMode():
            ref, loaded = self._test_serialization("TORCH_FUNCTION_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
        self._test_check_fn(ref, loaded, {"x": x}, False)
        with GlobalTorchFunctionMode():
            with torch._C.DisableTorchFunction():
                self._test_check_fn(ref, loaded, {"x": x}, False)
        with self.assertRaisesRegex(
            PackageError,
            "defined in local scope. Please define the class at global scope",
        ):
            with LocalTorchFunctionMode():
                ref, loaded = self._test_serialization("TORCH_FUNCTION_STATE", fn, x)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_training_state(self):
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        param_group = FSDPParamGroup(
            [],  # params: List[nn.Parameter],
            (torch.nn.Linear(1, 1),),  # module: nn.Module,
            None,  # mesh_info: FSDPMeshInfo,
            None,  # post_forward_mesh_info: Optional[FSDPMeshInfo],
            torch.device("cpu"),  # device: torch.device,
            None,  # shard_placement_fn: Optional[Callable],
            None,  # mp_policy: MixedPrecisionPolicy,
            None,  # offload_policy: OffloadPolicy,
        )

        def fn(x):
            with param_group.use_training_state(TrainingState.FORWARD):
                if param_group._training_state == TrainingState.FORWARD:
                    return x + 1
                else:
                    return x - 1

        x = torch.randn(3, 2)

        with torch.enable_grad():
            ref, loaded = self._test_serialization("FSDP_TRAINING_STATE", fn, x)
        with torch.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_default_device(self):
        device = torch.get_default_device()

        def fn(x):
            return x + 1

        x = torch.randn(3, 2)
        try:
            torch.set_default_device("cpu")
            ref, loaded = self._test_serialization("DEFAULT_DEVICE", fn, x)
            torch.set_default_device("meta")
            self._test_check_fn(ref, loaded, {"x": x}, False)
            torch.set_default_device("cpu")
            self._test_check_fn(ref, loaded, {"x": x}, True)
        finally:
            torch.set_default_device(device)

    def test_shape_env(self):
        def fn(x):
            return x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)

        x = torch.randn(3, 2)
        torch._dynamo.mark_dynamic(x, 0, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(2, 2)}, False)

        x = torch.randn(3, 3, 2)
        torch._dynamo.mark_dynamic(x, 1, min=3, max=10)
        ref, loaded = self._test_serialization("SHAPE_ENV", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 4, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 10, 2)}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 11, 2)}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3, 2, 2)}, False)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
