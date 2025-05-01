# Owner(s): ["module: dynamo"]

import dataclasses
import importlib
import pickle
import sys
import types

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.guards import CheckFunctionManager, CompileId
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing
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
        # define custom metadata guard logic
        # TODO: find a nice way to test logic that is not equivalent to the default behavior
        return meta1["extra"] == meta2["extra"]

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


class SubclassWithSubclassInnerTensors(torch.Tensor):
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
            SubclassWithSubclassInnerTensors, lambda x: x.a, args
        )
        kwargs_a = pytree.tree_map_only(
            SubclassWithSubclassInnerTensors, lambda x: x.a, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        if isinstance(out_a, torch.Tensor):
            assert isinstance(args[0], SubclassWithSubclassInnerTensors)
            return SubclassWithSubclassInnerTensors(
                out_a, extra=args[0].inner_sub.extra
            )
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
        return SubclassWithSubclassInnerTensors(a, extra, outer_size, outer_stride)


class TestGuardSerialization(torch._inductor.test_case.TestCase):
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
        self._frame_state = None
        sys.settrace(self._tracefunc)
        if isinstance(fn, torch.nn.Module):
            fn = fn.forward
        try:
            fn(*args, **kwargs)
        finally:
            sys.settrace(None)

        assert self._frame_state is not None

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
                [],  # TODO tf_mode_stack,
                code_options,
                lambda gm, *args, **kwargs: gm.forward,
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
            )
            with compile_context(CompileContext(CompileId(0, 0))), tracing(
                tracer.output.tracing_context
            ), tracer.set_current_tx(), get_metrics_context(), dynamo_timed(""):
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

        local_sub = LocalSubclass(torch.randn(3))
        with self.assertRaisesRegex(
            RuntimeError, "Please define the class at global scope"
        ):
            self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, local_sub)

        # use TwoTensor as an example of a subclass with None for extra metadata
        from torch.testing._internal.two_tensor import TwoTensor

        tt = TwoTensor(torch.randn(3), torch.randn(3))
        self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, tt)

        # example subclass with extra metadata
        extra_meta = {
            "foo": 5,
            "bar": "hello",
        }
        sub = SubclassWithMeta(torch.randn(3), extra=extra_meta)
        self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, sub)

        # example subclass with custom metadata guard logic
        sub2 = SubclassWithCustomMetadataGuard(torch.randn(3), extra=extra_meta)
        self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, sub2)

        # example subclass with subclass inner tensor
        sub3 = SubclassWithSubclassInnerTensors(torch.randn(3), extra=extra_meta)
        self._test_serialization("TENSOR_SUBCLASS_METADATA_MATCH", fn, sub3)

    def test_dict_version(self):
        def fn(x):
            return pytree.tree_leaves(x)[0] + 1

        with self.assertRaisesRegex(
            RuntimeError, "DICT_VERSION guard cannot be serialized."
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
            RuntimeError, "ID_MATCH guard cannot be serialized."
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
