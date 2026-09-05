# Owner(s): ["module: dynamo"]

import dataclasses
import functools
import io
import itertools
import pickle
import sys
import tempfile
import threading
import types
import unittest
import weakref
from collections.abc import Iterator
from typing import NamedTuple

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.fx.graph as fx_graph
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import (
    _Missing,
    CheckFunctionManager,
    CompileId,
    GuardsStatePickler,
)
from torch._dynamo.package import CompilePackage
from torch._dynamo.source import LocalSource
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_MACOS,
    parametrize,
    subtest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
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


class GlobalNestedModule(torch.nn.Module):
    def __init__(self, submodule=None):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.param = torch.nn.Parameter(torch.randn(3, 2))
        self.nested = submodule or GlobalModule()

    def forward(self, x):
        return self.linear(x) + 1


def global_func(x):
    return x + 1


def keep_defaults(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if len(func.__defaults__) == 2:
            x = x + 1
        return func(self, x)

    return wrapper


def keep_kwdefaults(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__kwdefaults__["scale"] == 2.0:
            x = x + 1
        return func(self, x)

    return wrapper


def keep_attribute(func):
    func.scale_flag = 2.0

    @functools.wraps(func)
    def wrapper(self, x):
        if func.scale_flag == 2.0:
            x = x + 1
        return func(self, x)

    return wrapper


def keep_name(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__name__ == "forward":
            x = x + 1
        return func(self, x)

    return wrapper


def keep_renamed_name(func):
    # __name__ differs from co_name, so a co_name fallback reads "forward".
    func.__name__ = "renamed_forward"

    @functools.wraps(func)
    def wrapper(self, x):
        if func.__name__ == "renamed_forward":
            x = x + 1
        return func(self, x)

    return wrapper


def keep_module(func):
    @functools.wraps(func)
    def wrapper(self, x):
        # Guarding __globals__ too forces the SNAPSHOT variant.
        if func.__module__ == func.__globals__["__name__"]:
            x = x + 1
        return func(self, x)

    return wrapper


FQN_MISMATCH_GLOBAL = 2


def keep_global(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__globals__["FQN_MISMATCH_GLOBAL"] == 2:
            x = x + 10
        return func(self, x)

    return wrapper


def keep_globals_length(func):
    @functools.wraps(func)
    def wrapper(self, x):
        return func(self, x) + len(func.__globals__)

    return wrapper


class UnpicklableDefault:
    def __reduce__(self):
        raise RuntimeError("unrelated default cannot pickle")


def _cell_is_empty(cell):
    try:
        cell.cell_contents
    except ValueError:
        return True
    return False


class DecoratedForwardModule(torch.nn.Module):
    # The undecorated forward is unreachable by reference, so it is rebuilt by value.
    @keep_defaults
    def forward(self, x, scale=2.0, shift=1.0):
        return x * scale + shift


class DecoratedKwdefaultsForwardModule(torch.nn.Module):
    @keep_kwdefaults
    def forward(self, x, *, scale=2.0):
        return x * scale


class DecoratedAttributeForwardModule(torch.nn.Module):
    @keep_attribute
    def forward(self, x):
        return x * 2


class DecoratedRenamedNameForwardModule(torch.nn.Module):
    @keep_renamed_name
    def forward(self, x):
        return x * 2


class DecoratedModuleForwardModule(torch.nn.Module):
    @keep_module
    def forward(self, x):
        return x * 2


class DecoratedGlobalForwardModule(torch.nn.Module):
    @keep_global
    def forward(self, x):
        return x * 2


class DecoratedGlobalsLengthForwardModule(torch.nn.Module):
    @keep_globals_length
    def forward(self, x):
        return x * 2


class DecoratedUnpicklableDefaultForwardModule(torch.nn.Module):
    @keep_name
    def forward(self, x, unused=UnpicklableDefault()):
        return x * 2


# Module-scope wrappers: the globals snapshot contains the wrappers themselves.
MODULE_SCOPE_CONST = 2
MODULE_SCOPE_CONST_B = 2


def module_scope_wrapper(func, const_name):
    @functools.wraps(func)
    def wrapper(x):
        if func.__globals__[const_name] == 2:
            x = x + 1
        return func(x)

    return wrapper


def _scope_base_a(x):
    return x * 2


def _scope_base_b(x):
    return x * 3


MODULE_SCOPE_WRAPPED_A = module_scope_wrapper(_scope_base_a, "MODULE_SCOPE_CONST")
MODULE_SCOPE_WRAPPED_B = module_scope_wrapper(_scope_base_b, "MODULE_SCOPE_CONST_B")


def self_referencing_wrapper(func):
    @functools.wraps(func)
    def wrapper(x):
        # Reaches itself through both a closure cell and __dict__["me"].
        if wrapper.flag == 2.0:
            x = x + 1
        return func(x)

    wrapper.flag = 2.0
    wrapper.me = wrapper
    return wrapper


def _self_referencing_base(x):
    return x * 2


SELF_REFERENCING_WRAPPED = self_referencing_wrapper(_self_referencing_base)


def none_cell_wrapper(func):
    scale = None

    @functools.wraps(func)
    def wrapper(x):
        if scale is None:
            x = x + 1
        return func(x)

    return wrapper


def _none_cell_base(x):
    return x * 2


NONE_CELL_WRAPPED = none_cell_wrapper(_none_cell_base)


def _doc_base(x):
    """base doc"""
    return x * 2


def doc_wrapper(func):
    @functools.wraps(func)
    def wrapper(x):
        return func(x)

    return wrapper


DOC_WRAPPED = doc_wrapper(_doc_base)


class UnpicklableGuardedDefault:
    def __init__(self):
        self.flag = 2.0

    def __reduce__(self):
        raise RuntimeError("guarded default cannot pickle")


def keep_default_attribute(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__defaults__[0].flag == 2.0:
            x = x + 1
        return func(self, x)

    return wrapper


class DecoratedUnpicklableGuardedDefaultForwardModule(torch.nn.Module):
    @keep_default_attribute
    def forward(self, x, cfg=UnpicklableGuardedDefault()):
        return x * 2


class RecursingGuardedDefault:
    flag = 2.0

    def __init__(self, inner=None):
        self.inner = inner

    def __reduce__(self):
        # Hands pickle a fresh instance every time, so nothing is ever memoized.
        return type(self), (type(self)(),)


class DecoratedRecursingGuardedDefaultForwardModule(torch.nn.Module):
    @keep_default_attribute
    def forward(self, x, cfg=RecursingGuardedDefault()):
        return x * 2


def keep_name_with_empty_cell(func):
    @functools.wraps(func)
    def wrapper(x):
        if func.__name__ == "renamed":
            x = x + 1
        if x is None:
            return unset
        return func(x)

    if func is None:
        unset = 1  # never runs, so the cell wrapper closes over stays EMPTY

    return wrapper


def _empty_cell_base(x):
    return x * 2


EMPTY_CELL_WRAPPED = keep_name_with_empty_cell(_empty_cell_base)


def keep_default_value(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__defaults__[0] == 2.0:
            x = x + 1
        return func(self, x)

    return wrapper


class DecoratedDefaultValueForwardModule(torch.nn.Module):
    @keep_default_value
    def forward(self, x, scale=2.0):
        return x * scale


class GuardedDefaultsTupleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def fn(x, a=2.0, b=1.0, *, c=3.0):
            return x * a + b + c

        self.fn = fn

    def forward(self, x):
        # EQUALS_MATCH on the containers themselves, with no per-element source.
        if self.fn.__defaults__ == (2.0, 1.0) and self.fn.__kwdefaults__ == {"c": 3.0}:
            x = x + 1
        return x + 2


def keep_dict_attribute(func):
    func.tag = 2.0
    func.cache = threading.Lock()  # unpicklable and unguarded

    @functools.wraps(func)
    def wrapper(self, x):
        if func.__dict__["tag"] == 2.0:
            x = x + 1
        return func(self, x)

    return wrapper


class DecoratedDictAttributeForwardModule(torch.nn.Module):
    @keep_dict_attribute
    def forward(self, x):
        return x * 2


# A module-level lambda's qualname is "<lambda>", which resolves to nothing.
GLOBAL_LAMBDA = lambda x: x * 2  # noqa: E731
GLOBAL_LAMBDA.scale_flag = 2.0


def keep_fn_name(func):
    @functools.wraps(func)
    def wrapper(x):
        if func.__name__ == "base":
            x = x + 1
        return func(x)

    return wrapper


def global_add(obj, x):
    return x + 1


# mutation: what to change on the undecorated function so the guard stops matching.
FQN_MISMATCH_CASES = [
    subtest(
        ("SEQUENCE_LENGTH", DecoratedForwardModule, ("__defaults__", (2.0,))),
        name="defaults_length",
    ),
    subtest(
        ("EQUALS_MATCH", DecoratedDefaultValueForwardModule, ("__defaults__", (3.0,))),
        name="default_value",
    ),
    subtest(
        (
            "EQUALS_MATCH",
            DecoratedKwdefaultsForwardModule,
            ("__kwdefaults__", {"scale": 3.0}),
        ),
        name="kwdefaults",
    ),
    subtest(
        (
            "EQUALS_MATCH",
            DecoratedKwdefaultsForwardModule,
            ("__kwdefaults__", {"other": 2.0}),
        ),
        name="kwdefaults_keys",
    ),
    subtest(
        ("EQUALS_MATCH", DecoratedAttributeForwardModule, ("scale_flag", 3.0)),
        name="attribute",
    ),
    subtest(
        ("EQUALS_MATCH", DecoratedRenamedNameForwardModule, ("__name__", "forward")),
        name="renamed_name",
    ),
    subtest(
        ("EQUALS_MATCH", DecoratedModuleForwardModule, ("__module__", "other")),
        name="module",
    ),
    subtest(
        ("EQUALS_MATCH", DecoratedUnpicklableDefaultForwardModule, ("__name__", "x")),
        name="name_beside_unpicklable_default",
    ),
]


class ModuleNotSerializable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(3, 2))

    def __getstate__(self):
        raise NotImplementedError("not serialzable")

    def forward(self, x):
        return x + self.param


class GlobalTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class MyClass:
    def __getstate__(self):
        raise RuntimeError("Cannot pickle")

    def add(self, x):
        return x + 1


class MyClassNotSerializable:
    def __getstate__(self):
        raise NotImplementedError

    def add(self, x):
        return x + 1


class Inputs:
    def __init__(self, x, unused):
        self.x = x
        self.unused = unused


def _global_func_wrong_fqn(x):
    return x + 1


global_func_wrong_fqn = _global_func_wrong_fqn
del _global_func_wrong_fqn


class FlatModule(torch.nn.Module):
    def forward(self, x):
        return x + 2


class ModWithDict(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d


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
            assert isinstance(args[0], SubclassWithMeta)  # noqa: S101
            return SubclassWithMeta(out_a, extra=args[0].extra)
        return out_a

    def __tensor_flatten__(self):
        # store extra in meta
        return ["a"], {"extra": self.extra}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert isinstance(meta, dict)  # noqa: S101
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
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
            assert isinstance(args[0], SubclassWithCustomMetadataGuard)  # noqa: S101
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
        assert isinstance(meta, dict)  # noqa: S101
        a = inner_tensors["a"]
        # pull out extra from meta
        extra = meta["extra"]
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
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
            assert isinstance(args[0], SubclassWithSubclassInnerTensor)  # noqa: S101
            return SubclassWithSubclassInnerTensor(out_a, extra=args[0].inner_sub.extra)
        return out_a

    def __tensor_flatten__(self):
        return ["a", "inner_sub"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None  # noqa: S101
        a = inner_tensors["a"]
        extra = inner_tensors["inner_sub"].extra
        if type(a) is torch.Tensor:
            assert outer_size is not None  # noqa: S101
            assert outer_stride is not None  # noqa: S101
        return SubclassWithSubclassInnerTensor(a, extra, outer_size, outer_stride)


# defines a custom __eq__() / __hash__() to be registered as a pytree constant type
class CustomConstantType(torch._custom_class_base.CustomClassBase):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        # custom eq ignores b
        return self.a == other.a

    def __hash__(self):
        # custom hash ignores b
        return hash(self.a)

    def __repr__(self):
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})"

    def __fx_repr__(self):
        return f"CustomConstantType(a={self.a!r}, b={self.b!r})", {
            "CustomConstantType": CustomConstantType
        }


torch._library.opaque_object.register_custom_class(CustomConstantType, typ="constant")


class TestGuardSerializationBase(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._fx_magic_methods_snapshot = fx_graph.magic_methods.copy()
        self._saved_default_device_context = getattr(
            torch._GLOBAL_DEVICE_CONTEXT, "device_context", None
        )

    def tearDown(self):
        fx_graph.magic_methods.clear()
        fx_graph.magic_methods.update(self._fx_magic_methods_snapshot)

        current_ctx = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
        if current_ctx is not self._saved_default_device_context:
            if self._saved_default_device_context is None:
                torch.set_default_device(None)
            else:
                torch.set_default_device(self._saved_default_device_context.device)

        super().tearDown()

    def _tracefunc(self, frame, event, arg):
        if event != "call":
            return

        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=frame.f_globals,
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args, **kwargs):
        # kwargs might contain a callable that generates kwargs
        torch._dynamo.reset()
        kwarg_gen_fn = kwargs.get("_gen_fn")
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

        if self._frame_state is None:
            raise AssertionError("Expected _frame_state to be set after tracing")

        # Set f_locals from regenerated kwargs to handle exhausted input iterators
        # NB: This is super janky and might cause unforeseen problems
        if kwarg_gen_fn is not None:
            kwargs = kwarg_gen_fn()
            for key in self._frame_state.f_locals:
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

            torch._dynamo.convert_frame.initial_global_state = (
                torch._C._dynamo.guards.GlobalStateGuard()
            )
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
                compile_context(
                    CompileContext(CompileId(frame_id=0, frame_compile_id=0))
                ),
                tracing(tracer.output.tracing_context),
                tracer.set_current_tx(),
                get_metrics_context(),
                dynamo_timed(""),
            ):
                tracer.run()

                ref_gm = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                ).guard_manager

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    save_guards=True,
                )
                guards_state = check_fn_manager.guards_state
                self._cached_guards_state = guards_state
                self._cached_f_code = self._frame_state.f_code
                self.assertIsNotNone(guards_state)
                guards_state = torch._dynamo.package.load_guards_state(guards_state)

                loaded_gm = torch._dynamo.package.load_guard_manager(
                    guards_state,
                    self._frame_state.f_code,
                    self._frame_state.f_globals,
                )

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            torch._dynamo.convert_frame.initial_global_state = None
            self._frame_state = None

        self.assertIsNotNone(ref_gm)
        self.assertIsNotNone(loaded_gm)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        self.assertIsInstance(inputs, dict)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))


class TestGuardsStatePickler(torch._inductor.test_case.TestCase):
    # Pickler-level: these drive GuardsStatePickler directly rather than
    # through a capture, so none of TestGuardSerialization's setup applies.

    def test_reducer_handles_an_empty_cell_reached_directly(self):
        # _prune_cell only sees cells of a reconstructed function. A cell
        # reached directly -- a guarded __closure__ tuple, or the cell itself
        # -- goes through reducer_override's CellType branch, which read
        # cell_contents unguarded and raised ValueError out of the pickler.
        # Pickler-level because a guard cannot root at a raw cell through a
        # capture: CLOSURE_MATCH is in UNSUPPORTED_SERIALIZATION_GUARD_TYPES.
        empty = [c for c in EMPTY_CELL_WRAPPED.__closure__ if _cell_is_empty(c)]
        self.assertEqual(len(empty), 1)
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"cell": empty[0]})
        self.assertTrue(_cell_is_empty(pickle.loads(buf.getvalue())["cell"]))

    def test_reduce_handles_an_empty_closure_cell(self):
        # Reading an EMPTY cell raised ValueError out of the reducer. It has to
        # come back empty: a cell holding a sentinel reads as an assigned
        # variable. See FunctionPicklerBase._reduce_cell.
        wrapped = EMPTY_CELL_WRAPPED
        empty = [i for i, c in enumerate(wrapped.__closure__) if _cell_is_empty(c)]
        self.assertEqual(len(empty), 1)
        buf = io.BytesIO()
        GuardsStatePickler({id(wrapped): wrapped}, {}, {}, buf).dump({"fn": wrapped})
        out = pickle.loads(buf.getvalue())["fn"]
        self.assertTrue(_cell_is_empty(out.__closure__[empty[0]]))

    def test_fqn_mismatched_function_keeps_a_shared_closure_cell_shared(self):
        # Two functions closing over one variable must still share the cell
        # after reload; rebuilding every cell silently unshares them.
        def outer():
            shared = torch.zeros(2)

            def a():
                return shared

            def b():
                return shared

            return a, b

        a, b = outer()
        self.assertIs(a.__closure__[0], b.__closure__[0])
        buf = io.BytesIO()
        cell = a.__closure__[0]
        gtv = {id(a): a, id(b): b, id(cell): cell}
        pickler = GuardsStatePickler(gtv, {}, {}, buf)
        pickler.dump({"a": a, "b": b})
        out = pickle.loads(buf.getvalue())
        self.assertIs(out["a"].__closure__[0], out["b"].__closure__[0])

    def test_snapshot_globals_function_preserves_module(self):
        # The snapshot variant builds the function with empty globals; see
        # FunctionPicklerBase._build_function.
        def outer():
            def inner():
                return FQN_MISMATCH_GLOBAL

            return inner

        fn = outer()
        self.assertIsNotNone(fn.__module__)
        buf = io.BytesIO()
        g = fn.__globals__
        gtv = {id(fn): fn, id(g): g}
        pickler = GuardsStatePickler(gtv, {}, {}, buf)
        pickler.dump(fn)
        out = pickle.loads(buf.getvalue())
        self.assertEqual(out.__module__, fn.__module__)
        # And the state really did arrive, so a guard on the scope's shape
        # (DICT_KEYS_MATCH, len) still sees the module it was captured from.
        self.assertEqual(out.__globals__.keys(), g.keys())

    def test_globals_snapshot_is_built_once_per_module_dict(self):
        # The snapshot prunes a whole module dict. Building one per function
        # grew the payload with functions x globals; built once, pickle
        # memoizes it and a second function adds a reference, not a copy.
        def make():
            def fn():
                return MODULE_SCOPE_CONST

            return fn

        fns = [make() for _ in range(8)]
        g = globals()

        def dump_size(chosen):
            gtv = {id(fn): fn for fn in chosen}
            gtv[id(g)] = g
            buf = io.BytesIO()
            GuardsStatePickler(gtv, {}, {}, buf).dump(chosen)
            return len(buf.getvalue())

        # Each extra function still costs its own reduce record, but eight of
        # them together must cost less than one more copy of the scope.
        one = dump_size(fns[:1])
        self.assertLess(dump_size(fns) - one, one // 2)

    def test_locals_function_prunes_unguarded_values(self):
        # A <locals> function is rebuilt by value too, and used to carry its
        # defaults and closure verbatim: one unpicklable unguarded neighbour
        # bypassed the whole package. Pruning has to keep the signature's
        # shape -- defaults length, kwdefaults keys -- so a guard reading the
        # structure rather than the values still rebuilds against it.
        def outer():
            captured = UnpicklableDefault()

            def inner(x, unused=UnpicklableDefault(), *, kw=UnpicklableDefault()):
                return x, captured

            return inner

        fn = outer()
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"fn": fn})
        out = pickle.loads(buf.getvalue())["fn"]
        self.assertIsInstance(out.__defaults__[0], _Missing)
        self.assertIsInstance(out.__kwdefaults__["kw"], _Missing)
        self.assertIsInstance(out.__closure__[0].cell_contents, _Missing)

    def test_unguarded_fqn_mismatched_function_is_pruned(self):
        # Rebuilding by value is only for functions a guard is rooted at; an
        # unguarded one stays a sentinel, so widening the set of rebuilt
        # functions does not drag their neighbourhoods into the pickle.
        inner = DecoratedForwardModule.forward.__wrapped__
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"fn": inner})
        self.assertIsInstance(pickle.loads(buf.getvalue())["fn"], _Missing)

    def test_snapshot_globals_share_one_missing_sentinel(self):
        # A globals snapshot prunes a whole module dict; a fresh _Missing per
        # pruned value bloated the pickle with thousands of identical
        # sentinels, none shared even across snapshots in the same pickle.
        a, b = MODULE_SCOPE_WRAPPED_A, MODULE_SCOPE_WRAPPED_B
        g = a.__globals__
        gtv = {id(a): a, id(b): b, id(g): g}
        buf = io.BytesIO()
        GuardsStatePickler(gtv, {}, {}, buf).dump({"a": a, "b": b})
        out = pickle.loads(buf.getvalue())
        sentinels = {
            id(v)
            for fn in out.values()
            for v in fn.__globals__.values()
            if isinstance(v, _Missing)
        }
        self.assertEqual(len(sentinels), 1)

    def test_reduce_keeps_a_none_valued_cell(self):
        # None is a value, not an empty cell; see
        # FunctionPicklerBase._set_cell_contents.
        def outer():
            scale = None

            def inner():
                return scale

            return inner

        fn = outer()
        cell = fn.__closure__[0]
        buf = io.BytesIO()
        GuardsStatePickler({id(fn): fn, id(cell): cell}, {}, {}, buf).dump({"fn": fn})
        out = pickle.loads(buf.getvalue())["fn"]
        self.assertFalse(_cell_is_empty(out.__closure__[0]))
        self.assertIsNone(out())

    def test_reduce_sentinels_an_unpicklable_annotation(self):
        # An annotation nothing guards can be an unpicklable local class; it must
        # be pruned to a sentinel rather than fail the whole dump (which silently
        # bypasses the package).
        class Local:
            pass

        def fn(x):
            return x

        fn.__annotations__ = {"x": Local, "return": Local}
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"fn": fn})
        out = pickle.loads(buf.getvalue())["fn"]
        self.assertIsInstance(out.__annotations__["x"], _Missing)
        self.assertIsInstance(out.__annotations__["return"], _Missing)

    def test_function_reaching_itself_through_its_dict(self):
        # wrapper.me = wrapper, and wrapper is its own free variable; identity
        # has to survive the round trip through both.
        w = SELF_REFERENCING_WRAPPED
        cell = [i for i, c in enumerate(w.__closure__) if c.cell_contents is w]
        self.assertEqual(len(cell), 1)
        buf = io.BytesIO()
        gtv = {id(w): w, id(w.__dict__): w.__dict__}
        GuardsStatePickler(gtv, {}, {}, buf).dump({"w": w})
        out = pickle.loads(buf.getvalue())["w"]
        self.assertIs(out.me, out)
        self.assertIs(out.__closure__[cell[0]].cell_contents, out)

    def test_bound_method_under_a_name_self_lacks(self):
        # types.MethodType can bind a function under a name self has no
        # attribute for. _reduce_bound_method looked that name up unguarded,
        # and the AttributeError bypassed the package instead of carrying the
        # function and self explicitly.
        m = types.MethodType(global_add, Inputs(1, 2))
        self.assertFalse(hasattr(m.__self__, "global_add"))
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"m": m})
        out = pickle.loads(buf.getvalue())["m"]
        self.assertIs(out.__func__, global_add)
        self.assertIsInstance(out.__self__, Inputs)


# NB config.patch subclasses the class it decorates, so it has to go outermost:
# instantiate_parametrized_tests deletes the template method it expands, which
# only works on the class that actually defines it.
@torch._dynamo.config.patch({"strict_precompile": True})
@instantiate_parametrized_tests
class TestGuardSerialization(TestGuardSerializationBase):
    def test_function_locals(self):
        def foo(x):
            return x + 1

        def fn(x, g):
            return g(x) + 1

        self._test_serialization("TENSOR_MATCH", fn, torch.randn(3), foo)

    def test_autocast_object_input(self):
        def fn(x, ac):
            with ac:
                return torch.mm(x, x)

        x = torch.randn(4, 4)
        ac = torch.autocast("cpu", dtype=torch.bfloat16)
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, ac)
        same = torch.autocast("cpu", dtype=torch.bfloat16)
        other = torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)
        self._test_check_fn(ref, loaded, {"x": x, "ac": same}, True)
        self._test_check_fn(ref, loaded, {"x": x, "ac": other}, False)

        # The EQUALS_MATCH filter above dropped TYPE_MATCH; without it a duck-typed object passes.
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, x, ac)

        class DuckAutocast:
            def __init__(self):
                self.device = "cpu"
                self.fast_dtype = torch.bfloat16
                self._enabled = True
                self._cache_enabled = True

        self._test_check_fn(ref, loaded, {"x": x, "ac": same}, True)
        self._test_check_fn(ref, loaded, {"x": x, "ac": DuckAutocast()}, False)

    def test_guard_rooted_at_module_scope_wrappers_that_reach_themselves(self):
        # Two functools.wraps helpers bound at module scope and called from one
        # frame is ordinary code; see the fixture for why both are rebuilt
        # against one snapshot. Rejecting through either global shows both
        # guards survived the round trip.
        global MODULE_SCOPE_CONST, MODULE_SCOPE_CONST_B

        def fn(x):
            return MODULE_SCOPE_WRAPPED_A(x) + MODULE_SCOPE_WRAPPED_B(x)

        x = torch.randn(3)
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        try:
            MODULE_SCOPE_CONST_B = 3
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
            MODULE_SCOPE_CONST_B = 2
            MODULE_SCOPE_CONST = 3
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            MODULE_SCOPE_CONST = MODULE_SCOPE_CONST_B = 2

    def test_guard_rooted_at_wrapper_that_reaches_itself(self):
        # A kept closure cell and a kept attribute both reach back to the
        # function being reduced; see FunctionPicklerBase for why that has to
        # travel as pickle state.
        def fn(x):
            return SELF_REFERENCING_WRAPPED(x)

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        SELF_REFERENCING_WRAPPED.flag = 3.0
        try:
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            SELF_REFERENCING_WRAPPED.flag = 2.0

    def test_guard_rooted_at_fqn_mismatched_function_with_an_empty_cell(self):
        # The wrapper owns the empty cell and is rebuilt by value, so the cell
        # goes through _prune_cell and _reduce_cell rebuilds it empty.
        def fn(x):
            return EMPTY_CELL_WRAPPED(x)

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        _empty_cell_base.__name__ = "renamed"
        try:
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            _empty_cell_base.__name__ = "_empty_cell_base"

    def test_guard_rooted_at_a_none_valued_closure_cell(self):
        # The cell is reached as a CELL through the wrapper's __closure__, so
        # _reduce_cell has to hand None back as a value, not an empty cell.
        def fn(x):
            return NONE_CELL_WRAPPED(x)

        wrapper = NONE_CELL_WRAPPED
        cell = wrapper.__closure__[wrapper.__code__.co_freevars.index("scale")]
        self.assertIsNone(cell.cell_contents)
        ref, loaded = self._test_serialization("CONSTANT_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        cell.cell_contents = 2.0
        try:
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            cell.cell_contents = None

    def test_guard_rooted_at_wrapper_preserves_copied_doc(self):
        # functools.wraps copies the wrapped function's __doc__ onto a wrapper
        # whose own code object has none. Rebuilt from the code object alone
        # the wrapper reads None, the EQUALS_MATCH rebakes against it at load,
        # and the guard fails forever with no load error.
        def fn(x):
            if DOC_WRAPPED.__doc__ == "base doc":
                x = x + 1
            return DOC_WRAPPED(x)

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        DOC_WRAPPED.__doc__ = "other"
        try:
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            DOC_WRAPPED.__doc__ = "base doc"

    def test_fqn_mismatched_function_rejects_a_new_global(self):
        # len(func.__globals__) installs SEQUENCE_LENGTH (derived
        # DICT_KEYS_MATCH), rebaked from the snapshot's size at load, so a key
        # added to the live module dict is rejected.
        mod = DecoratedGlobalsLengthForwardModule()
        ref, loaded = self._test_serialization("DICT_KEYS_MATCH", mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        inputs = {"self": mod, "x": torch.randn(3), "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)
        inner.__globals__["FQN_MISMATCH_NEW_GLOBAL"] = 1
        try:
            self._test_check_fn(ref, loaded, inputs, False)
        finally:
            del inner.__globals__["FQN_MISMATCH_NEW_GLOBAL"]

    def test_unserializable_guarded_value_is_a_package_error(self):
        # Whatever the pickler raises for a value some guard reads -- here a
        # RuntimeError from the value's own __reduce__ -- surfaces as a
        # PackageError: a bypass for non-strict callers, never a compiler
        # crash. strict_precompile is on for this class, so it re-raises.
        mod = DecoratedUnpicklableGuardedDefaultForwardModule()
        with self.assertRaisesRegex(PackageError, "guarded default cannot pickle"):
            self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))

    def test_recursing_guarded_value_is_not_a_package_error(self):
        # A reducer that never bottoms out is a pickler bug, not a value that
        # cannot be serialized, so dump lets RecursionError through.
        mod = DecoratedRecursingGuardedDefaultForwardModule()
        with self.assertRaises(RecursionError):
            self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))

    def test_fqn_mismatched_function_from_a_module_gone_at_load(self):
        # The rebuilt function's __module__ names a module that only ever lived
        # in sys.modules (exec-created, transformers_modules.*), so the load
        # cannot import it; see FunctionPicklerBase._unpickle_fn_from_module.
        name = "dynamo_test_guard_serialization_exec_module"
        mod = types.ModuleType(name)
        mod.keep_fn_name = keep_fn_name
        exec("@keep_fn_name\ndef base(x):\n    return x * 2\n", mod.__dict__)
        sys.modules[name] = mod
        try:
            ref, _ = self._test_serialization("EQUALS_MATCH", mod.base, torch.randn(3))
        finally:
            del sys.modules[name]
        inner = mod.base.__wrapped__
        self.assertEqual(inner.__module__, name)
        state = torch._dynamo.package.load_guards_state(self._cached_guards_state)
        f_code, f_globals = self._cached_f_code, keep_fn_name.__globals__
        loaded = torch._dynamo.package.load_guard_manager(state, f_code, f_globals)
        inputs = {"x": torch.randn(3), "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)
        inner.__name__ = "renamed"
        self._test_check_fn(ref, loaded, inputs, False)

    @parametrize("guard_type,cls,mutation", FQN_MISMATCH_CASES)
    def test_guard_rooted_at_fqn_mismatched_function(self, guard_type, cls, mutation):
        # The undecorated function the guard is rooted at is rebuilt by value
        # (see DecoratedForwardModule), with whatever the guard reads intact.
        mod = cls()
        ref, loaded = self._test_serialization(guard_type, mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        inputs = {"self": mod, "x": torch.randn(3), "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)
        # The guard must also REJECT: a reconstruction that pinned the wrong
        # thing, or nothing at all, still passes the positive check.
        attr, new_value = mutation
        old_value = getattr(inner, attr)
        try:
            setattr(inner, attr, new_value)
            self._test_check_fn(ref, loaded, inputs, False)
        finally:
            setattr(inner, attr, old_value)

    def test_fqn_mismatched_function_preserves_guarded_globals(self):
        global FQN_MISMATCH_GLOBAL

        mod = DecoratedGlobalForwardModule()
        x = torch.ones(1)
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, x)
        inner = type(mod).forward.__wrapped__
        inputs = {"self": mod, "x": x, "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)

        # The loaded guard baked the snapshot's 2 at load, so the live 3 that
        # the check reads through func.__globals__ no longer matches it.
        old_value = FQN_MISMATCH_GLOBAL
        try:
            FQN_MISMATCH_GLOBAL = 3
            self._test_check_fn(ref, loaded, inputs, False)
        finally:
            FQN_MISMATCH_GLOBAL = old_value

    def test_nested_function_preserves_a_guarded_defaults_tuple(self):
        # A guard on the container itself registers no per-element source, so
        # pruning the elements is a silent permanent cache miss, not a load
        # error; see the Note in guards.py.
        mod = GuardedDefaultsTupleModule()
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))
        self._test_check_fn(ref, loaded, {"self": mod, "x": torch.randn(3)}, True)
        mod.fn.__defaults__ = (3.0, 1.0)
        self._test_check_fn(ref, loaded, {"self": mod, "x": torch.randn(3)}, False)
        mod.fn.__defaults__ = (2.0, 1.0)
        mod.fn.__kwdefaults__ = {"c": 4.0}
        self._test_check_fn(ref, loaded, {"self": mod, "x": torch.randn(3)}, False)

    def test_fqn_mismatched_function_prunes_unpicklable_dict_attributes(self):
        # A guard through __dict__ registers the dict itself, which used to be
        # carried verbatim: one unpicklable unguarded attribute then bypassed
        # the whole package. The guarded attribute must still round-trip.
        mod = DecoratedDictAttributeForwardModule()
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        inputs = {"self": mod, "x": torch.randn(3), "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)
        inner.tag = 3.0
        try:
            self._test_check_fn(ref, loaded, inputs, False)
        finally:
            inner.tag = 2.0

    def test_guard_rooted_at_a_lambda(self):
        # A module-level lambda is an fqn mismatch too (see GLOBAL_LAMBDA) and
        # is rebuilt by value with the guarded attribute intact.
        def fn(x):
            if GLOBAL_LAMBDA.scale_flag == 2.0:
                x = x + 1
            return GLOBAL_LAMBDA(x)

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)
        GLOBAL_LAMBDA.scale_flag = 3.0
        try:
            self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, False)
        finally:
            GLOBAL_LAMBDA.scale_flag = 2.0

    def test_guard_rooted_at_fqn_mismatched_bound_method(self):
        # The undecorated forward bound directly: the method carries its
        # function explicitly (_reduce_bound_method), and that function is an
        # fqn mismatch rebuilt by value.
        mod = DecoratedAttributeForwardModule()
        inner = type(mod).forward.__wrapped__
        bound = types.MethodType(inner, mod)

        def fn(f, x):
            if f.scale_flag == 2.0:
                x = x + 1
            return f(x)

        x = torch.randn(3)
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, bound, x)
        self._test_check_fn(ref, loaded, {"f": bound, "x": x}, True)
        inner.scale_flag = 3.0
        try:
            self._test_check_fn(ref, loaded, {"f": bound, "x": x}, False)
        finally:
            inner.scale_flag = 2.0

    def test_guard_rooted_at_bound_method_under_a_name_self_lacks(self):
        # See TestGuardsStatePickler.test_bound_method_under_a_name_self_lacks.
        bound = types.MethodType(global_add, Inputs(1, 2))

        def fn(f, x):
            if callable(f):
                x = x + 1
            return f(x)

        x = torch.randn(3)
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, bound, x)
        self._test_check_fn(ref, loaded, {"f": bound, "x": x}, True)
        self._test_check_fn(ref, loaded, {"f": global_add, "x": x}, False)

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

        # Check verbose_code_parts from leaf guards (they include hints)
        def check_leaf_guards(mgr):
            for guard in mgr.get_leaf_guards():
                verbose_parts = guard.verbose_code_parts()
                verbose_str = " ".join(verbose_parts)
                if "___check_type_id" in verbose_str and "L['m']" in verbose_str:
                    self.assertIn(
                        "HINT: type",
                        verbose_str,
                        (
                            lambda msg: f"{msg}\n"
                            + (
                                "TYPE_MATCH guard should include 'HINT: type' "
                                f"annotation.\nGuard: {verbose_str}"
                            )
                        ),
                    )
                    self.assertIn(
                        "GlobalModule",
                        verbose_str,
                        (
                            lambda msg: f"{msg}\n"
                            + (
                                "TYPE_MATCH guard should include type name "
                                f"'GlobalModule'.\nGuard: {verbose_str}"
                            )
                        ),
                    )
            for child_mgr in mgr.get_child_managers():
                check_leaf_guards(child_mgr)

        check_leaf_guards(ref.root)

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
                assert meta is None  # noqa: S101
                a = inner_tensors["a"]
                if type(a) is torch.Tensor:
                    assert outer_size is not None  # noqa: S101
                    assert outer_stride is not None  # noqa: S101
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

    @unittest.skipIf(not torch.distributed.is_available(), "requires torch.distributed")
    def test_transparent_subclass_tensor_match(self):
        # AsyncCollectiveTensor is a transparent traceable wrapper subclass: its
        # __torch_dispatch__ desugars ops to the inner tensor, so
        # torch.empty_like(act) returns a plain Tensor and drops the subclass
        # type. Guard-state serialization must round-trip such an input by
        # unflattening through the recorded pytype rather than type(meta_tensor)
        # (which would be torch.Tensor, with no __tensor_unflatten__).
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def fn(w):
            return w.sum()

        base = torch.randn(3, 4)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", fn, AsyncCollectiveTensor(base)
        )
        self._test_check_fn(ref, loaded, {"w": AsyncCollectiveTensor(base)}, True)
        # The reloaded guard must also match the resolved plain Tensor -- the
        # ACT->Tensor reuse this relaxation enables, and the case that matters
        # for precompile under FSDP+TP.
        self._test_check_fn(ref, loaded, {"w": base}, True)

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

    def test_autocast_equals_match(self):
        class Module(torch.nn.Module):
            def __init__(self, ctx):
                super().__init__()
                self.ctx = ctx

            def forward(self, x):
                with self.ctx:
                    return x @ x

        module = Module(torch.amp.autocast("cpu", dtype=torch.bfloat16))
        x = torch.randn(4, 4)
        ref, loaded = self._test_serialization("EQUALS_MATCH", module, x)
        self._test_check_fn(ref, loaded, {"self": module, "x": x}, True)

        module.ctx = torch.amp.autocast("cpu", dtype=torch.bfloat16)
        self._test_check_fn(ref, loaded, {"self": module, "x": x}, True)

        module.ctx = torch.amp.autocast("cpu", dtype=torch.float16)
        self._test_check_fn(ref, loaded, {"self": module, "x": x}, False)

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

    def test_class_match(self):
        def fn(x):
            # usage of this context manager installs a FUNCTION_MATCH guard
            with torch.no_grad():
                y = x * 2
            return y

        x = torch.randn(3)

        # we don't support FUNCTION_MATCH because it adds an ID_MATCH guard, and we don't
        # support that in serialization
        with self.assertRaisesRegex(
            PackageError, "CLASS_MATCH guard cannot be serialized."
        ):
            self._test_serialization("CLASS_MATCH", fn, x)

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

    def test_count_iterator_match(self):
        def fn(x, counter):
            return x + next(counter)

        x = torch.randn(3)

        def _gen_kwargs(x=x):
            return {"x": x, "counter": itertools.count(2, 3)}

        ref, loaded = self._test_serialization(
            "COUNT_ITERATOR_MATCH", fn, _gen_fn=_gen_kwargs
        )

        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(2, 3)}, True
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(4), "counter": itertools.count(2, 3)},
            True,
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(5, 3)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "counter": itertools.count(2, 4)}, False
        )

    def test_supported_nodes_dict_keys_match(self):
        def fn(x):
            return pytree.tree_leaves(x)[0] + 1

        ref, loaded = self._test_serialization(
            "DICT_KEYS_MATCH", fn, {"t": torch.randn(3)}
        )
        self._test_check_fn(ref, loaded, {"x": {"t": torch.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)

        # Sticky flag must survive pickling so load keeps keys-match instead of
        # re-promoting SUPPORTED_NODES to DICT_VERSION.
        guards_state = torch._dynamo.package.load_guards_state(
            self._cached_guards_state
        )
        self.assertTrue(
            any(g._force_dict_keys_match for g in guards_state.output_graph.guards)
        )

        # Loaded keys-match guard must observe SUPPORTED_NODES key changes, not
        # only changes to the user input dict.
        class _TmpPytreeNode:
            def __init__(self, x):
                self.x = x

        inputs = {"x": {"t": torch.randn(3)}}
        self.assertTrue(loaded.check(inputs))
        try:
            pytree.register_pytree_node(
                _TmpPytreeNode,
                lambda n: ([n.x], None),
                lambda xs, _: _TmpPytreeNode(xs[0]),
            )
            self.assertFalse(loaded.check(inputs))
        finally:
            pytree._deregister_pytree_node(_TmpPytreeNode)

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

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_id_match_with_config(self):
        def fn(x):
            return x + id(x)

        ref, loaded = self._test_serialization("ID_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

        def fn(x):
            # usage of this context manager installs a CLASS_MATCH guard
            with torch.no_grad():
                y = x * 2
            return y

        ref, loaded = self._test_serialization("CLASS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

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
        ref, loaded = self._test_serialization("DUPLICATE_INPUT", fn, x, x)

        self._test_check_fn(ref, loaded, {"x": x, "x_": x}, True)
        self._test_check_fn(ref, loaded, {"x": x, "x_": torch.randn(3, 2)}, False)

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
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
        with torch.no_grad():
            self._test_check_fn(ref, loaded, {"x": x}, False)
        with torch.enable_grad():
            self._test_check_fn(ref, loaded, {"x": x}, True)

    def test_grad_mode_loading(self):
        def fn(x):
            return x + 1

        x = torch.randn(3, 2)
        with torch.enable_grad():
            ref, _ = self._test_serialization("GLOBAL_STATE", fn, x)
        with torch.no_grad():
            # Ensure guards state loading is not affected by the current global grad mode.
            guards_state = pickle.loads(self._cached_guards_state)
            check_fn_manager = CheckFunctionManager(
                self._cached_f_code,
                guards_state.output_graph,
                shape_code_parts=guards_state.shape_code_parts,
            )
            loaded = check_fn_manager.guard_manager
            self._test_check_fn(ref, loaded, {"x": x}, False)

    def test_deterministic_algorithms(self):
        def fn(x):
            return x + 1

        deterministic_restore = torch.are_deterministic_algorithms_enabled()
        try:
            x = torch.randn(3, 2)
            torch.use_deterministic_algorithms(True)
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
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
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
            self._test_check_fn(ref, loaded, {"x": x}, True)
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
            ref, loaded = self._test_serialization("GLOBAL_STATE", fn, x)
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

    def test_builtin_match(self):
        def fn(x):
            # usage of getattr() here installs a BUILTIN_MATCH guard
            s = getattr(x, "shape")  # noqa: B009
            return x + s[0]

        x = torch.randn(3)

        ref, loaded = self._test_serialization("BUILTIN_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": x}, True)
        getattr_original = getattr

        def getattr_new(*args, **kwargs):
            return getattr_original(*args, **kwargs)

        builtins_dict = (
            __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        )
        builtins_dict["getattr"] = getattr_new
        try:
            self._test_check_fn(ref, loaded, {"x": x}, False)
        finally:
            builtins_dict["getattr"] = getattr_original

    def test_skipped_objects(self):
        def foo():
            pass

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.code = foo.__code__
                self.foo = foo
                self.p = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x):
                z = x + 1
                for p in self.parameters():
                    z += p
                return z

        m = Module()
        ref, loaded = self._test_serialization("TENSOR_MATCH", m, torch.randn(3, 2))
        self._test_check_fn(ref, loaded, {"self": m, "x": torch.randn(3, 2)}, True)

    def test_bound_method_input(self):
        class MyModule(torch.nn.Module):
            def forward(self, foo, x):
                return x + id(type(foo))

        m = MyModule()
        ref, loaded = self._test_serialization(
            "TYPE_MATCH", m, MyClass().add, torch.randn(3, 2)
        )
        self._test_check_fn(
            ref, loaded, {"self": m, "foo": MyClass().add, "x": torch.randn(3, 2)}, True
        )

    def test_bound_methods_missing(self):
        class MyClass:
            def __getstate__(self):
                raise NotImplementedError

            def add(self, x):
                return x + 1

        def foo(x: torch.Tensor, y: list[MyClass]):
            assert len(y) == 1  # noqa: S101
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), [MyClass()]
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(3, 2), "y": [MyClass()]}, True
        )

    def test_bound_methods_empty(self):
        def foo(x, y):
            assert callable(y[0])  # noqa: S101
            return x + 1

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), [MyClassNotSerializable().add]
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(3, 2), "y": [MyClassNotSerializable().add]},
            True,
        )

    def test_ddp_module(self):
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")
        from torch.nn.parallel import DistributedDataParallel as DDP

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ddp_model = DDP(GlobalNestedModule())

            def foo(ddp, x):
                return ddp(x)

            unsupported = frozenset(
                torch._dynamo.guards.CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
            )
            x = torch.randn(10)
            package = CompilePackage(foo)
            torch._dynamo.optimize(
                package=package,
                guard_filter_fn=lambda gs: [
                    x.guard_type not in unsupported for x in gs
                ],
            )(foo)(ddp_model, x)
            self.assertEqual(len(package._codes[foo.__code__].guarded_codes), 1)
            torch._dynamo.package.load_guards_state(
                package._codes[foo.__code__].guarded_codes[0].guards_state
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_dict_keys_serialization(self):
        d = {1: 2, 3: 4}

        def foo(x, y):
            for k in y:
                x += k
            return x

        ref, loaded = self._test_serialization(
            "TYPE_MATCH", foo, torch.randn(3, 2), d.keys()
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": torch.randn(3, 2), "y": d.keys()},
            True,
        )

    def test_unserializable_sharded_tensor(self):
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        tmpfile = tempfile.NamedTemporaryFile()  # noqa:SIM115
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method=f"file://{tmpfile.name}"
        )
        try:
            ChunkShardingSpec = dist._shard.sharding_spec.ChunkShardingSpec
            ShardedTensor = dist._shard.sharded_tensor.ShardedTensor
            tensor = torch.arange(2, dtype=torch.int64)
            local_tensor = torch.unsqueeze(torch.cat([tensor, tensor + 2]), 0)

            sharding_dim = 0
            sharding_spec = ChunkShardingSpec(
                dim=sharding_dim,
                placements=[
                    "rank:0/cpu",
                ],
            )
            st = ShardedTensor._init_from_local_tensor(
                local_tensor, sharding_spec, [1, 4]
            )

            def foo(inputs):
                return inputs.x + 1

            ref, loaded = self._test_serialization(
                "TENSOR_MATCH", foo, Inputs(torch.randn(3, 2), st)
            )
            self._test_check_fn(
                ref, loaded, {"inputs": Inputs(torch.randn(3, 2), st)}, True
            )
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_function_with_wrong_fqn(self):
        def foo(inputs):
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, global_func_wrong_fqn)
        )
        self._test_check_fn(
            ref, loaded, {"inputs": Inputs(x, global_func_wrong_fqn)}, True
        )

    def test_c10d_work(self):
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        Work = dist.distributed_c10d.Work

        class DummyWork(Work):
            def __init__(self, should_succeed=True):
                super().__init__()
                self._done = False
                self._should_succeed = should_succeed

            def is_completed(self):
                return self._done

            def is_success(self):
                return self._should_succeed

            def wait(self, timeout=None):
                self._done = True
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return self

            def result(self):
                if not self._should_succeed:
                    raise RuntimeError("DummyWork failed")
                return "dummy_result"

        def foo(inputs):
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, DummyWork())
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, DummyWork())}, True)

    def test_unused_weakref(self):
        def foo(inputs):
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, weakref.ref(x))
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, weakref.ref(x))}, True)

    def test_unused_stream(self):
        if not torch.accelerator.is_available():
            self.skipTest("Accelerator is not available")

        def foo(inputs):
            return inputs.x + 1

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", foo, Inputs(x, torch.Stream())
        )
        self._test_check_fn(ref, loaded, {"inputs": Inputs(x, torch.Stream())}, True)

    def test_unused_process_group(self):
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")

        def foo(inputs):
            return inputs.x + 1

        tmpfile = tempfile.NamedTemporaryFile()  # noqa: SIM115
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmpfile.name}",
            rank=0,
            world_size=1,
        )

        try:
            pg = dist.distributed_c10d._get_default_group()
            x = torch.randn(3, 2)
            ref, loaded = self._test_serialization("TENSOR_MATCH", foo, Inputs(x, pg))
            self._test_check_fn(ref, loaded, {"inputs": Inputs(x, pg)}, True)
        finally:
            dist.destroy_process_group()
            tmpfile.close()

    def test_unserializable_submodule(self):
        def foo(mod, x):
            return mod(x)

        x = torch.randn(10, 10)
        mod = GlobalNestedModule(ModuleNotSerializable())
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, mod, x)
        self._test_check_fn(ref, loaded, {"mod": mod, "x": x}, True)

    def test_closure_var_missing(self):
        captured = torch.randn(3, 2)

        def bar(x):
            return x + captured

        def foo(f, x):
            return f(x)

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("TENSOR_MATCH", foo, bar, x)
        self._test_check_fn(ref, loaded, {"f": bar, "x": x}, True)

    def test_bound_method_patched_forward(self):
        def forward(x):
            return x + 1

        m = FlatModule()
        m_forward = m.forward
        m.forward = forward

        def foo(f, x):
            assert callable(f)  # noqa: S101
            return f(x)

        x = torch.randn(3, 2)
        ref, loaded = self._test_serialization("TYPE_MATCH", foo, m_forward, x)
        self._test_check_fn(ref, loaded, {"f": m_forward, "x": x}, True)

    def test_guard_on_key_order_with_cache(self):
        def foo(x, mod):
            for y in mod.d.values():
                x *= y
            return x

        x = torch.randn(3, 2)
        d = {"a": 1e9, "b": 1e-9}
        ref, loaded = self._test_serialization(
            "DICT_KEYS_MATCH", foo, x, ModWithDict(d)
        )
        self._test_check_fn(
            ref, loaded, {"x": x, "d": ModWithDict({"b": 1e-9, "a": 1e9})}, False
        )

    def test_global_state_guard_filter(self):
        def foo(x):
            return x + 1

        x = torch.randn(3, 2)

        with torch.no_grad():
            compiled_fn = torch.compile(
                foo,
                backend="eager",
                options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
            )
            compiled_fn(x)

        # Check global guards are gone.
        with torch.enable_grad(), torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_torch_function_state_filter(self):
        def foo(x):
            return x + 1

        x = torch.randn(3, 2)

        with GlobalTorchFunctionMode():
            compiled_fn = torch.compile(
                foo,
                backend="eager",
                options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
            )
            compiled_fn(x)

        # Check global guards are gone.
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(x), foo(x))

    def test_nested_named_tuple(self):
        class NestedTuple(NamedTuple):
            a: int
            b: int
            c: torch.Tensor

        def fn(x: NestedTuple):
            return x.a + x.b + x.c

        x = NestedTuple(1, 2, torch.randn(3, 2))

        ref, loaded = self._test_serialization("TENSOR_MATCH", fn, x)

    def test_sdp_backend_serialization(self):
        def fn(x, backend):
            # Use the backend enum in a guard-producing way
            if backend == torch.nn.attention.SDPBackend.MATH:
                return x + 1
            elif backend == torch.nn.attention.SDPBackend.FLASH_ATTENTION:
                return x + 2
            elif backend == torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION:
                return x + 3
            else:
                return x + 4

        x = torch.randn(3, 2)
        backend = torch.nn.attention.SDPBackend.MATH

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x, backend)

        # Test with the same backend
        self._test_check_fn(
            ref, loaded, {"x": x, "backend": torch.nn.attention.SDPBackend.MATH}, True
        )

        # Test with different backends
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.FLASH_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION},
            False,
        )
        self._test_check_fn(
            ref,
            loaded,
            {"x": x, "backend": torch.nn.attention.SDPBackend.CUDNN_ATTENTION},
            False,
        )

    def test_source_serialization(self):
        # Test that "equal" sources with different hashes serialize to the same result
        src1 = LocalSource("x")
        src2 = LocalSource("x")

        # Force different cached hashes to test that serialization excludes _hash
        object.__setattr__(src1, "_hash", 12345)
        object.__setattr__(src2, "_hash", 67890)

        self.assertEqual(pickle.dumps(src1), pickle.dumps(src2))

    def test_source_serialization_init_false_fields(self):
        # Test that source serialization handles fields that are not initialized
        from torch._dynamo.source import DefaultsSource, LocalSource

        base = LocalSource("x")
        source = DefaultsSource(base=base, idx_key=0, is_kw=False)

        # Round-trip through pickle should work even with init=False fields
        restored = pickle.loads(pickle.dumps(source))
        self.assertEqual(source, restored)


class SimpleModule(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.p = torch.nn.Parameter(torch.randn(3, 2))

    def forward(self, x):
        z = x + 1
        for p in self.parameters():
            z += p
        return z


if torch.distributed.is_available() and not IS_MACOS:
    from torch.testing._internal.common_fsdp import FSDPTestMultiThread

    @torch._dynamo.config.patch({"strict_precompile": True})
    class TestGuardSerializationFSDP(TestGuardSerializationBase, FSDPTestMultiThread):
        def setUp(self):
            TestGuardSerializationBase.setUp(self)
            FSDPTestMultiThread.setUp(self)

        @unittest.skipIf(
            TEST_WITH_ASAN or IS_LINUX or TEST_WITH_ROCM,
            "https://github.com/pytorch/pytorch/issues/162793",
        )
        def test_guard_serialization_fsdp_module(self):
            from torch.distributed._tensor import distribute_tensor, Replicate
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh(str(torch.get_default_device()), (1,))
            m = SimpleModule(42)
            m = fully_shard(m, mesh=mesh)
            inputs = distribute_tensor(torch.randn(3, 2), mesh, [Replicate()])
            ref, loaded = self._test_serialization("TENSOR_MATCH", m, inputs)
            self._test_check_fn(ref, loaded, {"self": m, "x": inputs}, True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
