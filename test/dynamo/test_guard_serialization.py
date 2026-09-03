# Owner(s): ["module: dynamo"]

import copyreg
import dataclasses
import functools
import io
import itertools
import pickle
import sys
import tempfile
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
from torch._dynamo.guards import CheckFunctionManager, CompileId
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
    # __name__ reassigned away from co_name, so a reconstruction that falls back
    # to code.co_name reads "forward" where the real function says otherwise.
    # keep_name alone cannot catch that: there the two agree.
    func.__name__ = "renamed_forward"

    @functools.wraps(func)
    def wrapper(self, x):
        if func.__name__ == "renamed_forward":
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


class DecoratedForwardModule(torch.nn.Module):
    # forward is the wrapper; the undecorated function it closes over has the
    # same __qualname__ but is unreachable from the module, which is what makes
    # it unpicklable by reference.
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


class DecoratedNameForwardModule(torch.nn.Module):
    @keep_name
    def forward(self, x):
        return x * 2


class DecoratedRenamedNameForwardModule(torch.nn.Module):
    @keep_renamed_name
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


# --- module-scope wrappers that reach themselves through their own globals ---
# `wrapped = deco(base)` at module scope: the wrapper is reachable from its own
# __globals__, so a globals snapshot passed as a reduce ARG contains the very
# object being reduced. pickle memoizes only after saving args, so that recursed
# forever; two wrappers referencing each other do it across the pair.
MODULE_SCOPE_CONST = 2


def module_scope_wrapper(func):
    @functools.wraps(func)
    def wrapper(x):
        # Roots a guard at func.__globals__, which forces the snapshot -- and
        # the snapshot contains the wrappers themselves.
        if func.__globals__["MODULE_SCOPE_CONST"] == 2:
            x = x + 1
        return func(x)

    return wrapper


def _module_scope_base_a(x):
    return x * 2


def _module_scope_base_b(x):
    return x * 3


MODULE_SCOPE_WRAPPED_A = module_scope_wrapper(_module_scope_base_a)
MODULE_SCOPE_WRAPPED_B = module_scope_wrapper(_module_scope_base_b)


# The pickler refuses a locally-defined class before __reduce__ ever runs, so
# a value whose __reduce__ is the thing under test has to live at module scope.
_REDUCE_RAISES: type[BaseException] = AssertionError


class _RaisesFromReduce:
    def __reduce__(self):
        raise _REDUCE_RAISES("from __reduce__")


def _cell_is_full(cell):
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


def _dump_through_pickle_guards_state(value):
    """Run ``value`` through pickle_guards_state's try/except.

    A capture cannot reach that handler with an arbitrary exception -- Dynamo
    would have to raise it from inside a guarded value's __reduce__ mid-trace
    -- so the state is assembled directly. Only the fields the function reads
    are populated; the point is which exceptions cross the handler, not what a
    real GuardsState contains.
    """
    from torch._dynamo.guards import GuardsState, pickle_guards_state

    graph = types.SimpleNamespace(
        guards=[],
        local_scope={"value": value},
        global_scope={},
        guard_on_key_order=set(),
    )
    # Kept, not pruned: an unguarded value is replaced by a _Missing sentinel
    # and its __reduce__ never runs, so the handler is never reached.
    builder = types.SimpleNamespace(guard_tree_values={id(value): value})
    state = GuardsState(output_graph=graph, shape_code_parts=None)
    return pickle_guards_state(state, builder)


def self_referencing_wrapper(func):
    # The wrapper reads an attribute off ITSELF, so `wrapper` becomes one of
    # its own free variables and the cell holds the object being reduced. A
    # counter or a cache on the wrapper is the ordinary way a decorator does
    # this; functools.wraps then makes it fqn-unreachable, which is exactly
    # the shape the reducer reconstructs.
    @functools.wraps(func)
    def wrapper(x):
        wrapper.calls += 1
        if func.__globals__["MODULE_SCOPE_CONST"] == 2:
            x = x + 1
        return func(x)

    wrapper.calls = 0
    wrapper.me = wrapper
    return wrapper


def _self_referencing_base(x):
    return x * 5


SELF_REFERENCING_WRAPPED = self_referencing_wrapper(_self_referencing_base)


def _make_function_cycle():
    # The wrapper closes over the base and the base holds the wrapper in its
    # __dict__, so the cycle runs through the closure of one and the attributes
    # of the other. Neither is reachable by fqn, so both are reconstructed.
    def base(x):
        return x * 2

    @functools.wraps(base)
    def wrapper(x):
        return base(x) + 1

    base.wrapper = wrapper
    return base, wrapper


CYCLE_BASE, CYCLE_WRAPPER = _make_function_cycle()


class CarriedPayload:
    def __init__(self, size):
        self.size = size


class ReconstructedByReduce:
    # __reduce__ decides what its arguments MEAN, so a pruned attribute arrives
    # as a constructor argument rather than as an attribute nobody reads. The
    # check makes that fail loudly instead of reconstructing a wrong object.
    def __init__(self, tag, payload):
        if not isinstance(payload, CarriedPayload):
            raise TypeError(f"payload must be a CarriedPayload, got {type(payload)}")
        self.tag = tag
        self.payload = payload

    def __reduce__(self):
        return (type(self), (self.tag, self.payload))


class ReconstructedBySetstate:
    # The default protocol hands __setstate__ the whole __dict__, so a pruned
    # attribute arrives as a field this __setstate__ reads.
    def __init__(self, tag, payload):
        self.tag = tag
        self.payload = payload

    def __setstate__(self, state):
        if not isinstance(state["payload"], CarriedPayload):
            raise TypeError(f"payload must be a CarriedPayload, got {state['payload']}")
        self.__dict__.update(state)


class ReconstructedByNewargs:
    # __getnewargs__ feeds a validating __new__ under the DEFAULT __reduce_ex__,
    # so a pruned attribute arrives as a constructor argument.
    def __new__(cls, payload):
        if not isinstance(payload, CarriedPayload):
            raise TypeError(f"payload must be a CarriedPayload, got {payload}")
        return super().__new__(cls)

    def __init__(self, payload):
        self.tag = "tag"
        self.payload = payload

    def __getnewargs__(self):
        return (self.payload,)


class CopyregReduced:
    # copyreg.pickle registers the reducer from OUTSIDE the class, so no pickle
    # hook is present on the class itself; _builds_its_own_pickle must still
    # detect it via the dispatch table, or a pruned attribute reaches the
    # validating rebuild as the sentinel instead of the value guarded siblings
    # of it were traced against.
    def __init__(self, tag, payload):
        if not isinstance(payload, CarriedPayload):
            raise TypeError(f"payload must be a CarriedPayload, got {payload}")
        self.tag = tag
        self.payload = payload


def _rebuild_copyreg(tag, payload):
    return CopyregReduced(tag, payload)


copyreg.pickle(CopyregReduced, lambda o: (_rebuild_copyreg, (o.tag, o.payload)))


class Point(NamedTuple):
    x: int
    y: int


class TaggedPoint(Point):
    # No __slots__, so an instance carries a __dict__ alongside its items, and
    # namedtuple's __getnewargs__ returns only the items.
    pass


class StandInOn:
    # Reduces the way GuardsStatePickler reduces a type-guarded Generator or
    # Stream, for a device the pickling host need not have.
    def __init__(self, cls, device):
        self.cls = cls
        self.device = device

    def __reduce__(self):
        from torch._dynamo.guards import _rebuild_type_stand_in

        return _rebuild_type_stand_in, (self.cls, self.device)


# --- an empty closure cell -------------------------------------------------
def keep_name_with_empty_cell(func):
    @functools.wraps(func)
    def wrapper(self, x):
        if func.__name__ == "forward":
            x = x + 1
        if x is None:
            return unset
        return func(self, x)

    if func is None:
        unset = 1  # never runs, so the cell wrapper closes over stays EMPTY

    return wrapper


def _empty_cell_base(self, x):
    return x * 2


EMPTY_CELL_WRAPPED = keep_name_with_empty_cell(_empty_cell_base)


# --- a guarded default whose VALUE must survive, not just the tuple length --
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


@torch._dynamo.config.patch({"strict_precompile": True})
@instantiate_parametrized_tests
class TestGuardSerialization(TestGuardSerializationBase):
    def test_function_locals(self):
        def foo(x):
            return x + 1

        def fn(x, g):
            return g(x) + 1

        self._test_serialization("TENSOR_MATCH", fn, torch.randn(3), foo)

    def test_guard_rooted_at_module_scope_wrappers_that_reach_themselves(self):
        # Driven through a real capture, not the pickler: two functools.wraps
        # helpers bound at module scope and called from one compiled frame is
        # ordinary code, and the globals snapshot each one carries contains
        # both of them. Passing that snapshot as a reduce ARG recursed until
        # RecursionError; it goes in reduce STATE, which pickle applies after
        # memoizing, so the references resolve to the functions it already
        # built.
        def fn(x):
            return MODULE_SCOPE_WRAPPED_A(x) + MODULE_SCOPE_WRAPPED_B(x)

        x = torch.randn(3)
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

    def test_guard_rooted_at_a_wrapper_that_reaches_itself(self):
        # The globals snapshot moved to reduce STATE, but a kept closure cell,
        # __dict__ entry or default still travelled in the reduce ARGS. pickle
        # memoizes an object only AFTER saving its args, so a wrapper holding
        # itself in any of those re-entered the reducer and recursed until
        # RecursionError -- a hard capture failure, since aot_compile passes
        # strict_error=True. Everything that reaches the function now goes in
        # state too, and __closure__, which is read-only after construction,
        # is built empty and filled by the state setter.
        def fn(x):
            return SELF_REFERENCING_WRAPPED(x)

        x = torch.randn(3)
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, x)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3)}, True)

    def test_a_reconstructed_wrapper_still_points_at_itself(self):
        # Deferring to state is only correct if the reference comes BACK. A
        # wrapper whose cell or attribute silently lost its self-reference
        # would raise NameError or AttributeError on the first served call
        # rather than at load, so assert the identity directly.
        from torch._dynamo.guards import GuardsStatePickler

        w = SELF_REFERENCING_WRAPPED
        buf = io.BytesIO()
        GuardsStatePickler(
            {id(w): w, id(w.__globals__): w.__globals__}, {}, {}, buf
        ).dump({"fn": w})
        got = pickle.loads(buf.getvalue())["fn"]
        self.assertIs(got.me, got)
        cells = [
            cell.cell_contents
            for cell in (got.__closure__ or ())
            if _cell_is_full(cell)
        ]
        self.assertTrue(any(contents is got for contents in cells))

    def test_a_shared_closure_cell_survives_a_self_reference(self):
        # A cell's CONTENTS travel in the owning function's state, so the cell
        # itself comes back empty -- that must not cost the sharing keeping the
        # cell object in the reduce args exists to preserve: two functions
        # closing over one variable must still close over ONE cell after the
        # round trip.
        from torch._dynamo.guards import GuardsStatePickler

        def pair():
            shared = [0]

            def f():
                return shared

            def g():
                return shared

            return f, g

        f, g = pair()
        buf = io.BytesIO()
        GuardsStatePickler(
            {id(f): f, id(g): g, id(f.__closure__[0]): f.__closure__[0]}, {}, {}, buf
        ).dump({"f": f, "g": g})
        out = pickle.loads(buf.getvalue())
        self.assertIs(out["f"].__closure__[0], out["g"].__closure__[0])

    def test_a_dynamo_control_flow_exception_is_not_a_package_bypass(self):
        # pickle_guards_state catches Exception so a user assert in a
        # __reduce__ becomes a bypass rather than a hard compile failure. Every
        # exception Dynamo steers compilation with -- RestartAnalysis,
        # SkipFrame, Unsupported -- derives from TorchDynamoException, which
        # derives from RuntimeError, so the broad catch swallowed those too and
        # a restart silently became a logged bypass. They must propagate.
        global _REDUCE_RAISES
        original = _REDUCE_RAISES
        try:
            for exc_type in (
                torch._dynamo.exc.RestartAnalysis,
                torch._dynamo.exc.SkipFrame,
                torch._dynamo.exc.Unsupported,
            ):
                _REDUCE_RAISES = exc_type
                with self.subTest(exc=exc_type.__name__):
                    with self.assertRaisesRegex(exc_type, "from __reduce__"):
                        _dump_through_pickle_guards_state(_RaisesFromReduce())

            # ... while an ordinary failure inside __reduce__ still becomes one.
            _REDUCE_RAISES = AssertionError
            with self.assertRaisesRegex(
                PackageError, "AssertionError: from __reduce__"
            ):
                _dump_through_pickle_guards_state(_RaisesFromReduce())
        finally:
            _REDUCE_RAISES = original

    def test_reducer_handles_an_empty_cell_reached_directly(self):
        # A cell in a reconstructed function's closure is filled from that
        # function's state. A cell reached directly -- a guarded __closure__
        # tuple, or the cell itself -- goes through reducer_override's branch,
        # which read cell_contents unguarded and raised ValueError out of the
        # pickler, i.e. a package bypass. Pickler-level because a guard cannot
        # root at a raw cell through a capture: CLOSURE_MATCH is dropped.
        from torch._dynamo.guards import GuardsStatePickler

        empty = [c for c in EMPTY_CELL_WRAPPED.__closure__ if not _cell_is_full(c)]
        self.assertEqual(len(empty), 1)
        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"cell": empty[0]})
        self.assertGreater(len(buf.getvalue()), 0)

    def test_reduce_handles_an_empty_closure_cell(self):
        # A free variable a decorator only assigns on a path that did not run
        # has no contents; reading it raised ValueError out of the reducer,
        # which reaches the caller as a package bypass.
        from torch._dynamo.guards import GuardsStatePickler

        wrapped = EMPTY_CELL_WRAPPED
        empty = [c for c in wrapped.__closure__ if not _cell_is_full(c)]
        self.assertEqual(len(empty), 1)
        gtv = {id(wrapped): wrapped}
        buf = io.BytesIO()
        GuardsStatePickler(gtv, {}, {}, buf).dump({"fn": wrapped})
        self.assertGreater(len(buf.getvalue()), 0)

    def test_fqn_mismatched_function_keeps_a_shared_closure_cell_shared(self):
        # Two functions closing over one variable must still share the cell
        # after reload; rebuilding every cell silently unshares them.
        from torch._dynamo.guards import GuardsStatePickler

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

    @parametrize(
        "module_type,guard_type",
        [
            # The defaults TUPLE's length, and separately the VALUE of a guarded
            # element: forcing every default to _Missing passes the first and
            # fails the second.
            subtest((DecoratedForwardModule, "SEQUENCE_LENGTH"), name="defaults"),
            subtest(
                (DecoratedDefaultValueForwardModule, "EQUALS_MATCH"),
                name="default_value",
            ),
            subtest(
                (DecoratedKwdefaultsForwardModule, "EQUALS_MATCH"), name="kwdefaults"
            ),
            subtest(
                (DecoratedAttributeForwardModule, "EQUALS_MATCH"), name="attribute"
            ),
            subtest((DecoratedNameForwardModule, "EQUALS_MATCH"), name="name"),
            subtest(
                (DecoratedGlobalsLengthForwardModule, "DICT_KEYS_MATCH"),
                name="globals_structure",
            ),
        ],
    )
    def test_guard_rooted_at_fqn_mismatched_function(self, module_type, guard_type):
        # forward is a functools.wraps wrapper, so the function it closes over is
        # unreachable by fqn and has to be reconstructed. Each case guards a
        # different piece of what the reconstruction must carry with it.
        mod = module_type()
        ref, loaded = self._test_serialization(guard_type, mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        self._test_check_fn(
            ref, loaded, {"self": mod, "x": torch.randn(3), "func": inner}, True
        )

    def test_fqn_mismatched_function_preserves_a_renamed_name(self):
        # keep_name's __name__ happens to equal co_name, so it passes even if
        # reconstruction falls back to co_name. This one does not.
        mod = DecoratedRenamedNameForwardModule()
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        self.assertNotEqual(inner.__name__, inner.__code__.co_name)
        self._test_check_fn(
            ref, loaded, {"self": mod, "x": torch.randn(3), "func": inner}, True
        )

    def test_fqn_mismatched_function_preserves_guarded_globals(self):
        global FQN_MISMATCH_GLOBAL

        mod = DecoratedGlobalForwardModule()
        x = torch.ones(1)
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, x)
        inner = type(mod).forward.__wrapped__
        inputs = {"self": mod, "x": x, "func": inner}
        self._test_check_fn(ref, loaded, inputs, True)

        try:
            FQN_MISMATCH_GLOBAL = 3
            self.assertFalse(ref.check(inputs))
            guards_state = torch._dynamo.package.load_guards_state(
                self._cached_guards_state
            )
            loaded = torch._dynamo.package.load_guard_manager(
                guards_state,
                self._cached_f_code,
                globals(),
            )
            self.assertFalse(loaded.check(inputs))
        finally:
            FQN_MISMATCH_GLOBAL = 2

    def test_fqn_mismatched_function_prunes_unguarded_defaults(self):
        mod = DecoratedUnpicklableDefaultForwardModule()
        ref, loaded = self._test_serialization("EQUALS_MATCH", mod, torch.randn(3))
        inner = type(mod).forward.__wrapped__
        self._test_check_fn(
            ref, loaded, {"self": mod, "x": torch.randn(3), "func": inner}, True
        )

    def test_two_functions_that_reference_each_other(self):
        # Deferring to state used to be conditional on a DIRECT self-reference,
        # which a PAIR does not have: the wrapper's cell reaches the base, whose
        # __dict__ reaches the wrapper, and neither is memoized while its own
        # args are being saved, so the pair recursed until RecursionError. Only
        # the code, the names and the cells travel in args now.
        from torch._dynamo.guards import GuardsStatePickler

        base, wrapper = CYCLE_BASE, CYCLE_WRAPPER
        cell = wrapper.__closure__[0]
        gtv = {id(base): base, id(wrapper): wrapper, id(cell): cell}
        buf = io.BytesIO()
        GuardsStatePickler(gtv, {}, {}, buf).dump({"wrapper": wrapper})
        got = pickle.loads(buf.getvalue())["wrapper"]
        inner = got.__closure__[0].cell_contents
        self.assertIs(inner.wrapper, got)
        self.assertEqual(got(torch.ones(1)), torch.full((1,), 3.0))

    def test_a_reconstructed_function_keeps_its_module(self):
        # A guarded __globals__ makes the function rebuild from a globals
        # SNAPSHOT, and FunctionType reads __module__ out of globals["__name__"],
        # which a snapshot has no reason to contain -- so __module__ came back
        # None and a guard reading it could never match.
        from torch._dynamo.guards import GuardsStatePickler

        w = SELF_REFERENCING_WRAPPED
        buf = io.BytesIO()
        gtv = {id(w): w, id(w.__globals__): w.__globals__}
        GuardsStatePickler(gtv, {}, {}, buf).dump({"fn": w})
        got = pickle.loads(buf.getvalue())["fn"]
        self.assertEqual(got.__module__, w.__module__)
        self.assertEqual(got.__doc__, w.__doc__)

    def test_guard_on_an_object_that_reconstructs_itself(self):
        # Pruning replaces an unguarded attribute by id, which only comes back
        # as a missing attribute under the DEFAULT protocol. A class with its own
        # __reduce__ gets the sentinel as a constructor ARGUMENT instead, so its
        # attributes must not be pruned at all.
        obj = ReconstructedByReduce("tag", CarriedPayload(3))

        def fn(x):
            if obj.tag == "tag":
                return x + 1
            return x - 1

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "obj": obj}, True)

    def test_guard_on_an_object_whose_setstate_or_new_reads_state(self):
        # __reduce__ is not the only hook that reads a pruned value back: the
        # default protocol hands __setstate__ the whole __dict__, and
        # __getnewargs__ feeds __new__. Either saw the sentinel and raised at
        # load, so both classes must keep every attribute.
        from torch._dynamo.guards import _builds_its_own_pickle

        self.assertFalse(_builds_its_own_pickle(CarriedPayload))
        for obj in (
            ReconstructedBySetstate("tag", CarriedPayload(3)),
            ReconstructedByNewargs(CarriedPayload(3)),
        ):
            self.assertTrue(_builds_its_own_pickle(type(obj)))

            def fn(x):
                if obj.tag == "tag":
                    return x + 1
                return x - 1

            with self.subTest(cls=type(obj).__name__):
                ref, loaded = self._test_serialization(
                    "EQUALS_MATCH", fn, torch.randn(3)
                )
                self._test_check_fn(
                    ref, loaded, {"x": torch.randn(3), "obj": obj}, True
                )

    def test_guard_on_a_copyreg_registered_class(self):
        # A reducer registered with copyreg.pickle decides what its arguments
        # mean, exactly as a __reduce__ on the class would, so pruning a sibling
        # attribute would hand the rebuild the sentinel. It must be kept whole.
        from torch._dynamo.guards import _builds_its_own_pickle

        self.assertTrue(_builds_its_own_pickle(CopyregReduced))
        obj = CopyregReduced("tag", CarriedPayload(3))

        def fn(x):
            if obj.tag == "tag":
                return x + 1
            return x - 1

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "obj": obj}, True)

    def test_guard_on_a_namedtuple_subclass_with_an_unpicklable_extra(self):
        # Every namedtuple has a __getnewargs__, but it returns the ITEMS, which
        # pruning never touches, so a subclass carrying __dict__ extras is pruned
        # like any other user object rather than pickled whole.
        from torch._dynamo.guards import _builds_its_own_pickle

        self.assertFalse(_builds_its_own_pickle(TaggedPoint))
        pt = TaggedPoint(1, 2)
        pt.scratch = (i for i in ())

        def fn(x):
            if pt.x == 1:
                return x + 1
            return x - 1

        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "pt": pt}, True)

    def test_a_guarded_unsupported_type_is_refused(self):
        # Substituting the sentinel for a value a guard READS is not a pruning:
        # a TYPE_MATCH rebuilds against type(_Missing()) and then never matches,
        # so the artifact would load and silently reject every call. Only a value
        # the guard tree never reached may become the sentinel, and only
        # pickle_guards_state -- which has checked every guard -- may hand the
        # pickler a type stand-in instead.
        from torch._dynamo.guards import _Missing, GuardsStatePickler

        gen = torch.Generator()
        with self.assertRaisesRegex(PackageError, "torch._C.Generator"):
            GuardsStatePickler({id(gen): gen}, {}, {}, io.BytesIO()).dump({"gen": gen})

        buf = io.BytesIO()
        GuardsStatePickler({}, {}, {}, buf).dump({"gen": gen})
        self.assertIsInstance(pickle.loads(buf.getvalue())["gen"], _Missing)

        buf = io.BytesIO()
        pickler = GuardsStatePickler(
            {id(gen): gen}, {}, {}, buf, type_stand_ins={id(gen)}
        )
        pickler.dump({"gen": gen})
        stand_in = pickle.loads(buf.getvalue())["gen"]
        self.assertIs(type(stand_in), torch.Generator)
        self.assertEqual(stand_in.device, gen.device)

    def test_a_stand_in_this_host_cannot_build_fails_the_load_by_name(self):
        # The stand-in is built at load, so a device the loading host lacks
        # raised the constructor's own error out of pickle.loads, with nothing
        # to say which type or device the artifact was captured with. A Stream
        # rather than a Generator: the Generator binds its device lazily.
        payload = pickle.dumps(StandInOn(torch.Stream, torch.device("cuda:99")))
        with self.assertRaisesRegex(
            PackageError,
            r"captured with a torch.Stream on cuda:99 that this host cannot create",
        ):
            pickle.loads(payload)

    def test_a_type_matched_generator_rebuilds_as_a_stand_in(self):
        # Dynamo guards a Generator with TYPE_MATCH, which needs the type alone,
        # so the artifact carries a fresh Generator of the same type and device
        # instead of refusing the frame.
        def fn(x, gen):
            if gen.device.type == "cpu":
                return x + 1
            return x - 1

        gen = torch.Generator()
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, torch.randn(3), gen)
        state = torch._dynamo.package.load_guards_state(self._cached_guards_state)
        stand_in = state.output_graph.local_scope["gen"]
        self.assertIs(type(stand_in), torch.Generator)
        inputs = {"x": torch.randn(3), "gen": torch.Generator()}
        self._test_check_fn(ref, loaded, inputs, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "gen": 1}, False)

        # A guard THROUGH the object on .device rebuilds too: the stand-in keeps it.
        ref, loaded = self._test_serialization("EQUALS_MATCH", fn, torch.randn(3), gen)
        self._test_check_fn(ref, loaded, inputs, True)

    def test_a_type_matched_stream_rebuilds_as_a_stand_in(self):
        # The stand-in for a stream is a fresh stream on the same device.
        def fn(x, s):
            if s.device.type == "cpu":
                return x + 1
            return x - 1

        s = torch.Stream(device="cpu")
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, torch.randn(3), s)
        state = torch._dynamo.package.load_guards_state(self._cached_guards_state)
        self.assertIs(type(state.output_graph.local_scope["s"]), torch.Stream)
        inputs = {"x": torch.randn(3), "s": torch.Stream(device="cpu")}
        self._test_check_fn(ref, loaded, inputs, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "s": None}, False)

    def test_a_value_guard_on_a_stream_is_refused(self):
        # s == t emits an EQUALS_MATCH on each stream, which compares the live
        # stream's identity. A stand-in would rebuild that guard against a
        # fresh stream, so the value guard keeps the refusal -- with the path.
        def fn(x, s, t):
            if s == t:
                return x + 1
            return x - 1

        s, t = torch.Stream(device="cpu"), torch.Stream(device="cpu")
        with self.assertRaisesRegex(
            PackageError,
            r"a guard reads a torch.Stream.*\n  reached via: local_scope\['s'\]",
        ):
            self._test_serialization("EQUALS_MATCH", fn, torch.randn(3), s, t)

    def test_a_guarded_process_group_is_refused_with_its_path(self):
        # No stand-in exists for a process group, so a guard reaching one still
        # refuses the frame, and pickle_guards_state appends where it lives.
        import torch.distributed as dist

        if not dist.is_available():
            self.skipTest("Torch distributed is not available")
        from torch.testing._internal.distributed.fake_pg import FakeStore

        dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
        try:
            with self.assertRaisesRegex(
                PackageError,
                r"a guard reads a torch.distributed.distributed_c10d.ProcessGroup, "
                r"which precompile cannot serialize\n  reached via: local_scope\['value'\]",
            ):
                _dump_through_pickle_guards_state(dist.group.WORLD)
        finally:
            dist.destroy_process_group()

    def test_wrapper_subclass_parameter_stays_a_parameter(self):
        # nn.Parameter of a wrapper subclass IS the subclass, with _is_param set,
        # and nn.Parameter.__instancecheck__ reads that flag. It is only
        # presence-tested, so nothing registers it in the guard tree and pruning
        # dropped it: the rebuilt tensor was no longer a Parameter.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            return x + 1

        p = torch.nn.Parameter(TwoTensor(torch.randn(3), torch.randn(3)))
        self.assertIsInstance(p, torch.nn.Parameter)
        ref, loaded = self._test_serialization("TENSOR_MATCH", f, p)
        state = torch._dynamo.package.load_guards_state(self._cached_guards_state)
        rebuilt = state.output_graph.local_scope["x"]
        self.assertIsInstance(rebuilt, TwoTensor)
        self.assertIsInstance(rebuilt, torch.nn.Parameter)
        self._test_check_fn(ref, loaded, {"x": p}, True)

    def test_live_guard_leaves_reach_every_open_recording(self):
        # Two sessions may record at once, so a live build feeds every open sink
        # rather than the latest one hiding the rest.
        from torch._dynamo.guards import record_live_guard_leaves

        def f(x):
            return x + 1

        def g(x, y):
            return x + y

        with record_live_guard_leaves() as outer:
            with record_live_guard_leaves() as inner:
                self._test_serialization("TENSOR_MATCH", f, torch.randn(3))
            self.assertTrue(inner)
            self.assertEqual(inner, outer)
            self._test_serialization("TENSOR_MATCH", g, torch.randn(3), torch.randn(3))
        self.assertGreater(len(outer), len(inner))
        self.assertTrue(outer > inner)

        # Closed by identity: a and b are EQUAL while both are empty, so removing
        # b by equality would close a instead and the wrong sink would fill.
        recording_a, recording_b = (
            record_live_guard_leaves(),
            record_live_guard_leaves(),
        )
        a = recording_a.__enter__()
        b = recording_b.__enter__()
        recording_b.__exit__(None, None, None)
        try:
            self._test_serialization("TENSOR_MATCH", f, torch.randn(3))
        finally:
            recording_a.__exit__(None, None, None)
        self.assertTrue(a)
        self.assertFalse(b)

        # A fresh set on every entry, so a new recording starts empty.
        with record_live_guard_leaves() as leaves:
            self._test_serialization("TENSOR_MATCH", f, torch.randn(3))
        self.assertEqual(leaves, a)

    def test_dimension_marking_range_survives(self):
        # _dynamo_dynamic_range is compared by VALUE, but only when the
        # _has_dynamo_dim_marking gate is PRESENT on the tensor the guard was
        # built from. The gate is never read, only hasattr-tested, so nothing
        # registers it in the guard tree and pruning dropped it -- leaving the
        # rebuilt guard with no range leaf at all, so a tensor declaring a
        # different range for the same dim reused the graph.
        def fn(x):
            return x + 1

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
        ref, loaded = self._test_serialization("TENSOR_MATCH", fn, x)

        y = torch.randn(4)
        torch._dynamo.mark_dynamic(y, 0, min=3, max=9)
        self._test_check_fn(ref, loaded, {"x": y}, False)

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

    def test_tensor_subclass_requires_grad_survives(self):
        # A wrapper subclass is rebuilt by __tensor_unflatten__, which derives
        # the outer's requires_grad from its inners -- so a subclass carrying
        # autograd metadata of its own reloaded as requires_grad=False and the
        # rebuilt guard then rejected every training input, permanently.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x: torch.Tensor):
            return x + 1

        tt = TwoTensor(torch.randn(3), torch.randn(3)).requires_grad_(True)
        self.assertFalse(tt.a.requires_grad)
        ref, loaded = self._test_serialization("TENSOR_MATCH", f, tt)
        self._test_check_fn(ref, loaded, {"x": tt}, True)
        self._test_check_fn(
            ref, loaded, {"x": TwoTensor(torch.randn(3), torch.randn(3))}, False
        )

    def test_tensor_match_through_a_python_attribute(self):
        # A tensor is reconstructed from its metadata, which does not include a
        # plain Python attribute someone assigned onto it -- so a guard whose
        # SOURCE traverses one could not be rebuilt at all, and the whole state
        # failed to load with AttributeError.
        def f(x: torch.Tensor):
            return x + x.companion

        x = torch.ones(2)
        x.companion = torch.ones(2)
        ref, loaded = self._test_serialization("TENSOR_MATCH", f, x)

        def with_companion(companion):
            t = torch.randn(2)
            t.companion = companion
            return {"x": t}

        self._test_check_fn(ref, loaded, with_companion(torch.randn(2)), True)
        self._test_check_fn(ref, loaded, with_companion(torch.randn(3)), False)
        self._test_check_fn(
            ref, loaded, with_companion(torch.randn(2, dtype=torch.float64)), False
        )

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
