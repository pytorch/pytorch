# mypy: allow-untyped-decorators
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import logging
import math
import os
import threading
import traceback
import types
import typing
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Literal,
    Optional,
    TYPE_CHECKING,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import Self
from weakref import ReferenceType

import torch
import torch._library.utils as library_utils
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C._functorch import is_functorch_wrapped_tensor, is_legacy_batchedtensor
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.fake_profile import MissingOpProfile
from torch._logging import dtrace_structured
from torch._prims_common import suggest_memory_format
from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)
from torch._utils import render_call
from torch.fx.immutable_collections import immutable_dict
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.types import IntLikeType, py_sym_types
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import KeyPath, keystr, PyTree, tree_map, tree_map_, TreeSpec
from torch.utils._stats import count
from torch.utils._traceback import CapturedTraceback

from ._fake_tensor_utils import _CacheKeyState, _PySymInputStub, _SymIntOutputStub


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
    from types import TracebackType

    from torch._guards import Source
    from torch._ops import OpOverload
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# TODO: Hack to unblock https://github.com/pytorch/pytorch/pull/108186
# Proper fix tracked by https://github.com/pytorch/pytorch/issues/120105
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        not_implemented_log = logging.getLogger(__name__ + ".not_implemented")
    else:
        raise e


DimList = list

pytree = torch.utils._pytree
T = TypeVar("T")

aten = torch._ops.ops.aten

CONSTANT_NUMEL_LIMIT = 1

RECURSION_COUNT = 0


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:
    def __init__(self) -> None:
        global RECURSION_COUNT
        RECURSION_COUNT += 1

    def __del__(self) -> None:
        global RECURSION_COUNT
        RECURSION_COUNT -= 1


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str


@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload


@dataclass
class DataDependentOutputException(RuntimeError):
    func: OpOverload


@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload


@dataclass
class UnsupportedMutationAliasingException(RuntimeError):
    reason: str


@dataclass
class MetadataMismatchError(RuntimeError):
    reason: str


class FakeTensorTLS(threading.local):
    # Default to None, otherwise it'll be used to override _all_
    # `FakeTensorMode.allow_non_fake_inputs` in this thread.
    allow_non_fake_inputs_override: Optional[bool]
    non_strict_export_fake_tensor_tracker: weakref.WeakSet

    def __init__(self) -> None:
        self.allow_non_fake_inputs_override = None
        self.non_strict_export_fake_tensor_tracker = weakref.WeakSet()


fake_tensor_tls = FakeTensorTLS()


def ordered_set(*items: T) -> dict[T, Literal[True]]:
    return dict.fromkeys(items, True)


@contextlib.contextmanager
def unset_fake_temporarily() -> Generator[Optional[TorchDispatchMode], None, None]:
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


@contextlib.contextmanager
def disable_fake_tensor_cache(fake_mode: FakeTensorMode) -> Generator[None, None, None]:
    old_value: bool = fake_mode.cache_enabled
    try:
        fake_mode.cache_enabled = False
        yield
    finally:
        fake_mode.cache_enabled = old_value


def get_plain_tensors(
    subclass: Tensor, *, out: list[Union[Tensor, int, SymInt]]
) -> list[Union[Tensor, int, SymInt]]:
    # This function is used in Runtime, do not add redundant asserts
    todo = [subclass]
    while todo:
        curr = todo.pop()
        if not is_traceable_wrapper_subclass(curr):
            out.append(curr)
            continue

        inner_keys, _ = curr.__tensor_flatten__()
        todo.extend(getattr(curr, key) for key in reversed(inner_keys))

    return out


def is_fake(x: object) -> TypeGuard[Tensor]:
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(x, FakeTensor):
        return True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)
        flattened_tensors = [getattr(x, attr) for attr in attrs]
        all_fake = all(is_fake(x) for x in flattened_tensors)
        any_fake = any(is_fake(x) for x in flattened_tensors)
        assert all_fake == any_fake, "got mixed fake and real tensors!"
        return all_fake
    elif isinstance(x, FunctionalTensor):
        return is_fake(x.elem)
    elif isinstance(x, Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)
        return is_fake(unwrapped)
    elif isinstance(x, Tensor) and is_functorch_wrapped_tensor(x):
        unwrapped = torch._C._functorch.get_unwrapped(x)
        return is_fake(unwrapped)
    return False


def maybe_get_fake_mode(t: object) -> Optional[FakeTensorMode]:
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FakeTensor):
        return t.fake_mode
    if is_traceable_wrapper_subclass(t):
        inner_tensor_names, _ = t.__tensor_flatten__()
        modes = [
            maybe_get_fake_mode(getattr(t, t_name)) for t_name in inner_tensor_names
        ]
        m = modes[0]
        assert all(m is x for x in modes)
        return m
    elif isinstance(t, FunctionalTensor):
        return maybe_get_fake_mode(t.elem)
    elif isinstance(t, Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)
        return maybe_get_fake_mode(unwrapped)
    elif isinstance(t, Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)
        return maybe_get_fake_mode(unwrapped)
    return None


@functools.cache
def get_schema_info(func: OpOverload) -> torch._C._SchemaInfo:
    return torch._C._SchemaInfo(func._schema)


# many of the decompositions registered to torch/_prims do not at the moment model
# aliasing or strides, so as an incremental step, just enable the decompositions in
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them
@functools.cache
def torch_decomp_decompositions(func: OpOverload) -> bool:
    from torch._decomp import decomposition_table

    decompositions = torch._decomp.decompositions
    # Note that the function in the decomposition table might be
    # different from the one in the module because of the difference
    # in out handling in aten API and torch public API
    return decomposition_table[func].__module__.startswith(
        "torch._decomp"
    ) and decomposition_table[func].__name__ in dir(decompositions)


def tree_flatten_only(ty: type[T], tree: PyTree) -> list[T]:
    flat_vals = pytree.tree_leaves(tree)
    return [elem for elem in flat_vals if isinstance(elem, ty)]


def _is_plain_tensor(t: object) -> bool:
    return (
        type(t) is Tensor
        and t.layout == torch.strided
        and not (
            t.is_sparse
            or t.is_nested
            or is_functorch_wrapped_tensor(t)
            or is_legacy_batchedtensor(t)
            or torch._is_functional_tensor(t)
        )
    )


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter:
    @property
    def tensor_memo(
        self,
    ) -> weakref.WeakValueDictionary:
        # not valid until py3.10
        # weakref.WeakValueDictionary["torch._subclasses.meta_utils.MetaTensorId", Optional["FakeTensor"]]
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: dict[StorageWeakRef, list[ReferenceType]]
    export: bool

    def __init__(self, *, copy_data: bool = False, export: bool = False) -> None:
        self.meta_converter = MetaConverter(copy_data=copy_data)
        self.export = export

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor: FakeTensor) -> None:
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor: Tensor) -> None:
        assert not isinstance(tensor, FakeTensor)

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def _get_memo(self, t: Tensor) -> Optional[FakeTensor]:
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t: Tensor, v: FakeTensor) -> None:
        tid = self.meta_converter.describer.get_tensor_id(t)
        self.meta_converter.tensor_memo[tid] = v

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def from_real_tensor(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        make_constant: bool = False,
        shape_env: Optional[ShapeEnv] = None,
        *,
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
        trace: bool = True,
    ) -> FakeTensor:
        # see note [Tensor Fakification and Symbol Caching]
        if not symbolic_context and not source and shape_env:
            if tracing_context := torch._guards.TracingContext.try_get():
                if t in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[t]
                    from torch.fx.experimental.symbolic_shapes import (
                        StatefulSymbolicContext,
                    )

                    assert isinstance(symbolic_context, StatefulSymbolicContext)
                    source = symbolic_context.tensor_source

        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        if type(t) is torch.nn.Parameter:
            assert not make_constant

        constant = t if make_constant else None

        # This callback is used by both subclass and inner tensors. Require the
        # caller to explicitly specify the device in case outer and inner tensors
        # have different devices.
        def mk_fake_tensor(
            make_meta_t: Callable[[], object], device: Union[torch.device, str]
        ) -> FakeTensor:
            # NB: don't use in_kernel_invocation_manager. to
            # ensure FakeTensor can internally do constant computation
            # as necessary.  Invocation manager is "more correct" as
            # it works for more operators in make_meta_t, but
            # invariant is that make_meta_t only calls factories
            # for which it is not strictly necessary to use the
            # invocation manager (I think!)
            with no_dispatch():
                return FakeTensor(
                    fake_mode,
                    # pyrefly: ignore [bad-argument-type]
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                    # TODO: callback might be used in recursive contexts, in
                    # which case using t is wrong!  BUG!
                    constant=constant,
                )

        out = self.meta_converter(
            t,
            shape_env=shape_env,
            callback=mk_fake_tensor,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )
        if out is NotImplemented:
            raise UnsupportedFakeTensorException("meta converter nyi")

        from torch._dynamo.source import RandomValueSource

        value = None
        if (
            not self.export
            and _is_plain_tensor(t)  # mostly, we want to know if item() works
            and t.dim() == 0
            and t.device.type == "cpu"
            # All integer types are fair game, because signed overflow is UB
            # (and even int64 can overflow, since integers in Python are
            # arbitrary precision). But only float64 is OK for float, because
            # switching between float32 and float64 changes semantics in an
            # observable way without hitting UB.
            and t.dtype
            in [torch.int64, torch.int32, torch.int16, torch.int8, torch.float64]
            and source is not None
            # Impede setting up item() on things coming from random.  These
            # are not "real" item() calls, instead UnspecializedPythonVariable
            # is unsafely pretending an int is a tensor, which can sometimes
            # implicitly cause an item call.  The problem is this is pretty
            # unsound: there's no reason substituting an int with a Tensor is
            # going to give the same results.  Today, you mostly get around
            # this by typically not having capture_scalar_outputs on and graph
            # breaking when someone tries to use the unspec variable in an
            # int-y context.  But allowing it through here would break that.
            # So don't.
            #
            # Once random values are setup to be represented as
            # SymNodeVariable, this condition can be removed.  To check if
            # you've done it right, this is a good test:
            #
            #   PYTORCH_TEST_WITH_DYNAMO=1 python test/test_reductions.py -k
            #   TestReductionsCPU.test_dim_reduction_fns_fn_name_amax_cpu_bfloat16
            and not isinstance(source, RandomValueSource)
            # In Dynamo, shape_env is never none (even with static shapes).
            # However, FakeTensorMode can be used by hand and in some cases
            # ShapeEnv is not allocated.
            and shape_env is not None
        ):
            from torch._dynamo.source import CallMethodItemSource, FloatTensorSource
            from torch.fx.experimental.symbolic_shapes import DimDynamic

            with no_dispatch():
                value = t.item()
            if not math.isnan(value) and not math.isinf(value):
                # Peephole strip out unnecessary torch.as_tensor(x).item()
                if isinstance(source, FloatTensorSource):
                    item_source = source.base
                else:
                    item_source = CallMethodItemSource(source)
                symbol = shape_env.create_unspecified_symbol(
                    value,
                    source=item_source,
                    dynamic_dim=DimDynamic.DYNAMIC,
                    symbolic_context=symbolic_context,
                )
                # NB: reusing item_memo here ensures that we invalidate on
                # mutation
                if t.dtype == torch.int64:
                    out.item_memo = shape_env.create_symintnode(
                        symbol,
                        hint=value,
                        source=item_source,
                    )
                elif t.dtype == torch.float64:
                    out.item_memo = shape_env.create_symfloatnode(
                        symbol,
                        hint=value,
                        source=item_source,
                    )
        if make_constant:
            self.add_constant_storage_mapping(out)
        # NB: meta_converter set the memo
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(
        self,
        fake_mode: FakeTensorMode,
        t: Tensor,
        device: torch.device,
        pytype: Optional[type[torch.Tensor]] = None,
        dispatch_keys: Optional[torch.DispatchKeySet] = None,
    ) -> FakeTensor:
        assert t.device.type == "meta", (
            f"tensor's device must be `meta`, got {t.device.type} instead"
        )
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(
            fake_mode, t, device, pytype=pytype, dispatch_keys=dispatch_keys
        )
        self.set_tensor_memo(t, out)
        return out


@functools.cache
def init_gpu_context(device: torch.device) -> None:
    # Backward will error with cuda Fake Tensors if no cuda tensors have been initialized first
    if torch.cuda.is_available() or torch.xpu.is_available():
        (
            torch.empty(1, device=device)
            if torch.version.hip is None
            else torch.zeros(1, device=device)
        )


@contextlib.contextmanager
def in_kernel_invocation_manager(
    fake_mode: FakeTensorMode,
) -> Generator[None, None, None]:
    # See: note [Fake Tensor Dispatch Keys]
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f"{meta_in_tls}, {prev_in_kernel}"

    with torch._C._DisableTorchDispatch():
        fake_mode.in_kernel_invocation = True
        # Unfortunately _set_meta_in_tls_dispatch_include(False) can leave
        # `Dense` turned on (because it's implied by `Meta`)
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield
            finally:
                fake_mode.in_kernel_invocation = prev_in_kernel
                # torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)


# Return if the function allows Python numbers to bind to Tensors
def should_allow_numbers_as_tensors(func: OpOverload) -> bool:
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


class FakeTensorConfig:
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", "0") == "1"


# This memorizes unbacked SymInt or SymFloats representing quantities like the
# number of nonzero elements in this tensor or learning rate. There is one
# instance of the descriptor per particular quantity to memoize.
#
# Memoization is helpful if you do something like x[mask] and y[mask];
# mask.nonzero() gets repeatedly called and should give a consistent unbacked
# SymInt. It needs to be invalidated in the same way constant is.
#
# Making this a descriptor may seem overly fancy, but actually it's the most
# convenient way to ensure access to FakeTensor during access, which is
# required for testing version counter and epoch validity.
class SymNumberMemoDescriptor:
    _name: str

    # By default, SymInts in this memo are invalidated across versions/epochs.
    # nested_ints however are preserved across epochs and across versions.
    # Preserving across versions is okay for nested int since the association
    # of a nested int is agnostic to the underlying data and nested ints are not
    # shared across multiple distinct tensors.
    _is_nested_int: bool

    def __init__(self, *, is_nested_int: bool = False) -> None:
        self._is_nested_int = is_nested_int

    def __set_name__(self, owner: str, name: str) -> None:
        self._name = name

    def _memo(self, obj: FakeTensor) -> str:
        return f"_{self._name}"

    def _memo_vc(self, obj: FakeTensor) -> str:
        return f"_{self._name}_vc"

    # When we retrace, we need to invalidate all the memos so that we can
    # accurately identify the first time unbacked SymInts are allocated.
    # This is only relevant for inputs; for intermediates, they will get fresh
    # fake tensors so you won't have a memo anyway
    def _memo_epoch(self, obj: FakeTensor) -> str:
        return f"_{self._name}_epoch"

    def __get__(
        self, obj: FakeTensor, objtype: Optional[type[FakeTensor]] = None
    ) -> Optional[Union[torch.SymInt, torch.SymFloat]]:
        if (r := getattr(obj, self._memo(obj))) is None:
            return None

        # If backed, it's ok to preserve memo since we know it won't renumber.
        if isinstance(r, torch.SymFloat) and r.node.hint is not None:
            return r

        # Version counter based tracking isn't 100% sound but it's close
        # enough
        if (
            not self._is_nested_int and getattr(obj, self._memo_vc(obj)) != obj._version
        ) or (
            not self._is_nested_int
            and getattr(obj, self._memo_epoch(obj)) != obj.fake_mode.epoch
        ):
            setattr(obj, self._memo(obj), None)
            return None
        return r

    def __set__(
        self, obj: FakeTensor, value: Optional[Union[torch.SymInt, torch.SymFloat]]
    ) -> None:
        if value is None:
            setattr(obj, self._memo(obj), None)
            setattr(obj, self._memo_vc(obj), None)
            setattr(obj, self._memo_epoch(obj), None)
        elif not obj.is_inference() or self._is_nested_int:
            setattr(obj, self._memo(obj), value)
            if not self._is_nested_int:
                setattr(obj, self._memo_vc(obj), obj._version)
            setattr(obj, self._memo_epoch(obj), obj.fake_mode.epoch)


class FakeTensor(Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    fake_device: torch.device
    fake_mode: FakeTensorMode
    constant: Optional[Tensor]
    real_tensor: Optional[Tensor]

    # TODO: Generalize this as needed, e.g., into a trie of memos, if
    # you do something like x[0].item()  (x[0] is fresh each time, so
    # memo mechanism here won't work)
    nonzero_memo = SymNumberMemoDescriptor()
    item_memo = SymNumberMemoDescriptor()
    unique_memo = SymNumberMemoDescriptor()
    unique_consecutive_memo = SymNumberMemoDescriptor()

    # We expect nested_int_memo to be None when an offsets is a graph
    # intermediate, or an input that has never been associated with a
    # nested int.
    nested_int_memo = SymNumberMemoDescriptor(is_nested_int=True)

    # FakeTensor doesn't fully emulate the original tensor's Python type
    # and dispatch key set, therefore sometimes we want to track them
    # separately.
    pytype: Optional[type[Tensor]]
    dispatch_keys: Optional[torch.DispatchKeySet]

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    # pyrefly: ignore [bad-override]
    def device(self) -> torch.device:
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    @device.setter
    def device(self, _: torch.device) -> None:
        raise NotImplementedError

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch_dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below
    # NOTE: this probably will not do the right thing for backends
    # that have dispatch keys which are higher than the "meta" key:
    # https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h#L189

    # We don't support named tensors; graph break
    @property
    # pyrefly: ignore [bad-override]
    def names(self) -> list[str]:
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    @names.setter
    def names(self, _: list[str]) -> None:
        raise NotImplementedError

    @staticmethod
    def __new__(
        cls,
        fake_mode: FakeTensorMode,
        elem: Tensor,
        device: torch.device,
        constant: Optional[Tensor] = None,
        real_tensor: Optional[Tensor] = None,
        pytype: Optional[type[Tensor]] = None,
        dispatch_keys: Optional[torch.DispatchKeySet] = None,
    ) -> Self:
        self = Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        if not fake_mode._allow_unsafe_data_ptr_access:
            torch._C._set_throw_on_mutable_data_ptr(self)
        else:
            torch._C._set_warn_deprecated_on_mutable_data_ptr(self)

        assert elem.device.type == "meta", elem.device.type
        device = device if isinstance(device, torch.device) else torch.device(device)
        # NB: it is fine, if a little confusing, for device to be meta
        # (we are faking a meta tensor in that case).  However, it often
        # indicates some sort of confusion (e.g., you accidentally passed
        # in a meta tensor when you should have passed in the real tensor).
        # So by default we disallow meta, and if you are working in a situation
        # where it is helpful (e.g., crossref testing) you can turn it back
        # on
        if not fake_mode.allow_meta:
            assert device.type != "meta"
        # normalize device.
        if device.type in ["cuda", "xpu"]:
            init_gpu_context(device)

        if (
            device.type
            in ["cuda", "hpu", "xpu", "mps", torch._C._get_privateuse1_backend_name()]
            and device.index is None
        ):
            if device.type != "mps" and getattr(torch, device.type).is_initialized():
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        # pyrefly: ignore [read-only]
        self.fake_device = device
        self.fake_mode = fake_mode
        self.constant = constant
        self.pytype = pytype
        self.dispatch_keys = dispatch_keys
        assert not isinstance(real_tensor, FakeTensor)
        self.real_tensor = real_tensor
        self.nonzero_memo = None
        self.item_memo = None
        self.unique_memo = None
        self.unique_consecutive_memo = None
        self.nested_int_memo = None

        if FakeTensorConfig.debug:
            self._debug_trace = CapturedTraceback.extract()  # type: ignore[attr-defined]
        return self

    # In some circumstances, a conventional Tensor constructor
    # will get rewritten to call into FakeTensor.  We must provide an
    # __init__ method that can accept the Python interpreters initialization
    # in such a situation; we must also be able to handle direct fake
    # tensor construction via FakeTensor().
    #
    # In particular, the __init__ call will look funny in the following case:
    #
    #   with FakeTensorMode():
    #       x = Tensor([1, 2, 3])
    #
    # this desugars into:
    #
    #   with FakeTensorMode():
    #       x = Tensor.__new__([1, 2, 3])
    #       # NB: x is a fake tensor, because of the mode!
    #       x.__init__([1, 2, 3])  # not the normal fake tensor args!
    #
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        if (
            torch.compiler.is_exporting()
            and torch._export.config.detect_non_strict_fake_tensor_leaks
        ):
            fake_tensor_tls.non_strict_export_fake_tensor_tracker.add(self)

    @staticmethod
    def from_tensor(t: Tensor, fake_mode: FakeTensorMode) -> FakeTensor:
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(  # type: ignore[override] # TODO
        cls,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func is torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # this handler must be done inside FakeTensor subclass, not mode, because
        # we can end up dispatching here when we have a fake tensor with
        # symbolic sizes running under in_kernel_invocation_manager.
        # The subclass is asked to handle this query because size (not
        # sym_size) was called, but we are unable to serve it directly because
        # there are symbolic sizes in the class.  The use of
        # in_kernel_invocation_manager means it's incorrect to activate a
        # mode to actually handle this (this caused
        # https://github.com/pytorch/pytorch/issues/122772).
        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break

        assert fake_mode is not None

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(
            torch._C._TorchDispatchModeKey.FAKE
        )
        if maybe_cur_fake_mode:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_mode,
                maybe_cur_fake_mode,
            )
            return NotImplemented

        assert not fake_mode.in_kernel_invocation

        with fake_mode:
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(
        func: OpOverload, flat_args: Sequence[object]
    ) -> tuple[torch.device, bool]:
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        # list of ops which can have args(tensor/tensorList) in mixed device
        mixed_device_fns = ordered_set(
            aten._foreach_copy.default,
        )

        # list of ops not using zero dim cpu tensor logic to align with the eager mode.
        bypass_zero_dim_cpu_tensor_check_ops = ordered_set(
            aten.nextafter.default,
        )

        def check_cpu_device(device: torch.device) -> bool:
            return device.type == "cpu"

        def cpu_zero_dim(t: Tensor) -> bool:
            return check_cpu_device(t.device) and t.dim() == 0

        def merge_devices(t: object) -> None:
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            is_bypass_zero_dim_cpu_tensor_check_op = (
                func in bypass_zero_dim_cpu_tensor_check_ops
            )

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim and not is_bypass_zero_dim_cpu_tensor_check_op:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # if still device mismatches we will check ops which can work
            # on different devices for ex. _foreach_copy, and one of the
            # device must be cpu in this case we will return from here without
            # throwing an error
            if func in mixed_device_fns:
                if any(map(check_cpu_device, (common_device, t.device))):
                    return

            # if prefer_device_type is set, prefer that device type over others
            prefer_device_type = torch._functorch.config.fake_tensor_prefer_device_type
            if prefer_device_type is not None:
                common_has_preferred = prefer_device_type in common_device.type
                t_has_preferred = prefer_device_type in t.device.type

                if not common_has_preferred and t_has_preferred:
                    # Switch to the preferred device type
                    common_device = t.device
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                    return
                elif common_has_preferred and not t_has_preferred:
                    # Keep the existing preferred device type
                    return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise RuntimeError(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        for arg in flat_args:
            merge_devices(arg)

        # some functions that allow Python numbers to bind to Tensors
        # if we have failed to find a device, and we're running one of these operators,
        # we must have scalar only inputs
        if should_allow_numbers_as_tensors(func) and common_device is None:
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        assert common_device is not None, f"Could not find common device for {func}"

        return common_device, has_scalar_only_inputs

    def get_nested_int(
        self,
        *,
        coeff: Union[int, torch.SymInt] = 1,
    ) -> torch.SymInt:
        if self.nested_int_memo is None:
            self.nested_int_memo = self.fake_mode.create_symbolic_nested_int(
                nt_tensor_id=None
            )
        assert isinstance(self.nested_int_memo, torch.SymInt)
        return self.nested_int_memo * coeff

    # Similar to FunctionalTensor.tolist
    def tolist(self) -> Any:
        if self.dim() == 0:
            return self.item()
        elif self.dim() == 1:
            return [elem.item() for elem in self]
        else:
            return [elem.tolist() for elem in self]


_MetadataIntLike = Union[IntLikeType, "_PySymInputStub", "_SymIntOutputStub"]


@dataclass(slots=True)
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    dtype: torch.dtype
    shape: tuple[_MetadataIntLike, ...]
    stride: tuple[_MetadataIntLike, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: _MetadataIntLike
    storage_bytes: Optional[_MetadataIntLike]
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool  # read: is sparse COO
    is_coalesced: Optional[bool]
    dense_dim: Optional[int]
    sparse_dim: Optional[int]

    def _flatten_into(
        self,
        result: list[object],
        mode: FakeTensorMode,
        state: _CacheKeyState,
    ) -> None:
        # Flatten the TensorMetadata out into `result`.  Make sure to call
        # state.convert_sym_int() on any SymInts.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (tuple, list, torch.Size)):
                # This will recursively flatten the iterable, calling
                # convert_sym_int() as necessary.
                id_hashed_objects: list[object] = []
                mode._prep_args_for_hash(result, value, state, id_hashed_objects)
                id_hashed_objects.clear()
            elif isinstance(value, SymInt):
                state.convert_sym_int(result, value)
            else:
                result.append(value)


def extract_tensor_metadata(t: Tensor) -> TensorMetadata:
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format = suggest_memory_format(t)
    # Don't call is_contiguous() on a Tensor which has symbolic sizes or things
    # will go badly (guards will be messed up?)
    if (
        t._has_symbolic_sizes_strides
        or is_sparse_any(t)
        or not t.is_contiguous(memory_format=memory_format)
    ):
        memory_format = None  # type: ignore[assignment]

    storage_offset = t.storage_offset()

    return TensorMetadata(
        t.dtype,
        t.shape,
        t.stride() if t.layout == torch.strided else (),
        t.device,
        t.layout,
        memory_format,
        storage_offset,
        # Only set storage_bytes for tensors that have storage (not sparse)
        t.untyped_storage().nbytes() if not is_sparse_any(t) else None,
        t.requires_grad,
        t.is_quantized,
        t.is_conj(),
        t.is_neg(),
        t.is_inference(),
        t.is_sparse,
        t.is_coalesced() if t.is_sparse else None,
        t.dense_dim() if is_sparse_any(t) else None,
        t.sparse_dim() if is_sparse_any(t) else None,
    )


@dataclass(slots=True)
class _DispatchCacheKey:
    """
    Key for the FakeTensor dispatch cache.
    """

    key: tuple[object, ...]
    hashvalue: int

    def __init__(self, tup: tuple[object, ...]) -> None:
        self.key = tup
        self.hashvalue = hash(tup)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DispatchCacheKey) and self.key == other.key

    def __hash__(self) -> int:
        return self.hashvalue

    def strip_shape_env(self) -> None:
        # We need to strip the ShapeEnv from any values before we store in the
        # cache so the cache doesn't keep our ShapeEnvs alive.
        for v in self.key:
            if isinstance(v, _PySymInputStub):
                v.strip_shape_env()


# Default value for constant_value in _DispatchCacheEntryOutputInfo. This is
# only for checking and differentiates from None.
class SingletonConstant:
    pass


@dataclass(frozen=True, slots=True)
class _DispatchCacheEntryOutputInfo:
    """
    Entry type for the FakeTensor dispatch cache for an output. Accounts for three
    possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a
       given index.
    2) We need to synthesize a new FakeTensor given tensor metadata. For view
       ops, we further capture the index of the arg to alias.
    3) if the tensor related fields are None, then it is a constant value (e.g.
    None or integer)
    """

    inplace_idx: Optional[int]
    metadata: Optional[TensorMetadata]
    view_idx: Optional[int]
    constant_value: Optional[Any] = SingletonConstant


@dataclass(frozen=True, slots=True)
class _DispatchCacheValidEntry:
    """
    Entry type for the FakeTensor dispatch cache. It supports two types of outputs
    1) tensor
    2) tuple of tensors

    is_output_tuple flag helps in differentiating the return type
    """

    output_infos: tuple[_DispatchCacheEntryOutputInfo]
    is_output_tuple: bool = False


@dataclass(frozen=True, slots=True)
class _DispatchCacheBypassEntry:
    """
    Entry type for a negative cache entry.
    """

    reason: str


if TYPE_CHECKING:
    _DispatchCacheEntry = Union[_DispatchCacheValidEntry, _DispatchCacheBypassEntry]


@dataclass(frozen=True, slots=True)
class _BypassDispatchCache(Exception):
    """
    Signals cases that should skip FakeTensor caching.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class DispatchCacheInfo:
    """
    Information about the state of the FakeTensor dispatch cache.
    """

    hits: int
    misses: int
    bypasses: dict[str, int]
    size: int


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.


class FakeTensorMode(TorchDispatchMode):
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: dict[str, int] = defaultdict(int)
    # Every time you retrace using the same fake tensor mode, you should
    # advance the epoch so we don't reuse unbacked memos
    epoch: int = 0
    in_kernel_invocation: bool = False
    static_shapes: bool
    shape_env: Optional[ShapeEnv]
    _stack: Optional[str]
    allow_meta: bool

    # NestedTensor uses a tensor_id_counter to uniquely identify offsets.
    # This counter is incremented when an offsets is used to create an NJT
    # for the first time. To avoid mutating eager state if we construct NJT
    # during tracing, we maintain a separate counter on the FakeTensorMode.
    # The initial count is set to the current eager tensor_id_counter value
    # upon initialization, and every time you retrace using the same fake tensor
    # mode, you should reset the counter to the initial count.
    nt_tensor_id_counter: int = -1
    nt_tensor_id_initial_count: int = -1

    def __init__(
        self,
        *,
        allow_fallback_kernels: bool = True,
        allow_non_fake_inputs: bool = False,
        shape_env: Optional[ShapeEnv] = None,
        static_shapes: Optional[bool] = None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        export: bool = False,
    ) -> None:
        log.debug("create_mode 0x%x", id(self))
        super().__init__()
        self.allow_fallback_kernels = allow_fallback_kernels

        import torch._dynamo.config
        import torch._functorch.config

        self.propagate_real_tensors = (
            torch._functorch.config.fake_tensor_propagate_real_tensors
        )
        self.fake_tensor_converter = FakeTensorConverter(
            copy_data=self.propagate_real_tensors,
            export=export,
        )

        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None

        # This is temporarily patched to True in Dynamo to grandfather in some
        # places where we unconditionally allow scalar outputs, TO BE REMOVED
        self.allow_scalar_outputs = False

        self._allow_unsafe_data_ptr_access = (
            torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access
        )
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        self.cache_enabled: bool = (
            torch._dynamo.config.fake_tensor_cache_enabled
            and not self.propagate_real_tensors
        )
        self.cache_crosscheck_enabled = (
            torch._dynamo.config.fake_tensor_cache_crosscheck_enabled
        )

        # A flag that controls, whether we want to invoke ops on mix of
        # real weights/global variables and fake inputs
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # when FakeTensor is invoked in user code, .device should return
        # the fake_device of the tensor so that code such as as `if x.is_cuda`
        # or torch.zeros([10, 10], device=x.device) continues to execute as if
        # the FakeTensor were real. However, within kernel execution, we return
        # the `Meta` device because all computation within the kernels should
        # behave as if the Tensors are on meta devices. Kernels should allocate
        # new tensors on meta devices, and checks like `is_meta` should return true.
        # within python refs, we always return the real device by defining
        # the device property
        self.in_kernel_invocation = False

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        # If another fake mode was already active when we enter, we also stash it here.
        # That way when we exit, we know to re-enable the previous fake mode.
        self.enter_stack: list[
            tuple[bool, Optional[TorchDispatchMode], Optional[bool]]
        ] = []

        self.shape_env = shape_env

        self._stack_trace = traceback.extract_stack()
        self._stack = None

        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE

        import torch.nested._internal.nested_tensor

        self.nt_tensor_id_initial_count = (
            torch.nested._internal.nested_tensor._tensor_id_counter
        )
        self.nt_tensor_id_counter = self.nt_tensor_id_initial_count

    def reset_nt_tensor_id_counter(self) -> None:
        self.nt_tensor_id_counter = self.nt_tensor_id_initial_count

    # Typically, there is only one fake tensor mode and you test for it by
    # doing an isinstance test.  However, in some situations, there might be
    # TWO fake tensor modes.  The canonical example of this is exporting
    # a fake model: there is an outer fake mode created by the user, and
    # an inner fake mode created by Dynamo.  The two phase process is required
    # because the outer fake mode typically won't have a ShapeEnv, even if
    # the user is interested in exporting with dynamic shapes (so the inner
    # fake mode will actually have a ShapeEnv and swap in symbolic sizes.)
    #
    # In this case, it's insufficient to test only one FakeTensor: you need
    # to distinguish between our fake tensor and other fake tensors.  That's
    # what this function does.
    def is_our_fake(self, t: object) -> TypeGuard[FakeTensor]:
        return isinstance(t, FakeTensor) and t.fake_mode is self

    # If we should avoid device init. This changes the behavior of various APIs:
    # - We avoid constant-prop on Tensors with ops that move them to another device
    # - We change the torch.tensor ctor contract to never materialize
    #   tensors on device
    #   (see NOTE: [torch.tensor, lift_fresh, and device movement])
    @property
    def avoid_device_init(self) -> bool:
        if torch.xpu._is_compiled():
            assert not torch.cuda._is_compiled()
            return not torch.xpu.is_available()

        return not (
            torch.cuda.is_available()
            or (hasattr(torch, "hpu") and torch.hpu.is_available())
        )

    @property
    def stack(self) -> str:
        if self._stack is None:
            self._stack = "".join(traceback.format_list(self._stack_trace))
        return self._stack

    @count
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        # FakeTensorMode should not be set when we're inside of it.
        assert (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None
        ), func
        try:
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception("fake tensor raised TypeError")
            raise

    # No-op if FakeTensorMode is already in use
    def __enter__(self) -> Self:
        import torch.nested._internal.nested_tensor

        prev_only_lift_cpu_tensors = None
        if self.avoid_device_init:
            # See NOTE: [torch.tensor, lift_fresh, and device movement]
            prev_only_lift_cpu_tensors = torch._C._only_lift_cpu_tensors()
            torch._C._set_only_lift_cpu_tensors(True)

            # In the case of CPU-only build or cuda device unavailable,
            # we patch the cuda device guard to use NoOpDeviceGuardImpl.
            # This enables us to trace over cuda kernels under FakeTensorMode.
            torch._C._ensureCUDADeviceGuardSet()

        maybe_prev_fake_mode = torch._C._unset_dispatch_mode(self._mode_key)
        if self is not maybe_prev_fake_mode:
            self.enter_stack.append(
                (True, maybe_prev_fake_mode, prev_only_lift_cpu_tensors)
            )
            return super().__enter__()
        else:
            # no-op (still need to re-set the fake mode though since we unset it)
            torch._C._set_dispatch_mode(self)
            self.enter_stack.append((False, None, prev_only_lift_cpu_tensors))

        return self

    def __exit__(
        self,
        a: Optional[type[BaseException]],
        b: Optional[BaseException],
        c: Optional[TracebackType],
    ) -> None:
        (
            live,
            maybe_prev_fake_mode,
            maybe_prev_only_lift_cpu_tensors,
        ) = self.enter_stack.pop()
        if live:
            super().__exit__(a, b, c)

            # Re-enable the previous fake mode, if there was one.
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)
            if maybe_prev_only_lift_cpu_tensors is not None:
                torch._C._set_only_lift_cpu_tensors(maybe_prev_only_lift_cpu_tensors)

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True

    @classmethod
    def cache_info(cls) -> DispatchCacheInfo:
        """
        Query the state of the dispatch cache.
        """
        return DispatchCacheInfo(
            FakeTensorMode.cache_hits,
            FakeTensorMode.cache_misses,
            dict(FakeTensorMode.cache_bypasses),
            len(FakeTensorMode.cache),
        )

    @classmethod
    def cache_clear(cls) -> None:
        """
        Clear the dispatch cache.
        """
        cls.cache_hits = 0
        cls.cache_misses = 0
        cls.cache_bypasses.clear()
        cls.cache.clear()

    def _cached_dispatch_impl(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> object:
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
        state = None
        key = None
        try:
            state = _CacheKeyState(self.shape_env)
            key = self._cache_key(state, func, args, kwargs)
        except _BypassDispatchCache as e:
            # We couldn't create the cache key at all
            if (
                isinstance(func, torch._ops.HigherOrderOperator)
                and func.name() == "invoke_subgraph"
            ):
                hc_log.debug(
                    "Fake tensor cache failed: identifier = %s, reason = %s",
                    args[1],
                    e.reason,
                )
            FakeTensorMode.cache_bypasses[e.reason] += 1

        if key is None:
            # Do this dispatch outside the above except handler so if it
            # generates its own exception there won't be a __context__ caused by
            # the caching mechanism.
            # pyrefly: ignore [bad-argument-type]
            return self._dispatch_impl(func, types, args, kwargs)

        assert state is not None
        if state.cache_on_shape_env():
            assert state.shape_env is not None
            cache = state.shape_env.fake_tensor_cache
            set_cache_key = _set_cache_key_for_shape_env
        else:
            cache = FakeTensorMode.cache
            set_cache_key = _set_cache_key
        entry = cache.get(key, None)

        if entry is not None:
            if isinstance(entry, _DispatchCacheBypassEntry):
                # This represents a negative cache entry - we already saw that the
                # output is uncachable. Compute it from first principals.
                FakeTensorMode.cache_bypasses[entry.reason] += 1
                # pyrefly: ignore [bad-argument-type]
                return self._dispatch_impl(func, types, args, kwargs)

            # We have a cache entry.
            # pyrefly: ignore [bad-argument-type]
            output = self._output_from_cache_entry(state, entry, key, func, args)
            FakeTensorMode.cache_hits += 1
            if self.cache_crosscheck_enabled:
                # For debugging / testing: Validate that the output synthesized
                # from the cache matches the output created by normal dispatch.
                with disable_fake_tensor_cache(self):
                    # pyrefly: ignore [bad-argument-type]
                    self._crosscheck_cache_output(output, func, types, args, kwargs)
            return output

        # We don't have a cache entry.
        # pyrefly: ignore [bad-argument-type]
        output = self._dispatch_impl(func, types, args, kwargs)

        try:
            # pyrefly: ignore [bad-argument-type]
            entry = self._make_cache_entry(state, key, func, args, kwargs, output)
        except _BypassDispatchCache as e:
            # We ran "extra" checks on the cache key and determined that it's no
            # good. Record the reason and mark it so we don't bother validating
            # again.
            if (
                isinstance(func, torch._ops.HigherOrderOperator)
                and func.name() == "invoke_subgraph"
            ):
                hc_log.debug(
                    "Fake tensor cache failed: identifier = %s, reason = %s",
                    args[1],
                    e.reason,
                )
            FakeTensorMode.cache_bypasses[e.reason] += 1
            set_cache_key(cache, key, _DispatchCacheBypassEntry(e.reason))
            return output

        set_cache_key(cache, key, entry)
        FakeTensorMode.cache_misses += 1
        return output

    def _cache_key(
        self,
        state: _CacheKeyState,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> _DispatchCacheKey:
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
        is_tracing = torch.fx.experimental.proxy_tensor.get_proxy_mode() is not None
        key_values = [
            func,
            # Capture the default_dtype mode since that can affect the output tensor,
            # e.g., when operating on constant float values.
            torch.get_default_dtype(),
            # Capture the current device to support, e.g., cache tensor creation,
            # where there isn't necessarily a tensor to take the device from.
            torch._C._get_default_device(),
            # We want to create tensors from cached metadata only when the inference
            # mode is the same.
            torch.is_inference_mode_enabled(),
            # Shape env settings could affect behavior. One example seen in the wild:
            # Disallowing dynamic shapes can introduce a DynamicOutputShapeException
            # where it wasn't seen on a previous instance of the same op.
            self.shape_env.settings if self.shape_env else None,
            # ProxyTorchDispatchMode needs to track how SymNodes are constructed
            # so we need to handle things a little different depending on
            # whether we're tracing or not.
            is_tracing,
        ]
        if state.known_symbols:
            # If there are symbols then include the epoch - this is really more
            # of a Shape env var which lives on the FakeTensorMode.
            # pyrefly: ignore [bad-argument-type]
            key_values.append(self.epoch)
        # Collect the id_hashed objects to attach a weakref finalize later
        id_hashed_objects: list[object] = []
        # Translate any FakeTensor args to metadata.
        if args:
            # pyrefly: ignore [bad-argument-type]
            self._prep_args_for_hash(key_values, args, state, id_hashed_objects)
        if kwargs:
            # pyrefly: ignore [bad-argument-type]
            self._prep_args_for_hash(key_values, kwargs, state, id_hashed_objects)
        key = _DispatchCacheKey(tuple(key_values))

        for id_hashed_obj in id_hashed_objects:
            weakref.finalize(
                id_hashed_obj, functools.partial(evict_fake_tensor_cache_key, key=key)
            )
        id_hashed_objects.clear()
        return key

    def _validate_cache_key(
        self,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
        from torch._higher_order_ops.utils import registered_hop_fake_fns

        # For hops, we perform the validity check in _make_cache_entry  because we
        # need to have the output tensor.
        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            return

        # Avoid caching for any ops that would require a more sophisticated
        # caching implementation, e.g., data dependent ops or ops that modify
        # the inputs.
        if torch.Tag.data_dependent_output in func.tags:
            raise _BypassDispatchCache("data dependent output")

        if torch.Tag.dynamic_output_shape in func.tags:
            if func is aten.index.Tensor:
                _, new_kwargs = normalize_function(  # type: ignore[misc]
                    func,
                    args=args,  # type: ignore[arg-type]
                    kwargs=kwargs,  # type: ignore[arg-type]
                    normalize_to_only_use_kwargs=True,
                )
                for index in new_kwargs["indices"]:
                    # index calls nonzero for bool or int8 tensors, and
                    # therefore has a dynamic shape output. For other dtypes,
                    # the output shape depends on the input shape (and not data)
                    if isinstance(index, torch.Tensor) and index.dtype in (
                        torch.bool,
                        torch.int8,
                    ):
                        raise _BypassDispatchCache("dynamic output shape")
                return

            raise _BypassDispatchCache("dynamic output shape")

        if torch.Tag.inplace_view in func.tags:
            raise _BypassDispatchCache("inplace view")

        if func is aten._unsafe_view.default:
            raise _BypassDispatchCache("unsafe view")

        if func in self.lift_fns:
            raise _BypassDispatchCache("lift")

        if func.name() == "inductor::resize_storage_bytes_":
            raise _BypassDispatchCache("inductor::resize_storage_bytes_")

        if not torch._library.utils.is_builtin(func):
            raise _BypassDispatchCache("non-builtin")

        # In order to handle storage aliasing, we need to establish the alias
        # for any view op on a cache hit. But CompositeImplicitAutograd ops may
        # or may not alias the input, so just punt on caching these.
        if func.is_view and torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            raise _BypassDispatchCache("CompositeImplicitAutograd")

    def _prep_args_for_hash(
        self,
        result: list[object],
        args: Union[Mapping[str, object], Sequence[object], Iterable[object]],
        state: _CacheKeyState,
        id_hashed_objects: list[object],
    ) -> None:
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
        from torch._higher_order_ops.auto_functionalize import (
            FunctionalCallableWithEpilogue,
        )
        from torch._higher_order_ops.utils import FunctionalizeCtxWrapper

        if isinstance(args, (list, tuple, dict)):
            result.append(type(args))
            result.append(f"length_{len(args)}")

        if isinstance(args, dict):
            self._prep_args_for_hash(result, args.keys(), state, id_hashed_objects)
            self._prep_args_for_hash(result, args.values(), state, id_hashed_objects)
            return

        for arg in args:
            if isinstance(arg, FakeTensor):
                if not self.is_our_fake(arg):
                    raise _BypassDispatchCache("not our fake")
                if arg.constant is not None:
                    raise _BypassDispatchCache("constant attribute")
                if is_sparse_any(arg):
                    raise _BypassDispatchCache(f"{arg.layout} tensor")
                metadata = extract_tensor_metadata(arg)
                metadata._flatten_into(result, self, state)
            elif isinstance(arg, Tensor):
                raise _BypassDispatchCache("non-fake tensor")
            elif isinstance(arg, SymInt):
                state.convert_sym_int(result, arg)
            elif isinstance(arg, (SymBool, SymFloat)):
                raise _BypassDispatchCache("symbolic shape")
            elif isinstance(arg, (list, tuple, dict)):
                self._prep_args_for_hash(result, arg, state, id_hashed_objects)
            elif isinstance(arg, types.FunctionType):
                raise _BypassDispatchCache("function argument")
            elif isinstance(arg, torch.fx.GraphModule):
                # This is used for invoke_subgraph where id(graph_module) allows
                # us to cache fake outputs
                result.append(type(arg))
                result.append(id(arg))
                id_hashed_objects.append(arg)
            elif isinstance(arg, FunctionalizeCtxWrapper):
                # Special case for AOT Dispatcher first pass, where the fake
                # tensor is called on the functional wrapper of the subgraph.
                result.append(hash(arg))
                # functional wrapper is destroyed after fake tensor prop. We
                # need to put the finalizer on the subgraph.
                id_hashed_objects.append(arg.subgraph)
            elif isinstance(arg, FunctionalCallableWithEpilogue):
                result.append(type(arg))
                result.append(hash(arg))
                id_hashed_objects.append(arg.orig_callable)
            else:
                # It's important to capture the type of the arg since, e.g., 1 and 1.0
                # hash to the same value, but can produce different dtypes for the
                # output tensor.
                result.append(type(arg))
                result.append(arg)

    def _validate_output_for_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: Optional[FakeTensor],
    ) -> None:
        # Is this even possible? According to the signature this can be None but
        # not `int`. So either the signature is a lie or (part of) this line is
        # unnecessary...
        if isinstance(output, (int, type(None))):
            return

        # Check for symbolic content that should bypass caching - raises
        # _BypassDispatchCache if necessary.
        _validate_symbolic_output_for_caching(state, output)

        # Some ops return tuples of Tensors, but it's rare, so avoid
        # the complexity of caching other types.
        if not isinstance(output, FakeTensor):
            raise _BypassDispatchCache("non-FakeTensor output")

        # Avoid caching FakeTensors with constants attached since those
        # can be invalidated.
        if output.constant is not None:
            raise _BypassDispatchCache("constant attribute")

        # TODO: support caching sparse outputs?
        if output.is_sparse:
            raise _BypassDispatchCache("sparse output")

        if is_sparse_compressed(output):
            raise _BypassDispatchCache("sparse compressed output")

        # Can an in-place op really reference a kwarg? If so, then we need
        # to extend the implementation to handle it.
        for kval in kwargs.values():
            if id(kval) == id(output):
                raise _BypassDispatchCache("kwarg aliases output")

    def _get_output_info_for_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: FakeTensor,
    ) -> _DispatchCacheEntryOutputInfo:
        if isinstance(output, (int, torch.SymInt, type(None))):
            return _DispatchCacheEntryOutputInfo(
                inplace_idx=None, metadata=None, view_idx=None, constant_value=output
            )

        # If this is an in-place op, the entry records which input arg is aliased.
        for idx in range(len(args)):
            if id(args[idx]) == id(output):
                return _DispatchCacheEntryOutputInfo(
                    inplace_idx=idx, metadata=None, view_idx=None
                )

        # Otherwise, create an entry that records the output tensor's metadata.
        view_idx = None
        if isinstance(func, torch._ops.OpOverload) and func.is_view:
            idxs = [i for i, t in enumerate(args) if isinstance(t, Tensor)]
            assert len(idxs) == 1
            view_idx = idxs[0]

        metadata = extract_tensor_metadata(output)
        metadata.shape = tuple(state.convert_output(v) for v in metadata.shape)
        metadata.stride = tuple(state.convert_output(v) for v in metadata.stride)
        metadata.storage_offset = state.convert_output(metadata.storage_offset)
        metadata.storage_bytes = (
            None
            if metadata.storage_bytes is None
            else state.convert_output(metadata.storage_bytes)
        )

        entry = _DispatchCacheEntryOutputInfo(
            inplace_idx=None,
            metadata=metadata,
            view_idx=view_idx,
        )

        # N.B.: Some checks for bypassing the cache would be performed on the
        # output tensor synthesized from the cached metadata. As an optimization,
        # we can synthesize a tensor here and do the checks on that instance.
        # This approach keeps the (more frequent) cache-hit path as lightweight
        # as possible.
        entry_for_synth_output = _DispatchCacheValidEntry(
            output_infos=(entry,), is_output_tuple=False
        )
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        try:
            synth_output = self._output_from_cache_entry(
                state, entry_for_synth_output, key, func, args
            )
        except GuardOnDataDependentSymNode:
            # This should probably never really happen. If it does it means that
            # although the original call didn't get a data-dependent error when
            # we tried to reconstruct the output we did - that's almost
            # certainly a bug.
            raise _BypassDispatchCache("data dependent symnode") from None

        # Make sure the dispatch_key_set from the synthesized output tensor will
        # be the same.
        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

        return entry

    def _make_cache_entry(
        self,
        state: _CacheKeyState,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        output: Optional[FakeTensor],
    ) -> _DispatchCacheValidEntry:
        """
        Make a cache entry object for the given 'output' Tensor. Raises
        _BypassDispatchCache if the output tensor has characteristics that
        prevent caching it.
        """
        from torch._higher_order_ops.utils import registered_hop_fake_fns
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        self._validate_cache_key(func, args, kwargs)

        # For hops, lets look at the output tensor to find any unbacked symints.
        # If there are none, then we rely on the existing checks to validate
        # caching.
        # NB: Note that the HOPs that sta alive till FakeTensor are functional,
        # once they support mutations, we will have to revisit this logic.
        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            assert isinstance(output, tuple)
            non_cacheable = any(
                isinstance(o, (torch.Tensor, torch.SymInt))
                and has_free_unbacked_symbols(o)
                for o in output
            )
            if non_cacheable:
                raise _BypassDispatchCache(f"unbacked symbol in HOP {func} output")

        if isinstance(output, (int, torch.SymInt, type(None))):
            output_info = _DispatchCacheEntryOutputInfo(
                inplace_idx=None, metadata=None, view_idx=None, constant_value=output
            )
            return _DispatchCacheValidEntry(
                output_infos=(output_info,), is_output_tuple=False
            )

        if isinstance(output, tuple):
            for out_element in output:
                self._validate_output_for_cache_entry(
                    state,
                    key,
                    # pyrefly: ignore [bad-argument-type]
                    func,
                    args,
                    kwargs,
                    out_element,
                )
        else:
            self._validate_output_for_cache_entry(
                state,
                key,
                # pyrefly: ignore [bad-argument-type]
                func,
                args,
                kwargs,
                output,
            )

        if isinstance(output, tuple):
            output_infos = [
                self._get_output_info_for_cache_entry(
                    state,
                    key,
                    # pyrefly: ignore [bad-argument-type]
                    func,
                    args,
                    kwargs,
                    out_elem,
                )
                for out_elem in output
            ]
            return _DispatchCacheValidEntry(
                # pyrefly: ignore [bad-argument-type]
                output_infos=tuple(output_infos),
                is_output_tuple=True,
            )

        else:
            output_info = self._get_output_info_for_cache_entry(
                state,
                key,
                # pyrefly: ignore [bad-argument-type]
                func,
                args,
                kwargs,
                output,
            )
            return _DispatchCacheValidEntry(
                output_infos=(output_info,), is_output_tuple=False
            )

    def _get_output_tensor_from_cache_entry(
        self,
        state: _CacheKeyState,
        entry: _DispatchCacheEntryOutputInfo,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
    ) -> Optional[FakeTensor]:
        if (
            entry.inplace_idx is None
            and entry.metadata is None
            and entry.view_idx is None
        ):
            assert entry.constant_value is not SingletonConstant
            return entry.constant_value
        if entry.inplace_idx is not None:
            # This is an in-place op; return the aliased arg.
            inplace_arg = args[entry.inplace_idx]
            assert isinstance(inplace_arg, FakeTensor)
            return inplace_arg

        # Synthesize a new FakeTensor with the cached metadata.
        metadata = entry.metadata
        if metadata is None:
            return None

        assert not is_sparse_any(metadata)

        def check_value(
            value: _MetadataIntLike, state: _CacheKeyState
        ) -> Union[IntLikeType]:
            if isinstance(value, _SymIntOutputStub):
                assert state.shape_env is not None
                return value.extract(key, state.shape_env)
            else:
                assert not isinstance(value, _PySymInputStub)
                return value

        shape = tuple(check_value(v, state) for v in metadata.shape)
        stride = tuple(check_value(v, state) for v in metadata.stride)
        storage_offset = check_value(metadata.storage_offset, state)
        if metadata.storage_bytes is not None:
            check_value(metadata.storage_bytes, state)

        maybe_suppress: Callable[[], typing.ContextManager] = contextlib.nullcontext
        if self.shape_env is not None:
            maybe_suppress = self.shape_env.suppress_guards

        with in_kernel_invocation_manager(self), maybe_suppress():
            empty = torch.empty_strided(
                shape,
                stride,
                dtype=metadata.dtype,
                layout=metadata.layout,
                device="meta",
                requires_grad=metadata.requires_grad,
            )

        if metadata.is_conj:
            torch._C._set_conj(empty, True)
        if metadata.is_neg:
            torch._C._set_neg(empty, True)

        if isinstance(func, torch._ops.OpOverload) and func.is_view:
            # For view ops, the storage should be the same as the tensor input.
            view_arg = args[cast(int, entry.view_idx)]
            assert isinstance(view_arg, FakeTensor)
            storage = view_arg.untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(storage, storage_offset, shape, stride)

        return FakeTensor(self, empty, metadata.device)

    def _output_from_cache_entry(
        self,
        state: _CacheKeyState,
        entry: _DispatchCacheValidEntry,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Sequence[object],
    ) -> Union[Optional[FakeTensor], tuple[Optional[FakeTensor], ...]]:
        """
        Create a new FakeTensor from the cache entry.
        """

        if entry.is_output_tuple:
            outputs = [
                self._get_output_tensor_from_cache_entry(
                    state, output_info, key, func, args
                )
                for output_info in entry.output_infos
            ]
            return tuple(outputs)
        else:
            return self._get_output_tensor_from_cache_entry(
                state, entry.output_infos[0], key, func, args
            )

    def _crosscheck_cache_output(
        self,
        output: Union[Optional[FakeTensor], tuple[Optional[FakeTensor], ...]],
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """

        def assert_helper(a: Any, b: Any) -> None:
            if isinstance(a, tuple):
                assert isinstance(b, tuple)
                assert len(a) == len(b)
                for l, r in zip(a, b):
                    assert_helper(l, r)
            elif isinstance(a, int):
                assert isinstance(b, int) and a == b
            elif a is None:
                assert b is None
            elif isinstance(a, py_sym_types):
                assert type(a) is type(b) and a.node is b.node
            elif isinstance(a, torch.Tensor):
                assert isinstance(b, torch.Tensor)
                assert_metadata_eq(assert_eq, a, b)
            else:
                raise RuntimeError(f"Unsupported type {type(a)}")

        try:
            true_output = self._dispatch_impl(func, types, args, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}: Dispatch raised={e}"
            ) from e
        try:
            assert_helper(true_output, output)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}"
            ) from e

    def dispatch(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        kwargs = kwargs or {}
        with no_dispatch():
            log.debug("%s %s %s", func, args, kwargs)

        if func in _DISPATCH_META_HANDLERS:
            return _DISPATCH_META_HANDLERS[func](args)

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "%sFakeTensorMode.__torch_dispatch__: %s", " " * RECURSION_COUNT, func
            )
            # NOTE: incr is intentionally unused for a RAII pattern
            incr = IncrementRecursionCount()  # noqa: F841

        # Some attribute queries that can be serviced directly
        # See Note [is_coalesced is dispatched]
        if func in _DISPATCH_HANDLE_DIRECTLY:
            # NB: no_dispatch is ok here too, this func is very simple
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)

        if self.cache_enabled:
            return self._cached_dispatch_impl(func, types, args, kwargs)
        else:
            return self._dispatch_impl(func, types, args, kwargs)

    def _maybe_infer_fake(
        self, func: OpOverload, path: KeyPath, fake: object, real: object
    ) -> tuple[Optional[object], bool]:
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched.
        Returns tuple of object & boolean, for whether or not it was overwrriten
        """
        import sympy

        from torch._subclasses.fake_utils import _check_fake_real_tensors

        def _check_fake_real_vals(fake: Any, real: Any) -> None:
            # use real values + ShapeEnv to check mismatches between potentially symbolic values
            if isinstance(fake, (SymInt, SymFloat)):
                # symbolic expression, ask ShapeEnv to substitute known backed/unbacked values
                assert self.shape_env is not None
                if (
                    not fake.node.expr.free_symbols
                    - self.shape_env.var_to_val.keys()
                    - self.shape_env.unbacked_var_to_val.keys()
                ):
                    if (
                        self.shape_env._maybe_evaluate_static(
                            sympy.Eq(fake.node.expr, real), compute_hint=True
                        )
                        is not sympy.S.true
                    ):
                        raise MetadataMismatchError(
                            f"mismatch between fake value {fake} and real value {real} "
                        )
            elif isinstance(
                fake, (int, float, bool)
            ):  # concrete value, check direct equality
                if fake != real:
                    raise MetadataMismatchError(
                        f"mismatch between fake value {fake} and real value {real} "
                    )

        if isinstance(fake, torch.Tensor):
            try:
                _check_fake_real_tensors(
                    real,  # type: ignore[arg-type]
                    fake,  # type: ignore[arg-type]
                    context="Real tensor propagation found",
                    sizes=False,  # manual check below
                    strides=False,  # skip strides
                    storage_offset=True,
                    requires_grad=False,  # issues with FakeTensorConverter preserving requires_grad
                )
            except MetadataMismatchError as exc:
                if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                    dtrace_structured(
                        "mismatched_fake_kernel",
                        metadata_fn=lambda: {
                            "op": str(func),
                            "reason": exc.reason,  # noqa: F821
                        },
                    )
                    return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
                raise MetadataMismatchError(
                    f"Real tensor propagation found a metadata mismatch between "
                    f"fake tensor {fake} and real tensor {real}, "
                    f" at output{keystr(path)}, for func: {func}"
                ) from exc

            for j, (s_fake, s_real) in enumerate(zip(fake.size(), real.size())):  # type: ignore[attr-defined]
                try:
                    _check_fake_real_vals(s_fake, s_real)
                except MetadataMismatchError as exc:
                    if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                        dtrace_structured(
                            "mismatched_fake_kernel",
                            metadata_fn=lambda: {
                                "op": str(func),
                                "reason": exc.reason,  # noqa: F821
                            },
                        )
                        return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
                    raise MetadataMismatchError(
                        f"Real tensor propagation found an output size mismatch between "
                        f"fake shape {s_fake} and real shape {s_real}, "
                        f"at output{keystr(path)}.size({j}), for func: {func}"
                    ) from exc
        elif fake is None and real is not None:
            if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                dtrace_structured(
                    "mismatched_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                        "reason": f"mismatch between fake value {fake} and real value {real}",  # noqa: F821
                    },
                )
                return _infer_fake_from_real_tensor(self, func, real), True  # type: ignore[arg-type]
            raise MetadataMismatchError(
                f"Real tensor propagation found a metadata mismatch between "
                f"fake tensor {fake} and real tensor {real}, "
                f" at output{keystr(path)}, for func: {func}"
            )
        else:
            try:
                _check_fake_real_vals(fake, real)
            except MetadataMismatchError as exc:
                raise MetadataMismatchError(
                    f"Real tensor propagation found an output value mismatch between "
                    f"fake output value {fake} and real output value {real}, "
                    f"at output{keystr(path)}, for func: {func}"
                ) from exc
        return fake, False

    def _maybe_infer_fake_kernel_from_pytree_out(
        self,
        func: OpOverload,
        fake_in: object,
        real_in: object,
        fake_out: object,
        real_out: object,
    ) -> Optional[object]:
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched, but at the kernel level.
        Means this handles pytree outputs & checks aliasing.
        """
        from torch._subclasses.fake_utils import _check_alias_info

        # we might have to clear pending unbacked symbols, if we override the kernel
        pending_unbacked = None
        if self.shape_env:
            pending_unbacked = list(self.shape_env.pending_fresh_unbacked_symbols)

        def _clear_pending_unbacked() -> None:
            self.shape_env.pending_fresh_unbacked_symbols = list(  # type: ignore[union-attr]
                set(self.shape_env.pending_fresh_unbacked_symbols).difference(  # type: ignore[union-attr]
                    pending_unbacked  # type: ignore[arg-type]
                )
            )

        fake_paths_leaves, fake_spec = pytree.tree_flatten_with_path(fake_out)
        real_leaves, _ = pytree.tree_flatten(real_out)
        try:
            # catch aliasing mismatches between fake/real tensors
            _check_alias_info(
                "Real tensor propagation found", real_out, real_in, fake_out, fake_in
            )
        except MetadataMismatchError as exc:
            # if mismatch found, optionally infer fake kernel
            if torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                dtrace_structured(
                    "mismatched_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                        "reason": (
                            f"Mismatched aliasing spec between fake kernel and real kernel: {exc.reason}"  # noqa: F821
                        ),
                    },
                )
                # if aliasing mismatches are found, it's likely that the fake tensor impl
                # is incorrectly aliasing, since we don't support aliasing custom ops.
                # in this case we can default to inferring non-aliasing fake kernels from the real outputs.
                _clear_pending_unbacked()
                return tree_map(
                    lambda x: _infer_fake_from_real_tensor(self, func, x), real_out
                )
            else:
                raise MetadataMismatchError(
                    f"Real tensor propagation found an aliasing mismatch between "
                    f"fake output {fake_out} and real output {real_out}, "
                    f" for func: {func}"
                ) from exc

        # if no errors raised, run cross checks on fake/real tensors,
        # optionally overriding individual fake tensors, if individual meta kernel output is incorrect.
        fake_leaves, overrides = zip(
            *[
                self._maybe_infer_fake(func, _fake_path, _fake_out, _real_out)
                for (_fake_path, _fake_out), _real_out in zip(
                    fake_paths_leaves, real_leaves
                )
            ]
        )
        if (
            any(overrides) and pending_unbacked
        ):  # only keep new pending unbacked symbols
            _clear_pending_unbacked()
        return pytree.tree_unflatten(fake_leaves, fake_spec)

    def _dispatch_impl(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> Optional[FakeTensor]:
        from torch._higher_order_ops.utils import registered_hop_fake_fns

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # DO NOT PUT LOGIC BEFORE UNRECOGNIZED TYPE CHECKING
        # We must throw NotImplemented in case of unrecognized types to handle subclasses.
        # Throwing the exception will pass the control to the next __torch_dispatch__.
        # See [subclass inputs] below
        # NB: If you're seeing a mysterious infinite loop involving fake
        # tensor, it might be related to this line.  Though I'm not sure
        # how you'll know to read this comment, as this line won't show up
        # in the stack trace.
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "FakeTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        flat_arg_fake_tensors = [t for t in flat_args if self.is_our_fake(t)]
        has_symbolic_sizes = any(
            i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors
        ) or any(isinstance(a, SymInt) for a in flat_args)

        converter = self.fake_tensor_converter

        is_lift_func = func in self.lift_fns

        # If we are trying to avoid device init, then we need to avoid constant
        # prop on constant tensors for ops that change devices.
        avoiding_device_init = False
        if self.avoid_device_init:
            if (
                func is torch.ops.aten._to_copy.default
                and "device" in kwargs
                and kwargs["device"].type != "cpu"  # type: ignore[attr-defined]
            ):
                avoiding_device_init = True
            if func is torch.ops.prims.device_put.default:
                avoiding_device_init = True

        # skip const prop for aten._to_copy if
        # 1. input tensor is on "meta" device
        # 2. destination device is unavailable, captured by `avoiding_device_init`
        device_conversion_skip_const_prop = (
            func is torch.ops.aten._to_copy.default
            and isinstance(args[0], torch.Tensor)
            and args[0].device.type == "meta"
        ) or avoiding_device_init

        # To constant propagate through these functions:
        # 1, If this is a lift due to a torch.tensor call,
        #    the input tensor is guaranteed to be a
        #    constant, so we keep a copy of the original argument along so
        #    we can query it if we're asked to item() it at some later point.
        #    (Note that you can always call a lift fn manually, so we do
        #    have to check if there are any fake tensors!)
        # 2, Some functions that allow Python numbers to bind to Tensors, e.g, torch.div
        if (is_lift_func and not flat_arg_fake_tensors) or (
            should_allow_numbers_as_tensors(func)
            and not has_symbolic_sizes
            and not flat_arg_fake_tensors
            and not device_conversion_skip_const_prop
        ):
            assert all(t.constant is not None for t in flat_arg_fake_tensors), (
                f"{func} should not have fake inputs without constants"
            )
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            out = func(*const_args, **const_kwargs)
            if type(out) is Tensor and self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                # NB: no_dispatch() here is VERY DANGEROUS (like, segfault
                # dangerous) if this is actually a wrapper subclass tensor,
                # therefore the exact type test above
                with no_dispatch():
                    out = out.clone()
                return converter.from_real_tensor(self, out, make_constant=True)

        # if we are in the dispatch mode, we will enter this function even if the inputs
        # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
        # and just support constructors.

        # this is generated from torch.tensor(), which does not use the
        # dispatcher, to allow wrapper subclasses to wrap the new tensor
        if is_lift_func:
            assert len(kwargs) == 0 and len(args) == 1, f"{args} {kwargs}"

            if type(args[0]) is Tensor:
                return converter.from_real_tensor(self, args[0])

        # Recompute flat_arg_fake_tensors here again in case some of the inputs
        # were real tensors and fakified in validate_and_convert_non_fake_tensors
        (flat_args, flat_arg_fake_tensors) = self.validate_and_convert_non_fake_tensors(
            func, converter, flat_args, args_spec
        )
        del args, kwargs  # Invalidated

        # The current constant handling only support tracing systems
        # (aot autograd, torchdynamo) where each operation is run consecutively.
        # Because each operation is run in order, we can trace out and support
        # sequences like: x = torch.tensor(0.); y = x.add_(1)
        # Whenever a constant is written to but with inputs that cannot be evaluated
        # statically, such as random_(), we invalidate all constants that alias the input
        # We will rely on functionalization for use of fake tensors constants as persistent
        # objects on an FX Graph.

        # We dispatch size/stride/numel on the FakeTensor not its constant, so bail on inplace_view
        all_constant = all(e.constant is not None for e in flat_arg_fake_tensors)
        if (
            isinstance(func, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded not in func.tags
            and torch.Tag.inplace_view not in func.tags
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
            and not avoiding_device_init
            and func is not aten._nested_tensor_from_tensor_list.default
        ):
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)

            # NB: not in_kernel_invocation_manager(self) as we want to do REAL
            # compute
            with no_dispatch():
                out = func(*const_args, **const_kwargs)

            flat_out = pytree.tree_leaves(out)
            flat_out_tensors = [t for t in flat_out if isinstance(t, Tensor)]
            all_constant = all(self.may_turn_const(t) for t in flat_out_tensors)

            if all_constant:
                return pytree.tree_map_only(
                    Tensor,
                    lambda t: converter.from_real_tensor(self, t, make_constant=True),
                    out,
                )

            # we weren't able to turn outputs to constants,
            # so invalidate all constants that might be aliases of the outputs
            for ten in flat_out_tensors:
                converter.invalidate_constant_aliases(ten)

        # we are falling through to running non constant tensors, any input constant that
        # is written to must be invalidated
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        if (
            isinstance(func, torch._ops.HigherOrderOperator)
            and func in registered_hop_fake_fns
        ):
            # Reenable the fake tensor mode for the registered fake function
            maybe_ignore_fresh_unbacked_symbols = (
                contextlib.nullcontext
                if self.shape_env is None
                else self.shape_env.ignore_fresh_unbacked_symbols
            )

            with self, maybe_ignore_fresh_unbacked_symbols():
                # pyrefly: ignore [index-error]
                return registered_hop_fake_fns[func](*args, **kwargs)

        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)

        def maybe_to_real_tensor(
            t: T,
        ) -> Optional[Union[T, Tensor, torch._C.ScriptObject]]:
            if isinstance(t, FakeTensor):
                return t.real_tensor
            elif isinstance(t, py_sym_types):
                assert self.shape_env is not None
                return t.node.pytype(
                    t.node.expr.xreplace(self.shape_env.var_to_val).xreplace(
                        self.shape_env.unbacked_var_to_val
                    )
                )
            elif isinstance(t, FakeScriptObject):
                return t.real_obj
            else:
                return t

        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            free_unbacked_symbols,
        )

        nil = object()

        real_out = nil
        if (
            self.propagate_real_tensors
            and all(e.real_tensor is not None for e in flat_arg_fake_tensors)
            and not any(
                (
                    isinstance(a, py_sym_types)
                    and (syms := free_unbacked_symbols(a))
                    and self.shape_env is not None
                    and any(s not in self.shape_env.unbacked_var_to_val for s in syms)
                )
                for a in flat_args
            )
        ):
            log.debug("propagate_real_tensors %s", func)
            real_flat_args = [maybe_to_real_tensor(a) for a in flat_args]
            real_args, real_kwargs = pytree.tree_unflatten(real_flat_args, args_spec)

            is_builtin = library_utils.is_builtin(func)
            if not is_builtin:
                mutation_checker = library_utils.MutationChecker(
                    func, real_flat_args, args_spec
                )

            try:
                real_out = func(*real_args, **real_kwargs)
            except ZeroDivisionError as exc:
                # we shouldn't broadly catch all errors here;
                # some come from real-kernel mutation/aliasing checks we want to run.
                # add more exception types as needed.
                log.debug(  # noqa: G200
                    "real-tensor fallback failed for %s: %s; silently ignoring",
                    func,
                    exc,
                )

            if not is_builtin:
                mutation_checker.check()  # type: ignore[possibly-undefined]
                library_utils.check_aliasing_constraint(func._name, flat_args, real_out)

        elif self.propagate_real_tensors:
            # This can happen occasionally legitimately, specifically when you
            # are inside the meta of a data dependent operation and you create
            # a tensor on an unbacked SymInt; at this point in time we don't
            # know what the unbacked SymInt is, but we will know later.
            # However, if there's a bug in the condition above, this condition
            # will also trigger.
            log.debug(
                "SKIPPED propagate_real_tensors %s(%s, %s) %s",
                func,
                flat_arg_fake_tensors,
                flat_args,
                self.shape_env.unbacked_var_to_val if self.shape_env else None,
            )

        def maybe_propagate_real_tensors(fake_out: T) -> T:
            import sympy

            log.debug("maybe_propagate_real_tensors %s", func)

            def go(t: object, real_t: Tensor) -> None:
                if isinstance(t, FakeTensor):
                    # NB: unconditionally overwrite
                    log.debug(
                        "maybe_propagate_real_tensors %s -> %s", id(t), id(real_t)
                    )
                    t.real_tensor = real_t
                    for s, real_s in zip(t.size(), real_t.size()):
                        go(s, real_s)  # type: ignore[arg-type]
                    for s, real_s in zip(t.stride(), real_t.stride()):
                        go(s, real_s)  # type: ignore[arg-type]
                    go(t.storage_offset(), real_t.storage_offset())  # type: ignore[arg-type]
                elif isinstance(t, py_sym_types) and free_unbacked_symbols(t):
                    if isinstance(t.node.expr, sympy.Symbol):
                        assert self.shape_env is not None
                        self.shape_env.set_unbacked_var_to_val(t.node.expr, real_t)
                    elif (
                        isinstance(s := t.node.expr, sympy.Eq)
                        and isinstance(s.lhs, sympy.Symbol)
                        and s.rhs == 1
                    ):
                        assert self.shape_env is not None

                        self.shape_env.set_unbacked_var_to_val(s, int(real_t))

            if real_out is not nil:
                # cross check fake/real outputs, and optionally override fake kernel mismatches
                if not torch._functorch.config.generate_fake_kernels_from_real_mismatches:
                    self._maybe_infer_fake_kernel_from_pytree_out(
                        func,
                        (args, kwargs),
                        (real_args, real_kwargs),
                        fake_out,
                        real_out,
                    )
                else:
                    # this can override the output only when the flag is True
                    fake_out = self._maybe_infer_fake_kernel_from_pytree_out(  # type: ignore[assignment]
                        func,
                        (args, kwargs),
                        (real_args, real_kwargs),
                        fake_out,
                        real_out,
                    )

                # populate unbacked_var_to_val
                if (
                    not isinstance(fake_out, Tensor)
                    and not isinstance(real_out, Tensor)
                    and type(fake_out) is not type(real_out)
                ):
                    # This can happen when decompositions have different return types,
                    # e.g. namedtuple vs. tuple vs. list.
                    tree_map_(
                        go,
                        tuple(pytree.tree_flatten(fake_out)),
                        tuple(pytree.tree_flatten(real_out)),
                    )
                else:
                    tree_map_(go, fake_out, real_out)

                # If a data-dependent op is used in a decomposition, we
                # may need to get the unbacked settings "early"
                # TODO: Is this really needed?
                compute_unbacked_bindings(self.shape_env, fake_out, peek=True)

            # pyrefly: ignore [bad-return]
            return fake_out

        # Try for fastpath
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return maybe_propagate_real_tensors(fast_impl(self, *args, **kwargs))

        # If there's a Python meta, prefer that over the decomposition
        from torch._decomp import meta_table

        if (
            func not in meta_table
            and not self.cpp_meta_supports_symint(func)
            and not (
                has_symbolic_sizes and func in self._unbacked_special_fake_handling_ops
            )
        ):
            from torch._decomp import decomposition_table

            # Prefer Python decompositions over C++ ones
            if func in decomposition_table and (
                has_symbolic_sizes
                or (
                    # TODO: Remove these exclusions, so that we can remove
                    # this leg entirely
                    torch_decomp_decompositions(func)
                    and all(not is_sparse_any(e) for e in flat_arg_fake_tensors)
                )
            ):
                with self:
                    return maybe_propagate_real_tensors(
                        decomposition_table[func](*args, **kwargs)
                    )

            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return maybe_propagate_real_tensors(r)

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we dont need do anything but run them
        # and ensure that Meta kernels are dispatched to (see)
        # Fake Tensor Dispatch Keys
        # TODO - we should be use the prim aten impl
        # TODO - fix prims complex ops
        if (
            "prims::" in func._schema.name
            and hasattr(func, "prim_meta_impl")
            and not stride_incorrect_op(func)
        ):
            with self:
                return maybe_propagate_real_tensors(
                    func.prim_meta_impl(*args, **kwargs)
                )

        profiles = torch._dynamo.config._custom_ops_profile
        if profiles is not None:
            if func in profiles.data:
                return profiles.generic_fake_kernel(func, self, *args, **kwargs)

        if (
            self.propagate_real_tensors
            and real_out is not nil
            and not library_utils.is_builtin(func)
            and self.shape_env is not None
        ):
            # Automatically infer a Fake kernel if there isn't one.
            if not library_utils.has_fake_kernel(func):
                result = inferred_fake_kernel_from_real_out(self, func, real_out)

                dtrace_structured(
                    "missing_fake_kernel",
                    metadata_fn=lambda: {
                        "op": str(func),
                    },
                )
                return maybe_propagate_real_tensors(result)

        # Users can register FakeTensor rules for custom operators
        # Call them if they exist.
        maybe_fake_impl = torch._library.simple_registry.singleton.find(
            func.name()
        ).fake_impl.kernel
        if maybe_fake_impl:
            try:
                ctx = torch._library.fake_impl.FakeImplCtx(self, func)
                with torch._library.fake_impl.set_ctx_getter(lambda: ctx), self:
                    result = maybe_fake_impl(*args, **kwargs)
                    return maybe_propagate_real_tensors(result)

            except MissingOpProfile as e:
                # If we have a fake kernel registered generated from OpProfiles
                # but there doesn't exist a profile for the existing inputs, and we are in
                if (
                    self.propagate_real_tensors
                    and real_out is not nil
                    and not library_utils.is_builtin(func)
                    and self.shape_env is not None
                ):
                    result = inferred_fake_kernel_from_real_out(self, func, real_out)

                    dtrace_structured(
                        "missing_fake_kernel",
                        metadata_fn=lambda: {
                            "op": str(func),
                        },
                    )
                    return maybe_propagate_real_tensors(result)
                else:
                    raise e

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations_checks:
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out is not NotImplemented:
                    return maybe_propagate_real_tensors(op_impl_out)

        def maybe_run_unsafe_fallback(
            error: Optional[RuntimeError] = None,
        ) -> Optional[FakeTensor]:
            # We infer the meta of a custom ops that return None to just
            # return None. custom ops are not allowed to mutate metadata
            # of their inputs, so this is safe.
            if torch._library.utils.can_generate_trivial_fake_impl(func):
                return None
            # no meta kernel registered, fallback to kernel for the device
            if has_symbolic_sizes or not self.can_run_unsafe_fallback(func):
                raise UnsupportedOperatorException(func)
            if error is None:
                error = UnsupportedOperatorException(func)
            return run_fallback_kernel(self, func, flat_args, args_spec, error)

        # Optimization: If there is no Meta kernel, it takes a surprisingly long
        # amount of time to catch the NotImplementedError, so we check it here.
        if not has_meta(func):
            fallback = maybe_run_unsafe_fallback()
            return maybe_propagate_real_tensors(fallback)

        # run kernel registered to meta for func, which include
        # python meta registrations, prims, decomps, and c++ meta fns (structured kernels)
        # It's possible that the kernel will return NotImplementedError
        try:
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
        except NotImplementedError as not_implemented_error:
            return maybe_run_unsafe_fallback(not_implemented_error)
        except Exception:
            log.exception("failed while attempting to run meta for %s", func)
            raise

        return maybe_propagate_real_tensors(
            self.wrap_meta_outputs_with_default_device_logic(
                r, func, flat_args, device=kwargs.get("device")
            )
        )

    # WARNING: DO NOT add any additional namespaces/operators here if they refer to operators
    # outside of the pytorch/pytorch library! Any pre-existing things here
    # are either in the pytorch/pytorch library or have been grandfathered in.
    # The fallback does not always work and MAY CRASH and emit unreadable error messages
    # so it should not be allowed by default.
    _can_run_unsafe_fallback_allowed_namespaces = ordered_set(
        "debugprims",
        "prims",
        "aten",
        "xla",
        "vision",
        "torchtext",
        "torchaudio",
        "quantized",
    )

    def can_run_unsafe_fallback(self, func: OpOverload) -> bool:
        if not self.allow_fallback_kernels:
            return False
        # It's OK to try the fallback for built-in ops (e.g. aten, prims)
        # because we control and test these but the fallback leads to unexpected behavior
        # in user-defined custom ops
        return (
            func.namespace in self._can_run_unsafe_fallback_allowed_namespaces
            or func.name() == "fbgemm::gmm"
        )

    def validate_and_convert_non_fake_tensors(
        self,
        func: OpOverload,
        converter: FakeTensorConverter,
        flat_args: Sequence[object],
        args_spec: TreeSpec,
    ) -> tuple[list[object], list[FakeTensor]]:
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors: list[FakeTensor] = []

        def validate(x: T) -> Union[T, FakeTensor]:
            if not isinstance(x, Tensor):
                return x

            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if hasattr(func, "tags") and torch.Tag.inplace_view in func.tags:
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                allow_non_fake_inputs = (
                    self.allow_non_fake_inputs
                    if fake_tensor_tls.allow_non_fake_inputs_override is None
                    else fake_tensor_tls.allow_non_fake_inputs_override
                )
                if not allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError("Mixing fake modes NYI")
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                out = converter.from_real_tensor(self, x)
            else:
                out = x

            flat_arg_fake_tensors.append(out)
            return out

        validated_args = [validate(a) for a in flat_args]
        return validated_args, flat_arg_fake_tensors

    def wrap_meta_outputs_with_default_device_logic(
        self,
        r: object,
        func: OpOverload,
        flat_args: Sequence[object],
        device: torch.device,
    ) -> PyTree:
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None
        has_scalar_only_inputs = False

        def wrap(e: T) -> Union[T, FakeTensor]:
            nonlocal common_device
            nonlocal has_scalar_only_inputs

            if not isinstance(e, Tensor):
                return e

            if common_device is None:
                (
                    common_device,
                    has_scalar_only_inputs,
                ) = FakeTensor._find_common_device(func, flat_args)

            is_our_fake = self.is_our_fake(e)
            if is_our_fake:
                torch._check(
                    e.device == common_device,
                    lambda: f"FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}",
                )
                return cast(T, e)
            elif converter is not None:
                if has_scalar_only_inputs:
                    # Under FakeTensorMode, op accepts scalar only inputs, such as aten.add/sub/mul/div,
                    # returns a real scalar tensor on CPU. See TensorMeta() in _prims/__init__.py for details.
                    # We thus directly convert real tensor to fake tensor.
                    return converter.from_real_tensor(self, e)
                else:
                    return converter.from_meta_and_device(
                        self, e, device or common_device
                    )
            else:
                # pyrefly: ignore [bad-return]
                return e

        return tree_map(wrap, r)

    def create_symbolic_nested_int(
        self, *, nt_tensor_id: Optional[int] = None
    ) -> torch.SymInt:
        # See Note: [Creating symbolic nested int]
        # Returned nested int always has coeff=1; multiply the result by coeff if needed
        import torch.nested._internal.nested_tensor
        from torch.nested._internal.nested_int import NestedIntNode

        if nt_tensor_id is None:
            nt_tensor_id = self.nt_tensor_id_counter
            assert self.enter_stack, "should only called while FakeTensorMode is active"
            self.nt_tensor_id_counter += 1
        hint = torch.SymInt(NestedIntNode(nt_tensor_id, 1))

        src = torch._dynamo.source.EphemeralSource("intermediate_offsets_or_lengths")
        assert self.shape_env is not None
        ret = self.shape_env.create_symintnode(
            sym=self.shape_env.create_symbol(
                val=hint,
                source=src,
            ),
            hint=hint,
            source=src,
        )
        return ret

    _cpp_meta_supports_symint = ordered_set(
        aten.empty.memory_format,
        aten.empty_strided.default,
        aten.as_strided_scatter.default,
        aten.as_strided.default,
        aten.as_strided_.default,
        aten.zeros.default,
        aten.detach.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
        aten.set_.source_Storage_storage_offset,
        aten._sparse_coo_tensor_with_dims_and_tensors.default,
    )

    _unbacked_special_fake_handling_ops = ordered_set(
        aten.view.default,
        aten._unsafe_view.default,
        aten.slice.Tensor,
    )

    def cpp_meta_supports_symint(self, func: OpOverload) -> bool:
        if torch.Tag.view_copy in func.tags:
            return True
        return func in self._cpp_meta_supports_symint

    lift_fns = ordered_set(aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t: Tensor) -> bool:
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT
            and not is_sparse_any(t)
            and not self.is_our_fake(t)
            and t.device.type != "meta"
        )

    def invalidate_written_to_constants(
        self,
        func: OpOverload,
        flat_arg_fake_tensors: Sequence[FakeTensor],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        schema_info = get_schema_info(func)
        if any_constant and schema_info.is_mutable():
            _, new_kwargs = normalize_function(  # type: ignore[misc]
                func,
                args=args,  # type: ignore[arg-type]
                kwargs=kwargs,  # type: ignore[arg-type]
                normalize_to_only_use_kwargs=True,
            )
            for k, v in new_kwargs.items():
                k = k if (k != "input" or schema_info.has_argument(k)) else "self"
                if (
                    self.is_our_fake(v)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(
        self,
        tensor: Tensor,
        *,
        static_shapes: Optional[bool] = None,
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
        trace: bool = True,
    ) -> FakeTensor:
        shape_env: Optional[ShapeEnv] = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert symbolic_context is None, (
                "cannot set both static_shapes and symbolic_context"
            )
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )


_StoragePointer = object


def _validate_symbolic_output_for_caching(
    state: _CacheKeyState, output: FakeTensor
) -> None:
    """
    Validate symbolic content in output and raise _BypassDispatchCache if
    caching should be bypassed.

    Args:
        state: Cache key state containing known symbols
        output: Output to validate
        proxy_mode_active: Whether PROXY dispatch mode is currently active

    Raises: _BypassDispatchCache: If output contains symbolic content that
        prevents caching

    Details:

    If our output contains any symbols that didn't appear in the input then we
    need to bypass. Usually this will be unbacked symbols which can't be
    properly reconstructed but there could be "weird" cases where backed symbols
    spontaneously appear (from non-input state)?

    If we're proxy (symbol) tracing and the output contains ANY symbols then we
    need to bypass. The problem is that ProxyTorchDispatchMode relies on SymNode
    object identity and being able to see the construction of SymNodes.

    We could improve the proxy tracing case in a few ways:

    1. If the output SymNodes are directly copied from inputs then this is
       actually fine - they're already tracked. This would probably be the
       biggest bang/buck.

    2. If the output (tensors) are all direct copies of the inputs then this is
       also fine - since they're inputs they must be tracked. We already compute
       this we just don't plumb it around enough.

    3. If the output SymNodes are already tracked by the proxy then this is also
       actually fine - they're properly tracked. This probably wouldn't be
       common since for most outputs we use torch.empty_strided() and recompute
       strides.

    4. We could use the proxy to track "how" the SymNodes were computed and when
       using the cache we could "replay" them properly to teach the proxy how to
       build them.
    """
    from torch.fx.experimental.symbolic_shapes import _iterate_exprs, _iterate_nodes

    is_tracing = torch.fx.experimental.proxy_tensor.get_proxy_mode() is not None
    if is_tracing:
        # Check for SymNode types in PROXY mode - this should bypass caching
        # regardless of whether symbols are known or not
        for _ in _iterate_nodes(output):
            raise _BypassDispatchCache("Proxy mode with SymNode output")
    else:
        # Check for unrepresented symbols in tensor expressions
        for s in _iterate_exprs(output):
            for symbol in s.free_symbols:
                if symbol not in state.known_symbols:
                    raise _BypassDispatchCache("unrepresented symbol in output")


# NB: returns fake tensors
def run_fallback_kernel(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    flat_args: Sequence[object],
    args_spec: PyTree,
    orig_not_implemented_exception: RuntimeError,
) -> FakeTensor:
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e: T) -> Union[T, Tensor]:
            if fake_mode.is_our_fake(e):
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        flat_args = [to_real_tensor(a) for a in flat_args]
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        r = func(*args, **kwargs)

    storages: set[_StoragePointer] = set()

    for e in flat_args:
        if isinstance(e, Tensor):
            if not is_sparse_any(e):
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e: T) -> Union[T, FakeTensor]:
        if id(e) not in inp_impls and (
            isinstance(e, Tensor)
            and not is_sparse_any(e)
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    return pytree.tree_map(map_out, r)


def _set_cache_key_for_shape_env(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    key.strip_shape_env()
    cache[key] = entry


def _set_cache_key(
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry],
    key: _DispatchCacheKey,
    entry: _DispatchCacheEntry,
) -> None:
    cache[key] = entry


# Just for use to allow copying a module to fake tensors,
# does not apply elsewhere
class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        self.fake_mode = fake_mode

    def __torch_function__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Optional[Mapping[str, object]] = None,
    ) -> FakeTensor:
        kwargs = kwargs if kwargs else {}

        # clone will get called in Parameter deepcopy
        if func is torch._C.TensorBase.clone:
            assert isinstance(args[0], Tensor)
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func is Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor = cast(Tensor, args[0])
            memo = cast(dict[int, FakeTensor], args[1])

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


def _device_handler(args: Sequence[object]) -> torch.device:
    # NB: Don't use is_our_fake, just serve the fake information
    # as is.  Notice we don't use 'self'; we use args[0].fake_mode
    # because they may not be the same.  It would also be possible
    # to return NotImplemented here, in which case the FakeTensor
    # handler on args[0] would handle it, but we're being nice and
    # short-circuiting quickly.
    assert len(args) == 1 and isinstance(args[0], FakeTensor)
    if args[0].fake_mode.in_kernel_invocation:
        return torch.device("meta")
    else:
        return args[0].fake_device


# [subclass inputs]
# Suppose we enable fake tensor mode.  This means that fake tensor
# mode will run first.  But what if we do an operation that
# involves a tensor subclass that will desugar into normal tensor
# operations?  Without returning NotImplemented, fake tensor mode will run first,
# decide that a conversion was made (since there was a non fake
# tensor argument), and report an error that converting non
# fake tensor is not supported.  What we actually wanted to happen
# was to give the subclass a chance to figure out what it wants to
# before erroring out. Returning NotImplemented here allows this.
def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, FakeTensor)
        and isinstance(x, Tensor)
        and type(x) is not Tensor
        and type(x) is not torch.nn.Parameter
    )


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).size()
    ),
    torch.ops.aten.stride.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).stride()
    ),
    torch.ops.aten.storage_offset.default: lambda args: int(
        cast(Tensor, args[0]).storage_offset()
    ),
}

_DISPATCH_HANDLE_DIRECTLY = ordered_set(
    torch.ops.aten.is_coalesced.default,
    torch.ops.aten.dense_dim.default,
    torch.ops.aten.sparse_dim.default,
    # _RecordFunction doesn't support __eq__ so make sure not to attempt to
    # cache it.
    torch.ops.profiler._record_function_exit._RecordFunction,
)

from torch._subclasses.fake_impls import (  # noqa: F401
    _device_not_kwarg_ops,
    _is_tensor_constructor,
    _like_tensor_constructors,
    contains_tensor_types,
    get_fast_op_impls,
    has_meta,
    op_implementations_checks,
    stride_incorrect_op,
)


def evict_fake_tensor_cache_key(key: _DispatchCacheKey) -> None:
    if key in FakeTensorMode.cache:
        FakeTensorMode.cache.pop(key)


@atexit.register
def dump_cache_stats() -> None:
    log.info("FakeTensor cache stats:")
    log.info("  cache_hits: %s", FakeTensorMode.cache_hits)
    log.info("  cache_misses: %s", FakeTensorMode.cache_misses)
    bypasses = FakeTensorMode.cache_bypasses
    if bypasses:
        log.info("  cache_bypasses:")
        width = max(len(k) for k in bypasses)
        for k, v in sorted(bypasses.items(), key=lambda i: -i[1]):
            log.info("    %-*s %s", width + 1, f"{k}:", v)


def _infer_fake_from_real_tensor(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: torch.Tensor
) -> torch.Tensor:
    def unsupported(reason: str) -> None:
        raise RuntimeError(
            f"propagate_real_tensors: we cannot infer a Fake kernel "
            f"(meta kernel) for operator {op._name} because {reason}. "
            f"Please use torch.library.register_fake to add a Fake kernel."
        )

    if real_out.storage_offset() != 0:
        unsupported(
            f"a return has a non-zero storage offset {real_out.storage_offset()}"
        )

    # Since PT2 is rank specialized, there's no such thing as a symbolic
    # output rank. So we can assume the fake tensor has the same number of
    # dimensions as the real tensor output.
    #
    # We shouldn't assume the Fake sizes/strides are exactly what we see on
    # the real tensor output (perhaps we should give users a lever to toggle
    # this). This is because there's a good amount of operators that return
    # outputs with data-dependent output shape.
    # So we infer the output sizes to all be unbacked symints
    fake_shape = [
        torch._library.fake_impl.allocate_size(mode.shape_env)
        for _ in range(real_out.dim())
    ]

    # We infer what the strides are. We had a couple of options for this:
    # - assume the strides are computable from the sizes
    # - use new fresh unbacked symints in the strides
    #   This doesn't work that well (PT2 doesn't support unbacked symint strides well)
    # - use the real strides
    #   This can only be used if we assume the strides are static.
    # We went with the first option.
    fake_strides = [-1] * real_out.dim()
    strides = [(s, idx) for idx, s in enumerate(real_out.stride())]
    strides.sort(key=lambda x: (x[0], -x[1]))
    expected = 1
    fake_stride = expected
    for s, idx in strides:
        if s != expected:
            unsupported(
                f"a return was not dense in memory (sizes {real_out.shape} strides {real_out.stride()})"
            )
        fake_strides[idx] = fake_stride
        expected = expected * real_out.shape[idx]
        fake_stride = fake_stride * fake_shape[idx]

    with mode:
        return torch.empty_strided(
            fake_shape,
            fake_strides,
            device=real_out.device,
            dtype=real_out.dtype,
            layout=real_out.layout,
        )


def inferred_fake_kernel_from_real_out(
    mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: Any
) -> Any:
    assert mode.shape_env is not None

    # Only support operators that have all Tensor outputs
    # This is a general limitation on custom ops that we impose for PT2
    # to avoid baking non-symbolic float/int outputs into the graph.
    real_flat_out, spec = pytree.tree_flatten(real_out)
    if not all(isinstance(t, torch.Tensor) for t in real_flat_out):
        raise RuntimeError(
            f"propagate_real_tensors: we don't support operators that return "
            f"non-Tensors. Got {op._schema}"
        )

    fake_flat_out = [_infer_fake_from_real_tensor(mode, op, t) for t in real_flat_out]
    return pytree.tree_unflatten(fake_flat_out, spec)
