from __future__ import annotations

import contextlib
import dataclasses
import functools
import threading
import typing
import weakref
from abc import abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Generic,
    NewType,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import override, TypedDict, TypeIs, Unpack

import torch
from torch._C._autograd import CreationMeta
from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_functional_tensor,
    _wrap_functional_tensor,
    get_unwrapped,
    is_batchedtensor,
    is_functorch_wrapped_tensor,
    is_gradtrackingtensor,
    is_legacy_batchedtensor,
    maybe_get_bdim,
    maybe_get_level,
    peek_interpreter_stack,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._logging import trace_structured
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.weak import WeakIdKeyDictionary


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from torch._C._functorch import CInterpreter
    from torch._guards import Source
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    # Import here to avoid cycle
    # Import the following modules during type checking to enable code intelligence features,
    # Do not import unconditionally, as they import sympy and importing sympy is very slow
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext


def _is_fake_tensor(t: object) -> TypeIs[FakeTensor]:
    from torch._subclasses.fake_tensor import FakeTensor

    return isinstance(t, FakeTensor)


DimList = list
_TensorLikeT = TypeVar("_TensorLikeT", "MetaTensorDesc", torch.Tensor)
_T = TypeVar("_T")
_TensorT = TypeVar("_TensorT", bound=torch.Tensor)
_TensorT_cov = TypeVar("_TensorT_cov", bound=torch.Tensor, covariant=True)


def safe_is_leaf(t: Union[MetaTensorDesc, torch.Tensor]) -> bool:
    try:
        return t.is_leaf
    except RuntimeError:
        # inference mode can trigger this
        return False


def safe_grad(t: _TensorLikeT) -> Optional[_TensorLikeT]:
    with torch._logging.hide_warnings(torch._logging._internal.safe_grad_filter):
        # pyrefly: ignore  # bad-return
        return t.grad


def _expect_safe_grad(t: _TensorLikeT) -> _TensorLikeT:
    grad = safe_grad(t)
    assert grad is not None
    return grad


def assert_eq(a: _T, b: _T) -> None:
    assert a == b, f"{a} != {b}"


tls = threading.local()
# Turns off inference mode for fake tensor propagation. This is turned to True
# only for `torch.compile`. Also look at
# _dynamo.config.fake_tensor_disable_inference_mode
tls.disable_inference_mode = False


@contextmanager
def disable_inference_mode_for_fake_prop() -> Generator[None, None, None]:
    prior = getattr(tls, "disable_inference_mode", False)
    tls.disable_inference_mode = True
    try:
        yield
    finally:
        tls.disable_inference_mode = prior


def assert_metadata_eq(
    assert_eq: Callable[[object, object], None],
    m1: Union[MetaTensorDesc, torch.Tensor],
    m2: torch.Tensor,
    *,
    skip_symbolic: bool = False,
    skip_leaf: bool = False,
) -> None:
    m1 = (
        MetaTensorDescriber().describe_tensor(m1)
        if isinstance(m1, torch.Tensor)
        else m1
    )

    def go(m1: MetaTensorDesc, m2: torch.Tensor) -> None:
        assert_eq(m1.dtype, m2.dtype)
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        assert_eq(m1.requires_grad, m2.requires_grad)
        if not skip_leaf:
            assert_eq(m1.is_leaf, m2.is_leaf)
        # MetaTensorDesc doesn't store grad_fn; inferred from leaf
        # assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        if not getattr(tls, "disable_inference_mode", False):
            assert_eq(m1.is_inference, m2.is_inference())
        else:
            assert_eq(m1.is_inference, False)
        assert_eq(m1.is_conj, m2.is_conj())
        assert_eq(m1.is_neg, m2.is_neg())
        assert_eq(m1.grad is not None, safe_grad(m2) is not None)
        if m1.grad is not None:
            go(m1.grad, _expect_safe_grad(m2))
        # TODO: move "assert_eq(m1.layout, m2.layout)" out of sparse
        #       branches (but not ready for prime time yet)...
        if m1.is_sparse:
            assert_eq(m1.layout, m2.layout)
            assert_eq(m1.dense_dim, m2.dense_dim())
            assert_eq(m1.sparse_dim, m2.sparse_dim())
            assert_eq(m1.is_coalesced, m2.is_coalesced())
        elif is_sparse_compressed(m1):
            assert_eq(m1.layout, m2.layout)
            assert_eq(m1.dense_dim, m2.dense_dim())
            assert_eq(m1.sparse_dim, m2.sparse_dim())
        else:
            if not skip_symbolic:
                assert_eq(m1.stride, m2.stride())
                assert_eq(m1.storage_offset, m2.storage_offset())
            assert_eq(m1.is_view, m2._is_view())
            if m1.is_view:
                assert m1.base is not None
                assert m2._base is not None
                go(m1.base, m2._base)
        # TODO: test if is resizable (no direct query for this atm)
        # TODO: audit AutogradMeta to see if it matches
        # TODO: test forward AD

    return go(m1, m2)


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_coo(t: object) -> TypeGuard[torch.Tensor]:
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


def is_sparse_compressed_layout(layout: torch.layout) -> bool:
    return layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_compressed(t: object) -> TypeGuard[torch.Tensor]:
    return isinstance(t, torch.Tensor) and is_sparse_compressed_layout(t.layout)


# TypeGuard (not TypeIs): False does not imply !torch.Tensor
def is_sparse_any(t: object) -> TypeGuard[torch.Tensor]:
    return is_sparse_coo(t) or is_sparse_compressed(t)


def _checked_cast(ty: type[_T], obj: object) -> _T:
    assert isinstance(obj, ty), f"expected {ty} but got {type(obj)}"
    return obj


def _get_real_storage(base: torch.UntypedStorage) -> torch.UntypedStorage:
    return base.real_storage  # type: ignore[attr-defined]


def _set_real_storage(
    base: torch.UntypedStorage, real_storage: torch.UntypedStorage
) -> None:
    base.real_storage = real_storage  # type: ignore[attr-defined]


# Don't use id() directly, because those can get reallocated over time.
MetaStorageId = NewType("MetaStorageId", int)
MetaTensorId = NewType("MetaTensorId", int)


_DescriberId = NewType("_DescriberId", int)
DESCRIBER_NEXT_ID = _DescriberId(0)


class MetaTensorDescriber:
    """
    Given a Tensor/Storage, generate a MetaTensorDesc/MetaStorageDesc
    for it, which is enough information to reconstruct a meta tensor/fake tensor
    corresponding to a Tensor as faithfully as possible.

    This is a stateful conversion object because we keep track of the IDs
    of the tensors/storages passed to us, so we can consistently give
    the same ID when we see the same tensor/storage.
    """

    def __init__(self, *, copy_data: bool = False) -> None:
        global DESCRIBER_NEXT_ID
        self.id = DESCRIBER_NEXT_ID
        DESCRIBER_NEXT_ID = _DescriberId(DESCRIBER_NEXT_ID + 1)
        self.next_tensor_id: MetaTensorId = MetaTensorId(0)
        self.next_storage_id: MetaStorageId = MetaStorageId(0)
        # Tensor -> int
        self.lookup_tensor = WeakIdKeyDictionary()
        # Storage -> int
        self.lookup_storage = WeakIdKeyDictionary()
        self.copy_data = copy_data
        self.traced_tensors: set[int] = set()
        self.traced_storages: set[int] = set()

    def get_tensor_id(self, t: torch.Tensor) -> MetaTensorId:
        if t not in self.lookup_tensor:
            self.lookup_tensor[t] = self.next_tensor_id
            self.next_tensor_id = MetaTensorId(self.next_tensor_id + 1)
        return self.lookup_tensor[t]

    def get_storage_id(self, s: torch.UntypedStorage) -> MetaStorageId:
        if s not in self.lookup_storage:
            self.lookup_storage[s] = self.next_storage_id
            self.next_storage_id = MetaStorageId(self.next_storage_id + 1)
        return self.lookup_storage[s]

    def describe_storage(
        self, s: torch.UntypedStorage, *, trace: bool = False
    ) -> MetaStorageDesc:
        r = MetaStorageDesc(
            id=self.get_storage_id(s),
            size=s.size(),
            # NB: We don't do the copy yet; copy happens when we start
            # creating the new storages
            data=s if self.copy_data else None,
        )
        if trace and r.id not in self.traced_storages:
            trace_structured(
                "describe_storage",
                metadata_fn=lambda: r.as_json(self.id),
            )
            self.traced_storages.add(r.id)
        return r

    def describe_tensor(
        self, t: torch.Tensor, *, recurse: bool = True, trace: bool = False
    ) -> MetaTensorDesc:
        is_leaf = safe_is_leaf(t)
        is_view = t._is_view()
        is_sparse = t.is_sparse
        layout = t.layout
        is_nested = t.is_nested
        is_traceable_wrapper_subclass_v = is_traceable_wrapper_subclass(t)
        is_functorch_wrapped = is_functorch_wrapped_tensor(t)
        is_mkldnn = t.is_mkldnn
        is_batchedtensor_v = is_batchedtensor(t)
        is_legacy_batchedtensor_v = is_legacy_batchedtensor(t)
        is_gradtrackingtensor_v = is_gradtrackingtensor(t)
        is_functional = torch._is_functional_tensor(t)

        storage = None
        # NB: For compatibility, I default this to zero, as sometimes people
        # still have stuffed zero into storage offset even though the tensor
        # doesn't meaningfully have an offset
        storage_offset = 0
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
            or is_mkldnn
            # TODO: TBH, functorch wrapped tensors probably should have
            # storage associated with them
            or is_functorch_wrapped
            or is_legacy_batchedtensor_v
        ):
            # NB: We actually don't use storage to do views, but might as well
            # put it in for accuracy
            storage = self.describe_storage(t.untyped_storage(), trace=trace)
            storage_offset = t.storage_offset()  # type: ignore[assignment]

        stride = None
        if not (
            is_sparse
            or is_sparse_compressed_layout(layout)
            or (is_nested and not is_traceable_wrapper_subclass_v)
        ):
            # stride/storage_offset are called from is_functorch_wrapped,
            # view_from_base, empty_create_subclass,
            # sym_sizes_strides_storage_offset (empty_create)
            stride = t.stride()

        # NB: this technically should refer to functorch unwrapped tensor, but
        # I am (perhaps abusively) using it to store both the functorch and
        # non-functorch functional tensor
        unwrapped = None
        autograd_meta_from = None
        current_level = None
        if is_batchedtensor_v or is_gradtrackingtensor_v:
            unwrapped = self.describe_tensor(get_unwrapped(t), trace=trace)
        # xla and lazy tensors present as functional tensors, but we want them
        # to be handled specially
        elif is_functional and t.device.type not in ("xla", "lazy"):
            if t._is_view():
                raise RuntimeError(
                    "Cannot safely fakify a view because this process drops the view information right now."
                )
            if not is_functorch_wrapped:
                torch._sync(t)
                unwrapped = self.describe_tensor(
                    torch._from_functional_tensor(t), trace=trace
                )
                autograd_meta_from = t
            else:
                reapply_views = torch._C._functionalization_reapply_views_tls()
                # NB: has side effects!
                unwrapped = self.describe_tensor(
                    _unwrap_functional_tensor(t, reapply_views), trace=trace
                )
                # TODO: It's pretty suspicious that functional tensors don't have
                # valid level and thus we just grab whatever the current level
                # is
                current_level = torch._C._functorch.current_level()

        maybe_functorch_stack = None
        if is_functorch_wrapped:
            with (
                torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
            ) as maybe_functorch_stack:
                pass

        attrs = None
        ctx = None
        type_v = None
        if is_traceable_wrapper_subclass_v:
            assert hasattr(t, "__tensor_flatten__")
            raw_attrs, ctx = t.__tensor_flatten__()
            attrs = {
                attr: self.describe_tensor(getattr(t, attr), trace=trace)
                for attr in raw_attrs
            }
            type_v = type(t)

        from torch.nested._internal.nested_tensor import _tensor_symint_registry

        view_func = ViewFunc.from_tensor(t)

        # TODO: Is it important to enable torch.inference_mode before querying
        # these values?
        is_inference_mode_disabled = getattr(tls, "disable_inference_mode", False)
        r: MetaTensorDesc = MetaTensorDesc(
            id=self.get_tensor_id(t),
            storage=storage,
            is_inference=False if is_inference_mode_disabled else t.is_inference(),
            is_leaf=is_leaf,
            requires_grad=t.requires_grad,
            # NB: ndim should be OK too but there is a disaster at
            # python test/dynamo/test_subclasses.py -k test_user_overridden_property_unsupported
            # Actually, this means that we have a little bit of a problem
            # here, which is that there is some sensitivity to how exactly an
            # access is done if you have a __torch_function__ subclass.  Maybe
            # should disable torch function before doing accesses?
            ndim=t.dim(),
            dtype=t.dtype,
            is_sparse=is_sparse,
            is_mkldnn=is_mkldnn,
            is_functorch_wrapped=is_functorch_wrapped,
            is_batchedtensor=is_batchedtensor_v,
            is_legacy_batchedtensor=is_legacy_batchedtensor_v,
            is_gradtrackingtensor=is_gradtrackingtensor_v,
            is_view=is_view,
            is_conj=t.is_conj(),
            is_neg=t.is_neg(),
            is_parameter=isinstance(t, torch.nn.Parameter),
            is_traceable_wrapper_subclass=is_traceable_wrapper_subclass_v,
            is_nested=is_nested,
            nested_int=(
                _tensor_symint_registry[t].node.nested_int()
                if t in _tensor_symint_registry
                else None
            ),
            is_functional=is_functional,
            layout=layout,
            device=t.device,
            size=t.size(),
            stride=stride,
            # pyrefly: ignore  # bad-argument-type
            storage_offset=storage_offset,
            dynamo_dynamic_indices=list(getattr(t, "_dynamo_dynamic_indices", set())),
            dynamo_hint_overrides=getattr(t, "_dynamo_hint_overrides", {}),
            sparse_dim=(
                t.sparse_dim() if t.is_sparse or is_sparse_compressed(t) else None
            ),
            dense_dim=t.dense_dim() if t.is_sparse or is_sparse_compressed(t) else None,
            is_coalesced=t.is_coalesced() if t.is_sparse else None,
            # TODO: I actually think recursing here is correct, but we have at
            # least an infinite cycle from base -> values -> base
            # https://github.com/pytorch/pytorch/issues/122089
            crow_indices=(
                self.describe_tensor(t.crow_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
                else None
            ),
            col_indices=(
                self.describe_tensor(t.col_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csr, torch.sparse_bsr}
                else None
            ),
            ccol_indices=(
                self.describe_tensor(t.ccol_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
                else None
            ),
            row_indices=(
                self.describe_tensor(t.row_indices(), recurse=False, trace=trace)
                if recurse and t.layout in {torch.sparse_csc, torch.sparse_bsc}
                else None
            ),
            values=(
                self.describe_tensor(t.values(), recurse=False, trace=trace)
                if recurse and is_sparse_compressed(t)
                else None
            ),
            grad=(
                self.describe_tensor(grad, trace=trace)
                if (grad := safe_grad(t)) is not None
                else None
            ),
            creation_meta=(
                torch._C._autograd._get_creation_meta(t) if t._is_view() else None
            ),
            unwrapped=unwrapped,
            level=(
                maybe_get_level(t)
                if is_batchedtensor_v or is_gradtrackingtensor_v
                else None
            ),
            bdim=maybe_get_bdim(t) if is_batchedtensor_v else None,
            base=(
                self.describe_tensor(t._base, trace=trace)
                if recurse and t._is_view() and t._base is not None
                else None
            ),
            fake_mode=torch._subclasses.fake_tensor.maybe_get_fake_mode(t),
            view_func=view_func,
            attrs=attrs,
            ctx=ctx,
            type=type_v,
            # NB: even if functorch is enabled, don't actually save the
            # interpreter stack here unless we are actually functorch wrapped;
            # it's irrelevant for non-functorch stuff
            functorch_stack=maybe_functorch_stack,
            autograd_meta_from=autograd_meta_from,
            current_level=current_level,
            data=t if self.copy_data else None,
        )
        if trace and r.id not in self.traced_tensors:
            trace_structured(
                "describe_tensor",
                metadata_fn=lambda: r.as_json(self.id),
            )
            self.traced_tensors.add(r.id)
        return r


@dataclass(frozen=True)
class MetaStorageDesc:
    id: MetaStorageId
    size: int
    # NB: this is only populated with copy_data True, it is not directly
    # serializable in JSON, you want to do something special here anyway
    data: Optional[torch.UntypedStorage]

    def as_json(self, describer_id: _DescriberId) -> dict[str, object]:
        return {
            "id": self.id,
            "describer_id": describer_id,
            "size": self.size if isinstance(self.size, int) else repr(self.size),
        }


@dataclass(frozen=True)
class ViewFunc(Generic[_TensorT]):
    @abstractmethod
    def apply(
        self,
        t: _TensorT,
        new_base: _TensorT,
        symint_visitor_fn: Optional[Callable[[int], int]] = None,
        tensor_visitor_fn: Optional[Callable[[torch.Tensor], _TensorT]] = None,
    ) -> _TensorT: ...

    @staticmethod
    def from_tensor(t: torch.Tensor) -> ViewFunc:
        if _is_fake_tensor(t):
            return _FakeTensorViewFunc()
        else:
            return _CustomViewFunc(t._view_func_unsafe)


@dataclass(frozen=True)
class _FakeTensorViewFunc(ViewFunc["FakeTensor"]):
    @override
    def apply(
        self,
        t: torch.Tensor,
        new_base: torch.Tensor,
        symint_visitor_fn: Optional[Callable[[int], int]] = None,
        tensor_visitor_fn: Optional[Callable[[torch.Tensor], FakeTensor]] = None,
    ) -> FakeTensor:
        return torch._subclasses.fake_tensor.FakeTensor._view_func_unsafe(
            # pyrefly: ignore  # bad-argument-type
            t,
            new_base,
            symint_visitor_fn,
            tensor_visitor_fn,
        )


@dataclass(frozen=True)
class _CustomViewFunc(ViewFunc[_TensorT], Generic[_TensorT]):
    func: Callable[
        [
            torch.Tensor,
            Optional[Callable[[int], int]],
            Optional[Callable[[torch.Tensor], _TensorT]],
        ],
        _TensorT,
    ]

    @override
    def apply(
        self,
        t: torch.Tensor,
        new_base: torch.Tensor,
        symint_visitor_fn: Optional[Callable[[int], int]] = None,
        tensor_visitor_fn: Optional[Callable[[torch.Tensor], _TensorT]] = None,
    ) -> _TensorT:
        # ignore `t`
        return self.func(new_base, symint_visitor_fn, tensor_visitor_fn)


# A callback where the device is either optional or required.
# All of these satisfy this protocol:
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str])
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str] = "meta")
#   def mk(arg: Callable[[], torch.Tensor], device: Optional[Union[torch.device, str]] = None)
class _MetaTensorCallback(Protocol, Generic[_TensorT_cov]):
    def __call__(
        self, arg: Callable[[], torch.Tensor], /, *, device: Union[torch.device, str]
    ) -> _TensorT_cov: ...


class _MetaTensorCallbackKwargs(TypedDict, total=False):
    device: Union[torch.device, str]


# A callback where the device may not be provided (is optional).
# All of these satisfy this protocol:
#   def mk(arg: Callable[[], torch.Tensor], device: Union[torch.device, str] = "meta")
#   def mk(arg: Callable[[], torch.Tensor], device: Optional[Union[torch.device, str]] = None)
class _MetaTensorCallbackOptDevice(Protocol, Generic[_TensorT_cov]):
    def __call__(
        self,
        arg: Callable[[], torch.Tensor],
        /,
        **kwargs: Unpack[_MetaTensorCallbackKwargs],
    ) -> _TensorT_cov: ...


@dataclass(frozen=True)
class MetaTensorDesc(Generic[_TensorT]):
    id: MetaTensorId
    ndim: int
    dtype: torch.dtype
    device: torch.device

    # NB: Sometimes, size, stride and storage_offset contain SymInt, in which
    # case this is NOT serializable.  That only happens when you're
    # re-fakeifying a fake tensor with an existing ShapeEnv... maybe we
    # can get rid of this use case entirely.  Notably, even if we are
    # fakeifying a real tensor into a fake tensor with symbolic shapes, the
    # size here is NOT dynamic
    # NB: These also contain SymInt because wrap_meta_outputs_with_default_device_logic
    # goes through this codepath.  But it really should not LOL.
    # NB: size could potentially be None as you can override it and make it
    # throw an error, but we don't currently have any subclasses that do this
    # except C++ nested tensor but we're going to have nested int to make this
    # defined on NJT
    size: tuple[int, ...]
    dynamo_dynamic_indices: list[int]
    dynamo_hint_overrides: dict[int, int]

    layout: torch.layout = torch.strided
    is_inference: bool = False
    is_leaf: bool = False
    requires_grad: bool = False
    is_sparse: bool = False
    is_mkldnn: bool = False
    is_functorch_wrapped: bool = False
    is_batchedtensor: bool = False
    is_legacy_batchedtensor: bool = False
    is_gradtrackingtensor: bool = False
    is_view: bool = False
    is_nested: bool = False
    # We eagerly symbolicize the associated nested int for e.g. offsets / lengths
    # metadata if that offsets is already associated with a nested int.
    # See test_construct_from_jagged_with_input_offsets_mixed_case.
    nested_int: Optional[int] = None
    is_traceable_wrapper_subclass: bool = False
    is_functional: bool = False
    is_conj: bool = False
    is_neg: bool = False
    is_parameter: bool = False
    stride: Optional[tuple[int, ...]] = None
    storage_offset: int = 0
    # NB: We have a choice whether or not to store the id or a direct pointer
    # to the data structure.  For ease of use, we store the data structure,
    # but this means that when we serialize, we have to swizzle these pointers
    # back into ids (so we have accurate aliasing relationships)
    storage: Optional[MetaStorageDesc] = None
    sparse_dim: Optional[int] = None  # is_sparse, is_sparse_compressed
    dense_dim: Optional[int] = None  # is_sparse, is_sparse_compressed
    is_coalesced: Optional[bool] = None  # is_sparse
    crow_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    col_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    ccol_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    row_indices: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    values: Optional[MetaTensorDesc] = None  # is_sparse_compressed
    unwrapped: Optional[MetaTensorDesc] = None  # is_functorch_wrapped
    bdim: Optional[int] = None  # is_functorch_wrapped
    base: Optional[MetaTensorDesc] = None  # is_view
    attrs: Optional[dict[str, MetaTensorDesc]] = None  # is_traceable_wrapper_subclass
    creation_meta: Optional[CreationMeta] = None
    grad: Optional[MetaTensorDesc] = None

    # Everything below is NOT serializable, need some more work

    _UNSERIALIZABLE: ClassVar[set[str]] = {
        "ctx",
        "type",
        "fake_mode",
        # view_func isn't serializable when it's a _CustomViewFunc
        "view_func",
        "level",
        "current_level",
        "functorch_stack",
        "autograd_meta_from",
        "data",
        "nested_int",
    }

    ctx: Optional[object] = None  # is_traceable_wrapper_subclass
    type: Optional[type] = None  # is_traceable_wrapper_subclass
    fake_mode: Optional[FakeTensorMode] = None
    view_func: Optional[ViewFunc] = None
    # level looks serializable, but actually it is meaningless without
    # the functorch_stack below
    level: Optional[int] = None  # is_functorch_wrapped
    current_level: Optional[int] = None
    functorch_stack: Optional[list[CInterpreter]] = None
    autograd_meta_from: Optional[torch.Tensor] = None

    # This is only populated on copy_data, and typically is not used at all,
    # except for some of our meta-ification paths that don't properly use
    # storage (pro-tip: you should use storage)
    data: Optional[torch.Tensor] = None

    # Faithfully serializing functorch tensors will not be too difficult.
    # We only need to consider grad/vmap interpreters, and their internal
    # state is only bools (mostly what the grad enabled/disabled state
    # should be in the lower layer).  Beyond that, tensors just need to
    # precisely indicate which particular interpreter they correspond
    # to (we then replace level with a pointer to the interpreter stack.)
    # However, this use of functorch is very "non-lexical" so it's not
    # entirely clear how to make it all lexical again, so we haven't done
    # it for now.

    # NB: This will reference numeric IDs, and it is assumed that you've
    # already serialized everything this recursively references
    def as_json(self, describer_id: _DescriberId) -> dict[str, object]:
        def json(k: str, v: object) -> object:
            # Some best-effort debugging serialization for unserializable
            # fields (feel free to add other special cases as appropriate)
            if k in ["data", "autograd_meta_from"]:
                return None  # never repr these
            if k in MetaTensorDesc._UNSERIALIZABLE:
                return repr(v)
            if isinstance(v, (torch.device, torch.dtype, torch.layout)):
                return repr(v)
            if isinstance(v, torch.SymInt):
                return repr(v)
            if isinstance(v, (tuple, list)):
                return [json(k, v1) for v1 in v]
            if isinstance(v, (MetaStorageDesc, MetaTensorDesc)):
                return v.id
            if isinstance(v, CreationMeta):
                return str(v)
            if k == "attrs" and isinstance(v, dict):
                return {k1: v1.id for k1, v1 in v.items()}
            return v

        r = {
            field.name: json(field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
            if not (
                getattr(self, field.name) is field.default
                or (
                    field.name == "dynamo_dynamic_indices"
                    and not getattr(self, field.name)
                )
            )
        }
        r.update({"describer_id": describer_id})
        return r

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size


# A more faithful reproduction would do a copy on the entire
# storage, but this needs to be done carefully because the
# underlying storage could have larger extent than is implied
# by size/stride.  The real fix is to properly call
# meta_storage recursively here.
#
# These "safe" functions are intended to be used under no_dispatch() mode.
# The no_dispatch() here is intended to prevent ambient fake tensor mode from
# fakeifying the operation.  But if we are given an honest to goodness
# FakeTensor as src, we MUST NOT run the copy/clone operation.  A better way
# to do this would be to not use no_dispatch and instead just disable fake
# tensor mode only (allowing for subclass dispatch to occur)
def _safe_copy(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
    if type(src) is not torch.Tensor:
        return
    dst.copy_(src)


def _safe_clone(src: torch.Tensor) -> Optional[torch.Tensor]:
    if type(src) is not torch.Tensor:
        return None
    return src.clone()


# This is a class for converting multiple tensors into meta tensors which
# share the same view/storage structure.  The operation model is you allocate
# one of these, and then call it repeatedly on all the tensors you want to
# convert.  It's important to use the same object for tensors you want to
# share storage because this is how we correlate shared storages to the same
# meta storages. This class will hold weak references to cached tenosrs
# and tensor storages.
class MetaConverter(Generic[_TensorT]):
    def __init__(self, *, copy_data: bool = False) -> None:
        # Maps MetaStorageId to UntypedStorage
        self.storage_memo: weakref.WeakValueDictionary[
            MetaStorageId, torch.UntypedStorage
        ] = weakref.WeakValueDictionary()
        # Maps MetaTensorId to torch.Tensor (typically a meta tensor or
        # FakeTensor)
        self.tensor_memo: weakref.WeakValueDictionary[MetaTensorId, _TensorT] = (
            weakref.WeakValueDictionary()
        )
        self.hit = 0
        self.miss = 0
        self.del_hook = None
        self.arg_cnt = 0
        # Ensures real_storage/real_tensor are populated on the resulting
        # metaified storage/tensor.  The naming of this attribute is load
        # bearing: FakeTensor relies on real tensor being set to exactly this
        # value
        self.copy_data = copy_data
        self.describer = MetaTensorDescriber(copy_data=copy_data)

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def get_tensor_memo(self, t: MetaTensorDesc) -> Optional[torch.Tensor]:
        return self.tensor_memo.get(t.id, None)

    def _checked_get_tensor_memo(self, t: MetaTensorDesc) -> _TensorT:
        r = self.tensor_memo.get(t.id, None)
        assert r is not None
        return r

    def set_tensor_memo(self, t: MetaTensorDesc, v: _TensorT) -> None:
        self.tensor_memo[t.id] = v

    def get_storage_memo(self, s: MetaStorageDesc) -> Optional[torch.UntypedStorage]:
        return self.storage_memo.get(s.id, None)

    def set_storage_memo(self, s: MetaStorageDesc, v: torch.UntypedStorage) -> None:
        self.storage_memo[s.id] = v

    def meta_storage(
        self,
        s: MetaStorageDesc,
        callback: Callable[[Callable[[], torch.Tensor]], _TensorT],
    ) -> torch.UntypedStorage:
        # If we are fakeifying a tensor that has a secretly-zero-sized storage,
        # Need to make sure to resize the meta storage too.
        if (memo := self.get_storage_memo(s)) is None:
            r_s = callback(
                lambda: torch.empty(s.size, dtype=torch.uint8, device="meta"),
            ).untyped_storage()
            if self.copy_data:
                # NB: no_dispatch is needed because internally storage copy is
                # implemented as Tensor operations
                with torch.no_grad(), no_dispatch():
                    assert s.data is not None
                    _set_real_storage(r_s, s.data.clone())
            self.set_storage_memo(s, r_s)
            return r_s
        else:
            return memo

    @classmethod
    def _checked_cast_tensor_t(cls, t: torch.Tensor) -> _TensorT:
        # TODO: how to check _TensorT?
        return typing.cast(_TensorT, t)

    @classmethod
    def _identity_callable(
        cls,
        t: Callable[[], torch.Tensor],
        device: Optional[Union[torch.device, str]] = None,
    ) -> _TensorT:
        return cls._checked_cast_tensor_t(t())

    @classmethod
    def _backward_error(cls, t: _TensorT) -> _TensorT:
        errfn = torch._C._functions.DelayedError(
            "Internal error: Tried to backward() through example input",
            1,
        )
        err = errfn(t)
        return typing.cast(_TensorT, err)

    # This function assumes that it's possible to do the conversion
    # NB: name here is used in a conventional way by Dynamo; it corresponds
    # precisely to the Source.name() of the tensor we're fakeifying and
    # corresponds to a valid Python expression.  When we construct sub-names
    # as part of this process, we will maintain this invariant!  (Even though
    # other users of this may not need it this property to be upheld.)
    def meta_tensor(
        self,
        t: MetaTensorDesc,
        shape_env: Optional[ShapeEnv],
        callback_: _MetaTensorCallback[_TensorT],
        source: Optional[Source],
        symbolic_context: Optional[SymbolicContext],
    ) -> _TensorT:
        callback: _MetaTensorCallbackOptDevice = functools.partial(
            callback_, device=t.device
        )
        if source is None:
            from torch._dynamo.source import ConstantSource

            # TODO: make a dedicated UnknownSource for this?
            source = ConstantSource(
                f"__meta_utils_unknown_tensor{len(self.tensor_memo)}"
            )

        msg = (
            " This indicates you set no_dispatch() before calling into this"
            " function.  This is an error: we may be creating fake tensors and"
            " will perform operations on them which need fake tensor mode to"
            " be active.  You will segfault if you are in a no_dispatch() block."
        )
        assert not torch._C._dispatch_tls_local_exclude_set().has(
            torch._C.DispatchKey.Python
        ), msg
        self.arg_cnt += 1

        # When we make as_strided calls, we end up generating a guard
        # that the new as_strided tensor is in bounds for the old storage
        # for the base (since as_strided calls can "bust" out of their
        # bounding box.)  This guard is unnecessary: if a user is able
        # to provide us a tensor with the view base setup this way, we
        # don't need to produce a guard, because the fact that they
        # were able to produce the view base means its in bounds.
        #
        # Now, ordinarily, this guard would be harmless.  However, the
        # generated guard refers to variables bound on the base variable.
        # At the moment, Dynamo doesn't actually guard on x._base, because
        # according to Voz this results in a lot of spurious invalidations,
        # and also if the user doesn't directly make use of _base, its
        # pointless anyway (because programs should be parametric over
        # whether or not the input tensor is a view or not--unless you're
        # mutating the input, but that's a whole 'nother ballgame).  So
        # for expediency, we suppress these guards so we don't have to
        # deal with this (yet, anyway.)
        #
        # NB: An old version of this code suppressed guards for ALL operations
        # happening during meta conversion, not just as_strided calls.
        # This is too aggressive: we do duck sizing and 0/1 simplification
        # as we allocate variables, and we do need to register guards for
        # these cases.
        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if shape_env is not None:
            maybe_suppress = shape_env.suppress_guards

        def sym_sizes_strides_storage_offset(
            t: MetaTensorDesc,
            src: torch._guards.Source,
            symbolic_context: Optional[
                torch.fx.experimental.symbolic_shapes.SymbolicContext
            ] = symbolic_context,
        ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
            assert t.stride is not None
            if shape_env is not None:
                fake_mode = t.fake_mode
                if fake_mode is not None and fake_mode.shape_env is shape_env:
                    # Don't reallocate the sizes; the shape envs are the same,
                    # so reuse the old sizes/strides/etc
                    return (t.size, t.stride, t.storage_offset)
                else:
                    # TODO: deduplicate this
                    t_size = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sz)
                        for sz in t.size
                    )
                    t_stride = tuple(
                        shape_env._maybe_specialize_sym_int_with_hint(sd)
                        for sd in t.stride
                    )
                    t_storage_offset = shape_env._maybe_specialize_sym_int_with_hint(
                        t.storage_offset
                    )
                    return shape_env._create_symbolic_sizes_strides_storage_offset(
                        t_size,
                        t_stride,
                        t_storage_offset,
                        [d in t.dynamo_dynamic_indices for d in range(t.ndim)],
                        src,
                        symbolic_context=symbolic_context,
                        hint_overrides=t.dynamo_hint_overrides,
                    )
            else:
                return (t.size, t.stride, t.storage_offset)

        def empty_create(
            inner_t: MetaTensorDesc,
            inner_src: torch._guards.Source,
            symbolic_context: Optional[
                torch.fx.experimental.symbolic_shapes.SymbolicContext
            ] = symbolic_context,
        ) -> torch.Tensor:
            (
                inner_sizes,
                inner_strides,
                _inner_storage_offset,
            ) = sym_sizes_strides_storage_offset(inner_t, inner_src, symbolic_context)
            return torch.empty_strided(
                inner_sizes,
                inner_strides,
                dtype=inner_t.dtype,
                device="meta",
            )

        # Creates a subclass instance with empty inner tensors according to the specified
        # symbolic context.
        def empty_create_subclass(
            t: MetaTensorDesc,
            outer_size: tuple[int, ...],
            outer_stride: tuple[int, ...],
            symbolic_context: Optional[
                torch.fx.experimental.symbolic_shapes.SymbolicContext
            ] = symbolic_context,
            source: Optional[torch._guards.Source] = source,
        ) -> _TensorT:
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext

            assert t.attrs is not None
            assert t.type is not None
            # NB: t.ctx could be None if the subclass in question has no
            # meaningful context

            # Note: transform_subclass will use __tensor_unflatten__ to generate
            # a fresh subclass wrapper with outer sizes / strides according to the
            # outer symbolic context (passed in to this function). Inner size / stride
            # / storage offset symbols are allocated according to the appropriate inner
            # symbolic contexts, after which the checks in transform_subclass() will
            # relate them to the outer metadata as possible.
            #
            # Morally, the code here is same as transform_subclass, but we've
            # written it from scratch to read EmptyCreateSubclass
            outer_size = outer_size if outer_size is not None else t.size
            # pyrefly: ignore  # bad-assignment
            outer_stride = outer_stride if outer_stride is not None else t.stride

            assert symbolic_context is None or isinstance(
                symbolic_context, SubclassSymbolicContext
            )

            def _empty_create_subclass(
                t: MetaTensorDesc,
                outer_size: Optional[tuple[int, ...]],
                outer_stride: Optional[tuple[int, ...]],
                symbolic_context: Optional[
                    torch.fx.experimental.symbolic_shapes.SymbolicContext
                ],
                callback: _MetaTensorCallbackOptDevice[_TensorT],
                source: torch._guards.Source,
            ) -> _TensorT:
                # We are hitting plain meta_desc tensor so actually
                # create a tensor here.
                if t.attrs is None:
                    return self.meta_tensor(
                        t,
                        shape_env,
                        callback,
                        source,
                        symbolic_context,
                    )

                inner_tensors = {}
                for attr, meta_tensor_desc in t.attrs.items():
                    current_context = None
                    if symbolic_context is not None:
                        assert isinstance(symbolic_context, SubclassSymbolicContext)
                        if (
                            current_context_ := symbolic_context.inner_contexts[attr]
                        ) is not None:
                            current_context = _checked_cast(
                                torch.fx.experimental.symbolic_shapes.SymbolicContext,
                                current_context_,
                            )

                    current_source = AttrSource(source, attr)
                    inner_callback = functools.partial(
                        callback, device=meta_tensor_desc.device
                    )
                    new_empty_tensor = _empty_create_subclass(
                        meta_tensor_desc,
                        meta_tensor_desc.size,
                        meta_tensor_desc.stride,
                        current_context,
                        inner_callback,
                        current_source,
                    )
                    inner_tensors[attr] = new_empty_tensor

                assert t.type is not None
                return t.type.__tensor_unflatten__(  # type: ignore[attr-defined]
                    inner_tensors, t.ctx, outer_size, outer_stride
                )

            assert source is not None
            sub = _empty_create_subclass(
                t, outer_size, outer_stride, symbolic_context, callback, source
            )

            # NB: Purposefully guard here to simplify the inner / outer symbols.
            # Using sym_eq() for symbolic comparison can result in an expression that's too
            # difficult to guard on, so we use == here.
            assert sub.shape == outer_size, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"shape equal to {outer_size}, but got: {sub.shape}"
            )
            assert sub.stride() == outer_stride, (
                f"Expected return value from {t.type}__tensor_unflatten__() to have "
                f"stride equal to {outer_stride}, but got: {sub.stride()}"
            )

            return sub

        # Returns an all-dynamic symbolic context used for metafying the given tensor with
        # fully dynamic dims. This is useful when fake-ifying intermediate tensors in
        # closed-over ViewFunc state, as we don't have symbolic contexts for them, but we
        # don't want to over-specialize during view replay.
        def all_dynamic_symbolic_context(
            t: MetaTensorDesc,
            source: torch._guards.Source,
            shape_env: Optional[torch.fx.experimental.symbolic_shapes.ShapeEnv],
            callback: _MetaTensorCallback[_TensorT],
        ) -> torch.fx.experimental.symbolic_shapes.SymbolicContext:
            from torch._dynamo.source import AttrSource
            from torch.fx.experimental.symbolic_shapes import (
                DimDynamic,
                StatelessSymbolicContext,
                SubclassSymbolicContext,
            )

            view_base_context: Optional[
                torch.fx.experimental.symbolic_shapes.SymbolicContext
            ] = None
            if t.is_view:
                assert t.base is not None
                view_base_context = all_dynamic_symbolic_context(
                    t.base, AttrSource(source, "_base"), shape_env, callback
                )

            t_symbolic_context: torch.fx.experimental.symbolic_shapes.SymbolicContext
            t_dynamic_sizes = [DimDynamic.DYNAMIC] * t.ndim
            if t.is_traceable_wrapper_subclass:
                assert t.attrs is not None
                inner_contexts: dict[
                    str, torch.fx.experimental.symbolic_shapes.SymbolicContext
                ] = {}
                for attr, inner in t.attrs.items():
                    assert isinstance(attr, str)
                    inner_contexts[attr] = all_dynamic_symbolic_context(
                        inner, AttrSource(source, attr), shape_env, callback
                    )
                t_symbolic_context = SubclassSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    inner_contexts=inner_contexts,  # type: ignore[arg-type]
                    tensor_source=source,
                    view_base_context=view_base_context,
                )
            else:
                t_symbolic_context = StatelessSymbolicContext(
                    dynamic_sizes=t_dynamic_sizes,
                    constraint_sizes=[None] * t.ndim,
                    view_base_context=view_base_context,
                )

            return t_symbolic_context

        # Returns a fake-ified version of an input view tensor t, given an already fake-ified
        # base. At a high level, we want two things:
        #   1. fake_t should have the same view relationship to the given fake base as the
        #      input t has to its _base.
        #   2. fake_t should have symbolic sizes / strides / storage offset according to the
        #      appropriate symbolic context (i.e. from the automatic dynamic algorithm).
        #
        # We currently take different strategies across view types:
        #   * For dense -> dense views, accomplish both (1) and (2) simultaneously via an
        #     as_strided() call on the fake-ified base, passing symbolic metadata.
        #   * For views involving subclasses, perform view replay using view funcs to
        #     achieve (1). It's necessary for (2) to swap out any closed-over state in
        #     the view funcs with symbolicized SymInts and fake-ified tensors. Doing this
        #     avoids specialization (and thus over-eager simplification of symbols) that
        #     could occur during view replay on the fake-ified base.
        #
        # Examples:
        #   * t.unsqueeze(-1) with dense t is a dense -> dense view. It can be modeled
        #     with an as_strided() call on the fake base passing symbolic metadata.
        #   * sub.select(dim=0, index=3) is a subclass -> subclass view. The index arg
        #     is made symbolic to avoid invalid specialization and view replay is then
        #     done to reconstruct the view.
        #   * _nested_from_jagged(values, offsets) is a dense -> subclass view
        #     that returns a subclass instance from a dense values tensor. The offsets
        #     tensor is closed over in the view func, as it can be considered view metadata.
        #     First, the offsets tensor is fake-ified according to the inner symbolic
        #     context and with the correct relationship to the outer size / stride metadata.
        #     Then view replay is done, swapping in the fake offsets so the view replay output
        #     is fully fake with no invalid specialization.
        def view_from_base(
            base: _TensorT,
            t: MetaTensorDesc,
            shape_env: Optional[
                torch.fx.experimental.symbolic_shapes.ShapeEnv
            ] = shape_env,
        ) -> _TensorT:
            with enable_python_dispatcher():
                # fake-ify t's metadata according to the outer symbolic context
                (sizes, strides, storage_offset) = sym_sizes_strides_storage_offset(
                    t, source
                )
                if (
                    not t.is_traceable_wrapper_subclass
                    and not is_traceable_wrapper_subclass(base)
                ):
                    # Dense -> Dense view case uses as_strided() to construct view relationship.
                    # TODO: Change this logic to use view replay for consistency?
                    # It's likely there is no view func available.
                    with maybe_suppress():
                        return self._checked_cast_tensor_t(
                            base.as_strided(sizes, strides, storage_offset)
                        )

                from torch._dynamo.source import EphemeralSource
                from torch.fx.experimental.symbolic_shapes import (
                    StatelessSymbolicContext,
                    sym_eq,
                )

                def symint_visitor_fn(s: int) -> int:
                    nonlocal symbolic_context
                    from torch.fx.experimental.symbolic_shapes import DimDynamic

                    all_static_sizes = (
                        symbolic_context is not None
                        and isinstance(symbolic_context, StatelessSymbolicContext)
                        and all(
                            x is DimDynamic.STATIC
                            for x in symbolic_context.dynamic_sizes
                        )
                    )
                    # Can't just rely on shape env being None - dynamo always initializes it
                    if all_static_sizes or shape_env is None:
                        return s

                    # NB: The symbol here is expected to be simplified out because we a priori
                    # allocate inner and outer symbols according to the appropriate symbolic
                    # contexts and prefer those over this symbol during symbol simplification
                    # (via usage of EphemeralSource below). This -shouldn't- happen, but if
                    # this symbol somehow leaks out beyond the view tensor's shape metadata, our
                    # assumption of it being simplified out will fail and it may be guarded on,
                    # which will hard error.
                    sym_source = EphemeralSource("symint_visitor_fn")

                    symbol = shape_env.create_symbol(s, sym_source, positive=None)
                    return shape_env.create_symintnode(
                        symbol, hint=s, source=sym_source
                    )

                real_to_fake_mapping = {}
                if t.is_traceable_wrapper_subclass:
                    assert t.attrs is not None
                    # NB: t.ctx could be None if the subclass in question has no
                    # meaningful context
                    assert t.type is not None

                    # Fake-ify t naively here; this is only done so we can get fake-ified inner
                    # tensors with the correct relationships to the outer sizes / strides for use
                    # in view replay. It's done beforehand here because it's not easy to do when
                    # visiting tensors one-by-one during view replay.
                    #
                    # Example:
                    #   Consider a Dense -> NJT view. NJT has (values, offsets) components and we
                    #   want a view of values with the offsets closed over. As the offsets component
                    #   is needed to describe the output view, it's important that it's fakeified
                    #   correctly.
                    fake_t: _TensorT = empty_create_subclass(
                        t, outer_size=sizes, outer_stride=strides
                    )
                    attrs, _ = fake_t.__tensor_flatten__()  # type: ignore[attr-defined]
                    for attr in attrs:
                        real_to_fake_mapping[t.attrs[attr].id] = getattr(fake_t, attr)

                def tensor_visitor_fn(
                    visited_t: torch.Tensor,
                    # These arguments are never passed, we just use them to close
                    # over these relevant values
                    shape_env: Optional[
                        torch.fx.experimental.symbolic_shapes.ShapeEnv
                    ] = shape_env,
                    callback: _MetaTensorCallbackOptDevice[_TensorT] = callback,
                ) -> torch.Tensor:
                    # It's possible to close over an undefined tensor (e.g. NJT's lengths).
                    if visited_t is None:
                        # pyrefly: ignore  # bad-return
                        return None

                    # NB: visited_t being a Tensor here is very naughty!  Should
                    # have already been described

                    # Fake inner tensors of view subclasses will come from the mapping built above.
                    visited_id = self.describer.get_tensor_id(visited_t)
                    fake_visited_t = real_to_fake_mapping.get(visited_id, None)
                    if fake_visited_t is not None:
                        return fake_visited_t

                    visited_desc = self.describer.describe_tensor(visited_t)

                    # For other closed-over tensor state, fake-ify it as all dynamic with an
                    # ephemeral source. This avoids invalid specialization during view replay.
                    # If we find that in practice the usage of ephemeral sources isn't enough
                    # to guarantee that we don't have guards on these symbols, we may need to
                    # explicitly suppress guards (as is done for _base in the dense -> dense
                    # view case).
                    temp_source = EphemeralSource("tensor_visitor_fn")
                    return self.meta_tensor(
                        visited_desc,
                        shape_env,
                        callback,
                        temp_source,
                        all_dynamic_symbolic_context(
                            visited_desc, temp_source, shape_env, callback
                        ),
                    )

                # Replay the view, swapping out any non-symbolic SymInts or real tensors
                # for symbolic SymInts or fake tensors.
                assert t.view_func is not None
                # NB: we do NOT suppress guards here, we need to remove ephemeral
                # sources
                fake_t = t.view_func.apply(
                    t, base, symint_visitor_fn, tensor_visitor_fn
                )

                # Ensure the output has symbolic shapes according to the outer symbolic context.
                # These checks should simplify out any symbols created for closed-over view func
                # SymInts.
                torch._check(sym_eq(fake_t.size(), sizes))
                torch._check(sym_eq(fake_t.stride(), strides))
                torch._check(sym_eq(fake_t.storage_offset(), storage_offset))
                return fake_t

        if self.get_tensor_memo(t) is None:
            GRAD_TENSOR_SENTINEL_VALUE = -2

            with torch.inference_mode(t.is_inference):
                if t.is_sparse:
                    is_leaf = t.is_leaf

                    # The lambda function below is similar to
                    # `t.to(device='meta')` except the latter
                    # preserves nnz value
                    r = callback(
                        lambda: torch.ops.aten._sparse_coo_tensor_with_dims(
                            t.sparse_dim,
                            t.dense_dim,
                            t.size,
                            dtype=t.dtype,
                            layout=torch.sparse_coo,
                            device="meta",
                        )
                    )
                    if self.copy_data:
                        # Pray that sparse clone doesn't lose information
                        assert t.data is not None
                        with torch.no_grad(), no_dispatch():
                            assert _is_fake_tensor(r)
                            r.real_tensor = _safe_clone(t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    # Note [is_coalesced is dispatched]
                    # Strangely enough, is_coalesced() is a dispatched operator,
                    # which means that it will get caught by fake tensor mode.
                    # Ordinarily this would error, but there's some logic in
                    # fake tensor ensure this doesn't happen.
                    r._coalesced_(bool(t.is_coalesced))
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        # This should probably use DelayedError,
                        # but clone is fine for now for sparse tensors.
                        # (DelayedError does not work for sparse because it causes
                        # the Fake sparse tensor to "lose" its fakeness)
                        r = self._checked_cast_tensor_t(r.clone())
                        with torch.enable_grad():
                            r._coalesced_(bool(t.is_coalesced))
                elif is_sparse_compressed_layout(t.layout):
                    is_leaf = t.is_leaf

                    if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                        assert t.sparse_dim is not None
                        assert t.dense_dim is not None
                        assert t.values is not None
                        batch_dim = t.ndim - t.sparse_dim - t.dense_dim
                        blocksize = t.values.shape[batch_dim + 1 : batch_dim + 3]
                    else:
                        blocksize = ()
                    if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        assert t.crow_indices is not None
                        index_dtype = t.crow_indices.dtype
                    else:
                        assert t.ccol_indices is not None
                        index_dtype = t.ccol_indices.dtype

                    r = callback(
                        lambda: torch.ops.aten._sparse_compressed_tensor_with_dims(
                            0,
                            t.dense_dim,
                            t.shape,
                            blocksize,
                            index_dtype,
                            layout=t.layout,
                            dtype=t.dtype,
                            device="meta",
                        )
                    )
                    if self.copy_data:
                        # Pray sparse clone doesn't lose information
                        assert t.data is not None
                        with torch.no_grad(), no_dispatch():
                            assert _is_fake_tensor(r)
                            r.real_tensor = _safe_clone(t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        # pyrefly: ignore  # bad-argument-type
                        r = self._backward_error(r)
                elif t.is_nested and not t.is_traceable_wrapper_subclass:
                    # TODO: Handle this better in Dynamo?
                    # There are checks there now, but this can still be triggered by a dense
                    # tensor graph input that is a view of a strided NT.
                    from torch._dynamo.exc import unimplemented

                    unimplemented(
                        "strided nested tensors are not supported by meta conversion"
                    )
                elif t.is_mkldnn:
                    is_leaf = t.is_leaf
                    (
                        sizes,
                        strides,
                        _storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source)
                    # TODO: This doesn't seem right, where's the MKLDNN'ness
                    # lol
                    r = callback(
                        lambda: torch.empty_strided(
                            sizes, strides, dtype=t.dtype, device="meta"
                        )
                    )
                    if self.copy_data:
                        with torch.no_grad(), no_dispatch():
                            assert t.size is not None
                            assert t.stride is not None
                            assert _is_fake_tensor(r)
                            r.real_tensor = torch.empty_strided(
                                t.size, t.stride, dtype=t.dtype, device=t.device
                            )
                            assert t.data is not None
                            _safe_copy(r.real_tensor, t.data)
                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = True
                    if t.requires_grad and not is_leaf:
                        # pyrefly: ignore  # bad-argument-type
                        r = self._backward_error(r)
                elif t.is_functorch_wrapped:
                    if t.is_view:
                        from torch._dynamo.exc import unimplemented

                        unimplemented(
                            "view functorch tensors are not supported by meta conversion"
                        )

                    # Wraps a functorch tensor class (BatchedTensor, GradTrackingTensor)
                    # in a FakeTensor
                    def _to_fake_tensor(t: MetaTensorDesc) -> _TensorT:
                        # TODO: why aren't the recursive calls going to
                        # meta_tensor
                        r: _TensorT
                        if t.is_batchedtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            assert t.bdim is not None
                            ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            bdim = t.bdim
                            # You cannot create functorch tensors without
                            # having the ambient funtorch interpreter stack
                            # available, as the level refers to things in the
                            # stack
                            with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                t.functorch_stack
                            ):
                                r = self._checked_cast_tensor_t(
                                    _add_batch_dim(ft, bdim, lvl)
                                )
                        elif t.is_gradtrackingtensor:
                            assert t.unwrapped is not None
                            assert t.level is not None
                            disable_functorch = torch._C._DisableFuncTorch
                            with disable_functorch():
                                ft = _to_fake_tensor(t.unwrapped)
                            lvl = t.level
                            if lvl == GRAD_TENSOR_SENTINEL_VALUE:
                                r = ft
                            else:
                                with torch._functorch.pyfunctorch.temporarily_restore_interpreter_stack(
                                    t.functorch_stack
                                ):
                                    r = self._checked_cast_tensor_t(
                                        torch._C._functorch._wrap_for_grad(ft, lvl),
                                    )

                            is_leaf = t.is_leaf
                            if t.requires_grad and safe_is_leaf(r):
                                r.requires_grad = True
                            elif t.requires_grad and not is_leaf:
                                r = self._backward_error(r)
                        elif t.is_functional:
                            assert t.unwrapped is not None
                            assert t.current_level is not None
                            ft = self.meta_tensor(
                                t.unwrapped,
                                shape_env,
                                callback,
                                # NB: reuse these exactly, we treat the
                                # functional tensor as "invisible".
                                # TODO: Actually this all probably doesn't
                                # work, take a closer look.
                                source,
                                symbolic_context,
                            )
                            r = self._checked_cast_tensor_t(
                                _wrap_functional_tensor(ft, t.current_level),
                            )
                            # TODO: is_leaf/requires_grad?
                        else:
                            assert t.stride is not None

                            sizes = t.size
                            strides = t.stride
                            r = callback(
                                lambda: torch.empty_strided(
                                    sizes,
                                    strides,
                                    dtype=t.dtype,
                                    device="meta",
                                ),
                                # device="meta",
                            )
                            if self.copy_data:
                                with torch.no_grad(), no_dispatch():
                                    r.real_tensor = torch.empty_strided(  # type: ignore[attr-defined]
                                        t.size,
                                        t.stride,
                                        dtype=t.dtype,
                                        device=t.device,
                                    )
                                    assert t.data is not None
                                    _safe_copy(r.real_tensor, t.data)  # type: ignore[attr-defined]
                        # pyrefly: ignore  # bad-return
                        return r

                    r = _to_fake_tensor(t)

                elif t.is_functional and t.device.type not in ["xla", "lazy"]:
                    assert t.unwrapped is not None
                    assert not t.is_functorch_wrapped  # handled above
                    unwrapped = self.meta_tensor(
                        t.unwrapped,
                        shape_env,
                        callback,
                        source,
                        symbolic_context,
                    )
                    r = self._checked_cast_tensor_t(
                        torch._to_functional_tensor(unwrapped)
                    )
                    torch._mirror_autograd_meta_to(t.autograd_meta_from, r)  # type: ignore[attr-defined]

                elif t.is_view:
                    # Construct views in two steps: recursively meta-fy their
                    # base, and then create view(s) off that.  NB: doing it
                    # directly from storage is WRONG because this won't cause
                    # version counters to get shared.

                    assert t.base is not None

                    base_symbolic_context = None
                    if shape_env and symbolic_context is not None:
                        from torch.fx.experimental.symbolic_shapes import (
                            StatelessSymbolicContext,
                        )

                        assert isinstance(symbolic_context, StatelessSymbolicContext)
                        # NB: This should generally be set when the input is a view,
                        # but the exception right now is for fake-ifying grads, which is
                        # a work in progress.
                        if symbolic_context.view_base_context is not None:
                            base_symbolic_context = symbolic_context.view_base_context

                    base = self.meta_tensor(
                        t.base,
                        shape_env,
                        callback,
                        torch._dynamo.source.AttrSource(source, "_base"),
                        base_symbolic_context,
                    )

                    def is_c_of_r(
                        complex_dtype: torch.dtype, real_dtype: torch.dtype
                    ) -> bool:
                        return (
                            utils.is_complex_dtype(complex_dtype)
                            and utils.corresponding_real_dtype(complex_dtype)
                            == real_dtype
                        )

                    # In some situations, MetaConverter may be called in a
                    # context where autograd is disabled.  For the _is_view
                    # assert to pass, we have to setup the autograd view
                    # metadata anyway.  Do this by reenabling the
                    # ADInplaceOrView key.  This is kind of a hack.
                    old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView
                    )
                    torch._C._dispatch_tls_set_dispatch_key_excluded(
                        torch._C.DispatchKey.ADInplaceOrView, False
                    )
                    try:
                        if base.dtype == t.dtype:
                            pass
                        elif is_c_of_r(base.dtype, t.dtype):
                            base = self._checked_cast_tensor_t(torch.view_as_real(base))
                        elif is_c_of_r(t.dtype, base.dtype):
                            base = self._checked_cast_tensor_t(
                                torch.view_as_complex(base)
                            )
                        else:
                            # This is not guaranteed to succeed.  If it fails, it
                            # means there is another dtype-converting view function
                            # that hasn't been handled here
                            base = self._checked_cast_tensor_t(base.view(t.dtype))

                        # This is very tricky.  Naively, you might expect this
                        # to hold:
                        #
                        #   if t.requires_grad and not safe_is_leaf(t)
                        #       assert t._base.requires_grad
                        #
                        # But it's not true!  As you can see in the following
                        # program:
                        #
                        #   x = torch.zeros(4)
                        #   y = x.view(1, 4)
                        #   y.requires_grad = True
                        #   z = y.view(1, 1, 4)
                        #   assert z._base is x
                        #
                        # So we may have to do *two* views out of the base to
                        # recreate this situation.
                        if t.is_leaf:
                            # Leaf views that track view metadata are created by
                            # creating a view inside a no_grad block
                            with torch.no_grad():
                                r = view_from_base(base, t)
                            # As it's a leaf, we can directly assign requires_grad
                            r.requires_grad = t.requires_grad
                        else:
                            if t.base.requires_grad == t.requires_grad:
                                # Easy case, just run the view op
                                with torch.enable_grad():
                                    r = view_from_base(base, t)

                                # NB: We don't actually faithfully replicate
                                # autograd connectivity, but that doesn't matter
                                # today. See following for more info:
                                # https://gist.github.com/soulitzer/e03f015b314c3f5fcf80888c69390913
                            else:
                                # Obscure case.  Create a leaf view and give it the
                                # correct requires_grad, then do the final view.
                                # NB: Can't have a non-leaf without requiring grad!
                                assert t.requires_grad
                                with torch.no_grad(), enable_python_dispatcher():
                                    mid = self._checked_cast_tensor_t(
                                        base.view(base.shape)
                                    )
                                mid.requires_grad = t.requires_grad
                                with torch.enable_grad():
                                    r = view_from_base(mid, t)
                        # The CreationMeta influences whether or not inplace
                        # mutation is an error or not.  So we need to make
                        # sure we properly propagate this as well.
                        assert t.creation_meta is not None
                        torch._C._autograd._set_creation_meta(r, t.creation_meta)
                    finally:
                        torch._C._dispatch_tls_set_dispatch_key_excluded(
                            torch._C.DispatchKey.ADInplaceOrView, old_exclude
                        )

                    r.fake_device = t.device  # type: ignore[attr-defined]

                else:
                    is_leaf = t.is_leaf

                    # Graph-Break for wrapped tensors
                    if (
                        not (t.is_batchedtensor or t.is_gradtrackingtensor)
                        and t.is_functorch_wrapped
                    ) or t.is_legacy_batchedtensor:
                        # pyrefly: ignore  # bad-return
                        return NotImplemented

                    (
                        sizes,
                        strides,
                        storage_offset,
                    ) = sym_sizes_strides_storage_offset(t, source, symbolic_context)

                    # If we have a subclass that desugars into dense tensors,
                    # perform our callback on each inner tensor.
                    if t.is_traceable_wrapper_subclass:
                        r = empty_create_subclass(
                            t, outer_size=sizes, outer_stride=strides
                        )
                    else:
                        r = callback(
                            lambda: torch.empty_strided(
                                sizes,
                                strides,
                                dtype=t.dtype,
                                device="meta",
                            )
                        )
                        if self.copy_data:
                            with torch.no_grad(), no_dispatch():
                                assert t.size is not None
                                assert t.stride is not None
                                assert _is_fake_tensor(r)
                                r.real_tensor = torch.empty_strided(
                                    t.size, t.stride, dtype=t.dtype, device=t.device
                                )
                                _safe_copy(r.real_tensor, t.data)

                    assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                    if t.requires_grad:
                        r.requires_grad = t.requires_grad
                        if not is_leaf:
                            # Fake up some autograd history.
                            # Note: we *used* to call .clone() here to mock up some autograd history.
                            # This is bad for subclasses.
                            # Consider the case where you have a wrapper subclass that is contiguous,
                            # but its inner tensor is noncontiguous().
                            # .clone() (or other ops) will have the side effect of changing
                            # the metadata of the inner tensor.
                            # So instead, we now have a dedicated fn to set autograd history,
                            # without inadvertently changing other metadata.
                            # pyrefly: ignore  # bad-argument-type
                            r = self._backward_error(r)

                    s = t.storage
                    assert s is not None
                    if s.id not in self.storage_memo and (
                        r.is_nested
                        or (
                            r.stride() == strides
                            and r.storage_offset() == storage_offset
                        )
                    ):
                        # You're normal and happy, install the fresh storage into the memo
                        self.set_storage_memo(s, r.untyped_storage())
                        if self.copy_data:
                            assert _is_fake_tensor(r)
                            assert r.real_tensor is not None
                            _set_real_storage(
                                r.untyped_storage(), r.real_tensor.untyped_storage()
                            )
                    else:
                        # You're in crazy town; somehow you gave us a tensor
                        # that wasn't a view, but had nonzero storage offset,
                        # nontrivial strides (such that clone() couldn't
                        # preserve them), or already aliases with another
                        # tensor's storage.  The most typical way to end
                        # up here is with set_.  So use set_ to bludgeon this
                        # in.
                        r_s = self.meta_storage(s, callback=callback)
                        # NB: In principle, this should always work, but there
                        # is some subtle difference in the autograd metadata
                        # that means we will backprop the set_ call, even if
                        # r is declared as an input to grad.
                        # See https://github.com/pytorch/pytorch/issues/87956
                        # for the reproducer.
                        # NB: The in_kernel_invocation_manager here is necessary
                        # for fake tensor.  If we run the set_ call with fake
                        # tensor on, r will improperly report that it is NOT a
                        # meta tensor but a cpu tensor, and then the set_ call
                        # will fail due to device mismatch.  no_dispatch() is
                        # not enough, because the fake tensor will still claim
                        # to be a CPU tensor and you'll end up in the CPU
                        # kernel.  Arguably this is a hack; a cleaner way to
                        # solve this is to have a FakeStorage concept which
                        # would report it's CPU device--no problem now!  But
                        # this is difficult to do because we don't have storage
                        # subclasses.  Relevant test is
                        # DynamicShapesFunctionTests::test_add_dynamic_shapes in
                        # test/dynamo/test_dynamic_shapes.py
                        maybe_fake_mgr: AbstractContextManager[None] = (
                            contextlib.nullcontext()
                        )
                        from torch._subclasses.fake_tensor import (
                            in_kernel_invocation_manager,
                            maybe_get_fake_mode,
                        )

                        mb_fake_mode = maybe_get_fake_mode(r)
                        if mb_fake_mode is not None:
                            maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                        with torch.no_grad(), maybe_suppress():
                            with maybe_fake_mgr:
                                r.set_(r_s, storage_offset, sizes, strides)
                            if self.copy_data:
                                with torch.no_grad(), no_dispatch():
                                    assert _is_fake_tensor(r)
                                    assert r.real_tensor is not None
                                    assert t.stride is not None
                                    r.real_tensor.set_(
                                        _get_real_storage(r_s),
                                        t.storage_offset,
                                        t.size,
                                        t.stride,
                                    )

                if t.grad is not None:
                    from torch._dynamo.source import AttrSource

                    # TODO: Use a valid grad-specific symbolic context instead of recycling
                    # the one from t. This isn't correct if e.g. t._is_view() != t.grad._is_view().
                    # pyrefly: ignore  # unbound-name
                    r.grad = self.meta_tensor(
                        t.grad,
                        shape_env,
                        callback,
                        AttrSource(source, "grad"),
                        symbolic_context,
                    )
                # pyrefly: ignore  # unbound-name
                torch._C._set_conj(r, t.is_conj)
                # pyrefly: ignore  # unbound-name
                torch._C._set_neg(r, t.is_neg)
            # This can be skipped if necessary for performance reasons
            skip_leaf = (
                t.is_gradtrackingtensor and t.level == GRAD_TENSOR_SENTINEL_VALUE
            )
            # pyrefly: ignore  # unbound-name
            assert_metadata_eq(assert_eq, t, r, skip_symbolic=True, skip_leaf=skip_leaf)
            # Thanks to storage resizing, it's possible to end up with a tensor
            # that advertises a real size, but has a storage that actually has zero bytes.
            # Need to reflect this in the generated FakeTensor.
            from torch.fx.experimental.symbolic_shapes import guard_or_false

            if t.storage is not None and guard_or_false(t.storage.size == 0):
                # pyrefly: ignore  # unbound-name
                r.untyped_storage().resize_(0)

            if t.is_parameter:
                # pyrefly: ignore  # unbound-name
                r._is_param = True

            # See Note: [Creating symbolic nested int]
            if t.nested_int is not None:
                # pyrefly: ignore  # unbound-name
                assert _is_fake_tensor(r)
                # pyrefly: ignore  # unbound-name
                r.nested_int_memo = r.fake_mode.create_symbolic_nested_int(
                    nt_tensor_id=t.nested_int
                )

            # pyrefly: ignore  # bad-argument-type
            self.set_tensor_memo(t, r)

        return self._checked_get_tensor_memo(t)

    def __call__(
        self,
        t: torch.Tensor,
        shape_env: Optional[ShapeEnv] = None,
        *,
        callback: Optional[_MetaTensorCallback[_TensorT]] = None,
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
        # Controls whether or not we should dump the tensor metadata to structured logs
        # when source is not None.  Because we refakify after Dynamo is done,
        # we don't want to dump info again from AOTAutograd, it is redundant.
        trace: bool = True,
    ) -> _TensorT:
        callback_: _MetaTensorCallback[_TensorT]
        if callback is None:
            callback_ = self._identity_callable
        else:
            callback_ = callback
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        # Filter out cases we don't support
        # TODO: This can probably be simplified quite a bit
        if isinstance(t, torch.Tensor):
            if (
                # Lazy tensors are not supported.  Note that XLA is
                # implemented on top of lazy tensor, not excluded here; we
                # have some special handling for it; this is for XLA Dynamo
                # integration
                t.device.type == "lazy"
                or
                # Quantization is not supported
                t.is_quantized
                or
                # Views out of sparse tensors not currently supported (plain
                # sparse is supported htough)
                (t._is_view() and t._base is not None and t._base.is_sparse)
            ):
                self.miss += 1
                # pyrefly: ignore  # bad-return
                return NotImplemented
            else:
                self.hit += 1
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            # pyrefly: ignore  # bad-return
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss
            return t

        if source is None:
            trace = False

        # Describe the tensor.  NB: do NOT disable ambient modes, we may need
        # to query them when figuring out what to put in here
        t_desc = self.describer.describe_tensor(t, trace=trace)

        if trace:
            assert source is not None
            trace_structured(
                "describe_source",
                metadata_fn=lambda: {
                    "describer_id": self.describer.id,
                    "id": t_desc.id,
                    "source": source.name(),
                },
            )

        # Do the meta-fication.  Here, we disable all the ambient modes, to
        # better simulate what would be like to re-fakeify from a fresh
        # process
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(torch._dispatch.python.suspend_functionalization())
            st = peek_interpreter_stack()
            if st is not None:
                exit_stack.enter_context(
                    torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
                )

            r = self.meta_tensor(
                t_desc,
                shape_env,
                callback_,
                source,
                symbolic_context,
            )

        if type(t) is torch.nn.Parameter:
            # NB: Cannot directly use Parameter constructor
            # because that would force a detach, not desirable
            r._is_param = True

        # TODO: return the description for later
        return r


import torch._prims_common as utils
