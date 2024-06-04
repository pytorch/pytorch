# mypy: allow-untyped-defs
import contextlib
import functools
import logging
import os
import traceback
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from weakref import ReferenceType

import torch
import torch._custom_op
import torch._logging
from torch._C._functorch import is_functorch_wrapped_tensor, is_legacy_batchedtensor

from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import suggest_memory_format
from torch._subclasses.meta_utils import (
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import PyTree, tree_map, tree_map_
from torch.utils._stats import count
from torch.utils._traceback import CapturedTraceback

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.types import _bool


class _Unassigned:
    pass


def _is_plain_tensor(t):
    return (
        type(t) is torch.Tensor
        and t.layout == torch.strided
        and not (
            t.is_sparse
            or t.is_nested
            or is_functorch_wrapped_tensor(t)
            or is_legacy_batchedtensor(t)
            or torch._is_functional_tensor(t)
        )
    )


_UNASSIGNED = _Unassigned()

DimList = List

log = logging.getLogger(__name__)

# TODO: Hack to unblock https://github.com/pytorch/pytorch/pull/108186
# Proper fix tracked by https://github.com/pytorch/pytorch/issues/120105
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        import logging as not_implemented_log
    else:
        raise e

pytree = torch.utils._pytree
T = TypeVar("T")
TensorWeakRef = Any

aten = torch._ops.ops.aten

CONSTANT_NUMEL_LIMIT = 1

RECURSION_COUNT = 0


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:
    def __init__(self):
        global RECURSION_COUNT
        RECURSION_COUNT += 1

    def __del__(self):
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


def ordered_set(*items):
    return dict.fromkeys(items, True)


@contextlib.contextmanager
def unset_fake_temporarily():
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


def is_fake(x):
    if isinstance(x, FakeTensor):
        return True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)
        flattened_tensors = [getattr(x, attr) for attr in attrs]
        # need to recurse because we could have nested subclasses
        all_fake = all(is_fake(x) for x in flattened_tensors)
        any_fake = any(is_fake(x) for x in flattened_tensors)
        assert all_fake == any_fake, "got mixed fake and real tensors!"
        return all_fake
    elif isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)
        return is_fake(unwrapped)
    elif isinstance(x, torch.Tensor) and is_functorch_wrapped_tensor(x):
        unwrapped = torch._C._functorch.get_unwrapped(x)
        return is_fake(unwrapped)
    return False


def maybe_get_fake_mode(t):
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
    elif isinstance(t, torch.Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)
        return maybe_get_fake_mode(unwrapped)
    elif isinstance(t, torch.Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)
        return maybe_get_fake_mode(unwrapped)
    return None


@functools.lru_cache(None)
def get_schema_info(func):
    return torch._C._SchemaInfo(func._schema)  # type: ignore[attr-defined]


# many of the decompositions registered to torch/_prims do not at the moment model
# aliasing or strides, so as an incremental step, just enable the decompositions in
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them
@functools.lru_cache(None)
def torch_decomp_decompositions(func):
    from torch._decomp import decomposition_table

    decompositions = torch._decomp.decompositions
    # Note that the function in the decomposition table might be
    # different from the one in the module because of the difference
    # in out handling in aten API and torch public API
    return decomposition_table[func].__module__.startswith(
        "torch._decomp"
    ) and decomposition_table[func].__name__ in dir(decompositions)


def tree_flatten_only(ty: Type[T], tree: PyTree):
    flat_vals = pytree.tree_leaves(tree)
    return [elem for elem in flat_vals if isinstance(elem, ty)]


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter:
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: Dict[StorageWeakRef, List[ReferenceType]]
    export: bool

    def __init__(self, *, copy_data=False, export=False):
        self.meta_converter = MetaConverter(copy_data=copy_data)
        self.export = export

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor):
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

    def invalidate_constant_aliases(self, tensor):
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

    def _get_memo(self, t):
        tid = self.meta_converter.describer.lookup_tensor.get(t)
        if tid is None:
            return None
        return self.tensor_memo.get(tid)

    def set_tensor_memo(self, t, v):
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
        fake_mode,
        t,
        make_constant=False,
        shape_env=None,
        *,
        source=None,
        symbolic_context=None,
        trace=True,
    ):
        # see note [Tensor Fakification and Symbol Caching]
        if not symbolic_context and not source and shape_env:
            if tracing_context := torch._guards.TracingContext.try_get():
                if t in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[t]
                    source = symbolic_context.tensor_source

        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        existing_device = t.device
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        if type(t) is torch.nn.Parameter:
            assert not make_constant

        def mk_fake_tensor(make_meta_t):
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
                    make_meta_t(),
                    existing_device,
                    # TODO: callback might be used in recursive contexts, in
                    # which case using t is wrong!  BUG!
                    constant=t if make_constant else None,
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
            # Peephole strip out unnecessary torch.as_tensor(x).item()
            if isinstance(source, FloatTensorSource):
                item_source = source.base
            else:
                item_source = CallMethodItemSource(source)
            symbol = shape_env.create_unspecified_symbol(
                value,
                source=item_source,
                dynamic_dim=DimDynamic.DYNAMIC,
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
    def from_meta_and_device(self, fake_mode, t, device):
        assert (
            t.device.type == "meta"
        ), f"tensor's device must be `meta`, got {t.device.type} instead"
        # This is a bit abusive (this is not the "real" tensor) but whatever,
        # the meta tensor should be fresh so there's no way to get it wrong
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out


@functools.lru_cache(None)
def init_cuda_context():
    # Backward will error with cuda Fake Tensors if no cuda tensors have been initialized first
    if torch.cuda.is_available():
        torch.empty(1, device="cuda") if torch.version.hip is None else torch.zeros(
            1, device="cuda"
        )


@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode):
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
def should_allow_numbers_as_tensors(func: OpOverload):
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


class FakeTensorConfig:
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", "0") == "1"


# This memorizes the unbacked SymInt representing quantities like the number
# of nonzero elements in this tensor.  There is one instance of the descriptor
# per particular quantity to memoize.
#
# Memoization is helpful if you do something like x[mask] and y[mask];
# mask.nonzero() gets repeatedly called and should give a consistent unbacked
# SymInt.  It needs to be invalidated in the same way constant is.
#
# Making this a descriptor may seem overly fancy, but actually it's the most
# convenient way to make sure we have access to FakeTensor during access,
# which is required for testing version counter and epoch validity
class UnbackedMemoDescriptor:
    _name: str

    def __set_name__(self, owner, name):
        self._name = name

    def _memo(self, obj):
        return f"_{self._name}"

    def _memo_vc(self, obj):
        return f"_{self._name}_vc"

    # When we retrace, we need to invalidate all the memos so that we can
    # accurately identify the first time unbacked SymInts are allocated.
    # This is only relevant for inputs; for intermediates, they will get fresh
    # fake tensors so you won't have a memo anyway
    def _memo_epoch(self, obj):
        return f"_{self._name}_epoch"

    def __get__(self, obj: "FakeTensor", objtype=None):
        if (r := getattr(obj, self._memo(obj))) is None:
            return None
        # Version counter based tracking isn't 100% sound but it's close
        # enough
        if (
            getattr(obj, self._memo_vc(obj)) != obj._version
            or getattr(obj, self._memo_epoch(obj)) != obj.fake_mode.epoch
        ):
            setattr(obj, self._memo(obj), None)
            return None
        return r

    def __set__(self, obj, value):
        if value is None:
            setattr(obj, self._memo(obj), None)
            setattr(obj, self._memo_vc(obj), None)
            setattr(obj, self._memo_epoch(obj), None)
        elif not torch.is_inference_mode_enabled():
            setattr(obj, self._memo(obj), value)
            setattr(obj, self._memo_vc(obj), obj._version)
            setattr(obj, self._memo_epoch(obj), obj.fake_mode.epoch)


class FakeTensor(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    fake_device: torch.device
    fake_mode: "FakeTensorMode"
    constant: Optional[torch.Tensor]
    real_tensor: Optional[torch.Tensor]

    # TODO: Generalize this as needed, e.g., into a trie of memos, if
    # you do something like x[0].item()  (x[0] is fresh each time, so
    # memo mechanism here won't work)
    nonzero_memo = UnbackedMemoDescriptor()
    item_memo = UnbackedMemoDescriptor()
    unique_memo = UnbackedMemoDescriptor()

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    def device(self):
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch__dispatch__ sits below
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
    def names(self):
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    @staticmethod
    def __new__(cls, fake_mode, elem, device, constant=None, real_tensor=None):
        self = torch.Tensor._make_subclass(
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
        if device.type == "cuda":
            init_cuda_context()

        if (
            device.type
            in ["cuda", "hpu", "xpu", torch._C._get_privateuse1_backend_name()]
            and device.index is None
        ):
            if getattr(torch, device.type).is_initialized():
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        self.fake_device = device  # type: ignore[attr-defined]
        self.fake_mode = fake_mode  # type: ignore[attr-defined]
        self.constant = constant  # type: ignore[attr-defined]
        assert not isinstance(real_tensor, FakeTensor)
        self.real_tensor = real_tensor  # type: ignore[attr-defined]
        self.nonzero_memo = None
        self.item_memo = None
        self.unique_memo = None

        if FakeTensorConfig.debug:
            self._debug_trace = CapturedTraceback.extract()  # type: ignore[attr-defined]
        return self

    # In some circumstances, a conventional torch.Tensor constructor
    # will get rewritten to call into FakeTensor.  We must provide an
    # __init__ method that can accept the Python interpreters initialization
    # in such a situation; we must also be able to handle direct fake
    # tensor construction via FakeTensor().
    #
    # In particular, the __init__ call will look funny in the following case:
    #
    #   with FakeTensorMode():
    #       x = torch.Tensor([1, 2, 3])
    #
    # this desugars into:
    #
    #   with FakeTensorMode():
    #       x = torch.Tensor.__new__([1, 2, 3])
    #       # NB: x is a fake tensor, because of the mode!
    #       x.__init__([1, 2, 3])  # not the normal fake tensor args!
    #
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func == torch.ops.prim.device.default:
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
            t for t in types if not issubclass(t, FakeTensor) and t is not torch.Tensor
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

        with fake_mode:  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(func, flat_args) -> Tuple[torch.device, bool]:
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        def cpu_zero_dim(t):
            return t.device.type == "cpu" and t.dim() == 0

        def merge_devices(t):
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

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
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

    # We must handle tolist in a special way for FakeTensors here in the case
    # where tolist is called from torch dispatch for tensor subclasses.
    # Ordinarily, if a program calls .tolist compiling still works because there is
    # special handling in dynamo, but for tensor subclasses if .tolist is called
    # inside torch dispatch, the .tolist call may be directly on a FakeTensor.
    # This would result in an error since wrapper subclasses don't have storage.
    # To avoid this, we handle the FakeTensor case by (1) specializing on the size
    # of the tensor to create the output Python list, and (2) creating unbacked
    # symints for each element of the list.
    def tolist(self):
        assert self.dim() == 1, "NYI for higher dims"
        shape_env = self.fake_mode.shape_env
        out = []
        # Specialize on the length of the list
        for _ in range(self.shape[0]):
            s = shape_env.create_unbacked_symint()
            # max value?
            torch._check_is_size(s)
            torch._check(s >= 2)
            out.append(s)
        return out


@dataclass(frozen=True)
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    dtype: torch.dtype
    shape: torch.Size
    stride: Tuple[Any, ...]
    device: torch.device
    layout: torch.layout
    memory_format: Optional[torch.memory_format]
    storage_offset: int
    storage_bytes: Optional[int]
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool  # read: is sparse COO
    is_coalesced: Optional[bool]
    dense_dim: Optional[int]
    sparse_dim: Optional[int]


def extract_tensor_metadata(t: torch.Tensor) -> "TensorMetadata":
    """
    Extract the TensorMetadata of a tensor.
    """
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    if is_sparse_any(t) or not t.is_contiguous(memory_format=memory_format):
        memory_format = None

    return TensorMetadata(
        dtype=t.dtype,
        shape=t.shape,
        stride=t.stride() if t.layout == torch.strided else (),
        device=t.device,
        layout=t.layout,
        memory_format=memory_format,
        storage_offset=t.storage_offset(),
        # Only set storage_bytes for tensors that have storage (not sparse)
        storage_bytes=t.untyped_storage().nbytes() if not t.is_sparse else None,
        requires_grad=t.requires_grad,
        is_quantized=t.is_quantized,
        is_conj=t.is_conj(),
        is_neg=t.is_neg(),
        is_inference=t.is_inference(),
        is_sparse=t.is_sparse,
        is_coalesced=t.is_coalesced() if t.is_sparse else None,
        dense_dim=t.dense_dim() if t.is_sparse else None,
        sparse_dim=t.sparse_dim() if t.is_sparse else None,
    )


class _DispatchCacheKey(list):
    """
    Key for the FakeTensor dispatch cache. Inspired by (copied from)
    _HashedSeq from the functools.lru_cache implementation.
    """

    __slots__ = "hashvalue"  # noqa: PLC0205

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


@dataclass(frozen=True)
class _DispatchCacheEntry:
    """
    Entry type for the FakeTensor dispatch cache. Accounts for two possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a given
    index. 2) We need to synthesize a new FakeTensor given tensor metadata. For view
    ops, we further capture the index of the arg to alias.
    """

    inplace_idx: Optional[int] = None
    metadata: Optional[TensorMetadata] = None
    view_idx: Optional[int] = None


@dataclass(frozen=True)
class _BypassDispatchCache(Exception):
    """
    Signals cases that should skip FakeTensor caching.
    """

    reason: str


@dataclass(frozen=True)
class DispatchCacheInfo:
    """
    Information about the state of the FakeTensor dispatch cache.
    """

    hits: int
    misses: int
    bypasses: Dict[str, int]
    size: int


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will no be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.


class FakeTensorMode(TorchDispatchMode):
    cache: Dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: Dict[str, int] = defaultdict(int)
    # Every time you retrace using the same fake tensor mode, you should
    # advance the epoch so we don't reuse unbacked memos
    epoch: int = 0
    in_kernel_invocation: bool = False

    def __init__(
        self,
        *,
        allow_fallback_kernels=True,
        allow_non_fake_inputs=False,
        shape_env=None,
        static_shapes=None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        export=False,
    ):
        log.debug("create_mode 0x%x", id(self))
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
        self.cache_enabled = (
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
        self.enter_stack: List[
            Tuple[bool, Optional[TorchDispatchMode], Optional[_bool]]
        ] = []

        self.shape_env: ShapeEnv = shape_env

        self._stack_trace = traceback.extract_stack()
        self._stack = None

        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE

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
    def is_our_fake(self, t):
        return isinstance(t, FakeTensor) and t.fake_mode is self

    # If we should avoid device init. This changes the behavior of various APIs:
    # - We avoid constant-prop on Tensors with ops that move them to another device
    # - We change the torch.tensor ctor contract to never materialize
    #   tensors on device
    #   (see NOTE: [torch.tensor, lift_fresh, and device movement])
    @property
    def avoid_device_init(self):
        return not torch.cuda.is_available()

    @property
    def stack(self):
        if self._stack is None:
            self._stack = "".join(traceback.format_list(self._stack_trace))
        return self._stack

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
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
    def __enter__(self):
        prev_only_lift_cpu_tensors = None
        if self.avoid_device_init:
            # See NOTE: [torch.tensor, lift_fresh, and device movement]
            prev_only_lift_cpu_tensors = torch._C._only_lift_cpu_tensors()
            torch._C._set_only_lift_cpu_tensors(True)
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

    def __exit__(self, a, b, c):
        (
            live,
            maybe_prev_fake_mode,
            maybe_prev_only_lift_cpu_tensors,
        ) = self.enter_stack.pop()
        if live:
            out = super().__exit__(a, b, c)
            # Re-enable the previous fake mode, if there was one.
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)
            if maybe_prev_only_lift_cpu_tensors is not None:
                torch._C._set_only_lift_cpu_tensors(maybe_prev_only_lift_cpu_tensors)

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
    def cache_clear(cls):
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
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
        output: Union[FakeTensor, _Unassigned] = _UNASSIGNED
        try:
            key = self._cache_key(func, args, kwargs)
            entry = FakeTensorMode.cache.get(key, None)
            if entry is not None:
                output = self._output_from_cache_entry(entry, func, args)
                FakeTensorMode.cache_hits += 1
                if self.cache_crosscheck_enabled:
                    # For debugging / testing: Validate that the output synthesized
                    # from the cache matches the output created by normal dispatch.
                    self._crosscheck_cache_output(output, func, types, args, kwargs)
            else:
                self._validate_cache_key(func, args, kwargs)
                output = self._dispatch_impl(func, types, args, kwargs)
                entry = self._make_cache_entry(key, func, args, kwargs, output)
                FakeTensorMode.cache[key] = entry
                FakeTensorMode.cache_misses += 1
        except _BypassDispatchCache as e:
            FakeTensorMode.cache_bypasses[e.reason] += 1

        if output is _UNASSIGNED:
            output = self._dispatch_impl(func, types, args, kwargs)

        return output

    def _cache_key(
        self,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> _DispatchCacheKey:
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
        key_values = (
            func,
            # Translate any FakeTensor args to metadata.
            self._prep_args_for_hash(args) if args else (),
            self._prep_args_for_hash(kwargs) if kwargs else (),
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
        )
        return _DispatchCacheKey(key_values)

    def _validate_cache_key(
        self,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
        # Avoid caching for any ops that would require a more sophisticated
        # caching implementation, e.g., data dependent ops or ops that modify
        # the inputs.
        if torch.Tag.data_dependent_output in func.tags:
            raise _BypassDispatchCache("data dependent output")

        if torch.Tag.dynamic_output_shape in func.tags:
            raise _BypassDispatchCache("dynamic output shape")

        if torch.Tag.inplace_view in func.tags:
            raise _BypassDispatchCache("inplace view")

        if func == aten._unsafe_view.default:
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

    def _prep_args_for_hash(self, args: Any) -> Any:
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
        if isinstance(args, dict):
            args = list(args.keys()) + list(args.values())

        result: List[Any] = []
        for arg in args:
            if isinstance(arg, FakeTensor):
                if not self.is_our_fake(arg):
                    raise _BypassDispatchCache("not our fake")
                if arg._has_symbolic_sizes_strides:
                    raise _BypassDispatchCache("symbolic shape")
                if arg.constant is not None:
                    raise _BypassDispatchCache("constant attribute")
                if arg.is_sparse:
                    raise _BypassDispatchCache("sparse tensor")
                if arg.layout in [
                    torch.sparse_csr,
                    torch.sparse_csc,
                    torch.sparse_bsr,
                    torch.sparse_bsc,
                ]:
                    # Does this subsume arg.is_sparse?
                    raise _BypassDispatchCache("sparse tensor layout")
                # sparse tensors don't have storage, so check is after
                if isinstance(arg.untyped_storage().nbytes(), torch.SymInt):
                    raise _BypassDispatchCache("symbolic nbytes")
                if is_sparse_compressed(arg):
                    raise _BypassDispatchCache("sparse compressed tensor")
                result.append(extract_tensor_metadata(arg))
            elif isinstance(arg, torch.Tensor):
                raise _BypassDispatchCache("non-fake tensor")
            elif isinstance(arg, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                raise _BypassDispatchCache("symbolic shape")
            elif isinstance(arg, (list, tuple, dict)):
                result.extend(self._prep_args_for_hash(arg))
            else:
                # It's important to capture the type of the arg since, e.g., 1 and 1.0
                # hash to the same value, but can produce different dtypes for the
                # output tensor.
                result.append((type(arg), arg))

        return tuple(result)

    def _make_cache_entry(
        self,
        key: _DispatchCacheKey,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: FakeTensor,
    ) -> _DispatchCacheEntry:
        """
        Make a cache entry object for the given 'output' Tensor. Raises
        _BypassDispatchCache if the output tensor has characteristics that
        prevent caching it.
        """
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

        # If this is an in-place op, the entry records which input arg is aliased.
        for idx in range(len(args)):
            if id(args[idx]) == id(output):
                return _DispatchCacheEntry(
                    inplace_idx=idx, metadata=None, view_idx=None
                )

        # Otherwise, create an entry that records the output tensor's metadata.
        view_idx = None
        if func.is_view:
            idxs = [i for i, t in enumerate(args) if isinstance(t, torch.Tensor)]
            assert len(idxs) == 1
            view_idx = idxs[0]

        metadata = extract_tensor_metadata(output)
        entry = _DispatchCacheEntry(
            inplace_idx=None, metadata=metadata, view_idx=view_idx
        )

        # N.B.: Some checks for bypassing the cache would be performed on the
        # output tensor synthesized from the cached metadata. As an optimization,
        # we can synthesize a tensor here and do the checks on that instance.
        # This approach keeps the (more frequent) cache-hit path as lightweight
        # as possible.
        synth_output = self._output_from_cache_entry(entry, func, args)

        # Make sure the dispatch_key_set from the synthesized output tensor will
        # be the same.
        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

        return entry

    def _output_from_cache_entry(
        self, entry: _DispatchCacheEntry, func: OpOverload, args: Tuple[Any, ...]
    ) -> FakeTensor:
        """
        Create a new FakeTensor from the cache entry.
        """
        if entry.inplace_idx is not None:
            # This is an in-place op; return the aliased arg.
            return args[entry.inplace_idx]

        # Synthesize a new FakeTensor with the cached metadata.
        metadata = entry.metadata
        assert metadata and not metadata.is_sparse

        empty = torch.empty_strided(
            metadata.shape,
            metadata.stride,
            dtype=metadata.dtype,
            layout=metadata.layout,
            device="meta",
            requires_grad=metadata.requires_grad,
        )

        if metadata.is_conj:
            torch._C._set_conj(empty, True)
        if metadata.is_neg:
            torch._C._set_neg(empty, True)

        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if self.shape_env is not None:
            maybe_suppress = self.shape_env.suppress_guards

        if func.is_view:
            # For view ops, the storage should be the same as the tensor input.
            storage = args[cast(int, entry.view_idx)].untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        elif metadata.storage_offset != 0:
            storage = empty.untyped_storage()
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        if metadata.storage_bytes == 0:
            empty.untyped_storage().resize_(0)

        return FakeTensor(self, empty, metadata.device)

    def _crosscheck_cache_output(
        self,
        output: FakeTensor,
        func: OpOverload,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """
        try:
            true_output = self._dispatch_impl(func, types, args, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}: Dispatch raised={e}"
            ) from e
        try:
            assert_metadata_eq(assert_eq, true_output, output)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}"
            ) from e

    def dispatch(self, func, types, args=(), kwargs=None):
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
            incr = IncrementRecursionCount()

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

    def _dispatch_impl(self, func, types, args, kwargs) -> FakeTensor:
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        flat_arg_fake_tensors = [t for t in flat_args if self.is_our_fake(t)]
        has_symbolic_sizes = any(
            i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors
        ) or any(isinstance(a, torch.SymInt) for a in flat_args)

        converter = self.fake_tensor_converter

        is_lift_func = func in self.lift_fns

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
        ):
            assert all(
                t.constant is not None for t in flat_arg_fake_tensors
            ), f"{func} should not have fake inputs without constants"
            const_flat_args = [
                a.constant if self.is_our_fake(a) else a for a in flat_args
            ]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            out = func(*const_args, **const_kwargs)
            if type(out) is torch.Tensor and self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                # NB: no_dispatch() here is VERY DANGEROUS (like, segfault
                # dangerous) if this is actually a wrapper subclass tensor,
                # therefore the exact type test above
                with no_dispatch():
                    out = out.clone()
                return converter.from_real_tensor(self, out, make_constant=True)

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

        # if we are in the dispatch mode, we will enter this function even if the inputs
        # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
        # and just support constructors.

        # this is generated from torch.tensor(), which does not use the
        # dispatcher, to allow wrapper subclasses to wrap the new tensor
        if is_lift_func:
            assert len(kwargs) == 0 and len(args) == 1, f"{args} {kwargs}"

            if type(args[0]) is torch.Tensor:
                return converter.from_real_tensor(self, args[0])

        # If we are trying to avoid device init, then we need to avoid constant
        # prop on constant tensors for ops that change devices.
        avoiding_device_init = False
        if self.avoid_device_init:
            if (
                func == torch.ops.aten._to_copy.default
                and "device" in kwargs
                and kwargs["device"] != "cpu"
            ):
                avoiding_device_init = True
            if func == torch.ops.prims.device_put.default:
                avoiding_device_init = True

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
        # Whenver a constant is written to but with inputs that cannot be evaluated
        # statically, such as random_(), we invalidate all constants that alias the input
        # We will rely on functionalization for use of fake tensors constants as persistent
        # objects on an FX Graph.

        # We dispatch size/stride/numel on the FakeTensor not its constant, so bail on inplace_view
        all_constant = all(e.constant is not None for e in flat_arg_fake_tensors)
        if (
            torch.Tag.nondeterministic_seeded not in func.tags
            and torch.Tag.inplace_view not in func.tags
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
            and not avoiding_device_init
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
            flat_out_tensors = [t for t in flat_out if isinstance(t, torch.Tensor)]
            all_constant = all(self.may_turn_const(t) for t in flat_out_tensors)

            if all_constant:
                return pytree.tree_map_only(
                    torch.Tensor,
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
        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)

        def maybe_to_real_tensor(t):
            if isinstance(t, FakeTensor):
                return t.real_tensor
            elif isinstance(t, SymTypes):
                return t.node.pytype(
                    t.node.expr.xreplace(self.shape_env.var_to_val).xreplace(
                        self.shape_env.unbacked_var_to_val
                    )
                )
            else:
                return t

        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            free_unbacked_symbols,
            SymTypes,
        )

        nil = object()

        real_out = nil
        if (
            self.propagate_real_tensors
            and all(e.real_tensor is not None for e in flat_arg_fake_tensors)
            # TODO: Handle SymFloat/SymBool
            and not any(
                (
                    isinstance(a, torch.SymInt)
                    and (syms := free_unbacked_symbols(a))
                    and any(s not in self.shape_env.unbacked_var_to_val for s in syms)
                )
                for a in flat_args
            )
        ):
            real_flat_args = [maybe_to_real_tensor(a) for a in flat_args]
            real_args, real_kwargs = pytree.tree_unflatten(real_flat_args, args_spec)
            real_out = func(*real_args, **real_kwargs)
        elif self.propagate_real_tensors:
            # This can happen occasionally legitimately, specifically when you
            # are inside the meta of a data dependent operation and you create
            # a tensor on an unbacked SymInt; at this point in time we don't
            # know what the unbacked SymInt is, but we will know later.
            # However, if there's a bug in the condition above, this condition
            # will also trigger.
            log.debug(
                "propagate_real_tensors skipped %s(%s, %s) %s",
                func,
                flat_arg_fake_tensors,
                flat_args,
                self.shape_env.unbacked_var_to_val if self.shape_env else None,
            )

        def maybe_propagate_real_tensors(fake_out):
            import sympy

            def go(t, real_t):
                if isinstance(t, FakeTensor):
                    # NB: unconditionally overwrite
                    t.real_tensor = real_t
                elif isinstance(t, SymTypes) and free_unbacked_symbols(t):
                    if isinstance(t.node.expr, sympy.Symbol):
                        self.shape_env.set_unbacked_var_to_val(t.node.expr, real_t)

            if real_out is not nil:
                tree_map_(go, fake_out, real_out)

                # If a data-dependent op is used in a decomposition, we
                # may need to get the unbacked settings "early"
                # TODO: Is this really needed?
                compute_unbacked_bindings(self.shape_env, fake_out, peek=True)

            return fake_out

        # Try for fastpath
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return maybe_propagate_real_tensors(fast_impl(self, *args, **kwargs))

        # If there's a Python meta, prefer that over the decomposition
        from torch._decomp import meta_table as meta_table

        if func not in meta_table and not self.cpp_meta_supports_symint(func):
            from torch._decomp import decomposition_table

            # Prefer Python decompositions over C++ ones
            if func in decomposition_table and (
                has_symbolic_sizes
                or (
                    # TODO: Remove these exclusions, so that we can remove
                    # this leg entirely
                    torch_decomp_decompositions(func)
                    and all(not e.is_sparse for e in flat_arg_fake_tensors)
                )
            ):
                with self:
                    return decomposition_table[func](*args, **kwargs)

            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

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

        # Users can register FakeTensor rules for custom operators
        # Call them if they exist.
        maybe_abstract_impl = torch._library.simple_registry.singleton.find(
            func.name()
        ).abstract_impl.kernel
        if maybe_abstract_impl:
            ctx = torch._library.abstract_impl.AbstractImplCtx(self, func)
            with torch._library.abstract_impl.set_ctx_getter(lambda: ctx), self:
                result = maybe_abstract_impl(*args, **kwargs)
                return maybe_propagate_real_tensors(result)

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations_checks:
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out != NotImplemented:
                    return maybe_propagate_real_tensors(op_impl_out)

        def maybe_run_unsafe_fallback(error=None):
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
            return maybe_propagate_real_tensors(maybe_run_unsafe_fallback())

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

    def can_run_unsafe_fallback(self, func: OpOverload):
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
        self, func, converter, flat_args, args_spec
    ):
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors: List[Any] = []

        def validate(x):
            if not isinstance(x, torch.Tensor):
                return x

            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if torch.Tag.inplace_view in func.tags:
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                if not self.allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError("Mixing fake modes NYI")
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise AssertionError(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                x = converter.from_real_tensor(self, x)

            flat_arg_fake_tensors.append(x)
            return x

        validated_args = [validate(a) for a in flat_args]
        return validated_args, flat_arg_fake_tensors

    def wrap_meta_outputs_with_default_device_logic(self, r, func, flat_args, device):
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None
        has_scalar_only_inputs = False

        def wrap(e):
            nonlocal common_device
            nonlocal has_scalar_only_inputs

            if not isinstance(e, torch.Tensor):
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
                return e
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
                return e

        return tree_map(wrap, r)

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

    def cpp_meta_supports_symint(self, func):
        if torch.Tag.view_copy in func.tags:
            return True
        return func in self._cpp_meta_supports_symint

    lift_fns = ordered_set(aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t):
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT
            and not t.is_sparse
            and not self.is_our_fake(t)
            and not t.device.type == "meta"
        )

    def invalidate_written_to_constants(
        self, func, flat_arg_fake_tensors, args, kwargs
    ):
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        schema_info = get_schema_info(func)
        if any_constant and schema_info.is_mutable():
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
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
        tensor,
        *,
        static_shapes=None,
        source: Optional[Source] = None,
        symbolic_context=None,
        trace=True,
    ):
        shape_env: Optional[ShapeEnv] = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert (
                symbolic_context is None
            ), "cannot set both static_shapes and symbolic_context"
            shape_env = None
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )


# NB: returns fake tensors
def run_fallback_kernel(
    fake_mode, func, flat_args, args_spec, orig_not_implemented_exception
):
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e):
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

    tensor_impls = set()
    storages = set()

    for e in flat_args:
        if isinstance(e, torch.Tensor):
            if not e.is_sparse:
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl

    def map_out(e):
        if id(e) not in inp_impls and (
            isinstance(e, torch.Tensor)
            and not e.is_sparse
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, torch.Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    return pytree.tree_map(map_out, r)


# Just for use to allow copying a module to fake tensors,
# does not apply elsewhere
class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode):
        self.fake_mode = fake_mode

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # clone will get called in Parameter deepcopy
        if func == torch._C.TensorBase.clone:
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func == torch.Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor, memo = args

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


def _device_handler(args):
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
def _check_for_subclass(flat_args):
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x):
    return (
        not isinstance(x, FakeTensor)
        and isinstance(x, torch.Tensor)
        and type(x) is not torch.Tensor
        and type(x) is not torch.nn.Parameter
    )


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(int(s) for s in args[0].size()),
    torch.ops.aten.stride.default: lambda args: tuple(int(s) for s in args[0].stride()),
    torch.ops.aten.storage_offset.default: lambda args: int(args[0].storage_offset()),
}

_DISPATCH_HANDLE_DIRECTLY = ordered_set(
    torch.ops.aten.is_coalesced.default,
    torch.ops.aten.dense_dim.default,
    torch.ops.aten.sparse_dim.default,
)

from torch._subclasses.fake_impls import (  # noqa: F401
    _device_not_kwarg_ops,  # noqa: F401
    _is_tensor_constructor,  # noqa: F401
    _like_tensor_constructors,  # noqa: F401
    contains_tensor_types,  # noqa: F401
    get_fast_op_impls,
    has_meta,
    op_implementations_checks,
    stride_incorrect_op,
)
