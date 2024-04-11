import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, Set, Type, Union

import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
from torch.utils._python_dispatch import TorchDispatchMode

# Query `hasattr` only once.

_SET_GLOBAL_FLAGS = hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags")


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    if not _SET_GLOBAL_FLAGS:
        yield
        return
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)


class OperatorBase:
    """
    Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator
    (which represents Python-only operators that are unrepresentable in TorchScript).
    """

    def __init__(self):
        # The dispatch cache precomputes a mapping of dispatch key that the
        # dispatcher wants to dispatch to, to an actual implementation of the
        # dispatch key.  Confusingly, the actual implementation could *also* be a
        # dispatch key, but in this case, this refers to the C++ kernel that
        # was registered to some dispatch key.  Aliases are permitted in the
        # latter but not the former; for example, you might lookup the
        # entry for AutogradCPU, and this maps you to the Autograd key for
        # the generic autograd kernel that works for all devices.  Since this
        # is the Python dispatcher, you can also put an arbitrary Python
        # callable to call instead.  This handler gets precisely the
        # args/kwargs that the operator was __call__'ed with.
        # NB: This name is hard-coded in torch/csrc/autograd/python_variable.cpp
        # for use with OpOverload; cache lookup is done entirely from C++
        # for speed.
        # TODO: The cache is NOT currently used by HigherOrderOperator, but it should!
        self._dispatch_cache: Dict[
            torch._C.DispatchKey, Union[torch._C.DispatchKey, Callable[..., Any]]
        ] = {}

        # This table allows you to override the behavior of a particular
        # dispatch key to call a custom Python function, rather than the
        # ordinary C++ configured behavior.  This is the raison d'etre of
        # Python dispatcher: to let you program the dispatcher from Python
        # in case you need something unusual, and don't want to clobber
        # the existing registrations using the Python operator registration
        # API.
        self.py_kernels: Dict[torch._C.DispatchKey, Callable[..., Any]] = {}

        # This table allows you to override the behavior of a particular
        # operator for a particular TorchDispatchMode.  In practice,
        # we are using this mostly for ProxyTensorMode.  Modes can be
        # thought of as an open world extension of dispatch keys, so it
        # makes sense that you should be able to register them, the same
        # way you can register dispatch keys.
        self.python_key_mode_table: Dict[
            Type[TorchDispatchMode], Callable[..., Any]
        ] = {}

        # This table allows you to override the behavior of functorch
        # transformations.  NB: this currently only does something for
        # HigherOrderOperator
        self.functorch_table = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False

    def py_impl(self, k):
        def inner(fn):
            if inspect.isclass(k) and issubclass(k, TorchDispatchMode):
                assert k not in self.python_key_mode_table
                # TODO(voz): Should we replace setting torch._C.DispatchKey.Python entirely with setting mode keys?
                self.python_key_mode_table[k] = fn
                self._dispatch_cache.clear()
                return fn

            if isinstance(k, torch._C._functorch.TransformType):
                assert k not in self.functorch_table
                self.functorch_table[k] = fn
                return fn

            assert isinstance(k, torch._C.DispatchKey)
            assert (
                k != torch._C.DispatchKey.Python
            ), "Please register a mode for the torch._C.DispatchKey.Python key instead."

            if k in self.py_kernels:
                raise RuntimeError(
                    f"Trying to override a python impl for {k} on operator {self.name()}"
                )
            self.py_kernels[k] = fn
            self._dispatch_cache.clear()
            return fn

        return inner

    # Registers an implementation to all **3** variants of functionalization that we have:
    # - DispatchKey.Functionalize
    # - functorch.TransformType.Functionalize
    # - FunctionalTensorMode
    # Example:
    #   @py_functionalize_impl
    #   def functionalize_rule(ctx, inner_f, *args):
    #       args_unwrapped = ctx.unwrap_tensors(args)
    #       with ctx.redispatch_to_next():
    #           out = ctx.functionalize(inner_f)(*args_unwrapped)
    #           return ctx.wrap_tensors(out)
    def py_functionalize_impl(self, fn):
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI as _CppFunctionalizeAPI,
            FunctorchFunctionalizeAPI as _FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI as _PythonFunctionalizeAPI,
        )

        # Construct our three flavors of functionalization,
        # each of which have slightly different wrap/unwrap/redispatch policies
        def functionalize_dk_fn(*args, **kwargs):
            return fn(_CppFunctionalizeAPI(), *args, **kwargs)

        def functionalize_dispatch_mode_fn(mode, *args, **kwargs):
            return fn(_PythonFunctionalizeAPI(mode), *args, **kwargs)

        def functionalize_functorch_fn(interpreter, *args, **kwargs):
            return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)

        self.py_impl(torch._C.DispatchKey.Functionalize)(functionalize_dk_fn)
        self.py_impl(torch._subclasses.functional_tensor.FunctionalTensorMode)(
            functionalize_dispatch_mode_fn
        )
        self.py_impl(torch._C._functorch.TransformType.Functionalize)(
            functionalize_functorch_fn
        )

        return fn

    def name(self):
        raise NotImplementedError()


is_included_in_alias = torch._C._dispatch_is_included_in_alias

DispatchKey = torch._C.DispatchKey


# Equivalent to computeDispatchTableEntryWithDebug
def resolve_key(op: OperatorBase, k: DispatchKey):  # type: ignore[valid-type]
    # 1. (Direct) operator registration
    if op.has_kernel_for_dispatch_key(k):
        return k
    # 2.1 Use CompositeExplicitAutogradNonFunctional kernel if available
    cand = DispatchKey.CompositeExplicitAutogradNonFunctional
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.2 Use CompositeExplicitAutograd kernel if available
    cand = DispatchKey.CompositeExplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    has_backend_kernel = op.has_kernel_for_any_dispatch_key(
        torch._C._dispatch_get_backend_keyset_from_autograd(k)
    ) or op.has_kernel_for_dispatch_key(DispatchKey.CompositeExplicitAutograd)
    # 2.3. Use CompositeImplicitAutograd kernel if available
    cand = DispatchKey.CompositeImplicitAutogradNestedTensor
    if (
        (k != DispatchKey.Undefined and is_included_in_alias(k, cand))
        and op.has_kernel_for_dispatch_key(cand)
        and not has_backend_kernel
    ):
        return cand
    cand = DispatchKey.CompositeImplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        if k == DispatchKey.AutogradOther and op.has_kernel_for_any_dispatch_key(
            torch._C._dispatch_autogradother_backends
        ):
            raise RuntimeError("ambiguous autogradother kernel")
        elif not has_backend_kernel:
            return cand
    # 2.4. For autograd backend keys, use kernel from DispatchKey::Autograd if available
    cand = DispatchKey.Autograd
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.5 Use kernel from DispatchKey::FuncTorchBatchedDecomposition if available
    cand = DispatchKey.FuncTorchBatchedDecomposition
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # Backend fallback
    if torch._C._dispatch_has_backend_fallback(k):
        # The dispatch key itself will implicitly route to backend fallback.
        # This is probably not great for the pure Python implementation.
        return k
    raise NotImplementedError(f"could not find kernel for {op} at dispatch key {k}")


_higher_order_ops: Dict[str, "HigherOrderOperator"] = {}

_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = [
    DispatchKey.PythonDispatcher,  # type: ignore[attr-defined]
    DispatchKey.PythonTLSSnapshot,  # type: ignore[attr-defined]
    DispatchKey.ADInplaceOrView,
    DispatchKey.BackendSelect,
    DispatchKey.AutocastCPU,  # type: ignore[attr-defined]
    DispatchKey.AutocastCUDA,  # type: ignore[attr-defined]
]


class HigherOrderOperator(OperatorBase):
    # The HigherOrderOperator will appear as torch.ops.higher_order.{name}
    #
    # If you're creating a new HigherOrderOperator, please do not change the
    # default. Adding operators to the global torch.ops namespace is a bad
    # practice due to name collisions.
    def __init__(self, name):
        super().__init__()
        self._name = name

        # Make _OPNamespace not scream, this whole name based association needs a good hard look
        self.__name__ = name
        _higher_order_ops[name] = self
        self._ns = "higher_order"

        # For a normal HigherOrderOperator instance, we will change its __module__ from torch._ops to
        # torch._ops.higher_order.
        # For an instance of subclass of HigherOrderOperator (e.g. customized higher order op),
        # the __module__ attribute will be kept unchanged.
        if self.__class__ is HigherOrderOperator:
            self_name_space = "." + self.namespace if self.namespace else ""
            self.__module__ = self.__module__ + self_name_space
        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        # [NOTE] We have to register pre-dispatch key implementation
        # because sometimes HOP use aot-dispatch tracing to detect certaion
        # mutations. This is problematic when we are functionalizing HOP
        # during pre-dispatch because when the inner tracer starts, it will see
        # that PreDispatch key is still active. In that case, we just redispatch
        # it to next key. This is only safe to do when PreDispatch key stack has no
        # active modes.

    def py_impl(self, k):
        if isinstance(k, torch._C.DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        return super().py_impl(k)

    @property
    def namespace(self):
        return self._ns

    def fallthrough(self, dispatch_key):
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def dispatch(self, dispatch_key, *args, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            assert not isinstance(kernel, torch._C.DispatchKey)
            return kernel(*args, **kwargs)

        if dispatch_key == torch._C.DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)

        if dispatch_key == torch._C.DispatchKey.Python:
            # The place to handle ProxyTorchDispatchMode, FakeTensorMode, etc
            from torch.utils._python_dispatch import _pop_mode_temporarily

            curr_mode = _get_current_dispatch_mode()
            assert (
                curr_mode is not None
            ), "Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode."
            assert (
                type(curr_mode) in self.python_key_mode_table
            ), f"Current active mode {curr_mode} not registered, python_key_mode_table: {self.python_key_mode_table}"
            handler = self.python_key_mode_table[type(curr_mode)]
            with _pop_mode_temporarily() as mode:
                return handler(mode, *args, **kwargs)

        functionality_key = torch._C._to_functionality_key(dispatch_key)  # type: ignore[attr-defined]
        if functionality_key == torch._C.DispatchKey.PreDispatch:
            from torch.utils._python_dispatch import _pop_mode_temporarily

            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                _len_torch_dispatch_stack_pre_dispatch() > 0
            ) and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                DispatchKey.Python
            ):
                curr_mode = _get_current_dispatch_mode_pre_dispatch()
                assert (
                    curr_mode is not None
                ), "Illegal invocation of dispatch on torch._C.DispatchKey.PreDispatch without a mode."
                assert (
                    type(curr_mode) in self.python_key_mode_table
                ), f"Current active mode {curr_mode} not registered"
                handler = self.python_key_mode_table[type(curr_mode)]
                with _pop_mode_temporarily(functionality_key) as mode:
                    return handler(mode, *args, **kwargs)

        final_key = resolve_key(self, dispatch_key)

        # This can current fail due to backend fallbacks.  You just have to
        # register them by hand for HigherOrderOperator.
        if final_key not in self.py_kernels:
            raise NotImplementedError(
                f"could not find kernel for HigherOrderOperator {self._name} "
                f"at dispatch key {final_key} (resolved from {dispatch_key})"
            )

        # [NOTE] We shouldn't cache PreDispatch kernel here because depending
        # on what modes are active, predispatch behaviour is different.
        # Also we do same thing for normal ops:
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        if dispatch_key != torch._C.DispatchKey.PreDispatch:
            self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]
        kernel = self.py_kernels[final_key]
        # It's illegal to register DispatchKey to py_kernels, since there's no
        # C++ kernel to call into
        assert not isinstance(kernel, torch._C.DispatchKey)
        return kernel(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            flat_args = _to_flat_tuple(args, kwargs)
            if torch.overrides.has_torch_function(flat_args):
                return torch.overrides.handle_torch_function(
                    self, flat_args, *args, **kwargs
                )

            dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
            return self.dispatch(
                dispatch_key_set.highestPriorityTypeId(), *args, **kwargs
            )

        return wrapper()

    def __str__(self):
        return f"{self.name()}"

    def name(self):
        return self._name


def _to_flat_tuple(args, kwargs):
    return pytree.arg_tree_leaves(*args, **kwargs)


def _compute_keyset(args, kwargs, non_fallthrough_keys):
    tensors = _get_tensors(args, kwargs)
    return key_extractor(tensors, non_fallthrough_keys)


def _get_tensors(args, kwargs):
    flat_all = _to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)


# Note - this should maintain identical impl to the C++ dispatcher key extraction logic
# at ATen/core/dispatch/DispatchKeyExtractor.h
def key_extractor(tensors, key_mask):
    key_set = torch._C._dispatch_tls_local_include_set()
    for tensor in tensors:
        key_set = key_set | torch._C._dispatch_keys(tensor)
    key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
    key_set = key_set & key_mask
    return key_set


# Mode stack for PreDispatchKey
# it should always have two keys with
# priority given to FunctionalTensorMode and
# then ProxyTorchDispatchMode. It means that
# slot 0 belongs to ProxyTorchDispatchMode and
# slot 1 belongs to FunctionalTensorMode.
class _ModeStackStateForPreDispatch:
    def __init__(self):
        self.__infra_modes = [None, None]

    def set(self, index, mode):
        assert index < len(self.__infra_modes)
        self.__infra_modes[index] = mode

    def get(self, index):
        assert index < len(self.__infra_modes)
        return self.__infra_modes[index]

    def count(self):
        return len([i for i in self.__infra_modes if i is not None])


_mode_stack_state_for_pre_dispatch = _ModeStackStateForPreDispatch()


def unset_mode_pre_dispatch(mode_key):
    current_mode_stack_pre_dispatch = mode_stack_state_for_pre_dispatch()
    assert mode_key in (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )

    def _unset_mode():
        if mode_key == torch._C._TorchDispatchModeKey.PROXY:
            current_mode = current_mode_stack_pre_dispatch.get(0)
            mode_stack_state_for_pre_dispatch().set(0, None)
            return current_mode
        else:
            current_mode = current_mode_stack_pre_dispatch.get(1)
            mode_stack_state_for_pre_dispatch().set(1, None)
            return current_mode

    current_mode = _unset_mode()

    new_pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()
    # When we are unsetting a mode, we need to check if there is
    # active mode left on the PreDispatch key. If there is nothing
    # active, we need to remove PreDispatch key from local dispatch include
    # set.
    if new_pre_dispatch_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(
            torch._C.DispatchKey.PreDispatch, False
        )

    return current_mode


def _set_mode_pre_dispatch(mode):
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    assert isinstance(mode, (FunctionalTensorMode, ProxyTorchDispatchMode))

    previous_mode_stack_len = _len_torch_dispatch_stack_pre_dispatch()
    if isinstance(mode, FunctionalTensorMode):
        current_mode = mode_stack_state_for_pre_dispatch().get(1)
        assert current_mode is None
        mode_stack_state_for_pre_dispatch().set(1, mode)
    else:
        current_mode = mode_stack_state_for_pre_dispatch().get(0)
        assert current_mode is None
        mode_stack_state_for_pre_dispatch().set(0, mode)

    # When we are setting a mode, we need to check if there is
    # active mode left on the PreDispatch key. If there was nothing
    # active before setting this mode, it means that PreDispatch key
    # was turned off. So we need to turn it on again.
    if previous_mode_stack_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(
            torch._C.DispatchKey.PreDispatch, True
        )


def _pop_mode_from_pre_dispatch():
    mode_stack = mode_stack_state_for_pre_dispatch()
    pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()

    if pre_dispatch_len == 0:
        raise AssertionError("Trying to pop empty mode stack")

    if mode_stack.get(1) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.FUNCTIONAL)

    if mode_stack.get(0) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)


def _len_torch_dispatch_stack_pre_dispatch():
    return mode_stack_state_for_pre_dispatch().count()


def _get_dispatch_mode_pre_dispatch(mode_key):
    assert mode_key in (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )
    if mode_key == torch._C._TorchDispatchModeKey.PROXY:
        return mode_stack_state_for_pre_dispatch().get(0)
    return mode_stack_state_for_pre_dispatch().get(1)


def _get_current_dispatch_mode_pre_dispatch():
    stack_len = mode_stack_state_for_pre_dispatch().count()
    if stack_len == 2:
        return mode_stack_state_for_pre_dispatch().get(1)
    if stack_len == 1:
        return (
            mode_stack_state_for_pre_dispatch().get(1)
            if mode_stack_state_for_pre_dispatch().get(1) is not None
            else mode_stack_state_for_pre_dispatch().get(0)
        )
    return None


def mode_stack_state_for_pre_dispatch():
    global _mode_stack_state_for_pre_dispatch
    return _mode_stack_state_for_pre_dispatch


cached_ops: Set["OpOverload"] = set()


def add_cached_op(op_overload):
    global cached_ops
    cached_ops.add(op_overload)


def reset_cached_ops():
    global cached_ops
    cached_ops.clear()


def get_cached_ops():
    global cached_ops
    return cached_ops


# Each OpOverload object contains pointer to a a specific operator overload, a pointer to the parent `OpOverloadPacket` object.
# You can obtain an OpOverload object through attribute query on OpOverloadPacket.
class OpOverload(OperatorBase):
    def __init__(self, overloadpacket, op, op_dk, schema, tags):
        super().__init__()
        self._op = op
        self._op_dk = op_dk
        self._schema = schema
        self._overloadpacket = overloadpacket
        self._tags = tags
        self._overloadname = (
            "default" if schema.overload_name == "" else schema.overload_name
        )
        self._name = self._schema.name
        if schema.overload_name:
            self._name += "." + schema.overload_name
        self.__name__ = f"{self._schema.name.split('::')[1]}.{self._overloadname}"
        self.__module__ = overloadpacket.__module__
        op.__module__ = overloadpacket.__module__
        self.__qualname__ = self._name
        self.__annotations__ = {}

        # If the OpOverload was constructed from a Library.def in Python.
        self._defined_in_python = self.__qualname__ in torch.library._defs

        # Logic replicated from aten/src/ATen/native/MathBitsFallback.h
        is_write = None
        for a in self._schema.arguments:
            if a.alias_info is None:
                continue
            if is_write is None:
                is_write = a.alias_info.is_write
            else:
                # We will conservatively call mixed mutable/non-mutable
                # aliased inputs as NOT a view
                is_write = a.alias_info.is_write or is_write
        self.is_view = is_write is not None and not is_write

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverload(op='{}.{}', overload='{}')>".format(
            *self._schema.name.split("::"), self._overloadname
        )

    def __call__(self_, *args, **kwargs):  # noqa: B902
        # use `self_` to avoid naming collide with aten ops arguments that
        # are named "self". This way, all the aten ops can be called by kwargs.
        return self_._op(*args, **kwargs)

    def __hash__(self):
        return hash(self._op)

    # `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    def has_kernel_for_dispatch_key(self, k):
        return super().has_kernel_for_dispatch_key(
            k
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    def has_kernel_for_any_dispatch_key(self, ks):
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(
            self.name(), ks
        ) or super().has_kernel_for_any_dispatch_key(ks)

    @property
    def namespace(self):
        return self._schema.name.split("::")[0]

    def _handle(self):
        return torch._C._dispatch_find_schema_or_throw(
            self._schema.name, self._schema.overload_name
        )

    def decompose(self, *args, **kwargs):
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if dk in self.py_kernels:
            # NB: This branch is not too necessary anymore, because we can
            # apply Python CompositeImplicitAutograd *before* tracing
            # using Python dispatcher (also taking advantage of the autograd
            # formula).  But it's included for completeness
            return self.py_kernels[dk](*args, **kwargs)
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        else:
            return NotImplemented

    # Remove a dispatch key from the dispatch cache.  This will force it to get
    # recomputed the next time.  Does nothing
    # WARNING: if you register a dispatch key to py_kernels of an OpOverload,
    # calling _del_dispatch on that key is NOT sufficient to apply your change,
    # because a single registration may affect MULTIPLE dispatch keys (e.g.,
    # registering Autograd affects AutogradCPU).  del_dispatch is to be used
    # only if you are specifically modifying how get_dispatch handles a
    # particular input 'key'.
    def _uncache_dispatch(self, key):
        self._dispatch_cache.pop(key, None)

    # This implements the pre-computation logic for the Python dispatcher.
    def _get_dispatch(self, key):
        # This is only called upon a cache miss
        assert key not in self._dispatch_cache, f"{self} {key}"

        if key == torch._C.DispatchKey.Python:
            if not self.python_key_mode_table:
                self._dispatch_cache[key] = key
                add_cached_op(self)
                return key

            def handler(*args, **kwargs):
                from torch.utils._python_dispatch import _get_current_dispatch_mode

                # TODO: We also need to handle tensor subclasses here
                # TODO(voz): We should walk all the nodes here / turn it into a list, topmode is ok for now.
                curr_mode = type(_get_current_dispatch_mode())
                assert (
                    curr_mode is not None
                ), "Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode."
                if curr_mode not in self.python_key_mode_table:
                    # TODO: This path is slow, should generally encourage this
                    # case to not happen
                    return self._op_dk(key, *args, **kwargs)
                # TODO(voz): The idea behind this is that we do not yet support dispatch by key + mode, only key.
                return self.python_key_mode_table[curr_mode](*args, **kwargs)

            self._dispatch_cache[key] = handler
            add_cached_op(self)
            return handler

        functionality_key = torch._C._to_functionality_key(key)  # type: ignore[attr-defined]
        if functionality_key == torch._C.DispatchKey.PreDispatch:
            curr_stack_len = _len_torch_dispatch_stack_pre_dispatch()
            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                curr_stack_len > 0
                and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                    DispatchKey.Python
                )
            ):

                def handler(*args, **kwargs):
                    @contextlib.contextmanager
                    def _temporarily_pop_modes_from_pre_dispatch():
                        top_mode = _pop_mode_from_pre_dispatch()
                        try:
                            yield top_mode
                        finally:
                            _set_mode_pre_dispatch(top_mode)

                    with _temporarily_pop_modes_from_pre_dispatch() as curr_mode:
                        assert isinstance(curr_mode, TorchDispatchMode)
                        overload_types = []
                        args_flattened, _ = torch.utils._pytree.tree_flatten(
                            (args, kwargs.values())
                        )
                        for a in args_flattened:
                            # TODO: need to double check the semantics of the "types" argument to torch_dispatch.
                            # It's generated in PyInterpreter.cpp, but seems to be generated in two places,
                            # where in one case we only include tensors with the python key, and in another
                            # we include **all** tensors.
                            if isinstance(a, torch.Tensor) and torch._C._dispatch_keys(
                                a
                            ).has(torch._C.DispatchKey.Python):
                                overload_types.append(type(a))
                        # TODO: check that I got these args correct (in C++, we pass in "0000"??)

                        return curr_mode.__torch_dispatch__(
                            self, overload_types, args, kwargs
                        )

                # Note [Not Caching Per-Dispatch-Key Mode Handlers]
                # Note that we're not caching this handler.  There isn't really a point, since the slow bit
                # is the handler itself (in python).
                # Also, not caching means that we don't have to reset the cache when any existing
                # modes go out of scope (which in of itself takes time to loop through all operators).
                return handler

        final_key = resolve_key(self, key)

        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        cache_result = key != torch._C.DispatchKey.PreDispatch

        # TODO: We could potentially have lots of debugging wrappers against
        # dispatch keys; design some general registration mechanism instead of
        # having if statement for each of them
        if key == torch._C.DispatchKey.Functionalize:
            import torch._dispatch.python as pydispatch

            if pydispatch.CROSSREF_FUNCTIONALIZE:
                handler = pydispatch.make_crossref_functionalize(self, final_key)
                if cache_result:
                    self._dispatch_cache[key] = handler
                    add_cached_op(self)
                return handler

        # print(self, key, final_key)
        r = self.py_kernels.get(final_key, final_key)
        if cache_result:
            self._dispatch_cache[key] = r
            add_cached_op(self)
        return r

    def name(self):
        return self._name

    @property
    def overloadpacket(self):
        return self._overloadpacket

    @property
    def op(self):
        return self._op

    @property
    def tags(self):
        return self._tags

    # TODO: add more methods to expose information about input and output arguments


# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket:
    def __init__(self, qualified_op_name, op_name, op, overload_names):
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names
        self._dir = []

    # it's a no-op since OpOverloadPacket object is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    @property
    def op(self):
        return self._op

    def __getattr__(self, key):
        # It is not a valid op_name when __file__ is passed in
        if key == "__file__":
            return "torch.ops"

        # ensure that query for dunder attributes that does not exist on
        # opoverloadpacket but instead exists on the self._op object does not unnecessarily call
        # `_get_operation_overload` (which is an expensive operation).
        # This is done to prevent any potential slowdown. This list can be extended
        # if there exists other attributes like `__name__` that only exist on self._op and not on the
        # opoverloadpacket.
        # This is ok since we are guaranteed that an overload name for an aten op can't start with '__'
        try:
            if key.startswith("__"):
                return getattr(self._op, key)
        except AttributeError:
            # for consistency because it seems weird to
            # throw an attribute error with a message containing
            # an object name different from the one the attribute
            # query was performed on.
            raise AttributeError(
                f"'{str(self)}' can't have an overload name beginning with '__' and the "
                f"underlying op {str(self._op)} has no attribute {key} either."
            ) from None

        try:
            # This is ok since we are guaranteed that an overload name for an aten op can't be 'default'
            use_key = "" if key == "default" else key
            # TODO: disallow access to overloads registered by JIT
            op_, op_dk_, tags = torch._C._get_operation_overload(
                self._qualified_op_name, use_key
            )
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload = OpOverload(self, op_, op_dk_, schema, tags)
            # cache the overload object
            setattr(self, key, overload)
            self._dir.append(key)
            return overload
        except RuntimeError:
            raise AttributeError(
                f"The underlying op of '{str(self)}' has no overload name '{key}'"
            ) from None

    def __iter__(self):
        return iter(self._dir)

    def __call__(self_, *args, **kwargs):  # noqa: B902
        # use `self_` to avoid naming collide with aten ops arguments that
        # named "self". This way, all the aten ops can be called by kwargs.

        # overloading __call__ to ensure torch.ops.foo.bar()
        # is still callable from JIT
        # We save the function ptr as the `op` attribute on
        # OpOverloadPacket to access it here.
        return self_._op(*args, **(kwargs or {}))

    # TODO: use this to make a __dir__
    def overloads(self):
        return [n if n else "default" for n in self._overload_names]


# Resolution of torch.fn is different from torch.ops.aten.fn
# torch.fn uses the Python argparser, matches with the
# appropriate schema, and calls into the unboxed version of the method
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT.
# JIT creates a stack of all the overloads and then tries to match the
# correct one at runtime and always calls into the boxed version of the method
# Autograd codegen creates VariableType, TracerType,
# inplace or view type and python bindings.
# Aten codegen generates tensor methods for the tensor class.

# _OpNamespace is a subclass of ModuleType because the torch script
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# to work from script, we need to ensure ops and foo are modules


class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """

    def __init__(self, name):
        super().__init__("torch.ops." + name)
        self.name = name
        self._dir = []

    def __iter__(self):
        return iter(self._dir)

    def __getattr__(self, op_name):
        # It is not a valid op_name when __file__ is passed in
        if op_name == "__file__":
            return "torch.ops"
        elif op_name in ["__origin__", "__self__"]:
            raise AttributeError(
                f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'"
            )

        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        namespace_name = self.name
        qualified_op_name = f"{namespace_name}::{op_name}"
        try:
            op, overload_names = torch._C._jit_get_operation(qualified_op_name)
            if op is None:
                raise AttributeError(
                    f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
                )
        except RuntimeError as e:
            # Turn this into AttributeError so getattr(obj, key, default)
            # works (this is called by TorchScript with __origin__)
            raise AttributeError(
                f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
            ) from e

        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        torch.jit._builtins._register_builtin(op, qualified_op_name)
        op.__module__ = self.__module__ + "." + namespace_name
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        setattr(self, op_name, opoverloadpacket)
        self._dir.append(op_name)
        return opoverloadpacket


class _PyOpNamespace(_OpNamespace):
    def __init__(self, name, ops):
        super().__init__(name)
        self._ops = ops

    def __getattr__(self, name):
        # Following _OpNamespace.__getattr__, we cache the op on the _PyOpNamespace object.
        op = self._ops.get(name, None)
        if op is None:
            raise AttributeError(
                f"'_PyOpNamespace' '{self.name}' object has no attribute '{name}'"
            )
        setattr(self, name, op)
        return op


class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super().__init__("torch.ops")
        self.loaded_libraries = set()
        self._higher_order_op_namespace = _PyOpNamespace(
            "torch.ops.higher_order", _higher_order_ops
        )
        self._dir = []

    def __getattr__(self, name):
        # Check if the name is a HigherOrderOperator
        if name == "higher_order":
            return self._higher_order_op_namespace

        # Here we are creating `torch.ops.my_namespace`
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace

    def __iter__(self):
        return iter(self._dir)

    def import_module(self, module):
        """
        Imports a Python module that has torch.library registrations.

        Generally, to extend PyTorch with custom operators, a user will
        create a Python module whose import triggers registration of
        the custom operators via a torch.ops.load_library call or a call
        to one or more torch.library.* APIs.

        It is unexpected for Python modules to have side effects, so some
        linters and formatters will complain. Use this API to import Python
        modules that contain these torch.library side effects.

        Args:
            module (str): The name of the Python module to import

        """
        importlib.import_module(module)

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        if torch._running_with_deploy():
            return

        path = _utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)


# The ops "namespace"
ops = _Ops()
