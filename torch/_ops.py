# mypy: allow-untyped-defs
import abc
import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Concatenate,
    final,
    Generic,
    Optional,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._C import _dispatch_is_included_in_alias as is_included_in_alias, DispatchKey
from torch._functorch.pyfunctorch import dispatch_functorch, TransformType
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


_T = TypeVar("_T", default=Any)
_P = ParamSpec("_P", default=...)


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
        self._dispatch_cache: dict[
            DispatchKey, Union[DispatchKey, Callable[..., Any]]
        ] = {}

        # This table allows you to override the behavior of a particular
        # dispatch key to call a custom Python function, rather than the
        # ordinary C++ configured behavior.  This is the raison d'etre of  # codespell:ignore
        # Python dispatcher: to let you program the dispatcher from Python
        # in case you need something unusual, and don't want to clobber
        # the existing registrations using the Python operator registration
        # API.
        self.py_kernels: dict[DispatchKey, Callable[..., Any]] = {}

        # This table allows you to override the behavior of a particular
        # operator for a particular TorchDispatchMode.  In practice,
        # we are using this mostly for ProxyTensorMode.  Modes can be
        # thought of as an open world extension of dispatch keys, so it
        # makes sense that you should be able to register them, the same
        # way you can register dispatch keys.
        self.python_key_table: dict[
            type[Union[TorchDispatchMode, torch.Tensor]], Callable[..., Any]
        ] = {}

        # This table allows you to override the behavior of functorch
        # transformations.  NB: this currently only does something for
        # HigherOrderOperator
        self.functorch_table = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False

    def py_impl(
        self,
        k: Union[
            type[TorchDispatchMode],
            type[torch.Tensor],
            TransformType,
            DispatchKey,
        ],
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        def inner(fn: Callable[_P, _T]) -> Callable[_P, _T]:
            if inspect.isclass(k) and (
                issubclass(k, TorchDispatchMode) or issubclass(k, torch.Tensor)
            ):
                assert k not in self.python_key_table
                # TODO(voz): Should we replace setting DispatchKey.Python entirely with setting mode keys?
                self.python_key_table[k] = fn
                self._dispatch_cache.clear()
                return fn

            if isinstance(k, TransformType):
                assert k not in self.functorch_table
                self.functorch_table[k] = fn
                return fn

            assert isinstance(k, DispatchKey)
            assert k != DispatchKey.Python, (
                "Please register a mode for the DispatchKey.Python key instead."
            )

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
    def py_functionalize_impl(
        self, fn: Callable[Concatenate["BaseFunctionalizeAPI", _P], _T]
    ) -> Callable[Concatenate["BaseFunctionalizeAPI", _P], _T]:
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI,
            FunctionalTensorMode,
            FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI,
        )

        # Construct our three flavors of functionalization,
        # each of which have slightly different wrap/unwrap/redispatch policies
        def functionalize_dk_fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return fn(CppFunctionalizeAPI(), *args, **kwargs)

        def functionalize_dispatch_mode_fn(
            mode: Optional[FunctionalTensorMode], *args: _P.args, **kwargs: _P.kwargs
        ) -> _T:
            return fn(PythonFunctionalizeAPI(mode), *args, **kwargs)

        def functionalize_functorch_fn(
            interpreter, *args: _P.args, **kwargs: _P.kwargs
        ) -> _T:
            return fn(FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)

        self.py_impl(DispatchKey.Functionalize)(functionalize_dk_fn)
        self.py_impl(FunctionalTensorMode)(functionalize_dispatch_mode_fn)
        self.py_impl(TransformType.Functionalize)(functionalize_functorch_fn)

        return fn

    def name(self):
        raise NotImplementedError


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


_higher_order_ops: dict[str, "HigherOrderOperator"] = {}

_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = [
    DispatchKey.PythonDispatcher,  # type: ignore[attr-defined]
    DispatchKey.PythonTLSSnapshot,  # type: ignore[attr-defined]
    DispatchKey.ADInplaceOrView,
    DispatchKey.BackendSelect,
    DispatchKey.AutocastCPU,  # type: ignore[attr-defined]
    DispatchKey.AutocastCUDA,  # type: ignore[attr-defined]
    DispatchKey.AutocastXPU,  # type: ignore[attr-defined]
]


class HigherOrderOperator(OperatorBase, abc.ABC):
    # The HigherOrderOperator will appear as torch.ops.higher_order.{name}
    #
    # If you're creating a new HigherOrderOperator, please do not change the
    # default. Adding operators to the global torch.ops namespace is a bad
    # practice due to name collisions.
    def __init__(self, name, *, cacheable=False):
        super().__init__()
        if type(self) is HigherOrderOperator:
            raise RuntimeError(
                "Direct instantiation of HigherOrderOperator is not allowed. Please subclass it."
            )
        self._name = name

        # Make _OPNamespace not scream, this whole name based association needs a good hard look
        self.__name__ = name
        _higher_order_ops[name] = self
        self._ns = "higher_order"
        self.__module__ = "torch.ops.higher_order"
        self._cacheable = cacheable

        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        # [NOTE] We have to register pre-dispatch key implementation
        # because sometimes HOP use aot-dispatch tracing to detect certain
        # mutations. This is problematic when we are functionalizing HOP
        # during pre-dispatch because when the inner tracer starts, it will see
        # that PreDispatch key is still active. In that case, we just redispatch
        # it to next key. This is only safe to do when PreDispatch key stack has no
        # active modes.

    def py_impl(
        self,
        k: Union[
            type[TorchDispatchMode],
            type[torch.Tensor],
            TransformType,
            DispatchKey,
        ],
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        if isinstance(k, DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        return super().py_impl(k)

    def py_autograd_impl(
        self,
        fn: Callable[_P, _T],
    ) -> Callable[_P, _T]:
        def maybe_run_autograd(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if not torch.is_grad_enabled() or pytree.tree_all_only(
                torch.Tensor,
                lambda t: not t.requires_grad,  # type: ignore[union-attr]
                (*args, kwargs),
            ):
                with torch._C._AutoDispatchBelowAutograd():
                    return self(*args, **kwargs)

            from torch._higher_order_ops.utils import _has_gen_schema

            if _has_gen_schema(self):
                schema = self.gen_schema(*args, **kwargs)
                if any(arg.is_write for arg in schema.arguments):
                    raise RuntimeError(
                        f"The {self.name()} HigherOrderOperator does not currently support training "
                        "with in-place input or buffer mutations "
                        "If you require this feature, please submit an issue to PyTorch. "
                        "Alternatively, consider creating your own custom autograd.Function. "
                    )

            return fn(*args, **kwargs)

        self.py_impl(DispatchKey.Autograd)(maybe_run_autograd)

        return fn

    @property
    def namespace(self):
        return self._ns

    @final
    def cacheable(self) -> bool:
        from torch._functorch.autograd_function import AutogradFunctionApply

        return (
            self._cacheable
            or f"{self.__module__}.{self.__name__}"
            in torch._inductor.config.unsafe_marked_cacheable_functions
            or (
                isinstance(self, AutogradFunctionApply)
                and torch._functorch.config.autograd_cache_allow_custom_autograd_functions
            )
        )

    def fallthrough(self, dispatch_key):
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    # Use positional-only argument to avoid naming collide with custom ops arguments
    # that are named "self".
    def dispatch(self, /, dispatch_key, *args, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            assert not isinstance(kernel, DispatchKey)
            return kernel(*args, **kwargs)

        if dispatch_key == DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)

        if dispatch_key == DispatchKey.Python:
            # Keep the following 1:1 with handle_torch_function_no_python_arg_parser
            # in torch/csrc/utils/python_arg_parser.cpp

            overloaded_args_list = []

            def has_python_key(tensor):
                return torch._C._dispatch_keys(tensor).has("Python")

            def check_overloaded(arg):
                if isinstance(arg, torch.Tensor) and has_python_key(arg):
                    overloaded_args_list.append(arg)

            for arg in (*args, *kwargs.values()):
                check_overloaded(arg)
                if isinstance(arg, (list, tuple)):
                    for a in arg:
                        check_overloaded(a)

            overloaded_args = tuple(overloaded_args_list)

            # Step 1: dispatch on any user TorchDispatchModes
            from torch.utils._python_dispatch import _pop_mode_temporarily

            curr_mode = _get_current_dispatch_mode()
            if curr_mode is not None:
                if type(curr_mode) in self.python_key_table:
                    handler = self.python_key_table[type(curr_mode)]
                    with _pop_mode_temporarily() as mode:
                        # "natural" calling convention: (mode, *args, **kwargs)
                        # TODO(rzou): we should support torch_dispatch calling convention too.
                        result = handler(mode, *args, **kwargs)
                else:
                    if curr_mode.supports_higher_order_operators:
                        with _pop_mode_temporarily() as mode:
                            return curr_mode.__torch_dispatch__(self, [], args, kwargs)
                    else:
                        raise NotImplementedError(
                            f"There was no rule registered for HigherOrderOperator {self._name} and mode {curr_mode}."
                            f"Hint: set {curr_mode}'s supports_higher_order_operators to True."
                            f" This causes all higher order operators to pass through {curr_mode}'s __torch_dispatch__,"
                            f" so handle them accordingly by"
                            f" adding support for HigerOrderOperators (in this case, {self._name}) in"
                            f" {curr_mode}.__torch_dispatch__ or"
                            f" returning NotImplemented when not supported."
                        )
                if result is not NotImplemented:
                    return result

            # Step 2: dispatch on any subclasses
            for arg in overloaded_args:
                subclass_type = type(arg)
                if (
                    subclass_type.__torch_dispatch__
                    is torch._C._disabled_torch_dispatch_impl
                ):
                    continue

                # In some case, people are using FakeTensor without a FakeTensorMode.
                # For example, some sparse arch model has a mix of FakeTensor and real
                # tensor for weights during lowering, and ppl tends to run eager evaluation
                # on the model without setting up the FakeTensorMode.
                # In this case, we pull FakeTensorMode impl.
                if subclass_type is torch._subclasses.fake_tensor.FakeTensor:
                    subclass_type = torch._subclasses.fake_tensor.FakeTensorMode  # type: ignore[assignment]
                    handler = self.python_key_table[subclass_type]
                    result = handler(arg.fake_mode, *args, **kwargs)  # type: ignore[attr-defined]
                    return result

                if subclass_type in self.python_key_table:
                    handler = self.python_key_table[subclass_type]
                    # "natural" calling convention: (*args, **kwargs)
                    # TODO(rzou): we should support torch_dispatch calling convention too.
                    result = handler(*args, **kwargs)
                else:
                    raise NotImplementedError(
                        f"There was no rule registered for HOP {self._name} and subclass {subclass_type}. "
                        f"We recommend filing an issue."
                    )
                if result is not NotImplemented:
                    return result

            # All handlers returned NotImplemented
            raise TypeError(
                f"HigherOrderOperator '{self._name}' is not supported for the given input types. "
                f"This typically happens when using custom tensor types or dispatch modes that don't "
                f"have implementations for this operation.\n\n"
                f"Current mode: {curr_mode}\n"
                f"Input types: {[type(a).__name__ for a in overloaded_args]}\n\n"
                f"To fix this, can add support for '{self._name}' in {curr_mode}'s __torch_dispatch__\n"
            )

        functionality_key = torch._C._to_functionality_key(dispatch_key)  # type: ignore[attr-defined]
        if functionality_key == DispatchKey.PreDispatch:
            from torch.utils._python_dispatch import _pop_mode_temporarily

            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                _len_torch_dispatch_stack_pre_dispatch() > 0
            ) and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                DispatchKey.Python
            ):
                curr_mode = _get_current_dispatch_mode_pre_dispatch()
                assert curr_mode is not None, (
                    "Illegal invocation of dispatch on DispatchKey.PreDispatch without a mode."
                )
                assert type(curr_mode) in self.python_key_table, (
                    f"Current active mode {curr_mode} not registered"
                )
                handler = self.python_key_table[type(curr_mode)]
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
        if dispatch_key != DispatchKey.PreDispatch:
            self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]
        kernel = self.py_kernels[final_key]
        # It's illegal to register DispatchKey to py_kernels, since there's no
        # C++ kernel to call into
        assert not isinstance(kernel, DispatchKey)
        return kernel(*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, /, *args, **kwargs):
        flat_args = _to_flat_tuple(args, kwargs)
        if torch.overrides.has_torch_function(flat_args):
            return torch.overrides.handle_torch_function(
                self, flat_args, *args, **kwargs
            )

        dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
        return self.dispatch(dispatch_key_set.highestPriorityTypeId(), *args, **kwargs)

    # NOTE [HigherOrderOperator Schema]
    # Each invocation of a HigherOrderOperator (hop) should have its own schema because
    # the subgraphs and the arguments can be different even for the same hop.
    #
    # Each hop should implement its own gen_schema method, which should
    # take the same input as the __call__ method and returns a FunctionSchema.
    # The schema provides a unified way to check if the hop mutates its inputs,
    # which can be useful in implementing optimizations.
    #
    # If the hop doesn't implement the gen_schema method,
    # we expect it to be functional. It should not mutate its inputs and there
    # are no input, output aliasing via views or direct referencing.
    def gen_schema(self, *args, **kwargs):
        raise NotImplementedError(
            f"HigherOrderOperator {self._name} does not implement a gen_schema. "
            f"This is OK as long as the hop is functional. "
            f"e.g. it should not mutate its inputs and there are no input, output aliasing "
            f"via views or direct referencing."
        )

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
# it should always have three keys with
# priority given to FunctionalTensorMode and
# then ProxyTorchDispatchMode. It means that
# slot 0 belongs to ProxyTorchDispatchMode and
# slot 1 belongs to FunctionalTensorMode.
#
# SchemaCheckMode is separate from the other 2,
# and is only valid when the stack is empty.
# SchemaCheckMode is for testing purposes, and
# is meant to run in eager mode on concrete inputs,
# checking for incorrect schemas in regards to
# aliasing or mutating ops.
class _ModeStackStateForPreDispatch:
    def __init__(self):
        self.__infra_modes = [None, None]
        self._schema_check_mode = None

    def set(self, index, mode):
        assert index < len(self.__infra_modes)
        self.__infra_modes[index] = mode

    def get(self, index):
        assert index < len(self.__infra_modes)
        return self.__infra_modes[index]

    def count(self):
        return len([i for i in self.__infra_modes if i is not None]) + int(
            self._schema_check_mode is not None
        )


_mode_stack_state_for_pre_dispatch = _ModeStackStateForPreDispatch()


def unset_mode_pre_dispatch(mode_key, schema_check=False):
    current_mode_stack_pre_dispatch = mode_stack_state_for_pre_dispatch()
    assert mode_key is None or mode_key in (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )
    if schema_check:
        assert mode_key is None

    def _unset_mode():
        # NOTE: Using `is` rather than `==` to work around slow enum comparison in
        # pybind11.
        if mode_key is torch._C._TorchDispatchModeKey.PROXY:
            current_mode = current_mode_stack_pre_dispatch.get(0)
            mode_stack_state_for_pre_dispatch().set(0, None)
            return current_mode
        elif mode_key is torch._C._TorchDispatchModeKey.FUNCTIONAL:
            current_mode = current_mode_stack_pre_dispatch.get(1)
            mode_stack_state_for_pre_dispatch().set(1, None)
            return current_mode
        else:
            current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
            mode_stack_state_for_pre_dispatch()._schema_check_mode = None
            return current_mode

    current_mode = _unset_mode()

    new_pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()
    # When we are unsetting a mode, we need to check if there is
    # active mode left on the PreDispatch key. If there is nothing
    # active, we need to remove PreDispatch key from local dispatch include
    # set.
    if new_pre_dispatch_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(DispatchKey.PreDispatch, False)

    return current_mode


def _set_mode_pre_dispatch(mode):
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    assert isinstance(
        mode,
        (
            FunctionalTensorMode,
            ProxyTorchDispatchMode,
            SchemaCheckMode,
        ),
    )

    previous_mode_stack_len = _len_torch_dispatch_stack_pre_dispatch()
    if isinstance(mode, SchemaCheckMode):
        current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
        if previous_mode_stack_len > 0:
            raise AssertionError(
                "SchemaCheckMode for pre-dispatch must be used exclusively, found other modes on the stack"
            )
        mode_stack_state_for_pre_dispatch()._schema_check_mode = mode
    elif isinstance(mode, FunctionalTensorMode):
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
        torch._C._dispatch_tls_set_dispatch_key_included(DispatchKey.PreDispatch, True)


def _pop_mode_from_pre_dispatch():
    mode_stack = mode_stack_state_for_pre_dispatch()
    pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()

    if pre_dispatch_len == 0:
        raise AssertionError("Trying to pop empty mode stack")

    if mode_stack._schema_check_mode is not None:
        return unset_mode_pre_dispatch(None, schema_check=True)
    if mode_stack.get(1) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    if mode_stack.get(0) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)


def _len_torch_dispatch_stack_pre_dispatch():
    return mode_stack_state_for_pre_dispatch().count()


def _get_dispatch_mode_pre_dispatch(mode_key):
    # NOTE: Using `is` rather than `==` to work around slow enum comparison in pybind11.
    if mode_key is torch._C._TorchDispatchModeKey.PROXY:
        return mode_stack_state_for_pre_dispatch().get(0)
    else:
        assert mode_key is torch._C._TorchDispatchModeKey.FUNCTIONAL
        return mode_stack_state_for_pre_dispatch().get(1)


def _get_current_dispatch_mode_pre_dispatch():
    if mode_stack_state_for_pre_dispatch()._schema_check_mode is not None:
        return mode_stack_state_for_pre_dispatch()._schema_check_mode
    else:
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


cached_ops: set["OpOverload"] = set()


def add_cached_op(op_overload):
    global cached_ops
    cached_ops.add(op_overload)


def reset_cached_ops():
    global cached_ops
    cached_ops.clear()


def get_cached_ops():
    global cached_ops
    return cached_ops


# Each OpOverload object contains pointer to a specific operator overload, a pointer to the parent `OpOverloadPacket` object.
# You can obtain an OpOverload object through attribute query on OpOverloadPacket.
class OpOverload(OperatorBase, Generic[_P, _T]):
    def __init__(
        self,
        overloadpacket: "OpOverloadPacket",
        op: Callable[_P, _T],
        op_dk: Callable[Concatenate[DispatchKey, _P], _T],
        schema: torch._C.FunctionSchema,
        tags: list[Any],
    ) -> None:
        super().__init__()
        self._op = op
        self._op_dk = op_dk
        self._schema = schema
        self._overloadpacket = overloadpacket
        self._tags = tags
        self._overloadname = (
            "default" if schema.overload_name == "" else schema.overload_name
        )
        if tags:
            self._nondeterministic_seeded = torch.Tag.nondeterministic_seeded in tags
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
        for a in self._schema.arguments:  # pyrefly: ignore  # bad-assignment
            if a.alias_info is None:
                continue
            if is_write is None:
                is_write = a.alias_info.is_write
            else:
                # We will conservatively call mixed mutable/non-mutable
                # aliased inputs as NOT a view
                is_write = a.alias_info.is_write or is_write
        self.is_view = is_write is not None and not is_write

    @cached_property
    def _namespace(self) -> str:
        return self._schema.name.split("::", maxsplit=1)[0]

    @cached_property
    def _opname(self) -> str:
        return self._schema.name.split("::", maxsplit=1)[1]

    @cached_property
    def _handle(self) -> torch._C._DispatchOperatorHandle:
        return torch._C._dispatch_find_schema_or_throw(
            self._schema.name, self._schema.overload_name
        )

    # it's a no-op since OpOverload object is immutable and must be unique for a given op overload.
    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f"<OpOverload(op='{self._namespace}.{self._opname}', overload='{self._overloadname}')>"

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        return self._op(*args, **kwargs)

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def redispatch(
        self, /, keyset: torch._C.DispatchKeySet, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        return self._handle.redispatch_boxed(keyset, *args, **kwargs)  # type: ignore[return-value]

    def __hash__(self):
        return hash(self._op)

    # `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    def has_kernel_for_dispatch_key(self, k: DispatchKey) -> bool:
        return super().has_kernel_for_dispatch_key(
            k
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    def has_kernel_for_any_dispatch_key(self, ks: torch._C.DispatchKeySet) -> bool:
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(
            self.name(), ks
        ) or super().has_kernel_for_any_dispatch_key(ks)

    @property
    def namespace(self) -> str:
        return self._namespace

    def _can_decompose(self) -> bool:
        dk = DispatchKey.CompositeImplicitAutograd
        return dk in self.py_kernels or torch._C._dispatch_has_kernel_for_dispatch_key(
            self.name(), dk
        )

    def decompose(self, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        dk = DispatchKey.CompositeImplicitAutograd
        if dk in self.py_kernels:
            # NB: This branch is not too necessary anymore, because we can
            # apply Python CompositeImplicitAutograd *before* tracing
            # using Python dispatcher (also taking advantage of the autograd
            # formula).  But it's included for completeness
            return self.py_kernels[dk](*args, **kwargs)
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        else:
            return NotImplemented  # pyrefly: ignore [bad-return]

    # Remove a dispatch key from the dispatch cache.  This will force it to get
    # recomputed the next time.  Does nothing
    # WARNING: if you register a dispatch key to py_kernels of an OpOverload,
    # calling _del_dispatch on that key is NOT sufficient to apply your change,
    # because a single registration may affect MULTIPLE dispatch keys (e.g.,
    # registering Autograd affects AutogradCPU).  del_dispatch is to be used
    # only if you are specifically modifying how get_dispatch handles a
    # particular input 'key'.
    def _uncache_dispatch(self, key: DispatchKey) -> None:
        self._dispatch_cache.pop(key, None)

    # This implements the pre-computation logic for the Python dispatcher.
    def _get_dispatch(self, key: DispatchKey) -> Union[DispatchKey, Callable[_P, _T]]:
        # This is only called upon a cache miss
        assert key not in self._dispatch_cache, f"{self} {key}"

        if key == DispatchKey.Python:
            if not isinstance(self, TorchBindOpOverload) and not self.python_key_table:
                self._dispatch_cache[key] = key
                add_cached_op(self)
                return key

            def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                from torch.utils._python_dispatch import _get_current_dispatch_mode

                # TODO: We also need to handle tensor subclasses here
                # TODO(voz): We should walk all the nodes here / turn it into a list, topmode is ok for now.
                curr_mode = type(_get_current_dispatch_mode())
                assert curr_mode is not None, (
                    "Illegal invocation of dispatch on DispatchKey.Python without a mode."
                )

                if curr_mode not in self.python_key_table:
                    if isinstance(self, TorchBindOpOverload):
                        with (
                            torch.utils._python_dispatch._pop_mode_temporarily() as mode
                        ):
                            return torch._library.utils.handle_dispatch_mode(
                                mode, self, *args, **kwargs
                            )
                    else:
                        return self._op_dk(key, *args, **kwargs)

                with torch.utils._python_dispatch._pop_mode_temporarily() as mode:
                    return self.python_key_table[curr_mode](mode, *args, **kwargs)  # type: ignore[index]

            self._dispatch_cache[key] = handler
            add_cached_op(self)
            return handler

        functionality_key = torch._C._to_functionality_key(key)  # type: ignore[attr-defined]
        if functionality_key == DispatchKey.PreDispatch:
            curr_stack_len = _len_torch_dispatch_stack_pre_dispatch()
            # The check for Python in the exclude set is so we properly respect `with no_dispatch()`
            # calls inside of a mode.
            if (
                curr_stack_len > 0
                and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                    DispatchKey.Python
                )
            ):

                def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
                    @contextlib.contextmanager
                    def _temporarily_pop_modes_from_pre_dispatch():
                        top_mode = _pop_mode_from_pre_dispatch()
                        try:
                            yield top_mode
                        finally:
                            _set_mode_pre_dispatch(top_mode)

                    with _temporarily_pop_modes_from_pre_dispatch() as curr_mode:
                        return torch._library.utils.handle_dispatch_mode(
                            curr_mode, self, *args, **kwargs
                        )

                # Note [Not Caching Per-Dispatch-Key Mode Handlers]
                # Note that we're not caching this handler.  There isn't really a point, since the slow bit
                # is the handler itself (in python).
                # Also, not caching means that we don't have to reset the cache when any existing
                # modes go out of scope (which in of itself takes time to loop through all operators).
                return handler

        final_key = resolve_key(self, key)

        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        cache_result = key != DispatchKey.PreDispatch

        # TODO: We could potentially have lots of debugging wrappers against
        # dispatch keys; design some general registration mechanism instead of
        # having if statement for each of them
        if key == DispatchKey.Functionalize:
            import torch._dispatch.python as pydispatch

            if pydispatch.CROSSREF_FUNCTIONALIZE:
                handler = pydispatch.make_crossref_functionalize(self, final_key)  # type: ignore[assignment]
                if cache_result:
                    self._dispatch_cache[key] = handler
                    add_cached_op(self)
                return handler

        r = self.py_kernels.get(final_key, final_key)
        if cache_result:
            self._dispatch_cache[key] = r  # pyrefly: ignore [unsupported-operation]
            add_cached_op(self)
        return r  # pyrefly: ignore [bad-return]

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


# TorchBindOpOverload are those custom ops which have at least one overload's
# schema consists of torch.ScriptObject (i.e. custom class) input.
# TorchBindOpOverload will skip C++ dispatcher and purely dispatched in python
# when its inputs contain FakeScriptObject in a similar way as higher order ops.
class TorchBindOpOverload(OpOverload[_P, _T]):
    def _fallthrough_keys(self) -> list[DispatchKey]:
        # TODO: we should be calling the fallback for these, but a fallthrough is almost close
        # enough to the fallback in most cases that we care about.
        _DEFAULT_FALLTHROUGH_KEYS = [
            DispatchKey.Autograd,
            DispatchKey.AutogradCPU,
            DispatchKey.AutogradCUDA,
            DispatchKey.ADInplaceOrView,
            DispatchKey.BackendSelect,
            DispatchKey.PythonTLSSnapshot,
            DispatchKey.PythonDispatcher,
        ]

        def _may_use_fallthrough_instead_of_fallback(key: DispatchKey):
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), key):
                return torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                    self.name(), key
                )

            return (
                key not in self.py_kernels
                or self.py_kernels[key] is torch.library.fallthrough_kernel
            )

        return [
            key
            for key in _DEFAULT_FALLTHROUGH_KEYS
            if _may_use_fallthrough_instead_of_fallback(key)
        ]

    @contextlib.contextmanager
    def _register_as_effectful_op_temporarily(self):
        from torch._higher_order_ops.effects import (
            _EffectType,
            _register_effectful_op,
            SIDE_EFFECTS,
        )

        try:
            if self not in SIDE_EFFECTS:
                _register_effectful_op(self, _EffectType.ORDERED)
            yield
        finally:
            if self in SIDE_EFFECTS:
                del SIDE_EFFECTS[self]

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if _must_dispatch_in_python(args, kwargs):
            # When any inputs are FakeScriptObject, we need to
            # skip c++ dispatcher and dispatch in python through _get_dispatch of python_dispatcher
            # because C++ dispatcher will check the schema and cannot recognize FakeScriptObject.
            #
            # Note:
            # 1. We only register the torchbind op temporarily as effectful op because we only want
            #    the effect token functionalization logic to be applied during tracing. Otherwise, the behavior
            #    of the eagerly executing the op might change after tracing.
            # 2. We don't want to register the op as effectful for all torchbind ops in ctor because this might
            #    cause unexpected behavior for some autograd.profiler ops e.g. profiler._record_function_exit._RecordFunction.
            with self._register_as_effectful_op_temporarily():
                return self._dispatch_in_python(
                    self._fallthrough_keys(), *args, **kwargs
                )
        return self._op(*args, **kwargs)

    def _dispatch_in_python(
        self, fallthrough_keys: list[DispatchKey], *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        non_fallthrough_keys = torch._C._dispatch_keyset_full()
        for key in fallthrough_keys:
            non_fallthrough_keys = non_fallthrough_keys.remove(key)

        dispatch_key_set = _compute_keyset(args, kwargs, non_fallthrough_keys)
        dispatch_key = dispatch_key_set.highestPriorityTypeId()

        handler = (
            self._get_dispatch(dispatch_key)
            if dispatch_key not in self._dispatch_cache
            else self._dispatch_cache[dispatch_key]
        )

        if isinstance(handler, DispatchKey):
            # fallthrough keys can be registered at runtime via torch.library.impl
            # so need to add it to fallthrough_keys and re-dispatch.
            if torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                self.name(), dispatch_key
            ):
                return self._dispatch_in_python(
                    fallthrough_keys + [dispatch_key],
                    *args,
                    **kwargs,
                )

            raise RuntimeError(
                f"Torchbind op {self} received a FakeScriptObject input when dispatching {handler}."
                f" but no python implementation is found."
                f" Please file an issue on this when you encounter this error."
                f" This error can happen when you export or compile the model."
                f" It can still happen even if a C++ implementation for {dispatch_key}. "
                f" has been registered. That's because FakeScriptObject purely lives in python and cannot work "
                f" with a C++ implementation."
            )

        assert isinstance(handler, Callable)  # type: ignore[arg-type]
        return handler(*args, **kwargs)  # pyrefly: ignore [bad-return]


def _must_dispatch_in_python(args, kwargs):
    return pytree.tree_any(
        lambda obj: isinstance(
            obj, torch._library.fake_class_registry.FakeScriptObject
        ),
        (args, kwargs),
    )


def _has_script_object_arg(schema: torch.FunctionSchema) -> bool:
    return any(isinstance(arg.type, torch.ClassType) for arg in schema.arguments)


# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket(Generic[_P, _T]):
    __file__: ClassVar[str] = "torch.ops"

    def __init__(
        self,
        qualified_op_name: str,
        op_name: str,
        op: Callable[_P, _T],
        overload_names: list[str],
    ) -> None:
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names
        self._dir: list[str] = []
        self._has_torchbind_op_overload = any(
            _has_script_object_arg(schema) for schema in self._schemas.values()
        )

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

    @property
    def _schemas(self):
        return {
            overload_name: torch._C._get_schema(self._qualified_op_name, overload_name)
            for overload_name in self._overload_names
        }

    def __getattr__(self, key: str) -> OpOverload[_P, _T]:
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
            op_dk_tags = torch._C._get_operation_overload(
                self._qualified_op_name, use_key
            )
            if op_dk_tags is None:
                raise AttributeError(
                    f"The underlying op of '{str(self)}' has no overload name '{key}'"
                )

            op_, op_dk_, tags = op_dk_tags
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            overload: OpOverload[_P, _T] = (
                OpOverload(self, op_, op_dk_, schema, tags)
                if not _has_script_object_arg(schema)
                else TorchBindOpOverload(self, op_, op_dk_, schema, tags)
            )
            # cache the overload object
            setattr(self, key, overload)
            self._dir.append(key)
            return overload
        except RuntimeError:
            raise AttributeError(
                f"The underlying op of '{str(self)}' has no overload name '{key}'"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # overloading __call__ to ensure torch.ops.foo.bar()
        # is still callable from JIT
        # We save the function ptr as the `op` attribute on
        # OpOverloadPacket to access it here.

        # Directly calling OverloadPacket goes into C++, which will check
        # the schema and cause an error for torchbind op when inputs consist of FakeScriptObject so we
        # intercept it here and call TorchBindOpverload instead.
        if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
            # pyrefly: ignore [bad-argument-type]
            return _call_overload_packet_from_python(self, *args, **kwargs)
        return self._op(*args, **kwargs)

    # TODO: use this to make a __dir__
    def overloads(self):
        return [n if n else "default" for n in self._overload_names]


# Note - this mirrors the logic of the cpp_function defined in jit/python/init.cpp
# _jit_get_operations, which calls _get_operation_for_overload_or_packet.
def _call_overload_packet_from_python(
    op: OpOverloadPacket[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    # Reuse the torch function handling logic in cpp
    torch_function_called, ret = torch._C._maybe_call_torch_function_for_op_packet(
        op, *args, **kwargs
    )

    if torch_function_called:
        return ret

    # The following mirrors getOpWithStack.
    # In cpp, we do a schema matching for the arguments, and call ToIValue to
    # to check whether the arguments are valid. But need to do similar things here
    # and check the schema whether the FakeScriptObject is the corresponding fake class
    # of the actual class used in schema.
    exceptions = {}
    found_op = None
    for overload_name in op.overloads():
        op_overload = getattr(op, overload_name)
        try:
            _ = torch._C._check_schema_allow_fake_script_object(
                op_overload._schema, *args, **kwargs
            )
            found_op = op_overload
            break
        except RuntimeError as e:
            exceptions[overload_name] = e

    if found_op:
        return found_op(*args, **kwargs)

    err_msg = (
        f"Fail to match any TorchBindOverload of {op} with following exceptions:\n"
    )
    for key, msg in exceptions.items():
        err_msg += f"Overload name {key}:\n {msg}\n"
    raise RuntimeError(err_msg)


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

    __file__ = "torch.ops"

    def __init__(self, name: str) -> None:
        super().__init__("torch.ops." + name)
        self.name = name
        self._dir: list[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    def __getattr__(self, op_name: str) -> OpOverloadPacket:
        if op_name in ("__origin__", "__self__"):
            raise AttributeError(
                f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'"
            )

        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        namespace_name = self.name
        qualified_op_name = f"{namespace_name}::{op_name}"
        module_name = self.__module__ + "." + namespace_name

        try:
            op, overload_names = _get_packet(qualified_op_name, module_name)
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

        op.__module__ = module_name
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        setattr(self, op_name, opoverloadpacket)
        self._dir.append(op_name)
        return opoverloadpacket


def _get_packet(qualname, op_module):
    op, overload_names = torch._C._jit_get_operation(qualname)
    if op is not None:
        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        torch.jit._builtins._register_builtin(op, qualname)
        op.__module__ = op_module
    return op, overload_names


def _refresh_packet(packet):
    op, overload_names = _get_packet(packet._qualified_op_name, packet._op.__module__)
    assert op is not None
    packet._op = op
    packet._overload_names = overload_names


class _HigherOrderNamespace(types.ModuleType):
    __file__ = "torch.ops"

    def __init__(self) -> None:
        super().__init__("torch.ops.higher_order")
        self._dir: list[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self._dir)

    def __getattr__(self, name: str) -> HigherOrderOperator:
        # Following _OpNamespace.__getattr__, we cache the op on this object.
        op = _higher_order_ops.get(name)
        if op is None:
            raise AttributeError(
                f"'_HigherOrderNamespace' 'torch.ops.higher_order' object has no attribute '{name}'"
            )
        setattr(self, name, op)
        self._dir.append(name)
        return op


class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        super().__init__("torch.ops")
        self.loaded_libraries = set()
        self.higher_order = _HigherOrderNamespace()
        self._dir = []

    def __getattr__(self, name: str) -> _OpNamespace:
        # Here we are creating `torch.ops.my_namespace`
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace

    def __iter__(self) -> Iterator[str]:
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
        path = _utils_internal.resolve_library_path(path)
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            try:
                ctypes.CDLL(path)
            except Exception as e:
                raise OSError(f"Could not load this library: {path}") from e
        self.loaded_libraries.add(path)


# The ops "namespace"
ops: _Ops = _Ops()
