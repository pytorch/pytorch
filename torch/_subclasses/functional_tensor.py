import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple

import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode

not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """

    elem: torch.Tensor
    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL

    # Note: The reason we add these extra keys to our FunctionalTensor subclass
    # is to mirror the behavior of C++ functionalization (we can choose to change this
    # later, as long as it doesn't break anything).
    # FunctionalTensorWrapper copies **all** dispatch keys from the inner tensor
    # to the wrapper, excluding functorch and python dispatch keys.
    # Here I'm trying to re-use the keyset the functorch wrapper subclasses copy,
    # except that they don't include ZeroTensor so I'm manually adding it in.
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(
        torch._C.DispatchKey.ZeroTensor
    )

    # These are all aten ops that correspond to metadata queries.
    # We want FunctionalTensor to be able to handle them directly.
    metadata_fns = [
        torch.ops.aten.is_contiguous.default,  # type: ignore[has-type]
        torch.ops.aten.is_contiguous.memory_format,  # type: ignore[has-type]
        torch.ops.aten.is_strides_like_format.default,  # type: ignore[has-type]
        torch.ops.aten.is_non_overlapping_and_dense.default,  # type: ignore[has-type]
        torch.ops.aten.size.default,  # type: ignore[has-type]
        torch.ops.aten.sym_size.default,  # type: ignore[has-type]
        torch.ops.aten.stride.default,  # type: ignore[has-type]
        torch.ops.aten.sym_stride.default,  # type: ignore[has-type]
        torch.ops.aten.storage_offset.default,  # type: ignore[has-type]
        torch.ops.aten.sym_storage_offset.default,  # type: ignore[has-type]
        torch.ops.aten.numel.default,  # type: ignore[has-type]
        torch.ops.aten.sym_numel.default,  # type: ignore[has-type]
        torch.ops.aten.dim.default,  # type: ignore[has-type]
    ]

    def __new__(cls, elem):
        assert torch._is_functional_tensor(elem)

        # In general, we'd like our functional tensor subclass to only be in charge of functionalization,
        # and defer to the inner subclass for all other functionality.
        # Example: If our inner tensor is a ZeroTensor, we would want to defer running the ZeroTensor fallback
        # until after we redispatch to our inner ZeroTensor.
        # However, there are a few keys that we need to mirror between the inner and outer tensors.
        #   Conjugate
        #   Negative
        # Why? These keys are used to test metadata queries, like `.is_conj()` and `.is_neg()`.
        # We **need** calls to is_conj() to return the same thing on the outer and inner tensors,
        # Because user code / framework code that branches like so needs to do the same thing
        # when it sees the outer FunctionalTensor:
        #     if (x.is_conj()) {
        #         return at::view_as_real(x.resolve_conj());
        #     } else {
        #         return at::view_as_real(x);
        #     }
        extra_dispatch_keys = (
            FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        )

        out = torch.Tensor._make_wrapper_subclass(  # type: ignore[arg-type, attr-defined]
            # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
            # Calling the overload that has kwargs causes us to go down the first overload path,
            # which will **always** specialize sizes.
            # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
            cls,
            elem.shape,  # sizes
            elem.stride(),  # strides
            elem.storage_offset(),  # storage_offset
            None,  # memory_format
            elem.dtype,  # dtype
            elem.layout,  # layout
            elem.device,  # device
            False,  # pin_memory
            elem.requires_grad,  # requires_grad
            "sizes",  # dispatch_sizes_strides_policy
            False,  # dispatch_device
            False,  # dispatch_layout
            extra_dispatch_keys,  # _extra_dispatch_keys
        )
        out.elem = elem
        return out

    # Need to disable default torch_function. Why?
    # Default torch_function will always wrap outputs into a subclass if they aren't already a subclass.
    # We actually.. don't want to do this sometimes, see Note [FunctionalTensorMode inputs are sometimes plain tensors]
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        unrecognized_types = [
            t
            for t in types
            if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
        # In theory we don't have to do this - but if we want to service metadata requests here,
        # we need to carefully make sure all metadata is accurate (including metadata mutations)
        if func in FunctionalTensor.metadata_fns:

            def unwrap(x):
                return x.elem

            assert len(args) == 1 and isinstance(args[0], FunctionalTensor)
            assert len(kwargs) == 0
            # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
            # about the problem of keeping metadata in sync between the wrapper and inner tensor.
            # This also alleviates us from having to manually handle metadata mutations on the wrapper.
            return func(args[0].elem)
        # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
        # - _make_wrapper_subclass requires a __torch_dispatch__
        # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
        #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
        # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
        #   which causes every subclass created above autograd to have autograd view metadata
        #   (in addition to also being a FunctionalTensorWrapper).
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    def __repr__(self):
        return f"FunctionalTensor({repr(self.elem)})"

    @staticmethod
    def to_functional(x):
        # We will do the wrapping for the user.
        assert not torch._is_functional_tensor(x)
        # The only autograd metadata we care about on the FunctionalTensor is:
        # - requires_grad (so autograd runs)
        # - is_leaf (so that mutations on graph inputs that are not leaves are allowed by the autograd engine)
        #   this is handled by FunctionalTensor.to_functional
        x_functional = torch._to_functional_tensor(x)
        torch._mirror_autograd_meta_to(x, x_functional)  # type: ignore[attr-defined]
        out = FunctionalTensor(x_functional)
        torch._mirror_autograd_meta_to(x_functional, out)  # type: ignore[attr-defined]
        return out

    def from_functional(self):
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)


class FunctionalTensorMode(TorchDispatchMode):
    def __init__(self):
        self.is_on_stack = False
        self.enter_stack = []
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        # This will be turned off later for pre-dispatch functionalization
        self.decompose_composite_implicit_ops = True

    # No-op if FunctionalTensorMode is already in use
    def __enter__(self):
        if (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
            is None
        ):
            self.enter_stack.append(True)

            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(a, b, c)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        unrecognized_types = [
            t
            for t in types
            if not issubclass(t, torch._subclasses.FakeTensor)
            and t not in [torch.Tensor, FunctionalTensor]
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        if (
            func not in FunctionalTensor.metadata_fns
            and self.decompose_composite_implicit_ops
            # Not all funcs from __torch_dispatch__ are actual dispatcher ops,
            # e.g. prim.device
            and torch._C._dispatch_has_kernel(func.name())
        ):
            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def assert_is_functional(x):
            assert torch._is_functional_tensor(x)

        def wrap(x):
            # Only wrap our outputs in subclasses if the inner functionalization call
            # also wrapped outputs into FunctionalTensorWrappers.
            # When can this happen? e.g. `torch.div(2, 2)`
            assert not isinstance(x, FunctionalTensor)
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x)
            return x

        any_functional_inputs = False

        def unwrap(x):
            any_functional_inputs = True
            return x.elem

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            FunctionalTensor, unwrap, (args, kwargs)
        )

        # Expectation: functionalization should not **already** be enabled above our mode.
        # Why would that be bad? when we return a FunctionalTensor here, we don't want functionalization
        # to run above this mode and further wrap that output in **another** C++ FunctionalTensorWrapper.
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Functionalize
        )
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize
        )
        assert is_excluded or not is_included
        include_to_set = (
            torch._C._dispatch_tls_local_include_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set().remove(
                torch._C.DispatchKey.Functionalize
            )
            - FunctionalTensor._extra_dispatch_keys
        )
        # All we want to do here is re-use the existing C++ functionalization logic.
        # This requires swizzling our TLS dispatch keys so that the Functionalize key is active.
        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            try:
                # By default for python functionalization (for AOTAutograd), we reapply views.
                old_apply_views = torch._functionalize_enable_reapply_views(True)  # type: ignore[attr-defined]
                outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
            finally:
                torch._disable_functionalization()
                torch._functionalize_enable_reapply_views(old_apply_views)  # type: ignore[attr-defined]

        is_included = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Functionalize
        )
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize
        )
        assert is_excluded or not is_included

        # If no outputs are our functional subclass, then don't try to fix up aliasing
        if not any(
            isinstance(x, FunctionalTensor)
            for x in pytree.tree_flatten(outs_wrapped)[0]
        ):
            return outs_wrapped
        # Wrapper tensor subclasses do not have correct aliasing info! Use this util to manually correct the output aliasing.
        # inplace ops like `aten.add_()` are expected to return inputs **directly**, instead of creating fresh tensor objects.
        # Use this util to figure out the right thing to return.
        # If none of our inputs were wrapped, then we have no FunctionalTensor outputs that we need to fix up storages for.
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)


@contextlib.contextmanager
def maybe_disable_functional_mode():
    maybe_func_mode = torch._C._unset_dispatch_mode(
        torch._C._TorchDispatchModeKey.FUNCTIONAL
    )
    try:
        yield
    finally:
        if maybe_func_mode is not None:
            torch._C._set_dispatch_mode(maybe_func_mode)


# TODO: clean up the redundancy here,
# unify on a single context manager for all mode keys.
@contextlib.contextmanager
def unset_functional_temporarily():
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)


# This is similar to torch.func.functionalize, but:
# - It uses FunctionalTensorMode, and FunctionalTensor (a python subclass).
#   One important advantage to using this mode is that it will let us
#   run functionalization underneath __torch_dispatch__,
#   which we need in AOTAutograd.
# - Doing so means that it does not automatically compose with other
#   functorch transforms, since these transforms always run above __torch_dispatch__.
#   That's why this util lives here, and not in functorch.
def dispatch_functionalize(func):
    # TODO: pull these from aot autograd
    def to_fun(t):
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        return t

    def from_fun(t):
        if not isinstance(t, FunctionalTensor):
            # quick sanity assert
            if isinstance(t, torch.Tensor):
                assert not torch._is_functional_tensor(t)
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t.elem)

    def inner(*args, **kwargs):
        func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
        func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)

        flattened_wrapped_args, _ = pytree.tree_flatten(func_args)
        flattened_wrapped_kwargs, _ = pytree.tree_flatten(func_kwargs)

        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        with disable_above, FunctionalTensorMode():
            func_outputs = func(*func_args, **func_kwargs)
            outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)

            return outputs

    return inner


class BaseFunctionalizeAPI(ABC):
    @abstractmethod
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        pass

    @abstractmethod
    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        pass

    @abstractmethod
    def functionalize(self, inner_f: Callable) -> Callable:
        pass

    @abstractmethod
    def redispatch_to_next(self) -> ContextManager:
        pass


class PythonFunctionalizeAPI(BaseFunctionalizeAPI):
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        return torch.utils._pytree.tree_map_only(
            FunctionalTensor, FunctionalTensor.to_functional, args
        )

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        return torch.utils._pytree.tree_map_only(
            FunctionalTensor, FunctionalTensor.from_functional, args
        )

    def functionalize(self, inner_f: Callable) -> Callable:
        return dispatch_functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        return unset_functional_temporarily()


class CppFunctionalizeAPI(BaseFunctionalizeAPI):
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=0)

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        return torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )


class FunctorchFunctionalizeAPI(BaseFunctionalizeAPI):
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=self.interpreter.level())

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(
            args, reapply_views=self.interpreter.functionalize_add_back_views()
        )

    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(
            inner_f,
            remove="mutations_and_views"
            if self.interpreter.functionalize_add_back_views()
            else "mutations",
        )

    def redispatch_to_next(self) -> ContextManager:
        return self.interpreter.lower()
