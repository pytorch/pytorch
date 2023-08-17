import contextlib

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode


class _ToFunctionalTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return FunctionalTensor(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        raise RuntimeError(
            "Attempted to backprop from a functional wrapper to its original tensor. This is unsupported behavior."
        )


class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding FunctionalTensorMode active.
    It doesn't bother defining a __torch_dispatch__, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """

    elem: torch.Tensor
    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C.TorchDispatchModeKey.FUNCTIONAL

    def __new__(cls, elem):
        assert torch._is_functional_tensor(elem)
        shape = elem.shape
        kwargs = {}
        kwargs["size"] = elem.shape
        kwargs["strides"] = elem.stride()
        kwargs["storage_offset"] = elem.storage_offset()
        kwargs["device"] = elem.device
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        kwargs["dtype"] = elem.dtype
        kwargs["dispatch_sizes_strides_policy"] = "sizes"
        out = torch.Tensor._make_wrapper_subclass(cls, **kwargs)
        out.elem = elem
        return out

    # Need to disable default torch_function. Why?
    # Default torch_function will always wrap outputs into a subclass if they aren't already a subclass.
    # We actually.. don't want to do this sometimes, see Note [FunctionalTensorMode inputs are sometimes plain tensors]
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
        # In theory we don't have to do this - but if we want to service metadata requests here,
        # we need to carefully make sure all metadata is accurate (including metadata mutations)
        if func in [
            torch.ops.aten.is_contiguous.default,
            torch.ops.aten.is_contiguous.memory_format,
            torch.ops.aten.is_strides_like_format.default,
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.size.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.storage_offset.default,
            torch.ops.aten.sym_storage_offset.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.dim.default,
        ]:

            def unwrap(x):
                return x.elem

            args_unwrapped = pytree.tree_map_only(FunctionalTensor, unwrap, args)
            # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
            # about the problem of keeping metadata in sync between the wrapper and inner tensor.
            # This also alleviates us from having to manually handle metadata mutations on the wrapper.
            return func(*args_unwrapped, **kwargs)
        # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
        # - _make_wrapper_subclass requires a __torch_dispatch__
        # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
        #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
        # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
        #   which causes every subclass created above autograd to have autograd view metadata (in addition to also being a FunctionalTensorWrapper).
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    def __repr__(self):
        return f"FunctionalTensor({str(self.elem)})"

    @staticmethod
    def to_functional(x):
        # We will do the wrapping for the user.
        assert not torch._is_functional_tensor(x)
        # The only autograd metadata we care about on the FunctionalTensor is:
        # - requires_grad (so autograd runs)
        # - is_leaf (so that mutations on graph inputs that are not leaves are allowed by the autograd engine)
        #   this is handled by FunctionalTensor.to_functional
        x_functional = torch._to_functional_tensor(x)
        torch._mirror_autograd_meta(x, x_functional)

        torch._functionalize_enable_reapply_views(True)
        if x.requires_grad and not x.is_leaf:
            out = _ToFunctionalTensor.apply(x_functional)
            return out
        else:
            out = FunctionalTensor(x_functional)
            out.requires_grad = x_functional.requires_grad
            return out

    def from_functional(self):
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)


class FunctionalTensorMode(TorchDispatchMode):
    def __init__(self):
        self.is_active = True
        self.is_on_stack = False
        self.enter_stack = []
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C.TorchDispatchModeKey.FUNCTIONAL

    # No-op if FunctionalTensorMode is already in use
    def __enter__(self):
        if (
            torch._C._get_dispatch_mode(torch._C.TorchDispatchModeKey.FUNCTIONAL)
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

        def assert_is_functional(x):
            assert torch._is_functional_tensor(x)

        def wrap(x):
            return FunctionalTensor(x)

        any_functional_inputs = False

        def unwrap(x):
            any_functional_inputs = True
            return x.elem

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            FunctionalTensor, unwrap, (args, kwargs)
        )

        if not self.is_active:
            assert not any_functional_inputs
            return func(*args_unwrapped, **kwargs)

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
        # All we want to do here is re-use the existing C++ functionalization logic.
        # This requires swizzling our TLS dispatch keys so that the Functionalize key is active.
        with torch._C._SetExcludeDispatchKeyGuard(
            torch._C.DispatchKey.Functionalize, False
        ), torch._C._IncludeDispatchKeyGuard(torch._C.DispatchKey.Functionalize):
            try:
                # By default for python functionalization (for AOTAutograd), we reapply views.
                torch._functionalize_enable_reapply_views(True)
                outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                pytree.tree_map_only(torch.Tensor, assert_is_functional, outs_unwrapped)

                outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
            finally:
                torch._disable_functionalization()

        is_included = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Functionalize
        )
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize
        )
        assert is_excluded or not is_included

        # Wrapper tensor subclasses do not have correct aliasing info! Use this util to manually correct the output aliasing.
        # inplace ops like `aten.add_()` are expected to return inputs **directly**, instead of creating fresh tensor objects.
        # Use this util to figure out the right thing to return.
        # If none of our inputs were wrapped, then we have no FunctionalTensor outputs that we need to fix up storages for.
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)


@contextlib.contextmanager
def maybe_disable_functional_mode():
    maybe_func_mode = torch._C._unset_dispatch_mode(
        torch._C.TorchDispatchModeKey.FUNCTIONAL
    )
    try:
        yield
    finally:
        if maybe_func_mode is not None:
            torch._C._set_dispatch_mode(maybe_func_mode)
