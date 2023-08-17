import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


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
        out = torch.Tensor._make_wrapper_subclass(cls, **kwargs)
        out.elem = elem
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
        # - _make_wrapper_subclass requires a __torch_dispatch__
        # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
        #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
        # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
        #   which causes every subclass created above autograd to have autograd view metadata (in addition to also being a FunctionalTensorWrapper).
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    @staticmethod
    def to_functional(x):
        # We will do the wrapping for the user.
        assert not torch._is_functional_tensor(x)
        # The only autograd metadata we care about on the FunctionalTensor is:
        # - requires_grad (so autograd runs)
        # - is_leaf (so that mutations on graph inputs that are not leaves are allowed by the autograd engine)
        #   this is handled by FunctionalTensor.to_functional
        x_functional = torch._to_functional_tensor(x, mirror_autograd_meta=True)

        torch._functionalize_enable_reapply_views(True)
        if x.requires_grad and not x.is_leaf:
            out = _ToFunctionalTensor.apply(x_functional)
            return out
        else:
            out = FunctionalTensor(x_functional)
            out.requires_grad = x_functional.requires_grad
            return out


class FunctionalTensorMode(TorchDispatchMode):
    def __init__(self):
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        _mode_key = torch._C.TorchDispatchModeKey.FUNCTIONAL

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def assert_is_functional(x):
            assert torch._is_functional_tensor(x)

        def wrap(x):
            return FunctionalTensor(x)

        def unwrap(x):
            return x.elem

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
        ):
            try:
                # By default for python functionalization (for AOTAutograd), we reapply views.
                torch._functionalize_enable_reapply_views(True)
                args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
                    FunctionalTensor, unwrap, (args, kwargs)
                )
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
        return outs_wrapped
