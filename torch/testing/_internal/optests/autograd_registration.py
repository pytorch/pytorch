import contextlib

import torch
import torch.utils._pytree as pytree


@contextlib.contextmanager
def set_autograd_fallback_mode(mode):
    prev = torch._C._get_autograd_fallback_mode()
    try:
        torch._C._set_autograd_fallback_mode(mode)
        yield
    finally:
        torch._C._set_autograd_fallback_mode(prev)


def autograd_registration_check(op, args, kwargs):
    """Check if autograd was registered correctly (for the operator).

    Operators should have "autograd support" registered directly to an
    autograd dispatch key.
    An incorrect registration may lead to unexpected silent incorrectness.
    Note that this check won't catch all problems but will catch
    the most common ones.

    Example usage:
        >>> x = torch.randn(3, requires_grad=True)
        >>> autograd_registration_check(torch.ops.aten.sin.default, (x,), {})

    Here are some best practices if you do find your autograd is
    registered incorrectly:
    - If the operator is composite (i.e. consists of other PyTorch ops)
      and you wish the operator to decompose and get autograd support
      that way, then please register the implementation to
      DispatchKey::CompositeImplicitAutograd
    - If you're adding an autograd formula for the operator, the correct
      thing to do is to register an autograd.Function to
      DispatchKey::Autograd (preferred) or one of the
      DispatchKey::Autograd<BACKEND> keys. It is NOT OK to register
      an autograd.Function to a backend (e.g. CPU/CUDA) key.
    - If your operator is non-differentiable, then you should register
      an implementation to the Autograd key that uses
      AutoDispatchBelowAutograd and re-invokes the operator.

    """
    assert isinstance(op, torch._ops.OpOverload)
    # Implementation details
    # -----------------------------------------------
    # If an operator doesn't have an autograd kernel at an autograd key,
    # and the operator does not return inputs as-is, then all of
    # the outputs should have requires_grad=False before we apply
    # special behaviors of our default autograd fallback.
    # (The default autograd fallback may set requires_grad=True on output
    # tensors in certain modes so that when they are backpropped through,
    # they raise an error).
    #
    # Our strategy for detecting if an operator doesn't have an autograd
    # kernel at the autograd key is:
    # - set the autograd fallback mode to "nothing" (so it does not change
    #   the required-gradness of outputs)
    # - run the operator
    # - Check if any outputs of the operator (that are not inputs) require
    #   grad. This would only happen if the user calls regular PyTorch
    #   operations in their backend key (this op should instead be
    #   CompositeImplicitAutograd or not an op) or if the user invokes
    #   an autograd.Function in the backend key.
    #
    # Note that it's already likely a bug if the operator directly returns
    # an input as output (because custom ops don't have a good way of
    # constructing true in-place or out variants), but we defer that
    # responsibility to a different test (schema_check).

    flat_args = pytree.tree_leaves((args, kwargs))
    all_tensors = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]
    if not any(t.requires_grad for t in all_tensors):
        raise RuntimeError(
            "autograd_registration_check: no inputs have requires_grad=True so "
            "we are unable to actually perform this test. Please pass inputs "
            "that do require grad."
        )

    # Determine which AutogradBACKEND key to check
    all_device_types = {arg.device.type for arg in all_tensors}
    if not all_device_types.issubset(["cpu", "cuda"]):
        # Don't want to support other keys yet
        raise NotImplementedError(
            f"autograd_registration_check: NYI devices other than CPU/CUDA, got {all_device_types}"
        )
    if "cuda" in all_device_types:
        key = "AutogradCUDA"
    elif "cpu" in all_device_types:
        key = "AutogradCPU"

    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), key):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Autograd"):
        return
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        op.name(), "CompositeImplicitAutograd"
    ):
        return

    # At this point, we know the operator doesn't have a kernel registered to an
    # autograd key. Let's proceed with our test.
    with set_autograd_fallback_mode("nothing"):
        all_outs = op(*args, **kwargs)

    inp_ids = {id(arg) for arg in flat_args}

    def not_an_input_and_requires_grad(tensor):
        if not tensor.requires_grad:
            return False
        if id(tensor) in inp_ids:
            return False
        return True

    if not pytree.tree_any_only(torch.Tensor, not_an_input_and_requires_grad, all_outs):
        return

    raise AssertionError(
        f"{op.name()}: at least one output of this operator has requires_grad=True "
        f"but the operator does not have an autograd kernel defined at an autograd "
        f"key (e.g. DispatchKey::Autograd). This could mean that you have "
        f"incorrectly registered an autograd kernel to a non-Autograd DispatchKey, "
        f"which may lead to silently incorrect results. If your operator consists "
        f"of regular PyTorch operations, consider not using an operator at all "
        f"or registering your operator as CompositeImplicitAutograd. If you have "
        f"an autograd.Function registered to a backend (CPU/CUDA) key, the correct "
        f"location for it is the Autograd key."
    )
