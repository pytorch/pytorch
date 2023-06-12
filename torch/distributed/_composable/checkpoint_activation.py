from contextlib import contextmanager
from functools import partial
from typing import Any, List, Optional, Tuple
from weakref import ref, ReferenceType, WeakKeyDictionary

import torch
import torch.nn as nn
from torch.utils.checkpoint import detach_variable, get_device_states, set_device_states

from .contract import contract


@contextmanager
def _no_hook(module: nn.Module):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    orig_enable_hook = checkpoint.state(module).enable_hook
    checkpoint.state(module).enable_hook = False
    try:
        yield
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook


class _ModuleHookCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: nn.Module, output: Any, *inputs: Any) -> Any:  # type: ignore[override]
        ctx.module = module

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                tensor_inputs.append(inp)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(inp)

        ctx.save_for_backward(*tensor_inputs)

        return output

    @staticmethod
    def backward(ctx, output_grads: Tuple[Optional[torch.Tensor]]) -> Any:  # type: ignore[override]
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an "
                "`inputs` parameter is passed to .backward(). Please use "
                ".backward() and do not pass its `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if checkpoint.state(ctx.module).had_cuda_in_fwd:
            rng_devices = checkpoint.state(ctx.module).fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(checkpoint.state(ctx.module).fwd_cpu_state)
            if checkpoint.state(ctx.module).had_cuda_in_fwd:
                set_device_states(
                    checkpoint.state(ctx.module).fwd_gpu_devices,
                    checkpoint.state(ctx.module).fwd_gpu_states,
                )
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), _no_hook(ctx.module):
                outputs = ctx.module(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        if isinstance(output_grads, torch.Tensor):
            output_grads = (output_grads,)

        # run backward() with only tensor that requires grad
        outputs_requires_grad: List[torch.Tensor] = []
        output_grad_tensors: List[torch.Tensor] = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_requires_grad.append(outputs[i])
                assert (
                    output_grads[i] is not None
                ), f"expecting grad for output at index {i}, but got None."
                output_grad_tensors.append(output_grads[i])  # type: ignore[arg-type]
        if len(outputs_requires_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_requires_grad, output_grad_tensors)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        # The two None is for forward argument module and output respectively.
        return (None, None) + grads


class _Holder:
    pass


def _pack(
    x: torch.Tensor,
    *,
    weak_holder_list: List[ReferenceType],
) -> _Holder:
    res = _Holder()
    weak_holder_list.append(ref(res))
    return res


def _unpack(
    holder: _Holder,
    *,
    storage: WeakKeyDictionary,
    weak_holder_list: List[ReferenceType],
    module: nn.Module,
    inputs: Tuple[Any],
) -> torch.Tensor:
    holder_index = 0
    if len(storage) == 0:

        def inner_pack(inner: torch.Tensor):
            nonlocal holder_index
            if weak_holder_list[holder_index]() is None:
                # If the holder went out of scope, the SavedVariable is dead
                # and so the value will never be read from the storage. Skip
                # filling it.
                pass
            else:
                # Use detach here to ensure we don't keep the temporary
                # autograd graph created during the second forward
                storage[weak_holder_list[holder_index]()] = inner.detach()
            holder_index += 1
            return

        def inner_unpack(holder: _Holder):
            raise RuntimeError(
                "You are calling backwards on a tensor that is never exposed. "
                "Please open an issue."
            )

        with _no_hook(
            module
        ), torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
            inner_pack, inner_unpack
        ):
            _unused = module(*inputs)

    if holder not in storage:
        raise RuntimeError(
            "Attempt to retrieve a tensor saved by autograd multiple times "
            "without checkpoint recomputation being triggered in between, this "
            "is not currently supported. Please open an issue with details on "
            "your use case so that we can prioritize adding this."
        )

    return storage[holder]


@contract()
def checkpoint(module: nn.Module, *, use_reentrant: bool = True) -> nn.Module:
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.
        use_reentrant (bool): Apply activation checkpointing using reentrant
            autograd.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if checkpoint.state(module).enable_hook:
            checkpoint.state(module).orig_grad_enabled = torch.is_grad_enabled()
            if checkpoint.state(module).use_reentrant:
                torch.set_grad_enabled(False)
                checkpoint.state(module).fwd_cpu_state = torch.get_rng_state()
                # Don't eagerly initialize the cuda context by accident.
                # (If the user intends that the context is initialized later, within their
                # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
                # we have no way to anticipate this will happen before we run the function.)
                checkpoint.state(module).had_cuda_in_fwd = False
                if torch.cuda._initialized:
                    checkpoint.state(module).had_cuda_in_fwd = True
                    (
                        checkpoint.state(module).fwd_gpu_devices,
                        checkpoint.state(module).fwd_gpu_states,
                    ) = get_device_states(*inputs)

            else:
                # The Holder object for each of the saved object is saved
                # directly on the SavedVariable and is cleared when reset_data()
                # is called on it. We MUST make sure that this is the only
                # object having an owning reference to ensure that the Tensor
                # stored in storage is deleted as soon as the corresponding
                # SavedVariable data is cleared.
                storage: WeakKeyDictionary = WeakKeyDictionary()
                weak_holder_list: List[ReferenceType] = []
                saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(
                    partial(_pack, weak_holder_list=weak_holder_list),
                    partial(
                        _unpack,
                        storage=storage,
                        weak_holder_list=weak_holder_list,
                        module=module,
                        inputs=inputs,
                    ),
                )
                saved_tensor_hooks.__enter__()
                checkpoint.state(module).saved_tensor_hooks = saved_tensor_hooks

    def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            torch.set_grad_enabled(checkpoint.state(module).orig_grad_enabled)
            if checkpoint.state(module).use_reentrant:
                return _ModuleHookCheckpointFunction.apply(module, output, *inputs)
            else:
                checkpoint.state(module).saved_tensor_hooks.__exit__()
                checkpoint.state(module).saved_tensor_hooks = None

        return output

    # This hook does the following things:
    # 1. detach outputs from the autograd graph to discard activations
    # 2. insert an autograd.Function after the forward pass to recompute
    #    activations during the backward pass.
    checkpoint.state(module).enable_hook = True
    checkpoint.state(module).use_reentrant = use_reentrant
    module.register_forward_pre_hook(forward_pre_hook)
    # Use prepend to make sure we restore the original grad enabled state right
    # after the module forward invocation.
    module.register_forward_hook(forward_hook, prepend=True)
    return module
