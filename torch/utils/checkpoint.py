# mypy: allow-untyped-defs
import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import *  # noqa: F403
import enum
from weakref import ReferenceType

import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from typing import NoReturn

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointError",
    "CheckpointFunction",
    "check_backward_validity",
    "detach_variable",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "CheckpointPolicy",
    "SelectiveCheckpointContext",
    "create_selective_checkpoint_contexts",
    "SAC_IGNORED_OPS",
    "GraphExecGroup",
]

_DEFAULT_DETERMINISM_MODE = "default"

_checkpoint_debug_enabled: Optional[bool] = None


@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: Optional[bool]):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None", stacklevel=2
        )


def _get_device_module(device="cuda"):
    if device == "meta":
        return torch.device("meta")
    device_module = getattr(torch, device)
    return device_module


class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type = None

    @staticmethod
    def set_device_type(device: str = "cuda") -> None:
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        if not DefaultDeviceType._default_device_type:
            DefaultDeviceType._default_device_type = acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"

        return DefaultDeviceType._default_device_type


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg) -> None:
        nonlocal device_types
        if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
            device_types.append(arg.device.type)
    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}", stacklevel=2
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types_set:
        return "cuda"
    else:
        return device_types[0]


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_device_ids = []

    def add_device_ids(arg) -> None:
        nonlocal fwd_device_ids
        if isinstance(arg, torch.Tensor) and arg.device.type not in {"cpu", "meta"}:
            fwd_device_ids.append(arg.get_device())
    tree_map(add_device_ids, args)

    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())

    return fwd_device_ids, fwd_device_states


def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states, strict=False):
        with device_module.device(device):
            device_module.set_rng_state(state)


def _get_autocast_kwargs(device_type="cuda"):
    if torch.amp.is_autocast_available(device_type):
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled('cpu'),
        "dtype": torch.get_autocast_dtype('cpu'),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
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
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads


def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()

# Note: [torch.compile and checkpoint]
# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.9 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        args: tuple containing inputs to the :attr:`function`

    Keyword args:
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.9 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        early_stop(bool, optional): If ``True``, non-reentrant checkpoint stops
            recomputation as soon as it has computed all needed Tensors. This
            argument is ignored if ``use_reentrant=True``. Can be overridden
            globally using :func:`set_checkpoint_early_stop` context manager.
            Default: ``True``.

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. Starting in PyTorch 2.9, calling checkpoint "
            "without use_reentrant will raise an exception. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, early_stop, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    r"""Checkpoint a sequential model to save memory.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will not store
    the intermediate activations. The inputs of each checkpointed segment will
    be saved for re-running the segment in the backward pass.

    .. warning::
        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.9 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True` variant, please see
        :func:`~torch.utils.checkpoint.checkpoint` for
        the important considerations and limitations of this variant. It is
        recommended that you use ``use_reentrant=False``.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint.checkpoint_sequential: the use_reentrant "
            "parameter should be passed explicitly. "
            "In version 2.9 we will raise an exception if use_reentrant "
            "is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.", stacklevel=2
        )
        use_reentrant = True

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions),
            input,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve,
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)


def _internal_assert(cond) -> None:
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch."
        )


# NOTE [ Nestable Checkpoint ]
#
# The semantics of nested checkpoint can be defined by two basic rules.
# Following the two rules leads to an important implication that is central
# to motivating the design.
#
# Rule 1. Saved tensors are managed by inner-most checkpoint only and hidden
#         from any outer layers of checkpoint.
#
# Rule 2. The inputs of inner checkpoints are treated as tensors saved to its
#         parent checkpoint.
#
# Implication: To recompute any given saved tensor, we need to recompute all of
#              the checkpoints wrapping it.
#
# Why is this implied? To unpack a saved tensor X during backward we need to
# recompute the inner-most checkpoint (#1), and in order to recompute that
# checkpoint I need to have its inputs, which are managed by that checkpoint's
# parent (#2), which thus also needs to be recomputed first. Continue this line
# of reasoning and we realize that in order to unpack X, all checkpoints that
# were active at the time X was saved need to be recomputed. (unless we have
# already done so in that backward for some other saved tensor).
#
# In practice, we use a noop autograd Function to save inputs as saved tensors.
# During unpack calling ctx.saved_tensor triggers the parent checkpoint to
# recompute.
#
# Rule 3. We should start recomputation as if there are no checkpoints currently
#         active. Checkpoints encountered during recomputation are still
#         respected.
#
# When we start recomputation, we push the saved variable hook meant for
# recomputation on the stack. See examples in Rule 6 for more context.
#
#                                  * * * *
#
# Beyond the basic semantics specific to nested checkpoint, we impose several
# more constraints that may apply to checkpointing in general.
#
# Rule 4. Lifetime of recomputed tensors
#
#         Recomputed tensors are considered specific to particular invocations
#         of backward and are always cleared immediately as they are unpacked
#         Particularly, we require this to happen even if retain_graph=True.
#
# [ Implementation details of Rule 4 ]
#
# If we were okay with recomputed tensors staying alive after backward is run
# with retain_graph=True, we would store recomputed variables as the values of a
# WeakKeyDictionary and pack strong references to the keys, so that as we
# backward, those packed keys would be cleared as long as retain_graph=False.
# Clearing the packed key clears the corresponding entry in the WKD.
#
# If we wish recomputed variables to be immediately cleared as we unpack them in
# the retain_graph=True case, we cannot rely on the packed keys to be cleared by
# backward automatically. Instead of packing the strong reference to the key
# directly, we pack a container object, which we manually clear as we unpack.
#
# An important detail is that if a second backward happens, the second
# recomputation needs to reset the container with a newly created key.
#
# Rule 5. Stop recomputation as soon as we've recomputed the saved tensors we
#         know we need.
#
# [ Implementation details of Rule 5 ]
#
# During recomputation, raise an exception if the number of recomputed tensors
# matches the number of tensors that we expected to recompute. We wrap the
# recomputation call with a try-catch to catch this specific exception. See
# Rule #6 below for some examples.
#
# Rule 6. We support doing backward inside checkpoint context
#
# [ retain_graph is True]
#
# def fn(x):
#   y = x.sin()
#   z = y.cos()
#   gx, = torch.autograd.grad(z, x, retains_grad=True)
#   return gx, z
#
# out = checkpoint(fn)(inp)
# out.backward()
#
# Because z is saved by cos while checkpoint is enabled, it would not be
# actually saved, and so the .grad() call inside must trigger a recomputation.
#
# During recomputation the "inner pack hook" has two responsibilities:
#
# 1) As usual, populating the WeakKeyDictionary storing recomputed tensors
# 2) Pack the actual tensor (detached) so that one may perform backward on the
#    recomputed graph. The tensors saved to this graph will live until the end
#    of recomputation, or die earlier if someone performs backward with
#    retain_graph=False.
#
# More generally performing backward on the recomputed graph occurs in the
# following cases:
# - If backward is performed inside forward,
#   - During the original forward IF early-stop is disabled
#   - During the original backward
# - If there are multiple .grad()/.backward() calls, we would perform backward
#   on the recomputed graph even if early-stop is enabled (see the example below)
#
# [ retain_graph is False ]
#
# The example below shows what happens if during recomputation we find that some
# of the tensors we are trying to recompute have already been cleared.
#
# Spoiler: we don't do anything special, we just skip over them!
#
# def fn(x):
#   y = x.sin()                           # (1)
#   z = y.cos()                           # (2)
#   gx, = torch.autograd.grad(z, x)       # (3)
#   return x.cos() * gx                   # (4)
#
# out = checkpoint(fn)(inp)
# out.backward()                          # (5)
#
# 1, 2. Don't save x and y since we are inside a checkpoint.
# 3. Trigger a recompute of fn since x and y weren't saved.
#    And depending on whether early stop is enabled, either stop at (2) or
#    continue running the function.
#    Because we are running backward with retain_graph=False, we clear x and y's
#    holders.
# 4. Don't save x since we are inside a checkpoint.
# 5. Calling backward triggers another recompute of fn. During recompute, we see
#    that x and y have already been cleared in the original graph as indicated
#    by holder=None. We skip over them. We still save x at (4) (since its holder
#    is still alive.)

_enable_checkpoint_early_stop: Optional[bool] = None


@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    """Context manager that sets whether checkpoint should stop recomputation early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Example::

    >>> # xdoctest: +SKIP(failing)
    >>> message = "saved tensors default hooks are disabled"
    >>> with set_checkpoint_early_stop(False):
    ...     # Any checkpoint under this context manager will respect this
    ...     # context manager, even if its backward is performed outside.
    ...     out = checkpoint(fn, inputs)
    ...
    >>> out.backward()
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev


class _Handle:
    pass


class _Holder:
    def __init__(self) -> None:
        self.handles: Dict[int, Optional[_Handle]] = {}


class _NoopSaveInputs(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(*args):
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        # Only tensors can be saved with ctx.save_for_backward, everything else
        # is captured by get_args, which is saved directly on ctx
        tensor_indices, tensors = zip(
            *[(i, o) for i, o in enumerate(inputs) if isinstance(o, torch.Tensor)], strict=False
        )
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        # args but with tensors replaced with None as placeholders
        args = [None if isinstance(o, torch.Tensor) else o for o in inputs]

        def get_args(saved_tensors):
            # restore the placeholders with the original tensors grabbed from
            # ctx.saved_tensors (which may be saved on a parent checkpoint if
            # this checkpoint is nested, and that would trigger a recursive
            # unpack!)
            ret = [
                saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o
                for i, o in enumerate(args)
            ]
            # grab the tail since we also saved the dummy to avoid having to explicitly
            # handle the case where there are no tensor inputs
            return ret[1:]

        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs) -> NoReturn:
        raise AssertionError("Did not expect to backward on this graph")


class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn) -> None:
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[ReferenceType] = []
        # We store this as a weakkeydictionary so that in the case of a partial
        # backward, the entries in the dict are cleared alongside the Holder
        # which will be removed when the SavedVariable is cleared.
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
        # We need both recomp_counter and recomputed since they can diverge
        # https://github.com/pytorch/pytorch/pull/90105#discussion_r1135889885
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)

        # See Rule 5
        self.early_stop = early_stop

        # Debugging
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

    def check_recomputed_tensors_match(self, gid) -> None:
        if self.ignore_saved_mismatch:
            # TODO: we can probably make this check stricter by checking that
            #       the metadata of the first tensors still match.
            return
        # NOTE [ Error handling for checkpoint ]
        #
        # At a high level, we need to check that the tensors saved
        # during original forward matches tensors saved during recompute
        # This means handling 3 cases:
        #
        # 1. During recompute, more tensors were saved.
        #
        #    Usually this is hidden due to the StopRecomputationError
        #    but if early stop is not enabled, or we would have errored
        #    anyway because there aren't enough weak_holders. But we
        #    do want to have a nice error. See the _recomputation_hook
        #    for details.
        if not len(self.weak_holders) == self.recomp_counter[gid]:
            # 2. During recompute, fewer tensors were saved
            #
            # We know that every time we save something do original forward
            # we append to weak_holder, and every time we save a tensor
            # during recompute we increment recompute_counter.
            raise CheckpointError(
                "torch.utils.checkpoint: A different number of tensors was saved "
                "during the original forward and recomputation.\n"
                f"Number of tensors saved during forward: {len(self.weak_holders)}\n"
                f"Number of tensors saved during recomputation: {self.recomp_counter[gid]}.\n"
                f"{_debug_tip_msg}"
            )

        # 3. During recompute, the same tensors were saved, but they
        #    have different metadata
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            # We've seen all holders since we iterate over them in order
            # For every holder that is still alive now, it must've been
            # alive when we saw it during recompute, therefore, the
            # gid must be set.
            _internal_assert(gid in holder.handles)
            # We know this is the first unpack, so it couldn't have been set
            # to None yet.
            _internal_assert(holder.handles[gid] is not None)
            # We always set these together in the recomputation hook
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            # see pack hook, x_metadata is 1:1 with weak_holders.
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))

        if len(nb_meta_different) > 0:
            mismatched_tensors = ""
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += (
                    f"tensor at position {idx}:\n"
                    f"saved metadata: {x_meta}\n"
                    f"recomputed metadata: {recomputed_meta}\n"
                )
            raise CheckpointError(
                "torch.utils.checkpoint: Recomputed values for the following tensors "
                "have different metadata than during the forward pass.\n"
                f"{mismatched_tensors}.\n"
                f"{_debug_tip_msg}"
            )


_debug_tip_msg = """
Tip: To see a more detailed error message, either pass `debug=True` to
`torch.utils.checkpoint.checkpoint(...)` or wrap the code block
with `with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):` to
enable checkpoint‑debug mode globally.
"""


_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `torch.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing

 You are seeing this error because you passed `debug=True` to checkpoint and
 tensors to be saved during the original forward and differ between those saved
 during recomputation. This can happen if different operators were ran in the
 original forward and in the recomputation.

 To identify where the mismatch may be coming from, you can do the following:

 1) Compare the operators ran during original forward and recomputation to
    see where they differ. These operators are printed above in the order they
    were executed.

 2) Review the stack trace for each operator to locate its invocation source.
    Each operator's stack trace is printed in their execution order.

 Note that the logs can be quite long. Here's how they are structured:
 (Tip: you can Ctrl-f for these headers)

 1. Stack traces of the operators that ran in the original forward
 2. Stack traces of the operators that ran during recomputation
 3. Log of operators in the original forward and recomputation
 4. Error message                                             <--- You are here
--------------------------------------------------------------------------------
"""

class CheckpointError(RuntimeError):
    pass


def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    # This function returns the context_fn and error_cb to be used by the
    # checkpointing mechanism. error_cb is invoked when an error is detected
    # during unpack.

    # record_context_cpp is not support on non-linux non-x86_64 platforms
    cpp_tb = platform.machine() == 'x86_64' and platform.system() == 'Linux'

    class CaptureLogs:
        def __init__(self) -> None:
            self.logs = None
            self.tbs = None

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    # pyrefly: ignore [bad-assignment]
                    self.logs, self.tbs = logs_and_tb
                    yield logs_and_tb
            return logging_mode()

    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    def unpack_error_cb(e: CheckpointError) -> NoReturn:
        def get_str_tb(label, capture_logs):
            out = ""
            total_len = len(capture_logs.logs)
            for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs, strict=False)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    # Start printing stack trace only after __torch_dispatch__ is found
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        if capture_logs_fwd.logs is None:
            raise AssertionError("capture_logs_fwd.logs is None")
        if capture_logs_recompute.logs is None:
            raise AssertionError("capture_logs_recompute.logs is None")
        raise CheckpointError(
            _checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs)
            )
        ) from e

    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    return context_fn, unpack_error_cb

def _default_meta_extractor(x: torch.Tensor) -> Dict[str, Any]:
    # These properties are fast to check, easy to understand
    return {
        "shape": x.shape,
        "dtype": x.dtype,
        "device": x.device
    }

_allowed_determinism_checks_to_fns: Dict[str, Callable[[torch.Tensor], Any]] = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,
    "none": lambda _: None,
}

# See Rule 5
class _StopRecomputationError(Exception):
    pass


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: Union["GraphExecGroup", int]) -> None:
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            target_frame = target_frame_ref()
            if target_frame is None:
                raise AssertionError("Internal error: target_frame reference is None")
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                if target_frame.early_stop:
                    raise AssertionError("Unexpected state: target_frame.early_stop is set")
                if not target_frame.forward_completed:
                    # We run into this case when early stop is not enabled and do
                    # grad within checkpoint.
                    # We need to set this flag, so we don't error out later when
                    # we check if the number of tensors saved during forward and
                    # recomputation match.
                    target_frame.ignore_saved_mismatch = True
                    return x
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass.\n"
                    f"{_debug_tip_msg}"
                )

            holder = target_frame.weak_holders[recomp_idx]()

            # This holder may have been cleared because someone may have called
            # backward within forward. If so, we don't need to save.
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x

            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError
            # See Rule 6: [ retain_graph is True ] above
            return x

        def unpack_hook(x):
            # See Rule 6: [ retain_graph is True ] above for an example of when
            # the graph created during recomputation could be backwarded.
            return x

        super().__init__(pack_hook, unpack_hook)


# torch._disable_dynamo creates a reference cycle with decorated function
# This function is used to ensure that the decorated function does not have
# a closure, so that other objects aren't also kept alive.
# https://github.com/pytorch/pytorch/issues/154642
# Note: does not work when fn is compiled
@torch._disable_dynamo
def _run_fn_with_dynamo_disabled(fn, *args, **kwargs):
    return fn(*args, **kwargs)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame) -> None:
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            # First check if we're inside a GraphExecGroup context
            gid: Union[GraphExecGroup, None, int] = GraphExecGroup._get_current_group()
            if gid is None:
                # Fallback to using the current graph task id
                gid = torch._C._current_graph_task_id()
                if gid == -1:
                    # generate a temporary id if we trigger unpack outside of a backward call
                    gid = int(uuid.uuid4())

            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        # See Note: [compiled autograd and checkpoint unpack hook]
                        _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                extra = ""
                if torch._C._get_graph_exec_group() is not None:
                    extra = (
                        "Performing two backward calls that overlap (i.e. require the same "
                        "saved activation in order to compute gradients) is not allowed while "
                        "under the torch.utils.checkpoint.GraphExecGroup context. "
                    )
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    f"unpacked once. {extra}If you are calling ctx.saved_tensors in backward, make sure "
                    "to do so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)


def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing
    # Checking that a functional mode is active should always do what we want
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val) -> None:
        self.val: Union[torch.Tensor, Any] = val
        self.version: Optional[int] = val._version if isinstance(val, torch.Tensor) else None

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val


def _maybe_detach(x, any_ret_has_alias_info):
    # We detach for two separate reasons:
    # - For view ops, we need to ensure that when the tensor is returned from
    #   CachedDispatchMode, as_view sees that the AutogradMeta is nullptr
    # - Avoid reference cycles
    # For case 1, it is not enough to check whether x has differentiable dtype
    # because non-differentiable dtype can have non-nullptr AutogradMeta, e.g.
    # when the tensor is a view.
    if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex() or any_ret_has_alias_info):
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False):
            # Ensure that view performed beneath autograd properly propagates
            # version counter. TODO: Use reentrant_dispatch instead of
            # manually manipulating dispatch keys. Using reentrant_dispatch
            # would respect inference_mode, though that is not relevant for
            # this case.
            x = x.detach()
    return x


class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.

    This class is used to pass relevant metadata to the policy function during
    selective checkpointing. The metadata includes whether the current invocation
    of the policy function is during recomputation or not.

    Example:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.is_recompute)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    def __init__(self, *, is_recompute) -> None:
        self.is_recompute = is_recompute


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3


def _policy_from_bool(b):
    # For backward compatibility
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE


SAC_IGNORED_OPS = {
    # AC inserts different number of detach during forward and recompute.
    torch.ops.aten.detach.default,
    # AC's determinism check invokes additional metadata ops during forward.
    # With subclasses involved, these metadata ops become dispatchable, this
    # can result in incorrectness if these ops are selected cached.
    torch.ops.prim.device.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)  # type: ignore[has-type]


class _CachingTorchDispatchMode(TorchDispatchMode):
    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage) -> None:
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False),
                                func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            # Overwrite each node's "recompute" tag to add in the user annotation.
            fx_traceback.current_meta["recompute"] = policy

        out = func(*args, **kwargs)

        # HOPs don't support func._schema
        # HOPs don't alias -> this is always true today and will be always true for a long time
        # TODO HOPs don't mutate -> this is always true today but will not be true forever
        if isinstance(func, torch._ops.HigherOrderOperator):
            any_ret_has_alias_info = False
        else:
            any_ret_has_alias_info = any(ret.alias_info is not None for ret in func._schema.returns)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            self.storage[func].append(tree_map(lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)), out))
        return out

class _CachedTorchDispatchMode(TorchDispatchMode):
    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=True),
                                func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            storage = self.storage.get(func)
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            out = tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0))
        else:
            out = func(*args, **kwargs)
        return out


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, List[Any]] = defaultdict(list)
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
    )

# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.

def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    *args,
    **kwargs
):
    """Checkpointing without reentrant autograd.

    Args:
        fn: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        early_stop(bool, optional): If ``True``, non-reentrant checkpoint stops
            recomputation as soon as it has computed all needed Tensors. Can be
            overridden globally using :func:`set_checkpoint_early_stop` context
            manager. Default: ``True``.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn is not noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args)
    device_module = _get_device_module(device_type)
    forward_context, recompute_context = context_fn()
    if _is_compiling(fn, args, kwargs) and context_fn is not noop_context_fn:
        if (
            not isinstance(forward_context, TorchDispatchMode)
            or not isinstance(recompute_context, TorchDispatchMode)
        ):
            raise AssertionError(
                "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` "
                "must generate a tuple of two `TorchDispatchMode`s."
            )
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs) -> None:
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, device_type=device_type
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=device_type, **device_autocast_kwargs
            ) if torch.amp.is_autocast_available(device_type) else contextlib.nullcontext()
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop if _enable_checkpoint_early_stop is not None else early_stop,
        unpack_error_cb,
        metadata_fn
    )
    dummy = torch.empty((0,), requires_grad=True)
    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return


class GraphExecGroup:
    """Any checkpointed regions encountered by backward under the same instance
    of this context manager will trigger recompute at most once, even if
    there are multiple calls to backward.

    Backward calls under the same instance of this context manager must execute
    over non-overlapping regions of the backward graph even if retain_graph=True.
    In particular, any two backward call cannot use the same saved activation for
    gradient computation.

    .. note::
        This context manager only affects checkpoint with use_reentrant=False, and
        is a no-op otherwise.
    """

    def __enter__(self) -> "GraphExecGroup":
        if torch._C._get_graph_exec_group() is not None:
            raise RuntimeError(
                "GraphExecGroup contexts cannot be nested. "
                f"Already inside group {torch._C._get_graph_exec_group()}"
            )
        torch._C._set_graph_exec_group(self)
        return self

    def __exit__(self, *args: object) -> None:
        torch._C._set_graph_exec_group(None)

    @classmethod
    def _get_current_group(cls) -> Optional["GraphExecGroup"]:
        # Private API to be used by utils like AC
        return torch._C._get_graph_exec_group()


# Note: [compiled autograd and checkpoint unpack hook]
# When tracing via compiled autograd, this hook will be visible to the
# compiler if the forward of this checkpointed region ran in eager.
# If the forward had ran under compile, it would have been wrapped in a
# higher order op. See Note: [torch.compile and checkpoint].
#
# Since we run the recomputation hook under a enable_grad context,
# AOTDispatch will trace a joint graph for this hook, and may
# save different activations than in eager. This conflicts with the
# strict activation count checks in `frame.check_recomputed_tensors_match`.
# So, we disable this hook to force it to recompute eager checkpointed regions
# in eager. This could be removed if we can disable the partitioner for this
# graph segment.
