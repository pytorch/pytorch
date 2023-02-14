import torch
import warnings
import weakref
from typing import Any, Iterable, List, Tuple, Dict, Optional, DefaultDict, NamedTuple
from collections import defaultdict

__all__ = [
    "checkpoint", "checkpoint_sequential", "CheckpointFunction",
    "check_backward_validity", "detach_variable", "get_device_states",
    "set_device_states",
]

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
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


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
    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def _get_autocast_kwargs():
    gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                           "dtype": torch.get_autocast_gpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                           "dtype": torch.get_autocast_cpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    return gpu_autocast_kwargs, cpu_autocast_kwargs

class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

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
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
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
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), \
                 torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
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
                " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, *args, use_reentrant: bool = True, **kwargs):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.


    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        If ``use_reentrant=True`` is specified, then if the checkpointed segment
        contains tensors detached from the computational graph by `detach()` or
        `torch.no_grad()`, the backward pass will raise an error. This is
        because `checkpoint` makes all the outputs require gradients which
        causes issues when a tensor is defined to have no gradient in the model.
        To circumvent this, detach the tensors outside of the `checkpoint`
        function. Note that the checkpointed segment can contain tensors
        detached from the computational graph if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, at least one of the inputs needs
        to have :code:`requires_grad=True` if grads are needed for model inputs,
        otherwise the checkpointed part of the model won't have gradients. At
        least one of the outputs needs to have :code:`requires_grad=True` as
        well. Note that this does not apply if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, checkpointing currently only
        supports :func:`torch.autograd.backward` and only if its `inputs`
        argument is not passed. :func:`torch.autograd.grad`
        is not supported. If ``use_reentrant=False`` is specified, checkpointing
        will work with :func:`torch.autograd.grad`.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function. Note that future
            versions of PyTorch will default to ``use_reentrant=False``.
            Default: ``True``
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if use_reentrant:
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        return _checkpoint_without_reentrant(
            function,
            preserve,
            *args,
            **kwargs,
        )


def checkpoint_sequential(functions, segments, input, use_reentrant=True, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

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
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function.
            Default: ``True``

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

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
            preserve_rng_state=preserve
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)

def _checkpoint_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    """Checkpointining without re-entrant autograd
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)

    # Custom class to be able to take weak references
    class Holder():
        pass
    # The Holder object for each of the saved object is saved directly on the
    # SavedVariable and is cleared when reset_data() is called on it. We MUST make
    # sure that this is the only object having an owning reference to ensure that
    # the Tensor stored in storage is deleted as soon as the corresponding SavedVariable
    # data is cleared.
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    def pack(x):
        # TODO(varal7): Instead of returning abstract object, we can return things metadata (such as
        # size, device, ...) to catch certain cases of undeterministic behavior of the forward
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res


    def unpack(x):
        unpack_counter = 0
        if len(storage) == 0:
            def inner_pack(inner):
                nonlocal unpack_counter
                unpack_counter += 1
                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return
                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed):
                raise RuntimeError("You are calling backwards on a tensor that is never exposed. Please open an issue.")

            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if preserve_rng_state and had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        set_device_states(fwd_gpu_devices, fwd_gpu_states)

                with torch.enable_grad(), \
                     torch.cuda.amp.autocast(**gpu_autocast_kwargs), \
                     torch.cpu.amp.autocast(**cpu_autocast_kwargs), \
                     torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args, **kwargs)

        if x not in storage:
            raise RuntimeError(
                "Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                " recomputation being triggered in between, this is not currently supported. Please"
                " open an issue with details on your use case so that we can prioritize adding this."
            )

        return storage[x]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
        if torch.cuda._initialized and preserve_rng_state and not had_cuda_in_fwd:
            # Cuda was not initialized before running the forward, so we didn't
            # stash the CUDA state.
            raise RuntimeError(
                "PyTorch's CUDA state was initialized in the forward pass "
                "of a Checkpoint, which is not allowed. Please open an issue "
                "if you need this feature.")

    return output


# NOTE: [Checkpoint that supports nesting]
#
# Contents:
# - Semantics
# - Mechanics
#
# Checkpointing Semantics (this should just be in the docs)
# =========================================================
#
# Under a checkpoint context, any variables that would've been saved by
# forward under this context would no longer be saved.
#
# Lifetime of recomputed variables
# --------------------------------
#
# Recomputed variables only stay alive for the lifetime of a particular
# backward.
#
# Backward within checkpoint
# --------------------------
#
# If backward is performed inside a checkpoint context...
#
# with checkpoint():
#    y = x.sin()                # saves x
#    z = y.exp()                # saves z
#    torch.autograd.grad(z, z)  # clears z, only x remains saved
#                               # As we exit, clears x only
#
# Nested checkpointing
# --------------------
#
# There is some specially handling for the nested case: the inputs to
# are treated as saved variables in the parent context.
#
# with checkpoint0():
#   with checkpoint1():          # saves `y` in check0
#     y = f(x)                   # f's saved variables are cleared by check1
#   with checkpoint2():          # saves `z` in check0
#     z = g(y)                   # g's saved variables are cleared by check1
#                                # exiting check0, clears `y` and `z`
#                                # whatever f and g save are hidden#
#
# Early stopping
# --------------
#
# Checkpointing Mechanics
# =======================
#
# 1) Doing backward inside
#
# This explains why someone might unpack something packed during recomputation.
# If someone tries to unpack this pack, that means that they are calling backward
# in a checkpoint. No need to be nested. It is okay if we just save the tensor normally,
# i.e. tie the tensor's lifetime to the recomputation graph because the recomputation graph will
# die as soon as we finish recomputation.
#
# 2) Doing backward inside, but we are also nested
#
# This happens when doing backward within checkpoint. I need to have the ability
# to recompute my top-level checkpoint WITHIN the recomputation stack. Consider the case:
#
# checkpoint
#    checkpoint
#       f
#    backward
#
# During recomputation, the inner checkpoint is no longer nested, but we still want
# to pretend as if it is, i.e., we need to save the inputs to parent, so that the
# indices match up. However, if we do backward, during recomputation, we aren't able
# able to recompute the outer parent to obtain the inputs that way since it isn't truly
# nested. What we do is directly save to the current frame as well.
#
# Why is this okay to do?
#
# 2) Two disjoint backwards
#
# This answers the question of how can handle be None?
#
# We want to save in the recomputation graph even if handle is None, i.e., x has already
# been cleared in the original graph. To see why this is the case,
# consider the case where we do two separate backwards on disjoint parts of the graph,
# we need to rerun that first backward to get to the second one.
#
#
# To be done before landing:
# - Test memory usage on CUDA and try to achieve potential the theoretical O(log(n))
#   memory usage reduction
# - Save and restore rng, autocast state
# - Maybe support non-Tensor inputs *args, **kwargs, should just be an API issue.
# - Finish writing the docs here


# NOTE: [Nested Checkpoint Input Handling]
#
# Checkpoint frames need to store the inputs in order to recompute saved tensors.
# We handle the storing of in two ways depending on whether the checkpoint is
# nested.
#
# (1) Non-nested case
# The simplest case is where there is no nesting, and we store args directly
# on the checkpoint frame. If this frame is nested, maybe_args is None.

# (2) Nested case
# In the nested case however, we need to let the parent manage what we save
# and this includes the inputs of the child checkpoints.
# To try to reuse the usual checkpoint logic to handle these, we rely on a dummy
# autograd Function to save inputs as saved tensors. However, Inputs differ from
# normal saved tensors in at that we don't unpack before using them. This means
# that (1) we cannot rely on the packed handle to identify them and so we need a
# couple extra fields so to distinguish them later. (2) we should not detach them.


class _Handle():
    pass

# Reimplementation of torch.distributed.utils.{_pack,_unpack}_kwargs to avoid a import cycle
def _pack_kwargs(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    kwarg_keys: List[str] = []
    flat_args: List[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)

def _unpack_kwargs(flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs

class NoopSaveInputs(torch.autograd.Function):
    # autograd Function that saves inputs and returns them as-is
    # TODO: We need to wrap this to emulate the requires_gradness of the inputs
    @staticmethod
    def forward(*args):
        return args

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs

def applyAutogradFunctionToSaveInputs(*args):
    # preserve the requires_grad-ness of the inputs
    idx_no_req_grad = [i for i, t in enumerate(args) if isinstance(t, torch.Tensor) \
                       and not t.requires_grad and (t.is_floating_point() or t.is_complex())]
    new_args = NoopSaveInputs.apply(*args)
    return tuple(t.detach() if i in idx_no_req_grad else t for i, t in enumerate(new_args))

class _CheckpointFrame():
    def __init__(self, wrapped_fn):
        # Stores a wrapped function, where the proper contexts are captured
        self.wrapped_fn = wrapped_fn

        # See NOTE: [Nested Checkpoint Input Handling]
        self.maybe_args: Optional[List[torch.Tensor]] = None
        self.args_idx: Optional[List[int]] = None
        self.child_args_idx : List[int] = []
        self.args_handles: Optional[List[Any]] = None

        self.weak_handles: List[weakref.ref[_Handle]] = []

        self.recomputed: DefaultDict[int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]] = \
            defaultdict(weakref.WeakKeyDictionary)
        self.recomp_counter: DefaultDict[int, int] = defaultdict(lambda: 0)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(lambda: False)

    def __repr__(self):
        return f"Frame({id(self)})"

    def get_args_from_parent(self, parent_frame, gid):
        assert self.args_idx is not None
        out = []
        for idx in self.args_idx:
            handle = parent_frame.weak_handles[idx]()
            assert handle is not None
            if handle:
                out.append(parent_frame.recomputed[gid][handle])
        return out

    def save_args_to_parent(self, parent_frame, args):
        parent_pre_len = len(parent_frame.weak_handles)
        new_args = applyAutogradFunctionToSaveInputs(*args)
        self.args_handles = parent_frame.weak_handles[parent_pre_len: len(parent_frame.weak_handles)]
        indices = list(range(parent_pre_len, len(parent_frame.weak_handles)))
        parent_frame.child_args_idx.extend(indices)
        self.args_idx = indices
        return new_args

class _CheckpointStack(NamedTuple):
    stack: List[_CheckpointFrame]
    is_recompute: bool

# This stack is synced with the saved_tensor_hook stack.
# When _recomputation_hook is pushed onto the hook stack, we also push a new
# empty _CheckpointStack
_checkpoint_stacks: List[_CheckpointStack] = \
    [_CheckpointStack(stack=[], is_recompute=False)]

class _StopRecomputationError(Exception):
    pass

class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: weakref.ref[_CheckpointFrame], gid: int):
        def pack_hook(x):
            target_frame = target_frame_ref()
            assert target_frame is not None
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1
            handle = target_frame.weak_handles[recomp_idx]()

            if handle is not None:
                # See Checkpointing Mechanics (3) to see when handle can be None
                target_frame.recomputed[gid][handle] = \
                    x if recomp_idx in target_frame.child_args_idx else x.detach()

            if target_frame.recomp_counter[gid] == len(target_frame.weak_handles):
                raise _StopRecomputationError()
            # See Checkpointing Mechanics (1)
            return x.detach()

        def unpack_hook(x):
            return x

        super().__init__(pack_hook, unpack_hook)

class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self):
        def pack_hook(x):
            # Snapshot the state of the current checkpoint stack
            current_frames, is_recompute = _checkpoint_stacks[-1]
            top_frame = current_frames[-1]
            handle = _Handle()
            top_frame.weak_handles.append(weakref.ref(handle))
            return handle, tuple(current_frames)

        def unpack_hook(saved):
            handle, frames = saved

            top_frame = frames[-1]
            gid = torch._C._current_graph_task_id()
            assert gid != -1, "checkpoint: expected unpack to be called inside a backward call"

            for i in range(len(frames)):
                frame = frames[i]
                if frame.is_recomputed[gid]:
                    continue
                # See NOTE [Nested Checkpoint Input Handling]
                if frame.maybe_args is None:
                    args = frame.get_args_from_parent(frames[i - 1], gid)
                else:
                    args = frame.maybe_args
                try:
                    _checkpoint_stacks.append(_CheckpointStack(stack=[], is_recompute=True))
                    # pass gid in in case we do reentrant backward
                    with _recomputation_hook(weakref.ref(frame), gid), torch.autograd.enable_grad():
                        frame.wrapped_fn(*args)
                except _StopRecomputationError as e:
                    _checkpoint_stacks.pop()
                    pass
                frame.is_recomputed[gid] = True

            return top_frame.recomputed[gid][handle]

        super().__init__(pack_hook, unpack_hook)

def _checkpoint(fn, *args, **kwargs):
    # Calls checkpoint_impl with a wrapped version of fn.
    #
    # The wrapper handles:
    # - kwargs. The inner checkpoint function only handles flat args
    # - capturing any global state (e.g. rng, autocast) that needs to be restored
    #   during recomputation if necessary
    preserve_rng_state = kwargs.pop('preserve_rng_state', True)

    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)

    # From checkpoint_wrapper.
    # We should modify to handle non-tensor, kwargs
    flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)
    def new_fn(*inputs):
        rng_devices = []
        if preserve_rng_state and had_cuda_in_fwd:
            rng_devices = fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_cuda_in_fwd:
                    set_device_states(fwd_gpu_devices, fwd_gpu_states)

        unpacked_args, unpacked_kwargs = _unpack_kwargs(
            inputs, kwarg_keys
        )
        with torch.cuda.amp.autocast(**gpu_autocast_kwargs), \
             torch.cpu.amp.autocast(**cpu_autocast_kwargs):
            return fn(*unpacked_args, **unpacked_kwargs)

    return _checkpoint_impl(new_fn, *flat_args)

def _checkpoint_impl(fn, *args):
    curr_stack, is_curr_stack_recompute = _checkpoint_stacks[-1]
    new_frame = _CheckpointFrame(fn)

    # See NOTE [Nested Checkpoint Input Handling]
    if len(curr_stack) > 0:
        args = new_frame.save_args_to_parent(curr_stack[-1], args)
    elif is_curr_stack_recompute:
        # See Checkpointing Mechanics (2)
        args = applyAutogradFunctionToSaveInputs(*args)
        new_frame.maybe_args = args  # type: ignore[assignment]
    else:
        new_frame.maybe_args = args  # type: ignore[assignment]

    curr_stack.append(new_frame)
    with _checkpoint_hook():
        ret = fn(*args)
    curr_stack.pop()
    return ret
