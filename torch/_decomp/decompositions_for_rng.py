from collections import defaultdict
from typing import Callable, Dict

import torch
import torch._decomp as decomp
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import disable_fake_tensor_mode_tracing

aten = torch.ops.aten

rng_decompositions: Dict[str, Dict[OpOverload, Callable]] = defaultdict(dict)
offset_calculator = {}


def throw_if_philox_offset_did_not_advance(fn):
    # Raises error if one forgot to advance the offset while adding a new decomposition.
    def inner(*args, **kwargs):
        old_relative_offset = PhiloxStateTracker.get_current_relative_offset()
        out = fn(*args, **kwargs)
        new_relative_offset = PhiloxStateTracker.get_current_relative_offset()
        if old_relative_offset == new_relative_offset:
            raise ValueError("Philox seed offset did not advance after decomposition")
        return out

    return inner


def register_rng_decomposition(aten_op):
    def inner(fn):
        return decomp.register_decomposition(aten_op, rng_decompositions)(
            throw_if_philox_offset_did_not_advance(fn)
        )

    return inner


# Offset calculations depdens on the distribution
def register_offset_calculation(aten_ops):
    def inner(fn):
        nonlocal aten_ops
        aten_ops = list(aten_ops)
        for aten_op in aten_ops:
            offset_calculator[aten_op] = fn
        return fn

    return inner


def get_default_stride(size):
    """
    A helper function to get the strides for a contiguous tensor of a given
    shape.
    """
    stride = [1] * len(size) + [1]
    for idx in reversed(range(len(size))):
        stride[idx] = stride[idx + 1] * size[idx]
    stride = stride[1:]
    return stride


class RNGFunctionalizationError(Exception):
    pass


def throw_on_cpu():
    raise RNGFunctionalizationError(
        "You are trying to functionalize a CPU RNG operator but CPU does not use "
        "Philox/counter-based PRNG. Therefore, functionalizing a CPU RNG operator "
        "is not supported. We are discussing the possibility of a Philox-based "
        "RNG implementation for CPU. We will revisit this assertion in future."
    )


@register_offset_calculation([aten.rand])
def rand_offset_calculator(shape):
    # For impl, look at calc_execution_policy
    numel = 1
    for dim_size in shape:
        numel *= dim_size

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_per_sm = int(device_property.max_threads_per_multi_processor / block_size)
    grid_size = int((numel + block_size - 1) / block_size)
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)
    offset = (
        int((numel - 1) / (block_size * grid_size * unroll) + 1) * curand4_engine_calls
    )
    return offset


# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    device = device or "cpu"

    if device.type == "cpu":
        throw_on_cpu()
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or torch.float32
    stride = get_default_stride(shape)
    r = torch.ops.prims.philox_rand(shape, seed, offset, stride, device, dtype)
    PhiloxStateTracker.advance_offset(offset_calculator[aten.rand](shape))
    return r


@register_rng_decomposition(aten.rand_like)
def rand_like(
    x: torch.Tensor,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    device = device or x.device
    if device.type == "cpu":
        throw_on_cpu()
    dtype = dtype or x.dtype
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    r = torch.ops.prims.philox_rand(x.shape, seed, offset, x.stride(), device, dtype)
    PhiloxStateTracker.advance_offset(offset_calculator[aten.rand](x.shape))
    return r


class PhiloxState:
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totaly consumed offset at
    trace time.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.seed = torch.Tensor(0)
        self.base_offset = torch.Tensor(0)
        self.relative_offset = 0

    def advance_offset(self, consumed_offset):
        self.relative_offset += consumed_offset

    def set_state(self, seed, base_offset, relative_offset=0):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        # Only needed because we override get_rng_state.
        seed_portion = self.seed.reshape(1)
        offset_portion = (self.base_offset + self.relative_offset).reshape(1)
        return torch.cat([seed_portion, offset_portion])

    def set_state_from_tensor(self, state):
        # Only needed because we override set_rng_state.
        seed, offset = torch.split(state, 1)
        self.seed = seed[0]
        self.base_offset = offset[0]
        self.relative_offset = 0


class PhiloxTotalOffsets:
    """
    PhiloxStateTracker computes the total fwd and bwd offsets for an AOT
    Autograd traced graph. However, PhiloxStateTracker is a singleton class, but
    the total offsets are specific to each traced graph. These offsets are
    stored as part of AOT graph. This class just encapsulates the fwd and bwd
    offsets to be used at runtime.
    """

    def __init__(self, fwd_offset=0, bwd_offset=0):
        self.total_fwd_offset = fwd_offset
        self.total_bwd_offset = bwd_offset


class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. There are stored as part of
    aot_config (PhiloxTotalOffsetsForRuntime), which is a config object per
    aot-tracing.
    """

    running_state = PhiloxState()
    fwd_state = PhiloxState()
    bwd_state = PhiloxState()

    @classmethod
    def reset(cls):
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        # Tells the tracker to use fwd_state as the running state
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        # Tells the tracker to use bwd_state as the running state
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        # Records the seed and offset tensors. These tensors are used to invoke
        # the philox_rand functional primitives.
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset)
        else:
            assert mode == "backward"
            cls.bwd_state.set_state(seed, offset)

    @classmethod
    def get_state_as_tensor(cls):
        # The only reason this exists is because we override get_rng_state and
        # set_rng_state during tracing. get_rng_state expects a tensor output,
        # so return (seed, offset) tuple upset other parts of the program like
        # ctx.saved_tensors.

        # A bad consequence is that if user saves and restores rng state, we
        # have little bit of ugliness in the generated code, where we first
        # concat the (seed, offset) to create a tensor for get_rng_state, and
        # then split it back to get (seed, offset) tuple in set_rng_state.

        # TODO: Investigate if there is be a better way to wrap the tuple in a
        # false Tensor object, and then desugar it later on.
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        # This is only needed because we override set_rng_state. Look at the
        # comment in get_state_from_tensor method.
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        cls.running_state.advance_offset(consumed_offset)

    @classmethod
    def get_current_relative_offset(cls):
        return cls.running_state.relative_offset

    @classmethod
    def get_accumulated_offsets(cls):
        fwd_offset = cls.fwd_state.relative_offset
        bwd_offset = cls.bwd_state.relative_offset
        return PhiloxTotalOffsets(fwd_offset, bwd_offset)


class RNGStateHelper:
    @staticmethod
    def get_torch_state_as_tuple():
        if not torch.cuda.is_available():
            return torch.tensor(0), torch.tensor(0)
        # torch.cuda.get_rng_state yields a real tensor, and upsets fake
        # tensor for the following ops.
        with disable_fake_tensor_mode_tracing():
            rng_state = torch.cuda.get_rng_state()
            seed = rng_state[800:808].view(dtype=torch.int64)[0]
            offset = rng_state[808:].view(dtype=torch.int64)[0]
            return seed, offset

    @staticmethod
    def set_torch_state_tensor(seed, offset):
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        prefix = torch.tensor([-1] * 800, dtype=torch.uint8)
        new_state = torch.cat([prefix, seed_portion, offset_portion])
        torch.cuda.set_rng_state(new_state)

    @staticmethod
    def advance_torch_state(relative_offset):
        rng_state = torch.cuda.get_rng_state()
        seed = rng_state[800:808].view(dtype=torch.int64)[0]
        offset = rng_state[808:].view(dtype=torch.int64)[0]
        new_offset = offset + relative_offset
        RNGStateHelper.set_torch_state_tensor(seed, new_offset)
