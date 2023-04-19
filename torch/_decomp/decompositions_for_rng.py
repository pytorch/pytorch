from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch._decomp as decomp
from torch._ops import OpOverload
from torch._prims_common import make_contiguous_strides_for

aten = torch.ops.aten

rng_decompositions: Dict[str, Dict[OpOverload, Callable]] = defaultdict(dict)


def register_rng_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, rng_decompositions)


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def rand_offset_calculator(shape):
    # For impl, look at the function calc_execution_policy in the file
    # aten/src/ATen/native/cuda/DistributionTemplates.h. The impl was copied at
    # commit hash ccc5d1daec46da82ce17fcb8e9dcc871e9fef9a2
    numel = 1
    for dim_size in shape:
        numel *= dim_size

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    max_threads_per_sm = 1536
    number_of_sm = 12
    blocks_per_sm = max_threads_per_sm // block_size
    grid_size = (numel + block_size - 1) // block_size
    grid_size = min(grid_size, number_of_sm * blocks_per_sm)
    offset = (
        (numel - 1) // (block_size * grid_size * unroll) + 1
    ) * curand4_engine_calls
    return offset


# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    if device and device.type != "cuda":
        throw_on_non_cuda(device)
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or torch.float32
    stride = make_contiguous_strides_for(shape)
    r = torch.ops.rngprims.philox_rand(shape, seed, offset, None, device, dtype)
    PhiloxStateTracker.advance_offset(rand_offset_calculator(shape))
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
    if device.type != "cuda":
        throw_on_non_cuda(device)
    dtype = dtype or x.dtype
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    r = torch.ops.rngprims.philox_rand(x.shape, seed, offset, None, device, dtype)
    PhiloxStateTracker.advance_offset(rand_offset_calculator(x.shape))
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
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0

    def validate_state(self):
        assert self.seed.numel() != 0 and self.base_offset.numel() != 0

    def advance_offset(self, consumed_offset):
        self.relative_offset += consumed_offset

    def set_state(self, seed, base_offset, relative_offset=0):
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        # Only needed because we override get_rng_state.
        self.validate_state()
        return torch.stack([self.seed, self.base_offset + self.relative_offset])

    def set_state_from_tensor(self, state):
        # Only needed because we override set_rng_state.
        self.seed, self.base_offset = torch.split(state, 1)
        self.relative_offset = 0


@dataclass
class PhiloxTotalOffsets:
    """
    PhiloxStateTracker computes the total fwd and bwd offsets for an AOT
    Autograd traced graph. However, PhiloxStateTracker is a singleton class, but
    the total offsets are specific to each traced graph. These offsets are
    stored as part of AOT graph. This class just encapsulates the fwd and bwd
    offsets to be used at runtime.
    """

    total_fwd_offset: int
    total_bwd_offset: int


class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. There are stored as part of
    aot_config (PhiloxTotalOffsets), which is a config object per aot-tracing.
    """

    running_state: PhiloxState
    fwd_state: PhiloxState
    bwd_state: PhiloxState

    def __enter__(self):
        PhiloxStateTracker.reset()
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        PhiloxStateTracker.reset()

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
            cls.mark_beginning_of_forward()
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
