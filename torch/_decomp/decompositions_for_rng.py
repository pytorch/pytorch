# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
from collections import defaultdict
from typing import Callable

import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload


aten = torch.ops.aten

rng_decompositions: dict[str, dict[OpOverload, Callable]] = defaultdict(dict)


def register_rng_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, rng_decompositions)


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    if device and device.type != "cuda":
        throw_on_non_cuda(device)
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or torch.float32
    out, offset_jump = torch.ops.rngprims.philox_rand(
        shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


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
    out, offset_jump = torch.ops.rngprims.philox_rand(
        x.shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


class PhiloxState:
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totally consumed offset at
    trace time.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0
        self.offset_advanced_alteast_once = False

    def validate_state(self):
        assert self.seed.numel() != 0 and self.base_offset.numel() != 0

    def advance_offset(self, consumed_offset):
        self.offset_advanced_alteast_once = True
        self.relative_offset = self.relative_offset + consumed_offset

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
        self.seed, self.base_offset = torch.unbind(state)
        self.relative_offset = 0


class PhiloxStateTracker:
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. For dynamic shapes, these
    offsets are a function of input shapes. Therefore, the AOT generated graphs
    have additional outputs that compute total consumed forward and backward
    offsets.
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

    @staticmethod
    def multiple_of_4(offset):
        # torch cuda rng state offset must be a multiple of 4. For inductor, as
        # we sum up all the numel, the result might not be a multiple of 4. This
        # method achieves that.
        return (offset + 3) // 4 * 4

    @classmethod
    def get_updated_fwd_offset(cls):
        # Short circuit if no rand ops were observed
        if not cls.fwd_state.offset_advanced_alteast_once:
            return cls.fwd_state.base_offset
        return cls.multiple_of_4(
            cls.fwd_state.base_offset + cls.fwd_state.relative_offset
        )

    @classmethod
    def get_updated_bwd_offset(cls):
        # Short circuit if no rand ops were observed
        if not cls.bwd_state.offset_advanced_alteast_once:
            return cls.bwd_state.base_offset
        return cls.multiple_of_4(
            cls.bwd_state.base_offset + cls.bwd_state.relative_offset
        )


# Adding more decompositions which eventually use rand_like inside decomps.
# Adding these in rng_decompositions ensures the functionalization of rand_like
# ops used in these decomps. The list is copied from inductor codebase, which
# uses it for similar purpose.
#
# Caution - These decomps do not have same accuracy as that of eager. However,
# we can't just disable them with a config flag like fallback_random, because
# for functionalization of rng ops, we have to decompose these ops.
extra_random_decomps = get_decompositions(
    [
        aten.cauchy,
        aten.cauchy_,
        aten.exponential,
        aten.exponential_,
        aten.geometric,
        aten.geometric_,
        aten.native_dropout,
        aten.normal,
        aten.normal_,
        aten.normal_functional,
        aten.log_normal,
        aten.log_normal_,
        aten.rrelu_with_noise,
        aten.rrelu_with_noise_,
        aten.uniform_,
    ]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    if self.device == torch.device("cpu"):
        return NotImplemented
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)


@register_extra_random_decomp([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    if self.device == torch.device("cpu"):
        return NotImplemented
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < p


rng_decompositions.update(extra_random_decomps)  # type: ignore[arg-type]
