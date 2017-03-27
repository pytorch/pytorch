from torch import _C
from . import _lazy_init


def get_rng_state():
    r"""Returns the random number generator state as a ByteTensor."""
    _lazy_init()
    return _C._cuda_getRNGState()


def set_rng_state(new_state):
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    _lazy_init()
    return _C._cuda_setRNGState(new_state)


def manual_seed(seed):
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int or long): The desired seed.
    """
    _lazy_init()
    return _C._cuda_manualSeed(seed)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all GPUs.

    Args:
        seed (int or long): The desired seed.
    """
    _lazy_init()
    return _C._cuda_manualSeedAll(seed)


def seed():
    r"""Sets the seed for generating random numbers to a random number."""
    _lazy_init()
    return _C._cuda_seed()


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all GPUs."""
    _lazy_init()
    return _C._cuda_seedAll()


def initial_seed():
    r"""Returns the current random seed"""
    _lazy_init()
    return _C._cuda_initialSeed()
