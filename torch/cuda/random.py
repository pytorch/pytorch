from torch import _C
from . import _lazy_init, _lazy_call


def get_rng_state():
    r"""Returns the random number generator state of the current
    GPU as a ByteTensor.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    return _C._cuda_getRNGState()


def set_rng_state(new_state):
    r"""Sets the random number generator state of the current GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    new_state_copy = new_state.clone()
    _lazy_call(lambda: _C._cuda_setRNGState(new_state_copy))


def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current GPU.

    Args:
        seed (int or long): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    _lazy_call(lambda: _C._cuda_manualSeed(seed))


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all GPUs.

    Args:
        seed (int or long): The desired seed.
    """
    _lazy_call(lambda: _C._cuda_manualSeedAll(seed))


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current GPU.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """
    _lazy_call(lambda: _C._cuda_seed())


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all GPUs."""
    _lazy_call(lambda: _C._cuda_seedAll())


def initial_seed():
    r"""Returns the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    return _C._cuda_initialSeed()
