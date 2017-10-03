import torch

from torch._C import default_generator


def set_rng_state(new_state):
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)


def get_rng_state():
    r"""Returns the random number generator state as a ByteTensor."""
    return default_generator.get_state()


def manual_seed(seed):
    r"""Sets the seed for generating random numbers. And returns a
    `torch._C.Generator` object.

    Args:
        seed (int or long): The desired seed.
    """
    import torch.cuda

    if not torch.cuda._in_bad_fork:
        torch.cuda.manual_seed_all(seed)

    return default_generator.manual_seed(seed)


def initial_seed():
    r"""Returns the initial seed for generating random numbers as a
    python `long`.
    """
    return default_generator.initial_seed()
