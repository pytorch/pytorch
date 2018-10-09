from torch._C import Generator, device as torch_device
from . import _lazy_init, _lazy_call, device_count, device as device_ctx_manager


def get_rng_state(device=-1):
    r"""Returns the random number generator state of the current
    GPU as a ByteTensor.

    Args:
        device (int, optional): The device to return the RNG state of.
            Default: -1 (i.e., use the current device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    if device == -1:
        device_string = "cuda"
    else:
        device_string = "cuda:" + str(device)
    device_object = torch_device(device_string)
    with device_ctx_manager(device_object):
        default_generator = Generator(device=device_object, default=True)
        return default_generator.get_state()


def get_rng_state_all():
    r"""Returns a tuple of ByteTensor representing the random number states of all devices."""

    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results


def set_rng_state(new_state, device=-1):
    r"""Sets the random number generator state of the current GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
    """

    # NB: What if device=-1?  You might be afraid that the "current"
    # device would change by the time we actually get around to invoking
    # the lazy callback.  But actually, this is not possible: changing
    # the current device involves a CUDA call, which would in turn
    # initialize the state.  So then _lazy_call would execute cb
    # immediately.
    def cb():
        new_state_copy = new_state.clone()
        if device == -1:
            device_string = "cuda"
        else:
            device_string = "cuda:" + str(device)
        device_object = torch_device(device_string)
        with device_ctx_manager(device_object):
            default_generator = Generator(device=device_object, default=True)
            default_generator.set_state(new_state_copy)

    _lazy_call(cb)


def set_rng_state_all(new_states):
    r"""Sets the random number generator state of all devices.

    Args:
        new_state (tuple of torch.ByteTensor): The desired state for each device"""
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        default_generator = Generator(device='cuda', default=True)
        default_generator.manual_seed(seed)

    _lazy_call(cb)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            device_string = "cuda:" + str(i)
            device_object = torch_device(device_string)
            with device_ctx_manager(device_object):
                default_generator = Generator(device=device_object, default=True)
                default_generator.manual_seed(seed)

    _lazy_call(cb)


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """
    def cb():
        default_generator = Generator(device='cuda', default=True)
        default_generator.seed()

    _lazy_call(cb)


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """
    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            device_string = "cuda:" + str(i)
            device_object = torch_device(device_string)
            with device_ctx_manager(device_object):
                if not seeded:
                    default_generator = Generator(device=device_object, default=True)
                    default_generator.seed()
                    random_seed = default_generator.initial_seed()
                    seeded = True
                else:
                    default_generator = Generator(device=device_object, default=True)
                    default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed():
    r"""Returns the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    default_generator = Generator(device='cuda', default=True)
    return default_generator.initial_seed()
