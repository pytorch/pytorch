import contextlib
import warnings

from torch._C import default_generator


def set_rng_state(new_state):
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)


def get_rng_state():
    r"""Returns the random number generator state as a `torch.ByteTensor`."""
    return default_generator.get_state()


def manual_seed(seed):
    r"""Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    import torch.cuda

    if not torch.cuda._in_bad_fork:
        torch.cuda.manual_seed_all(seed)

    return default_generator.manual_seed(seed)


def seed():
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()
    import torch.cuda

    if not torch.cuda._in_bad_fork:
        torch.cuda.manual_seed_all(seed)

    return seed


def initial_seed():
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller="fork_rng", _devices_kw="devices"):
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Arguments:
        devices (iterable of CUDA IDs): CUDA devices for which to fork
            the RNG.  CPU RNG state is always forked.  By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
    """

    import torch.cuda
    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = torch.cuda.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            warnings.warn(
                ("CUDA reports that you have {num_devices} available devices, and you "
                 "have used {caller} without explicitly specifying which devices are being used. "
                 "For safety, we initialize *every* CUDA device by default, which "
                 "can be quite slow if you have a lot of GPUs.  If you know that you are only "
                 "making use of a few CUDA devices, set the environment variable CUDA_VISIBLE_DEVICES "
                 "or the '{devices_kw}' keyword argument of {caller} with the set of devices "
                 "you are actually using.  For example, if you are using CPU only, "
                 "set CUDA_VISIBLE_DEVICES= or devices=[]; if you are using "
                 "GPU 0 only, set CUDA_VISIBLE_DEVICES=0 or devices=[0].  To initialize "
                 "all devices and suppress this warning, set the '{devices_kw}' keyword argument "
                 "to `range(torch.cuda.device_count())`."
                 ).format(num_devices=num_devices, caller=_caller, devices_kw=_devices_kw))
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    gpu_rng_states = []
    for device in devices:
        with torch.cuda.device(device):
            gpu_rng_states.append(torch.cuda.get_rng_state())

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, gpu_rng_state in zip(devices, gpu_rng_states):
            with torch.cuda.device(device):
                torch.cuda.set_rng_state(gpu_rng_state)
