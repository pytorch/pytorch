import torch
import torch_openreg
import torch_openreg._C

_initialized = False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = driver.exec("exchangeDevice", self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = driver.exec("uncheckedSetDevice", self.prev_idx)
        return False


def device_count() -> int:
    return torch_openreg._C._get_device_count()


def current_device():
    return torch_openreg._C._get_device()


def set_device(device) -> None:
    return torch_openreg._C._set_device(device)


def is_available():
    return True


def get_rng_state(device="openreg"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.get_state()


def set_rng_state(new_state, device="openreg"):
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.set_state(new_state)


def initial_seed() -> int:
    _lazy_init()
    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.initial_seed()


def manual_seed(seed: int) -> None:
    seed = int(seed)

    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    seed = int(seed)

    for idx in range(device_count()):
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.manual_seed(seed)


def _is_in_bad_fork():
    return False


def _lazy_init():
    if is_initialized():
        return
    torch_openreg._C._init()
    _initialized = True


def is_initialized():
    return _initialized
