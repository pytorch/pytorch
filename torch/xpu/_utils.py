import enum
import functools
from typing import Any

import torch

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index
from torch.types import Device


class XPUCodename(enum.Enum):
    PVC = "PVC"  # Intel® Data Center GPU Max Series
    BMG = "BMG"  # Intel® Arc™ Pro Battlemage Graphics


class XPUArch(enum.IntEnum):
    Unknown = 0
    Xe = 1  # Xe HPC
    Xe2 = 2


# device_id -> GPU codename
# From https://github.com/intel/intel-graphics-compiler/blob/master/inc/common/igfxfmid.h
_DEVICE_ID_TO_CODENAME = {
    0x0BD0: XPUCodename.PVC,
    0x0BD4: XPUCodename.PVC,
    0x0BD5: XPUCodename.PVC,
    0x0BD6: XPUCodename.PVC,
    0x0BD7: XPUCodename.PVC,
    0x0BD8: XPUCodename.PVC,
    0x0BD9: XPUCodename.PVC,
    0x0BDA: XPUCodename.PVC,
    0x0BDB: XPUCodename.PVC,
    0x0B69: XPUCodename.PVC,
    0x0B6E: XPUCodename.PVC,
    0xE202: XPUCodename.BMG,
    0xE20B: XPUCodename.BMG,
    0xE20C: XPUCodename.BMG,
    0xE20D: XPUCodename.BMG,
    0xE210: XPUCodename.BMG,
    0xE211: XPUCodename.BMG,
    0xE212: XPUCodename.BMG,
    0xE215: XPUCodename.BMG,
    0xE216: XPUCodename.BMG,
    0xE220: XPUCodename.BMG,
    0xE221: XPUCodename.BMG,
    0xE222: XPUCodename.BMG,
    0xE223: XPUCodename.BMG,
}

_CODENAME_TO_ARCH = {
    XPUCodename.PVC: XPUArch.Xe,
    XPUCodename.BMG: XPUArch.Xe2,
}


@functools.lru_cache(1)
def get_xpu_codename(device: Device = None) -> XPUCodename | None:
    device_id = torch.xpu.get_device_capability(device)["device_id"]
    return _DEVICE_ID_TO_CODENAME.get(device_id)


@functools.lru_cache(1)
def get_xpu_arch(device: Device = None) -> XPUArch:
    codename = get_xpu_codename(device)
    if codename is None:
        return XPUArch.Unknown
    return _CODENAME_TO_ARCH.get(codename, XPUArch.Unknown)


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a XPU device. Note that for a XPU device without a specified index,
    i.e., ``torch.device('xpu')``, this will return the current default XPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default XPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["xpu", "cpu"]:
                raise ValueError(f"Expected a xpu or cpu device, but got: {device}")
        elif device.type != "xpu":
            raise ValueError(f"Expected a xpu device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.xpu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
