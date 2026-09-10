import sys
from typing import Optional as _Optional, TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from torch.serialization import LoadEndianness as _LoadEndianess

from torch.utils._config_module import install_config_module as _install_config_module


class load:
    mmap: bool = False
    endianness: _Optional["_LoadEndianess"] = None
    # MAP_PRIVATE = 2
    mmap_flags: int | None = None if sys.platform == "win32" else 2
    calculate_storage_offsets: bool = False


class save:
    compute_crc32: bool = True
    use_pinned_memory_for_d2h: bool = False
    storage_alignment: int = 64


_install_config_module(sys.modules[__name__])
