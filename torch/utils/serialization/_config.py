import sys
from typing import Optional

from torch.serialization import LoadEndianness
from torch.utils._config_module import install_config_module


class load:
    mmap: bool = False
    endianness: Optional[LoadEndianness] = None
    # MAP_PRIVATE = 2
    mmap_flags: Optional[int] = 2 if sys.platform == "win32" else None


class save:
    compute_crc32: bool = True
    use_pinned_memory_for_d2h: bool = False


install_config_module(sys.modules[__name__])
