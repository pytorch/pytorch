import sys
from typing import Optional

from torch.serialization import LoadEndianness
from torch.utils._config_module import install_config_module


IS_WINDOWS = sys.platform == "win32"

if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None  # type: ignore[assignment]


class load:
    mmap: bool = False
    endianness: Optional[LoadEndianness] = None
    mmap_flags: Optional[int] = MAP_PRIVATE


class save:
    compute_crc32: bool = True


install_config_module(sys.modules[__name__])
