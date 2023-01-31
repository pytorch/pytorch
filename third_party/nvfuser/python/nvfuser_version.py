from typing import Any
from packaging.version import Version

from .version import _version_str

class NvfuserVersion(str):

    @classmethod
    def _convert_to_version(cls, ver: Any) -> Version:
        if isinstance(ver, str):
            return Version(ver.split('+')[0])
        elif isinstance(ver, Version.get_cls()):
            return ver
        else:
            raise ValueError("can't convert {} to Version".format(ver))

    def _cmp_version(self, other: Any, method: str) -> Version:
        return getattr(NvfuserVersion._convert_to_version(self), method)(NvfuserVersion._convert_to_version(other))

for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(NvfuserVersion, cmp_method, lambda x, y, method=cmp_method: x._cmp_version(y, method))

__version__ = NvfuserVersion(_version_str)
