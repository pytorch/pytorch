from typing import Any
from .version import _version_str

__all__ = ['NvfuserVersion', 'Version']

class _LazyImport:
    """Wraps around classes lazy imported from packaging.version
    Output of the function v in following snippets are identical:
       from packaging.version import Version
       def v():
           return Version('1.2.3')
    and
       Version = _LazyImport('Version')
       def v():
           return Version('1.2.3')
    The difference here is that in later example imports
    do not happen until v is called
    """
    def __init__(self, cls_name: str) -> None:
        self._cls_name = cls_name

    def get_cls(self):
        try:
            import packaging.version  # type: ignore[import]
        except ImportError:
            # If packaging isn't installed, try and use the vendored copy
            # in pkg_resources
            from pkg_resources import packaging  # type: ignore[attr-defined, no-redef]
        return getattr(packaging.version, self._cls_name)

    def __call__(self, *args, **kwargs):
        return self.get_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, self.get_cls())


Version = _LazyImport("Version")

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
