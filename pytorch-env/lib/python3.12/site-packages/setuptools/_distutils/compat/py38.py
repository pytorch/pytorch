import sys

if sys.version_info < (3, 9):

    def removesuffix(self, suffix):
        # suffix='' should not call self[:-0].
        if suffix and self.endswith(suffix):
            return self[: -len(suffix)]
        else:
            return self[:]

    def removeprefix(self, prefix):
        if self.startswith(prefix):
            return self[len(prefix) :]
        else:
            return self[:]
else:

    def removesuffix(self, suffix):
        return self.removesuffix(suffix)

    def removeprefix(self, prefix):
        return self.removeprefix(prefix)


def aix_platform(osname, version, release):
    try:
        import _aix_support  # type: ignore

        return _aix_support.aix_platform()
    except ImportError:
        pass
    return f"{osname}-{version}.{release}"
