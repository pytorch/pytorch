from torch.utils.version_string import VersionString
from torch.version import __version__ as internal_version


__all__ = ["VersionString"]

__version__ = VersionString(internal_version)
