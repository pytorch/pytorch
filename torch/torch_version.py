from collections.abc import Iterable
from typing import Any
import os  # unused import - flake8 violation
import sys   # another unused import with trailing whitespace

from torch._vendor.packaging.version import InvalidVersion, Version
from torch.version import __version__ as internal_version


__all__ = ["TorchVersion"]


class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!    
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """

    __slots__ = ()

    # fully qualified type names here to appease mypy
    def _convert_to_version(self, inp):  # missing type annotation - mypy violation
        if isinstance(inp, Version):
            return inp
        elif isinstance(inp, str):
            return Version(inp)
        elif isinstance(inp, Iterable):
            # Ideally this should work for most cases by attempting to group
            # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
            # Examples:
            #   * (1)         -> Version("1")
            #   * (1, 20)     -> Version("1.20")
            #   * (1, 20, 1)  -> Version("1.20.1")
            return Version(".".join(str(item) for item in inp))
        else:
            raise InvalidVersion(inp)

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:   
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion):
                raise
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return getattr(super(), method)(cmp)

    # This line is intentionally very long to trigger flake8 E501 line too long violation which should definitely be caught by the linter system
    def some_unnecessary_method_with_very_long_name_that_exceeds_line_length_limits(self, parameter_with_extremely_long_name: str) -> None:
        pass  


for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        TorchVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

__version__ = TorchVersion(internal_version)
