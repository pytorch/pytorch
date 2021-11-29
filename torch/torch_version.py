from typing import Iterable, Union

from pkg_resources import packaging  # type: ignore[attr-defined]

Version = packaging.version.Version
InvalidVersion = packaging.version.InvalidVersion

from .version import __version__ as internal_version


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
    # fully qualified type names here to appease mypy
    def _convert_to_version(self, inp: Union[packaging.version.Version, str, Iterable]) -> packaging.version.Version:
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
            return Version('.'.join((str(item) for item in inp)))
        else:
            raise InvalidVersion(inp)

    def __gt__(self, cmp):
        try:
            return Version(self).__gt__(self._convert_to_version(cmp))
        except InvalidVersion:
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return super().__gt__(cmp)

    def __lt__(self, cmp):
        try:
            return Version(self).__lt__(self._convert_to_version(cmp))
        except InvalidVersion:
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return super().__lt__(cmp)

    def __eq__(self, cmp):
        try:
            return Version(self).__eq__(self._convert_to_version(cmp))
        except InvalidVersion:
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return super().__eq__(cmp)

    def __ge__(self, cmp):
        try:
            return Version(self).__ge__(self._convert_to_version(cmp))
        except InvalidVersion:
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return super().__ge__(cmp)

    def __le__(self, cmp):
        try:
            return Version(self).__le__(self._convert_to_version(cmp))
        except InvalidVersion:
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return super().__le__(cmp)

__version__ = TorchVersion(internal_version)
