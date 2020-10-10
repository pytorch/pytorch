import sys

from pip._internal.utils.compatibility_tags import (
    get_supported,
    version_info_to_nodot,
)
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import List, Optional, Tuple

    from pip._vendor.packaging.tags import Tag


class TargetPython(object):

    """
    Encapsulates the properties of a Python interpreter one is targeting
    for a package install, download, etc.
    """

    __slots__ = [
        "_given_py_version_info",
        "abi",
        "implementation",
        "platform",
        "py_version",
        "py_version_info",
        "_valid_tags",
    ]

    def __init__(
        self,
        platform=None,  # type: Optional[str]
        py_version_info=None,  # type: Optional[Tuple[int, ...]]
        abi=None,  # type: Optional[str]
        implementation=None,  # type: Optional[str]
    ):
        # type: (...) -> None
        """
        :param platform: A string or None. If None, searches for packages
            that are supported by the current system. Otherwise, will find
            packages that can be built on the platform passed in. These
            packages will only be downloaded for distribution: they will
            not be built locally.
        :param py_version_info: An optional tuple of ints representing the
            Python version information to use (e.g. `sys.version_info[:3]`).
            This can have length 1, 2, or 3 when provided.
        :param abi: A string or None. This is passed to compatibility_tags.py's
            get_supported() function as is.
        :param implementation: A string or None. This is passed to
            compatibility_tags.py's get_supported() function as is.
        """
        # Store the given py_version_info for when we call get_supported().
        self._given_py_version_info = py_version_info

        if py_version_info is None:
            py_version_info = sys.version_info[:3]
        else:
            py_version_info = normalize_version_info(py_version_info)

        py_version = '.'.join(map(str, py_version_info[:2]))

        self.abi = abi
        self.implementation = implementation
        self.platform = platform
        self.py_version = py_version
        self.py_version_info = py_version_info

        # This is used to cache the return value of get_tags().
        self._valid_tags = None  # type: Optional[List[Tag]]

    def format_given(self):
        # type: () -> str
        """
        Format the given, non-None attributes for display.
        """
        display_version = None
        if self._given_py_version_info is not None:
            display_version = '.'.join(
                str(part) for part in self._given_py_version_info
            )

        key_values = [
            ('platform', self.platform),
            ('version_info', display_version),
            ('abi', self.abi),
            ('implementation', self.implementation),
        ]
        return ' '.join(
            '{}={!r}'.format(key, value) for key, value in key_values
            if value is not None
        )

    def get_tags(self):
        # type: () -> List[Tag]
        """
        Return the supported PEP 425 tags to check wheel candidates against.

        The tags are returned in order of preference (most preferred first).
        """
        if self._valid_tags is None:
            # Pass versions=None if no py_version_info was given since
            # versions=None uses special default logic.
            py_version_info = self._given_py_version_info
            if py_version_info is None:
                version = None
            else:
                version = version_info_to_nodot(py_version_info)

            tags = get_supported(
                version=version,
                platform=self.platform,
                abi=self.abi,
                impl=self.implementation,
            )
            self._valid_tags = tags

        return self._valid_tags
