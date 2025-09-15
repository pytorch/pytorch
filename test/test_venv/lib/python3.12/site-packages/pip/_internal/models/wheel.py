"""Represents a wheel file and provides access to the various parts of the
name that have meaning.
"""

import re
from typing import Dict, Iterable, List

from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import (
    InvalidWheelFilename as PackagingInvalidWheelName,
)
from pip._vendor.packaging.utils import parse_wheel_filename

from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.utils.deprecation import deprecated


class Wheel:
    """A wheel file"""

    wheel_file_re = re.compile(
        r"""^(?P<namever>(?P<name>[^\s-]+?)-(?P<ver>[^\s-]*?))
        ((-(?P<build>\d[^-]*?))?-(?P<pyver>[^\s-]+?)-(?P<abi>[^\s-]+?)-(?P<plat>[^\s-]+?)
        \.whl|\.dist-info)$""",
        re.VERBOSE,
    )

    def __init__(self, filename: str) -> None:
        """
        :raises InvalidWheelFilename: when the filename is invalid for a wheel
        """
        wheel_info = self.wheel_file_re.match(filename)
        if not wheel_info:
            raise InvalidWheelFilename(f"{filename} is not a valid wheel filename.")
        self.filename = filename
        self.name = wheel_info.group("name").replace("_", "-")
        _version = wheel_info.group("ver")
        if "_" in _version:
            try:
                parse_wheel_filename(filename)
            except PackagingInvalidWheelName as e:
                deprecated(
                    reason=(
                        f"Wheel filename {filename!r} is not correctly normalised. "
                        "Future versions of pip will raise the following error:\n"
                        f"{e.args[0]}\n\n"
                    ),
                    replacement=(
                        "to rename the wheel to use a correctly normalised "
                        "name (this may require updating the version in "
                        "the project metadata)"
                    ),
                    gone_in="25.1",
                    issue=12938,
                )

            _version = _version.replace("_", "-")

        self.version = _version
        self.build_tag = wheel_info.group("build")
        self.pyversions = wheel_info.group("pyver").split(".")
        self.abis = wheel_info.group("abi").split(".")
        self.plats = wheel_info.group("plat").split(".")

        # All the tag combinations from this file
        self.file_tags = {
            Tag(x, y, z) for x in self.pyversions for y in self.abis for z in self.plats
        }

    def get_formatted_file_tags(self) -> List[str]:
        """Return the wheel's tags as a sorted list of strings."""
        return sorted(str(tag) for tag in self.file_tags)

    def support_index_min(self, tags: List[Tag]) -> int:
        """Return the lowest index that one of the wheel's file_tag combinations
        achieves in the given list of supported tags.

        For example, if there are 8 supported tags and one of the file tags
        is first in the list, then return 0.

        :param tags: the PEP 425 tags to check the wheel against, in order
            with most preferred first.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        try:
            return next(i for i, t in enumerate(tags) if t in self.file_tags)
        except StopIteration:
            raise ValueError()

    def find_most_preferred_tag(
        self, tags: List[Tag], tag_to_priority: Dict[Tag, int]
    ) -> int:
        """Return the priority of the most preferred tag that one of the wheel's file
        tag combinations achieves in the given list of supported tags using the given
        tag_to_priority mapping, where lower priorities are more-preferred.

        This is used in place of support_index_min in some cases in order to avoid
        an expensive linear scan of a large list of tags.

        :param tags: the PEP 425 tags to check the wheel against.
        :param tag_to_priority: a mapping from tag to priority of that tag, where
            lower is more preferred.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        return min(
            tag_to_priority[tag] for tag in self.file_tags if tag in tag_to_priority
        )

    def supported(self, tags: Iterable[Tag]) -> bool:
        """Return whether the wheel is compatible with one of the given tags.

        :param tags: the PEP 425 tags to check the wheel against.
        """
        return not self.file_tags.isdisjoint(tags)
