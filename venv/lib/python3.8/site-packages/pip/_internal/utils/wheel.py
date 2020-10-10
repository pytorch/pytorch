"""Support functions for working with wheel files.
"""

from __future__ import absolute_import

import logging
from email.parser import Parser
from zipfile import ZipFile

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.pkg_resources import DistInfoDistribution
from pip._vendor.six import PY2, ensure_str

from pip._internal.exceptions import UnsupportedWheel
from pip._internal.utils.pkg_resources import DictMetadata
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from email.message import Message
    from typing import Dict, Tuple

    from pip._vendor.pkg_resources import Distribution

if PY2:
    from zipfile import BadZipfile as BadZipFile
else:
    from zipfile import BadZipFile


VERSION_COMPATIBLE = (1, 0)


logger = logging.getLogger(__name__)


class WheelMetadata(DictMetadata):
    """Metadata provider that maps metadata decoding exceptions to our
    internal exception type.
    """
    def __init__(self, metadata, wheel_name):
        # type: (Dict[str, bytes], str) -> None
        super(WheelMetadata, self).__init__(metadata)
        self._wheel_name = wheel_name

    def get_metadata(self, name):
        # type: (str) -> str
        try:
            return super(WheelMetadata, self).get_metadata(name)
        except UnicodeDecodeError as e:
            # Augment the default error with the origin of the file.
            raise UnsupportedWheel(
                "Error decoding metadata for {}: {}".format(
                    self._wheel_name, e
                )
            )


def pkg_resources_distribution_for_wheel(wheel_zip, name, location):
    # type: (ZipFile, str, str) -> Distribution
    """Get a pkg_resources distribution given a wheel.

    :raises UnsupportedWheel: on any errors
    """
    info_dir, _ = parse_wheel(wheel_zip, name)

    metadata_files = [
        p for p in wheel_zip.namelist() if p.startswith("{}/".format(info_dir))
    ]

    metadata_text = {}  # type: Dict[str, bytes]
    for path in metadata_files:
        # If a flag is set, namelist entries may be unicode in Python 2.
        # We coerce them to native str type to match the types used in the rest
        # of the code. This cannot fail because unicode can always be encoded
        # with UTF-8.
        full_path = ensure_str(path)
        _, metadata_name = full_path.split("/", 1)

        try:
            metadata_text[metadata_name] = read_wheel_metadata_file(
                wheel_zip, full_path
            )
        except UnsupportedWheel as e:
            raise UnsupportedWheel(
                "{} has an invalid wheel, {}".format(name, str(e))
            )

    metadata = WheelMetadata(metadata_text, location)

    return DistInfoDistribution(
        location=location, metadata=metadata, project_name=name
    )


def parse_wheel(wheel_zip, name):
    # type: (ZipFile, str) -> Tuple[str, Message]
    """Extract information from the provided wheel, ensuring it meets basic
    standards.

    Returns the name of the .dist-info directory and the parsed WHEEL metadata.
    """
    try:
        info_dir = wheel_dist_info_dir(wheel_zip, name)
        metadata = wheel_metadata(wheel_zip, info_dir)
        version = wheel_version(metadata)
    except UnsupportedWheel as e:
        raise UnsupportedWheel(
            "{} has an invalid wheel, {}".format(name, str(e))
        )

    check_compatibility(version, name)

    return info_dir, metadata


def wheel_dist_info_dir(source, name):
    # type: (ZipFile, str) -> str
    """Returns the name of the contained .dist-info directory.

    Raises AssertionError or UnsupportedWheel if not found, >1 found, or
    it doesn't match the provided name.
    """
    # Zip file path separators must be /
    subdirs = set(p.split("/", 1)[0] for p in source.namelist())

    info_dirs = [s for s in subdirs if s.endswith('.dist-info')]

    if not info_dirs:
        raise UnsupportedWheel(".dist-info directory not found")

    if len(info_dirs) > 1:
        raise UnsupportedWheel(
            "multiple .dist-info directories found: {}".format(
                ", ".join(info_dirs)
            )
        )

    info_dir = info_dirs[0]

    info_dir_name = canonicalize_name(info_dir)
    canonical_name = canonicalize_name(name)
    if not info_dir_name.startswith(canonical_name):
        raise UnsupportedWheel(
            ".dist-info directory {!r} does not start with {!r}".format(
                info_dir, canonical_name
            )
        )

    # Zip file paths can be unicode or str depending on the zip entry flags,
    # so normalize it.
    return ensure_str(info_dir)


def read_wheel_metadata_file(source, path):
    # type: (ZipFile, str) -> bytes
    try:
        return source.read(path)
        # BadZipFile for general corruption, KeyError for missing entry,
        # and RuntimeError for password-protected files
    except (BadZipFile, KeyError, RuntimeError) as e:
        raise UnsupportedWheel(
            "could not read {!r} file: {!r}".format(path, e)
        )


def wheel_metadata(source, dist_info_dir):
    # type: (ZipFile, str) -> Message
    """Return the WHEEL metadata of an extracted wheel, if possible.
    Otherwise, raise UnsupportedWheel.
    """
    path = "{}/WHEEL".format(dist_info_dir)
    # Zip file path separators must be /
    wheel_contents = read_wheel_metadata_file(source, path)

    try:
        wheel_text = ensure_str(wheel_contents)
    except UnicodeDecodeError as e:
        raise UnsupportedWheel("error decoding {!r}: {!r}".format(path, e))

    # FeedParser (used by Parser) does not raise any exceptions. The returned
    # message may have .defects populated, but for backwards-compatibility we
    # currently ignore them.
    return Parser().parsestr(wheel_text)


def wheel_version(wheel_data):
    # type: (Message) -> Tuple[int, ...]
    """Given WHEEL metadata, return the parsed Wheel-Version.
    Otherwise, raise UnsupportedWheel.
    """
    version_text = wheel_data["Wheel-Version"]
    if version_text is None:
        raise UnsupportedWheel("WHEEL is missing Wheel-Version")

    version = version_text.strip()

    try:
        return tuple(map(int, version.split('.')))
    except ValueError:
        raise UnsupportedWheel("invalid Wheel-Version: {!r}".format(version))


def check_compatibility(version, name):
    # type: (Tuple[int, ...], str) -> None
    """Raises errors or warns if called with an incompatible Wheel-Version.

    pip should refuse to install a Wheel-Version that's a major series
    ahead of what it's compatible with (e.g 2.0 > 1.1); and warn when
    installing a version only minor version ahead (e.g 1.2 > 1.1).

    version: a 2-tuple representing a Wheel-Version (Major, Minor)
    name: name of wheel or package to raise exception about

    :raises UnsupportedWheel: when an incompatible Wheel-Version is given
    """
    if version[0] > VERSION_COMPATIBLE[0]:
        raise UnsupportedWheel(
            "{}'s Wheel-Version ({}) is not compatible with this version "
            "of pip".format(name, '.'.join(map(str, version)))
        )
    elif version > VERSION_COMPATIBLE:
        logger.warning(
            'Installing from a newer Wheel-Version (%s)',
            '.'.join(map(str, version)),
        )
