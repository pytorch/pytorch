# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import absolute_import, division, print_function

import re

from .version import InvalidVersion, Version


_canonicalize_regex = re.compile(r"[-_.]+")


def canonicalize_name(name):
    # This is taken from PEP 503.
    return _canonicalize_regex.sub("-", name).lower()


def canonicalize_version(version):
    """
    This is very similar to Version.__str__, but has one subtle differences
    with the way it handles the release segment.
    """

    try:
        version = Version(version)
    except InvalidVersion:
        # Legacy versions cannot be normalized
        return version

    parts = []

    # Epoch
    if version.epoch != 0:
        parts.append("{0}!".format(version.epoch))

    # Release segment
    # NB: This strips trailing '.0's to normalize
    parts.append(re.sub(r"(\.0)+$", "", ".".join(str(x) for x in version.release)))

    # Pre-release
    if version.pre is not None:
        parts.append("".join(str(x) for x in version.pre))

    # Post-release
    if version.post is not None:
        parts.append(".post{0}".format(version.post))

    # Development release
    if version.dev is not None:
        parts.append(".dev{0}".format(version.dev))

    # Local version segment
    if version.local is not None:
        parts.append("+{0}".format(version.local))

    return "".join(parts)
