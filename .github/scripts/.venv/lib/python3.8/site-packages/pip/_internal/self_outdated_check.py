from __future__ import absolute_import

import datetime
import hashlib
import json
import logging
import os.path
import sys

from pip._vendor.packaging import version as packaging_version
from pip._vendor.six import ensure_binary

from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.utils.filesystem import (
    adjacent_tmp_file,
    check_path_owner,
    replace,
)
from pip._internal.utils.misc import (
    ensure_dir,
    get_distribution,
    get_installed_version,
)
from pip._internal.utils.packaging import get_installer
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    import optparse
    from typing import Any, Dict, Text, Union

    from pip._internal.network.session import PipSession


SELFCHECK_DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"


logger = logging.getLogger(__name__)


def _get_statefile_name(key):
    # type: (Union[str, Text]) -> str
    key_bytes = ensure_binary(key)
    name = hashlib.sha224(key_bytes).hexdigest()
    return name


class SelfCheckState(object):
    def __init__(self, cache_dir):
        # type: (str) -> None
        self.state = {}  # type: Dict[str, Any]
        self.statefile_path = None

        # Try to load the existing state
        if cache_dir:
            self.statefile_path = os.path.join(
                cache_dir, "selfcheck", _get_statefile_name(self.key)
            )
            try:
                with open(self.statefile_path) as statefile:
                    self.state = json.load(statefile)
            except (IOError, ValueError, KeyError):
                # Explicitly suppressing exceptions, since we don't want to
                # error out if the cache file is invalid.
                pass

    @property
    def key(self):
        # type: () -> str
        return sys.prefix

    def save(self, pypi_version, current_time):
        # type: (str, datetime.datetime) -> None
        # If we do not have a path to cache in, don't bother saving.
        if not self.statefile_path:
            return

        # Check to make sure that we own the directory
        if not check_path_owner(os.path.dirname(self.statefile_path)):
            return

        # Now that we've ensured the directory is owned by this user, we'll go
        # ahead and make sure that all our directories are created.
        ensure_dir(os.path.dirname(self.statefile_path))

        state = {
            # Include the key so it's easy to tell which pip wrote the
            # file.
            "key": self.key,
            "last_check": current_time.strftime(SELFCHECK_DATE_FMT),
            "pypi_version": pypi_version,
        }

        text = json.dumps(state, sort_keys=True, separators=(",", ":"))

        with adjacent_tmp_file(self.statefile_path) as f:
            f.write(ensure_binary(text))

        try:
            # Since we have a prefix-specific state file, we can just
            # overwrite whatever is there, no need to check.
            replace(f.name, self.statefile_path)
        except OSError:
            # Best effort.
            pass


def was_installed_by_pip(pkg):
    # type: (str) -> bool
    """Checks whether pkg was installed by pip

    This is used not to display the upgrade message when pip is in fact
    installed by system package manager, such as dnf on Fedora.
    """
    dist = get_distribution(pkg)
    if not dist:
        return False
    return "pip" == get_installer(dist)


def pip_self_version_check(session, options):
    # type: (PipSession, optparse.Values) -> None
    """Check for an update for pip.

    Limit the frequency of checks to once per week. State is stored either in
    the active virtualenv or in the user's USER_CACHE_DIR keyed off the prefix
    of the pip script path.
    """
    installed_version = get_installed_version("pip")
    if not installed_version:
        return

    pip_version = packaging_version.parse(installed_version)
    pypi_version = None

    try:
        state = SelfCheckState(cache_dir=options.cache_dir)

        current_time = datetime.datetime.utcnow()
        # Determine if we need to refresh the state
        if "last_check" in state.state and "pypi_version" in state.state:
            last_check = datetime.datetime.strptime(
                state.state["last_check"],
                SELFCHECK_DATE_FMT
            )
            if (current_time - last_check).total_seconds() < 7 * 24 * 60 * 60:
                pypi_version = state.state["pypi_version"]

        # Refresh the version if we need to or just see if we need to warn
        if pypi_version is None:
            # Lets use PackageFinder to see what the latest pip version is
            link_collector = LinkCollector.create(
                session,
                options=options,
                suppress_no_index=True,
            )

            # Pass allow_yanked=False so we don't suggest upgrading to a
            # yanked version.
            selection_prefs = SelectionPreferences(
                allow_yanked=False,
                allow_all_prereleases=False,  # Explicitly set to False
            )

            finder = PackageFinder.create(
                link_collector=link_collector,
                selection_prefs=selection_prefs,
            )
            best_candidate = finder.find_best_candidate("pip").best_candidate
            if best_candidate is None:
                return
            pypi_version = str(best_candidate.version)

            # save that we've performed a check
            state.save(pypi_version, current_time)

        remote_version = packaging_version.parse(pypi_version)

        local_version_is_older = (
            pip_version < remote_version and
            pip_version.base_version != remote_version.base_version and
            was_installed_by_pip('pip')
        )

        # Determine if our pypi_version is older
        if not local_version_is_older:
            return

        # We cannot tell how the current pip is available in the current
        # command context, so be pragmatic here and suggest the command
        # that's always available. This does not accommodate spaces in
        # `sys.executable`.
        pip_cmd = "{} -m pip".format(sys.executable)
        logger.warning(
            "You are using pip version %s; however, version %s is "
            "available.\nYou should consider upgrading via the "
            "'%s install --upgrade pip' command.",
            pip_version, pypi_version, pip_cmd
        )
    except Exception:
        logger.debug(
            "There was an error checking the latest version of pip",
            exc_info=True,
        )
