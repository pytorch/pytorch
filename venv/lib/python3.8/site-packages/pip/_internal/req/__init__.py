from __future__ import absolute_import

import collections
import logging

from pip._internal.utils.logging import indent_log
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet

if MYPY_CHECK_RUNNING:
    from typing import Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "RequirementSet", "InstallRequirement",
    "parse_requirements", "install_given_reqs",
]

logger = logging.getLogger(__name__)


class InstallationResult(object):
    def __init__(self, name):
        # type: (str) -> None
        self.name = name

    def __repr__(self):
        # type: () -> str
        return "InstallationResult(name={!r})".format(self.name)


def _validate_requirements(
    requirements,  # type: List[InstallRequirement]
):
    # type: (...) -> Iterator[Tuple[str, InstallRequirement]]
    for req in requirements:
        assert req.name, "invalid to-be-installed requirement: {}".format(req)
        yield req.name, req


def install_given_reqs(
    requirements,  # type: List[InstallRequirement]
    install_options,  # type: List[str]
    global_options,  # type: Sequence[str]
    root,  # type: Optional[str]
    home,  # type: Optional[str]
    prefix,  # type: Optional[str]
    warn_script_location,  # type: bool
    use_user_site,  # type: bool
    pycompile,  # type: bool
):
    # type: (...) -> List[InstallationResult]
    """
    Install everything in the given list.

    (to be called after having downloaded and unpacked the packages)
    """
    to_install = collections.OrderedDict(_validate_requirements(requirements))

    if to_install:
        logger.info(
            'Installing collected packages: %s',
            ', '.join(to_install.keys()),
        )

    installed = []

    with indent_log():
        for req_name, requirement in to_install.items():
            if requirement.should_reinstall:
                logger.info('Attempting uninstall: %s', req_name)
                with indent_log():
                    uninstalled_pathset = requirement.uninstall(
                        auto_confirm=True
                    )
            else:
                uninstalled_pathset = None

            try:
                requirement.install(
                    install_options,
                    global_options,
                    root=root,
                    home=home,
                    prefix=prefix,
                    warn_script_location=warn_script_location,
                    use_user_site=use_user_site,
                    pycompile=pycompile,
                )
            except Exception:
                # if install did not succeed, rollback previous uninstall
                if uninstalled_pathset and not requirement.install_succeeded:
                    uninstalled_pathset.rollback()
                raise
            else:
                if uninstalled_pathset and requirement.install_succeeded:
                    uninstalled_pathset.commit()

            installed.append(InstallationResult(req_name))

    return installed
