# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import logging
import os

from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip._internal.utils.misc import display_path, rmtree
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import VersionControl, vcs

if MYPY_CHECK_RUNNING:
    from typing import Optional, Tuple
    from pip._internal.utils.misc import HiddenText
    from pip._internal.vcs.versioncontrol import AuthInfo, RevOptions


logger = logging.getLogger(__name__)


class Bazaar(VersionControl):
    name = 'bzr'
    dirname = '.bzr'
    repo_name = 'branch'
    schemes = (
        'bzr', 'bzr+http', 'bzr+https', 'bzr+ssh', 'bzr+sftp', 'bzr+ftp',
        'bzr+lp',
    )

    def __init__(self, *args, **kwargs):
        super(Bazaar, self).__init__(*args, **kwargs)
        # This is only needed for python <2.7.5
        # Register lp but do not expose as a scheme to support bzr+lp.
        if getattr(urllib_parse, 'uses_fragment', None):
            urllib_parse.uses_fragment.extend(['lp'])

    @staticmethod
    def get_base_rev_args(rev):
        return ['-r', rev]

    def export(self, location, url):
        # type: (str, HiddenText) -> None
        """
        Export the Bazaar repository at the url to the destination location
        """
        # Remove the location to make sure Bazaar can export it correctly
        if os.path.exists(location):
            rmtree(location)

        url, rev_options = self.get_url_rev_options(url)
        self.run_command(
            make_command('export', location, url, rev_options.to_args())
        )

    def fetch_new(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        rev_display = rev_options.to_display()
        logger.info(
            'Checking out %s%s to %s',
            url,
            rev_display,
            display_path(dest),
        )
        cmd_args = (
            make_command('branch', '-q', rev_options.to_args(), url, dest)
        )
        self.run_command(cmd_args)

    def switch(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        self.run_command(make_command('switch', url), cwd=dest)

    def update(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        cmd_args = make_command('pull', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)

    @classmethod
    def get_url_rev_and_auth(cls, url):
        # type: (str) -> Tuple[str, Optional[str], AuthInfo]
        # hotfix the URL scheme after removing bzr+ from bzr+ssh:// readd it
        url, rev, user_pass = super(Bazaar, cls).get_url_rev_and_auth(url)
        if url.startswith('ssh://'):
            url = 'bzr+' + url
        return url, rev, user_pass

    @classmethod
    def get_remote_url(cls, location):
        urls = cls.run_command(['info'], cwd=location)
        for line in urls.splitlines():
            line = line.strip()
            for x in ('checkout of branch: ',
                      'parent branch: '):
                if line.startswith(x):
                    repo = line.split(x)[1]
                    if cls._is_local_repository(repo):
                        return path_to_url(repo)
                    return repo
        return None

    @classmethod
    def get_revision(cls, location):
        revision = cls.run_command(
            ['revno'], cwd=location,
        )
        return revision.splitlines()[-1]

    @classmethod
    def is_commit_id_equal(cls, dest, name):
        """Always assume the versions don't match"""
        return False


vcs.register(Bazaar)
