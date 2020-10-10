# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import logging
import os

from pip._vendor.six.moves import configparser

from pip._internal.exceptions import BadCommand, SubProcessError
from pip._internal.utils.misc import display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
    VersionControl,
    find_path_to_setup_from_repo_root,
    vcs,
)

if MYPY_CHECK_RUNNING:
    from pip._internal.utils.misc import HiddenText
    from pip._internal.vcs.versioncontrol import RevOptions


logger = logging.getLogger(__name__)


class Mercurial(VersionControl):
    name = 'hg'
    dirname = '.hg'
    repo_name = 'clone'
    schemes = (
        'hg', 'hg+file', 'hg+http', 'hg+https', 'hg+ssh', 'hg+static-http',
    )

    @staticmethod
    def get_base_rev_args(rev):
        return [rev]

    def export(self, location, url):
        # type: (str, HiddenText) -> None
        """Export the Hg repository at the url to the destination location"""
        with TempDirectory(kind="export") as temp_dir:
            self.unpack(temp_dir.path, url=url)

            self.run_command(
                ['archive', location], cwd=temp_dir.path
            )

    def fetch_new(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        rev_display = rev_options.to_display()
        logger.info(
            'Cloning hg %s%s to %s',
            url,
            rev_display,
            display_path(dest),
        )
        self.run_command(make_command('clone', '--noupdate', '-q', url, dest))
        self.run_command(
            make_command('update', '-q', rev_options.to_args()),
            cwd=dest,
        )

    def switch(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        repo_config = os.path.join(dest, self.dirname, 'hgrc')
        config = configparser.RawConfigParser()
        try:
            config.read(repo_config)
            config.set('paths', 'default', url.secret)
            with open(repo_config, 'w') as config_file:
                config.write(config_file)
        except (OSError, configparser.NoSectionError) as exc:
            logger.warning(
                'Could not switch Mercurial repository to %s: %s', url, exc,
            )
        else:
            cmd_args = make_command('update', '-q', rev_options.to_args())
            self.run_command(cmd_args, cwd=dest)

    def update(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        self.run_command(['pull', '-q'], cwd=dest)
        cmd_args = make_command('update', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)

    @classmethod
    def get_remote_url(cls, location):
        url = cls.run_command(
            ['showconfig', 'paths.default'],
            cwd=location).strip()
        if cls._is_local_repository(url):
            url = path_to_url(url)
        return url.strip()

    @classmethod
    def get_revision(cls, location):
        """
        Return the repository-local changeset revision number, as an integer.
        """
        current_revision = cls.run_command(
            ['parents', '--template={rev}'], cwd=location).strip()
        return current_revision

    @classmethod
    def get_requirement_revision(cls, location):
        """
        Return the changeset identification hash, as a 40-character
        hexadecimal string
        """
        current_rev_hash = cls.run_command(
            ['parents', '--template={node}'],
            cwd=location).strip()
        return current_rev_hash

    @classmethod
    def is_commit_id_equal(cls, dest, name):
        """Always assume the versions don't match"""
        return False

    @classmethod
    def get_subdirectory(cls, location):
        """
        Return the path to setup.py, relative to the repo root.
        Return None if setup.py is in the repo root.
        """
        # find the repo root
        repo_root = cls.run_command(
            ['root'], cwd=location).strip()
        if not os.path.isabs(repo_root):
            repo_root = os.path.abspath(os.path.join(location, repo_root))
        return find_path_to_setup_from_repo_root(location, repo_root)

    @classmethod
    def get_repository_root(cls, location):
        loc = super(Mercurial, cls).get_repository_root(location)
        if loc:
            return loc
        try:
            r = cls.run_command(
                ['root'],
                cwd=location,
                log_failed_cmd=False,
            )
        except BadCommand:
            logger.debug("could not determine if %s is under hg control "
                         "because hg is not available", location)
            return None
        except SubProcessError:
            return None
        return os.path.normpath(r.rstrip('\r\n'))


vcs.register(Mercurial)
