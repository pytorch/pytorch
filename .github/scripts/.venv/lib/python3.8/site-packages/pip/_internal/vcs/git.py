# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import logging
import os.path
import re

from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.six.moves.urllib import parse as urllib_parse
from pip._vendor.six.moves.urllib import request as urllib_request

from pip._internal.exceptions import BadCommand, SubProcessError
from pip._internal.utils.misc import display_path, hide_url
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.vcs.versioncontrol import (
    RemoteNotFoundError,
    VersionControl,
    find_path_to_setup_from_repo_root,
    vcs,
)

if MYPY_CHECK_RUNNING:
    from typing import Optional, Tuple
    from pip._internal.utils.misc import HiddenText
    from pip._internal.vcs.versioncontrol import AuthInfo, RevOptions


urlsplit = urllib_parse.urlsplit
urlunsplit = urllib_parse.urlunsplit


logger = logging.getLogger(__name__)


HASH_REGEX = re.compile('^[a-fA-F0-9]{40}$')


def looks_like_hash(sha):
    return bool(HASH_REGEX.match(sha))


class Git(VersionControl):
    name = 'git'
    dirname = '.git'
    repo_name = 'clone'
    schemes = (
        'git', 'git+http', 'git+https', 'git+ssh', 'git+git', 'git+file',
    )
    # Prevent the user's environment variables from interfering with pip:
    # https://github.com/pypa/pip/issues/1130
    unset_environ = ('GIT_DIR', 'GIT_WORK_TREE')
    default_arg_rev = 'HEAD'

    @staticmethod
    def get_base_rev_args(rev):
        return [rev]

    def is_immutable_rev_checkout(self, url, dest):
        # type: (str, str) -> bool
        _, rev_options = self.get_url_rev_options(hide_url(url))
        if not rev_options.rev:
            return False
        if not self.is_commit_id_equal(dest, rev_options.rev):
            # the current commit is different from rev,
            # which means rev was something else than a commit hash
            return False
        # return False in the rare case rev is both a commit hash
        # and a tag or a branch; we don't want to cache in that case
        # because that branch/tag could point to something else in the future
        is_tag_or_branch = bool(
            self.get_revision_sha(dest, rev_options.rev)[0]
        )
        return not is_tag_or_branch

    def get_git_version(self):
        VERSION_PFX = 'git version '
        version = self.run_command(['version'])
        if version.startswith(VERSION_PFX):
            version = version[len(VERSION_PFX):].split()[0]
        else:
            version = ''
        # get first 3 positions of the git version because
        # on windows it is x.y.z.windows.t, and this parses as
        # LegacyVersion which always smaller than a Version.
        version = '.'.join(version.split('.')[:3])
        return parse_version(version)

    @classmethod
    def get_current_branch(cls, location):
        """
        Return the current branch, or None if HEAD isn't at a branch
        (e.g. detached HEAD).
        """
        # git-symbolic-ref exits with empty stdout if "HEAD" is a detached
        # HEAD rather than a symbolic ref.  In addition, the -q causes the
        # command to exit with status code 1 instead of 128 in this case
        # and to suppress the message to stderr.
        args = ['symbolic-ref', '-q', 'HEAD']
        output = cls.run_command(
            args, extra_ok_returncodes=(1, ), cwd=location,
        )
        ref = output.strip()

        if ref.startswith('refs/heads/'):
            return ref[len('refs/heads/'):]

        return None

    def export(self, location, url):
        # type: (str, HiddenText) -> None
        """Export the Git repository at the url to the destination location"""
        if not location.endswith('/'):
            location = location + '/'

        with TempDirectory(kind="export") as temp_dir:
            self.unpack(temp_dir.path, url=url)
            self.run_command(
                ['checkout-index', '-a', '-f', '--prefix', location],
                cwd=temp_dir.path
            )

    @classmethod
    def get_revision_sha(cls, dest, rev):
        """
        Return (sha_or_none, is_branch), where sha_or_none is a commit hash
        if the revision names a remote branch or tag, otherwise None.

        Args:
          dest: the repository directory.
          rev: the revision name.
        """
        # Pass rev to pre-filter the list.

        output = ''
        try:
            output = cls.run_command(['show-ref', rev], cwd=dest)
        except SubProcessError:
            pass

        refs = {}
        for line in output.strip().splitlines():
            try:
                sha, ref = line.split()
            except ValueError:
                # Include the offending line to simplify troubleshooting if
                # this error ever occurs.
                raise ValueError('unexpected show-ref line: {!r}'.format(line))

            refs[ref] = sha

        branch_ref = 'refs/remotes/origin/{}'.format(rev)
        tag_ref = 'refs/tags/{}'.format(rev)

        sha = refs.get(branch_ref)
        if sha is not None:
            return (sha, True)

        sha = refs.get(tag_ref)

        return (sha, False)

    @classmethod
    def resolve_revision(cls, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> RevOptions
        """
        Resolve a revision to a new RevOptions object with the SHA1 of the
        branch, tag, or ref if found.

        Args:
          rev_options: a RevOptions object.
        """
        rev = rev_options.arg_rev
        # The arg_rev property's implementation for Git ensures that the
        # rev return value is always non-None.
        assert rev is not None

        sha, is_branch = cls.get_revision_sha(dest, rev)

        if sha is not None:
            rev_options = rev_options.make_new(sha)
            rev_options.branch_name = rev if is_branch else None

            return rev_options

        # Do not show a warning for the common case of something that has
        # the form of a Git commit hash.
        if not looks_like_hash(rev):
            logger.warning(
                "Did not find branch or tag '%s', assuming revision or ref.",
                rev,
            )

        if not rev.startswith('refs/'):
            return rev_options

        # If it looks like a ref, we have to fetch it explicitly.
        cls.run_command(
            make_command('fetch', '-q', url, rev_options.to_args()),
            cwd=dest,
        )
        # Change the revision to the SHA of the ref we fetched
        sha = cls.get_revision(dest, rev='FETCH_HEAD')
        rev_options = rev_options.make_new(sha)

        return rev_options

    @classmethod
    def is_commit_id_equal(cls, dest, name):
        """
        Return whether the current commit hash equals the given name.

        Args:
          dest: the repository directory.
          name: a string name.
        """
        if not name:
            # Then avoid an unnecessary subprocess call.
            return False

        return cls.get_revision(dest) == name

    def fetch_new(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        rev_display = rev_options.to_display()
        logger.info('Cloning %s%s to %s', url, rev_display, display_path(dest))
        self.run_command(make_command('clone', '-q', url, dest))

        if rev_options.rev:
            # Then a specific revision was requested.
            rev_options = self.resolve_revision(dest, url, rev_options)
            branch_name = getattr(rev_options, 'branch_name', None)
            if branch_name is None:
                # Only do a checkout if the current commit id doesn't match
                # the requested revision.
                if not self.is_commit_id_equal(dest, rev_options.rev):
                    cmd_args = make_command(
                        'checkout', '-q', rev_options.to_args(),
                    )
                    self.run_command(cmd_args, cwd=dest)
            elif self.get_current_branch(dest) != branch_name:
                # Then a specific branch was requested, and that branch
                # is not yet checked out.
                track_branch = 'origin/{}'.format(branch_name)
                cmd_args = [
                    'checkout', '-b', branch_name, '--track', track_branch,
                ]
                self.run_command(cmd_args, cwd=dest)

        #: repo may contain submodules
        self.update_submodules(dest)

    def switch(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        self.run_command(
            make_command('config', 'remote.origin.url', url),
            cwd=dest,
        )
        cmd_args = make_command('checkout', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)

        self.update_submodules(dest)

    def update(self, dest, url, rev_options):
        # type: (str, HiddenText, RevOptions) -> None
        # First fetch changes from the default remote
        if self.get_git_version() >= parse_version('1.9.0'):
            # fetch tags in addition to everything else
            self.run_command(['fetch', '-q', '--tags'], cwd=dest)
        else:
            self.run_command(['fetch', '-q'], cwd=dest)
        # Then reset to wanted revision (maybe even origin/master)
        rev_options = self.resolve_revision(dest, url, rev_options)
        cmd_args = make_command('reset', '--hard', '-q', rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)
        #: update submodules
        self.update_submodules(dest)

    @classmethod
    def get_remote_url(cls, location):
        """
        Return URL of the first remote encountered.

        Raises RemoteNotFoundError if the repository does not have a remote
        url configured.
        """
        # We need to pass 1 for extra_ok_returncodes since the command
        # exits with return code 1 if there are no matching lines.
        stdout = cls.run_command(
            ['config', '--get-regexp', r'remote\..*\.url'],
            extra_ok_returncodes=(1, ), cwd=location,
        )
        remotes = stdout.splitlines()
        try:
            found_remote = remotes[0]
        except IndexError:
            raise RemoteNotFoundError

        for remote in remotes:
            if remote.startswith('remote.origin.url '):
                found_remote = remote
                break
        url = found_remote.split(' ')[1]
        return url.strip()

    @classmethod
    def get_revision(cls, location, rev=None):
        if rev is None:
            rev = 'HEAD'
        current_rev = cls.run_command(
            ['rev-parse', rev], cwd=location,
        )
        return current_rev.strip()

    @classmethod
    def get_subdirectory(cls, location):
        """
        Return the path to setup.py, relative to the repo root.
        Return None if setup.py is in the repo root.
        """
        # find the repo root
        git_dir = cls.run_command(
            ['rev-parse', '--git-dir'],
            cwd=location).strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(location, git_dir)
        repo_root = os.path.abspath(os.path.join(git_dir, '..'))
        return find_path_to_setup_from_repo_root(location, repo_root)

    @classmethod
    def get_url_rev_and_auth(cls, url):
        # type: (str) -> Tuple[str, Optional[str], AuthInfo]
        """
        Prefixes stub URLs like 'user@hostname:user/repo.git' with 'ssh://'.
        That's required because although they use SSH they sometimes don't
        work with a ssh:// scheme (e.g. GitHub). But we need a scheme for
        parsing. Hence we remove it again afterwards and return it as a stub.
        """
        # Works around an apparent Git bug
        # (see https://article.gmane.org/gmane.comp.version-control.git/146500)
        scheme, netloc, path, query, fragment = urlsplit(url)
        if scheme.endswith('file'):
            initial_slashes = path[:-len(path.lstrip('/'))]
            newpath = (
                initial_slashes +
                urllib_request.url2pathname(path)
                .replace('\\', '/').lstrip('/')
            )
            url = urlunsplit((scheme, netloc, newpath, query, fragment))
            after_plus = scheme.find('+') + 1
            url = scheme[:after_plus] + urlunsplit(
                (scheme[after_plus:], netloc, newpath, query, fragment),
            )

        if '://' not in url:
            assert 'file:' not in url
            url = url.replace('git+', 'git+ssh://')
            url, rev, user_pass = super(Git, cls).get_url_rev_and_auth(url)
            url = url.replace('ssh://', '')
        else:
            url, rev, user_pass = super(Git, cls).get_url_rev_and_auth(url)

        return url, rev, user_pass

    @classmethod
    def update_submodules(cls, location):
        if not os.path.exists(os.path.join(location, '.gitmodules')):
            return
        cls.run_command(
            ['submodule', 'update', '--init', '--recursive', '-q'],
            cwd=location,
        )

    @classmethod
    def get_repository_root(cls, location):
        loc = super(Git, cls).get_repository_root(location)
        if loc:
            return loc
        try:
            r = cls.run_command(
                ['rev-parse', '--show-toplevel'],
                cwd=location,
                log_failed_cmd=False,
            )
        except BadCommand:
            logger.debug("could not determine if %s is under git control "
                         "because git is not available", location)
            return None
        except SubProcessError:
            return None
        return os.path.normpath(r.rstrip('\r\n'))


vcs.register(Git)
