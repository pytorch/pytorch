"""Base Command class, and related routines"""

from __future__ import absolute_import, print_function

import logging
import logging.config
import optparse
import os
import platform
import sys
import traceback

from pip._internal.cli import cmdoptions
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.cli.parser import (
    ConfigOptionParser,
    UpdatingDefaultsHelpFormatter,
)
from pip._internal.cli.status_codes import (
    ERROR,
    PREVIOUS_BUILD_DIR_ERROR,
    UNKNOWN_ERROR,
    VIRTUALENV_NOT_FOUND,
)
from pip._internal.exceptions import (
    BadCommand,
    CommandError,
    InstallationError,
    NetworkConnectionError,
    PreviousBuildDirError,
    SubProcessError,
    UninstallationError,
)
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filesystem import check_path_owner
from pip._internal.utils.logging import BrokenStdoutLoggingError, setup_logging
from pip._internal.utils.misc import get_prog, normalize_path
from pip._internal.utils.temp_dir import (
    global_tempdir_manager,
    tempdir_registry,
)
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.virtualenv import running_under_virtualenv

if MYPY_CHECK_RUNNING:
    from typing import List, Optional, Tuple, Any
    from optparse import Values

    from pip._internal.utils.temp_dir import (
        TempDirectoryTypeRegistry as TempDirRegistry
    )

__all__ = ['Command']

logger = logging.getLogger(__name__)


class Command(CommandContextMixIn):
    usage = None  # type: str
    ignore_require_venv = False  # type: bool

    def __init__(self, name, summary, isolated=False):
        # type: (str, str, bool) -> None
        super(Command, self).__init__()
        parser_kw = {
            'usage': self.usage,
            'prog': '{} {}'.format(get_prog(), name),
            'formatter': UpdatingDefaultsHelpFormatter(),
            'add_help_option': False,
            'name': name,
            'description': self.__doc__,
            'isolated': isolated,
        }

        self.name = name
        self.summary = summary
        self.parser = ConfigOptionParser(**parser_kw)

        self.tempdir_registry = None  # type: Optional[TempDirRegistry]

        # Commands should add options to this option group
        optgroup_name = '{} Options'.format(self.name.capitalize())
        self.cmd_opts = optparse.OptionGroup(self.parser, optgroup_name)

        # Add the general options
        gen_opts = cmdoptions.make_option_group(
            cmdoptions.general_group,
            self.parser,
        )
        self.parser.add_option_group(gen_opts)

        self.add_options()

    def add_options(self):
        # type: () -> None
        pass

    def handle_pip_version_check(self, options):
        # type: (Values) -> None
        """
        This is a no-op so that commands by default do not do the pip version
        check.
        """
        # Make sure we do the pip version check if the index_group options
        # are present.
        assert not hasattr(options, 'no_index')

    def run(self, options, args):
        # type: (Values, List[Any]) -> int
        raise NotImplementedError

    def parse_args(self, args):
        # type: (List[str]) -> Tuple[Any, Any]
        # factored out for testability
        return self.parser.parse_args(args)

    def main(self, args):
        # type: (List[str]) -> int
        try:
            with self.main_context():
                return self._main(args)
        finally:
            logging.shutdown()

    def _main(self, args):
        # type: (List[str]) -> int
        # We must initialize this before the tempdir manager, otherwise the
        # configuration would not be accessible by the time we clean up the
        # tempdir manager.
        self.tempdir_registry = self.enter_context(tempdir_registry())
        # Intentionally set as early as possible so globally-managed temporary
        # directories are available to the rest of the code.
        self.enter_context(global_tempdir_manager())

        options, args = self.parse_args(args)

        # Set verbosity so that it can be used elsewhere.
        self.verbosity = options.verbose - options.quiet

        level_number = setup_logging(
            verbosity=self.verbosity,
            no_color=options.no_color,
            user_log_file=options.log,
        )

        if (
            sys.version_info[:2] == (2, 7) and
            not options.no_python_version_warning
        ):
            message = (
                "pip 21.0 will drop support for Python 2.7 in January 2021. "
                "More details about Python 2 support in pip can be found at "
                "https://pip.pypa.io/en/latest/development/release-process/#python-2-support"  # noqa
            )
            if platform.python_implementation() == "CPython":
                message = (
                    "Python 2.7 reached the end of its life on January "
                    "1st, 2020. Please upgrade your Python as Python 2.7 "
                    "is no longer maintained. "
                ) + message
            deprecated(message, replacement=None, gone_in=None)

        # TODO: Try to get these passing down from the command?
        #       without resorting to os.environ to hold these.
        #       This also affects isolated builds and it should.

        if options.no_input:
            os.environ['PIP_NO_INPUT'] = '1'

        if options.exists_action:
            os.environ['PIP_EXISTS_ACTION'] = ' '.join(options.exists_action)

        if options.require_venv and not self.ignore_require_venv:
            # If a venv is required check if it can really be found
            if not running_under_virtualenv():
                logger.critical(
                    'Could not find an activated virtualenv (required).'
                )
                sys.exit(VIRTUALENV_NOT_FOUND)

        if options.cache_dir:
            options.cache_dir = normalize_path(options.cache_dir)
            if not check_path_owner(options.cache_dir):
                logger.warning(
                    "The directory '%s' or its parent directory is not owned "
                    "or is not writable by the current user. The cache "
                    "has been disabled. Check the permissions and owner of "
                    "that directory. If executing pip with sudo, you may want "
                    "sudo's -H flag.",
                    options.cache_dir,
                )
                options.cache_dir = None

        if getattr(options, "build_dir", None):
            deprecated(
                reason=(
                    "The -b/--build/--build-dir/--build-directory "
                    "option is deprecated."
                ),
                replacement=(
                    "use the TMPDIR/TEMP/TMP environment variable, "
                    "possibly combined with --no-clean"
                ),
                gone_in="20.3",
                issue=8333,
            )

        if 'resolver' in options.unstable_features:
            logger.critical(
                "--unstable-feature=resolver is no longer supported, and "
                "has been replaced with --use-feature=2020-resolver instead."
            )
            sys.exit(ERROR)

        try:
            status = self.run(options, args)
            assert isinstance(status, int)
            return status
        except PreviousBuildDirError as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)

            return PREVIOUS_BUILD_DIR_ERROR
        except (InstallationError, UninstallationError, BadCommand,
                SubProcessError, NetworkConnectionError) as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)

            return ERROR
        except CommandError as exc:
            logger.critical('%s', exc)
            logger.debug('Exception information:', exc_info=True)

            return ERROR
        except BrokenStdoutLoggingError:
            # Bypass our logger and write any remaining messages to stderr
            # because stdout no longer works.
            print('ERROR: Pipe to stdout was broken', file=sys.stderr)
            if level_number <= logging.DEBUG:
                traceback.print_exc(file=sys.stderr)

            return ERROR
        except KeyboardInterrupt:
            logger.critical('Operation cancelled by user')
            logger.debug('Exception information:', exc_info=True)

            return ERROR
        except BaseException:
            logger.critical('Exception:', exc_info=True)

            return UNKNOWN_ERROR
        finally:
            self.handle_pip_version_check(options)
