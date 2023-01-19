from __future__ import absolute_import

import logging
import os
import subprocess

from pip._vendor.six.moves import shlex_quote

from pip._internal.cli.spinners import SpinnerInterface, open_spinner
from pip._internal.exceptions import InstallationError
from pip._internal.utils.compat import console_to_str, str_to_display
from pip._internal.utils.logging import subprocess_logger
from pip._internal.utils.misc import HiddenText, path_to_display
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import (
        Any, Callable, Iterable, List, Mapping, Optional, Text, Union,
    )

    CommandArgs = List[Union[str, HiddenText]]


LOG_DIVIDER = '----------------------------------------'


def make_command(*args):
    # type: (Union[str, HiddenText, CommandArgs]) -> CommandArgs
    """
    Create a CommandArgs object.
    """
    command_args = []  # type: CommandArgs
    for arg in args:
        # Check for list instead of CommandArgs since CommandArgs is
        # only known during type-checking.
        if isinstance(arg, list):
            command_args.extend(arg)
        else:
            # Otherwise, arg is str or HiddenText.
            command_args.append(arg)

    return command_args


def format_command_args(args):
    # type: (Union[List[str], CommandArgs]) -> str
    """
    Format command arguments for display.
    """
    # For HiddenText arguments, display the redacted form by calling str().
    # Also, we don't apply str() to arguments that aren't HiddenText since
    # this can trigger a UnicodeDecodeError in Python 2 if the argument
    # has type unicode and includes a non-ascii character.  (The type
    # checker doesn't ensure the annotations are correct in all cases.)
    return ' '.join(
        shlex_quote(str(arg)) if isinstance(arg, HiddenText)
        else shlex_quote(arg) for arg in args
    )


def reveal_command_args(args):
    # type: (Union[List[str], CommandArgs]) -> List[str]
    """
    Return the arguments in their raw, unredacted form.
    """
    return [
        arg.secret if isinstance(arg, HiddenText) else arg for arg in args
    ]


def make_subprocess_output_error(
    cmd_args,     # type: Union[List[str], CommandArgs]
    cwd,          # type: Optional[str]
    lines,        # type: List[Text]
    exit_status,  # type: int
):
    # type: (...) -> Text
    """
    Create and return the error message to use to log a subprocess error
    with command output.

    :param lines: A list of lines, each ending with a newline.
    """
    command = format_command_args(cmd_args)
    # Convert `command` and `cwd` to text (unicode in Python 2) so we can use
    # them as arguments in the unicode format string below. This avoids
    # "UnicodeDecodeError: 'ascii' codec can't decode byte ..." in Python 2
    # if either contains a non-ascii character.
    command_display = str_to_display(command, desc='command bytes')
    cwd_display = path_to_display(cwd)

    # We know the joined output value ends in a newline.
    output = ''.join(lines)
    msg = (
        # Use a unicode string to avoid "UnicodeEncodeError: 'ascii'
        # codec can't encode character ..." in Python 2 when a format
        # argument (e.g. `output`) has a non-ascii character.
        u'Command errored out with exit status {exit_status}:\n'
        ' command: {command_display}\n'
        '     cwd: {cwd_display}\n'
        'Complete output ({line_count} lines):\n{output}{divider}'
    ).format(
        exit_status=exit_status,
        command_display=command_display,
        cwd_display=cwd_display,
        line_count=len(lines),
        output=output,
        divider=LOG_DIVIDER,
    )
    return msg


def call_subprocess(
    cmd,  # type: Union[List[str], CommandArgs]
    show_stdout=False,  # type: bool
    cwd=None,  # type: Optional[str]
    on_returncode='raise',  # type: str
    extra_ok_returncodes=None,  # type: Optional[Iterable[int]]
    command_desc=None,  # type: Optional[str]
    extra_environ=None,  # type: Optional[Mapping[str, Any]]
    unset_environ=None,  # type: Optional[Iterable[str]]
    spinner=None,  # type: Optional[SpinnerInterface]
    log_failed_cmd=True  # type: Optional[bool]
):
    # type: (...) -> Text
    """
    Args:
      show_stdout: if true, use INFO to log the subprocess's stderr and
        stdout streams.  Otherwise, use DEBUG.  Defaults to False.
      extra_ok_returncodes: an iterable of integer return codes that are
        acceptable, in addition to 0. Defaults to None, which means [].
      unset_environ: an iterable of environment variable names to unset
        prior to calling subprocess.Popen().
      log_failed_cmd: if false, failed commands are not logged, only raised.
    """
    if extra_ok_returncodes is None:
        extra_ok_returncodes = []
    if unset_environ is None:
        unset_environ = []
    # Most places in pip use show_stdout=False. What this means is--
    #
    # - We connect the child's output (combined stderr and stdout) to a
    #   single pipe, which we read.
    # - We log this output to stderr at DEBUG level as it is received.
    # - If DEBUG logging isn't enabled (e.g. if --verbose logging wasn't
    #   requested), then we show a spinner so the user can still see the
    #   subprocess is in progress.
    # - If the subprocess exits with an error, we log the output to stderr
    #   at ERROR level if it hasn't already been displayed to the console
    #   (e.g. if --verbose logging wasn't enabled).  This way we don't log
    #   the output to the console twice.
    #
    # If show_stdout=True, then the above is still done, but with DEBUG
    # replaced by INFO.
    if show_stdout:
        # Then log the subprocess output at INFO level.
        log_subprocess = subprocess_logger.info
        used_level = logging.INFO
    else:
        # Then log the subprocess output using DEBUG.  This also ensures
        # it will be logged to the log file (aka user_log), if enabled.
        log_subprocess = subprocess_logger.debug
        used_level = logging.DEBUG

    # Whether the subprocess will be visible in the console.
    showing_subprocess = subprocess_logger.getEffectiveLevel() <= used_level

    # Only use the spinner if we're not showing the subprocess output
    # and we have a spinner.
    use_spinner = not showing_subprocess and spinner is not None

    if command_desc is None:
        command_desc = format_command_args(cmd)

    log_subprocess("Running command %s", command_desc)
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)
    for name in unset_environ:
        env.pop(name, None)
    try:
        proc = subprocess.Popen(
            # Convert HiddenText objects to the underlying str.
            reveal_command_args(cmd),
            stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, cwd=cwd, env=env,
        )
        assert proc.stdin
        assert proc.stdout
        proc.stdin.close()
    except Exception as exc:
        if log_failed_cmd:
            subprocess_logger.critical(
                "Error %s while executing command %s", exc, command_desc,
            )
        raise
    all_output = []
    while True:
        # The "line" value is a unicode string in Python 2.
        line = console_to_str(proc.stdout.readline())
        if not line:
            break
        line = line.rstrip()
        all_output.append(line + '\n')

        # Show the line immediately.
        log_subprocess(line)
        # Update the spinner.
        if use_spinner:
            assert spinner
            spinner.spin()
    try:
        proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()
    proc_had_error = (
        proc.returncode and proc.returncode not in extra_ok_returncodes
    )
    if use_spinner:
        assert spinner
        if proc_had_error:
            spinner.finish("error")
        else:
            spinner.finish("done")
    if proc_had_error:
        if on_returncode == 'raise':
            if not showing_subprocess and log_failed_cmd:
                # Then the subprocess streams haven't been logged to the
                # console yet.
                msg = make_subprocess_output_error(
                    cmd_args=cmd,
                    cwd=cwd,
                    lines=all_output,
                    exit_status=proc.returncode,
                )
                subprocess_logger.error(msg)
            exc_msg = (
                'Command errored out with exit status {}: {} '
                'Check the logs for full command output.'
            ).format(proc.returncode, command_desc)
            raise InstallationError(exc_msg)
        elif on_returncode == 'warn':
            subprocess_logger.warning(
                'Command "%s" had error code %s in %s',
                command_desc,
                proc.returncode,
                cwd,
            )
        elif on_returncode == 'ignore':
            pass
        else:
            raise ValueError('Invalid value: on_returncode={!r}'.format(
                             on_returncode))
    return ''.join(all_output)


def runner_with_spinner_message(message):
    # type: (str) -> Callable[..., None]
    """Provide a subprocess_runner that shows a spinner message.

    Intended for use with for pep517's Pep517HookCaller. Thus, the runner has
    an API that matches what's expected by Pep517HookCaller.subprocess_runner.
    """

    def runner(
        cmd,  # type: List[str]
        cwd=None,  # type: Optional[str]
        extra_environ=None  # type: Optional[Mapping[str, Any]]
    ):
        # type: (...) -> None
        with open_spinner(message) as spinner:
            call_subprocess(
                cmd,
                cwd=cwd,
                extra_environ=extra_environ,
                spinner=spinner,
            )

    return runner
