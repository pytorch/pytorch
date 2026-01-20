#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import signal
import sys
from subprocess import Popen
from typing import Any

from torch.numa.binding import _maybe_wrap_command_args_with_numa_binding, NumaOptions


__all__ = ["SubprocessHandler"]

IS_WINDOWS = sys.platform == "win32"


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined] # noqa: F821
    else:
        return signal.SIGTERM


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: str | None,
        stderr: str | None,
        local_rank_id: int,
        numa_options: NumaOptions | None,
    ):
        self._stdout = open(stdout, "w") if stdout else None  # noqa: SIM115
        self._stderr = open(stderr, "w") if stderr else None  # noqa: SIM115
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        args_str = _maybe_wrap_command_args_with_numa_binding(
            args_str,
            gpu_index=local_rank_id,
            numa_options=numa_options,
        )

        self.local_rank_id = local_rank_id

        self.proc: Popen = self._popen(args_str, env_vars)

    def _popen(self, args: tuple, env: dict[str, str]) -> Popen:
        kwargs: dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True

        return Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )

    def close(self, death_sig: signal.Signals | None = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)
        else:
            os.killpg(self.proc.pid, death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()
