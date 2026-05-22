# mypy: allow-untyped-defs
# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Taken and modified from original source:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial


IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


logger = logging.getLogger(__name__)


_WIN32_STD_HANDLE = {
    "stdout": -11,  # STD_OUTPUT_HANDLE
    "stderr": -12,  # STD_ERROR_HANDLE
}


def get_libc():
    if IS_MACOS:
        logger.warning("NOTE: Redirects are currently not supported in MacOs.")
        return None
    elif IS_WINDOWS:
        for lib_name in ("ucrtbase", "msvcrt", "msvcr110", "msvcr100"):
            try:
                lib = ctypes.CDLL(lib_name)
                logger.debug("Loaded Windows C runtime: %s", lib_name)
                return lib
            except OSError:
                continue
        raise RuntimeError(
            "Could not load a C runtime DLL on Windows (tried: ucrtbase, msvcrt, "
            "msvcr110, msvcr100). Redirects cannot function without a CRT."
        )
    else:
        return ctypes.CDLL("libc.so.6")


libc = get_libc()


def _c_std(stream: str):
    if IS_WINDOWS:
        stream_index = 2 if stream == "stderr" else 1
        try:
            iob_func = libc.__acrt_iob_func
            iob_func.restype = ctypes.POINTER(ctypes.c_void_p)
            iob_func.argtypes = [ctypes.c_uint]
            return iob_func(stream_index)
        except AttributeError:
            pass
        try:
            legacy_index = 2 if stream == "stderr" else 1
            iob = (ctypes.POINTER(ctypes.c_void_p) * 3).in_dll(libc, "_iob")
            return iob[legacy_index]
        except (AttributeError, OSError) as err:
            raise RuntimeError(
                f"Could not resolve C-runtime FILE* for '{stream}'. "
                "Neither __acrt_iob_func nor _iob are available in the loaded CRT."
            ) from err
    return ctypes.c_void_p.in_dll(libc, stream)


def _python_std(stream: str):
    return {"stdout": sys.stdout, "stderr": sys.stderr}[stream]


_VALID_STD = {"stdout", "stderr"}


if IS_WINDOWS:  # libc is None on macOS; all of the below is Windows-only
    import io as _io
    import msvcrt as _msvcrt

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    _crt_dup = libc._dup
    _crt_dup2 = libc._dup2
    _crt_dup.restype = ctypes.c_int
    _crt_dup.argtypes = [ctypes.c_int]
    _crt_dup2.restype = ctypes.c_int
    _crt_dup2.argtypes = [ctypes.c_int, ctypes.c_int]

    @contextmanager
    def redirect(std: str, to_file: str):
        """
        Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file at ``to_file``.

        On Windows this performs a four-layer redirect:

        1. ``sys.stdout``/``sys.stderr`` -- rewired to a new TextIOWrapper so
           Python's ``print()`` writes to the destination file.
        2. CRT fd (``_dup2``) -- captures C ``printf`` and UCRT ``FILE*`` writers.
        3. Win32 ``SetStdHandle`` -- captures native code using ``WriteFile``/
           ``WriteConsole`` directly, including HIP/ROCm.
        4. ``fflush`` before each switch -- prevents lost output from CRT buffering.

        .. note:: If ROCm/HIP caches the Win32 HANDLE before this redirect runs
                  (e.g. at ``import torch`` time), set up the redirect *before*
                  importing torch/ROCm to capture all output.

        Directory of ``to_file`` is assumed to exist. The destination file is
        overwritten if it already exists.
        """
        if std not in _VALID_STD:
            raise ValueError(
                f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
            )

        std_fd = 1 if std == "stdout" else 2
        win32_handle_id = _WIN32_STD_HANDLE[std]
        orig_sys_std = getattr(sys, std)
        orig_fd_dup = _crt_dup(std_fd)
        if orig_fd_dup == -1:
            raise OSError(f"CRT _dup failed for {std} (fd={std_fd})")
        orig_win32_handle = _kernel32.GetStdHandle(win32_handle_id)

        with open(to_file, mode="w+b") as dst:
            dst_fd = dst.fileno()

            try:
                libc.fflush(_c_std(std))
            except Exception:
                pass
            try:
                orig_sys_std.flush()
            except Exception:
                pass

            _kernel32.SetStdHandle(
                win32_handle_id,
                _msvcrt.get_osfhandle(dst_fd),  # pyrefly: ignore [missing-attribute]
            )

            if _crt_dup2(dst_fd, std_fd) == -1:
                raise OSError(f"CRT _dup2 failed redirecting {std}")

            new_sys_std = _io.TextIOWrapper(
                open(dst_fd, mode="wb", closefd=False),  # noqa: SIM115
                encoding=orig_sys_std.encoding or "utf-8",
                errors="replace",
                line_buffering=True,
            )
            setattr(sys, std, new_sys_std)

            try:
                yield
            finally:
                try:
                    new_sys_std.flush()
                except Exception:
                    pass
                try:
                    libc.fflush(_c_std(std))
                except Exception:
                    pass

                setattr(sys, std, orig_sys_std)
                _crt_dup2(orig_fd_dup, std_fd)
                os.close(orig_fd_dup)
                _kernel32.SetStdHandle(win32_handle_id, orig_win32_handle)

else:

    @contextmanager
    def redirect(std: str, to_file: str):
        """
        Redirect ``std`` (one of ``"stdout"`` or ``"stderr"``) to a file in the path specified by ``to_file``.

        This method redirects the underlying std file descriptor (not just python's ``sys.stdout|stderr``).
        See usage for details.

        Directory of ``dst_filename`` is assumed to exist and the destination file
        is overwritten if it already exists.

        .. note:: Due to buffering cross source writes are not guaranteed to
                  appear in wall-clock order. For instance in the example below
                  it is possible for the C-outputs to appear before the python
                  outputs in the log file.

        Usage::

            # syntactic-sugar for redirect("stdout", "tmp/stdout.log")
            with redirect_stdout("/tmp/stdout.log"):
                print("python stdouts are redirected")
                libc = ctypes.CDLL("libc.so.6")
                libc.printf(b"c stdouts are also redirected")
                os.system("echo system stdouts are also redirected")

            print("stdout restored")
        """
        if std not in _VALID_STD:
            raise ValueError(
                f"unknown standard stream <{std}>, must be one of {_VALID_STD}"
            )

        c_std = _c_std(std)
        python_std = _python_std(std)
        std_fd = python_std.fileno()

        def _redirect(dst):
            libc.fflush(c_std)
            python_std.flush()
            os.dup2(dst.fileno(), std_fd)

        with os.fdopen(os.dup(std_fd)) as orig_std, open(to_file, mode="w+b") as dst:
            _redirect(dst)
            try:
                yield
            finally:
                _redirect(orig_std)


redirect_stdout = partial(redirect, "stdout")
redirect_stderr = partial(redirect, "stderr")
