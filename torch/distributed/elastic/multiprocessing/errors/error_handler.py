#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Optional


log = logging.getLogger(__name__)


def _write_error(e: BaseException, error_file: Optional[str]):
    data = {
        "message": {
            "message": f"{type(e).__name__}: {e}",
            "extraInfo": {
                "py_callstack": traceback.format_exc(),
                "timestamp": str(int(time.time())),
            },
        }
    }

    if error_file:
        with open(error_file, "w") as fp:
            json.dump(data, fp)
    else:
        log.error(json.dumps(data, indent=2))


class ErrorHandler:
    """
    Writes the provided exception object along with some other metadata about
    the error in a structured way in JSON format to an error file specified by the
    environment variable: ``TORCHELASTIC_ERROR_FILE``. If this environment
    variable is not set, then simply logs the contents of what would have been
    written to the error file.

    This handler may be subclassed to customize the handling of the error.
    Subclasses should override ``initialize()`` and ``record_exception()``.
    """

    def _get_error_file_path(self) -> Optional[str]:
        """
        Returns the error file path. May return ``None`` to have the
        structured error be logged only.
        """
        return os.environ.get("TORCHELASTIC_ERROR_FILE", None)

    def initialize(self) -> None:
        """
        Called prior to running code that we wish to capture errors/exceptions.
        Typically registers signal/fault handlers. Users can override this
        function to add custom initialization/registrations that aid in
        propagation/information of errors/signals/exceptions/faults.
        """
        try:
            faulthandler.enable(all_threads=True)
        except Exception as e:
            warnings.warn(f"Unable to enable fault handler. {type(e).__name__}: {e}")

    def _write_error_file(self, file_path: str, error_msg: str) -> None:
        """
        Writes error message to the file.
        """
        try:
            with open(file_path, "w") as fp:
                fp.write(error_msg)
        except Exception as e:
            warnings.warn(f"Unable to write error to file. {type(e).__name__}: {e}")

    def record_exception(self, e: BaseException) -> None:
        """
        Writes a structured information about the exception into an error file in
        JSON format. If the error file cannot be determined, then logs the content
        that would have been written to the error file.
        """
        _write_error(e, self._get_error_file_path())

    def dump_error_file(self, rootcause_error_file: str, error_code: int = 0):
        """
        Dumps parent error file from child process's root cause error and error code.
        """
        with open(rootcause_error_file, "r") as fp:
            rootcause_error = json.load(fp)
            # Override error code since the child process cannot capture the error code if it
            # is terminated by singals like SIGSEGV.
            if error_code:
                if "message" not in rootcause_error:
                    log.warning(
                        f"child error file ({rootcause_error_file}) does not have field `message`. \n"
                        f"cannot override error code: {error_code}"
                    )
                elif isinstance(rootcause_error["message"], str):
                    log.warning(
                        f"child error file ({rootcause_error_file}) has a new message format. \n"
                        f"skipping error code override"
                    )
                else:
                    rootcause_error["message"]["errorCode"] = error_code

            log.debug(
                f"child error file ({rootcause_error_file}) contents:\n"
                f"{json.dumps(rootcause_error, indent=2)}"
            )

        my_error_file = self._get_error_file_path()
        if my_error_file:
            # Guard against existing error files
            # This can happen when the child is created using multiprocessing
            # and the same env var (TORCHELASTIC_ERROR_FILE) is used on the
            # parent and child to specify the error files (respectively)
            # because the env vars on the child is set in the wrapper function
            # and by default the child inherits the parent's env vars, if the child
            # process receives a signal before the wrapper function kicks in
            # and the signal handler writes to the error file, then the child
            # will write to the parent's error file. In this case just log the
            # original error file contents and overwrite the error file.
            self._rm(my_error_file)
            self._write_error_file(my_error_file, json.dumps(rootcause_error))
            log.info(f"dumped error file to parent's {my_error_file}")
        else:
            log.error(
                f"no error file defined for parent, to copy child error file ({rootcause_error_file})"
            )

    def _rm(self, my_error_file):
        if os.path.isfile(my_error_file):
            # Log the contents of the original file.
            with open(my_error_file, "r") as fp:
                try:
                    original = json.dumps(json.load(fp), indent=2)
                    log.warning(
                        f"{my_error_file} already exists"
                        f" and will be overwritten."
                        f" Original contents:\n{original}"
                    )
                except json.decoder.JSONDecodeError as err:
                    log.warning(
                        f"{my_error_file} already exists"
                        f" and will be overwritten."
                        f" Unable to load original contents:\n"
                    )
            os.remove(my_error_file)
