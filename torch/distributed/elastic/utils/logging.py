#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import warnings
from typing import Optional

from torch.distributed.elastic.utils.log_level import get_log_level


def get_logger(name: Optional[str] = None):
    """
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """

    # Derive the name of the caller, if none provided
    # Use depth=2 since this function takes up one level in the call stack
    return _setup_logger(name or _derive_module_name(depth=2))


def _setup_logger(name: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOGLEVEL", get_log_level()))
    return logger


def _derive_module_name(depth: int = 1) -> Optional[str]:
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
    try:
        stack = inspect.stack()
        assert depth < len(stack)
        # FrameInfo is just a named tuple: (frame, filename, lineno, function, code_context, index)
        frame_info = stack[depth]

        module = inspect.getmodule(frame_info[0])
        if module:
            module_name = module.__name__
        else:
            # inspect.getmodule(frame_info[0]) does NOT work (returns None) in
            # binaries built with @mode/opt
            # return the filename (minus the .py extension) as modulename
            filename = frame_info[1]
            module_name = os.path.splitext(os.path.basename(filename))[0]
        return module_name
    except Exception as e:
        warnings.warn(
            f"Error deriving logger module name, using <None>. Exception: {e}",
            RuntimeWarning,
        )
        return None
