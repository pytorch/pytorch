# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

import os
import shutil
import tempfile
from functools import wraps
from typing import Callable
from typing_extensions import ParamSpec

import torch.distributed as dist


_P = ParamSpec("_P")
_F = Callable[_P, None]


def with_temp_dir(func: _F) -> _F:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: _P.args, **kwargs: _P.kwargs) -> None:
        if dist.is_initialized():
            # Only create temp_dir when rank is 0
            if dist.get_rank() == 0:
                temp_dir = tempfile.mkdtemp()
                print(f"Using temp directory: {temp_dir}")
            else:
                temp_dir = ""
            object_list = [temp_dir]

            # Broadcast temp_dir to all the other ranks
            os.sync()
            dist.broadcast_object_list(object_list)
            self.temp_dir = object_list[0]
            os.sync()
        else:
            temp_dir = tempfile.mkdtemp()
            print(f"No process group initialized, using temp directory: {temp_dir}")
            self.temp_dir = temp_dir

        try:
            func(self, *args, **kwargs)
        finally:
            if dist.is_initialized() and dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper
