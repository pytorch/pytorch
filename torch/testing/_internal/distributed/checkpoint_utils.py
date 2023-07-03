# Copyright (c) Meta Platforms, Inc. and affiliates

import shutil
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import torch.distributed as dist


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        # Only create temp_dir when rank is 0
        if dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]

        try:
            func(self)
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper
