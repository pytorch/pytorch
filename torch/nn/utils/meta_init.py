# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from itertools import chain
from typing import Iterator, Union

import torch
from torch import Tensor
from torch.nn import Module


@contextmanager
def meta_init() -> Iterator[None]:
    """Constructs all tensors within its scope using the meta device.

    This context manager is meant to be used if a module is too expensive or too
    big to construct on a single machine. It forces all tensors within its scope
    to use the meta device regardless of their real device. This way all modules
    constructed inside the context will be allocated on the meta backend.

    Such modules can be used for inspection purposes and then later materialized
    by calling the :func:`materialize` function.
    """
    torch._C._enable_meta_init(True)
    try:
        yield
    finally:
        torch._C._enable_meta_init(False)


def is_meta_init() -> bool:
    """Indicates whether the caller is within a meta-init context."""
    return torch._C._is_meta_init_enabled()


def materialize(obj: Union[Tensor, Module], keep_cache: bool = False) -> None:
    """Materializes ``obj``.

    Args:
        tensor:
            The ``Tensor`` or ``Module`` instance to materialize.
        keep_cache:
            A boolean value indicating whether to clear the meta-init cache once
            the object is materialized.
    """
    if not isinstance(obj, Tensor) and not isinstance(obj, Module):
        raise ValueError("Only `Tensor` and `Module` instances can be materialized.")

    try:
        if isinstance(obj, Tensor):
            torch._C._materialize_tensor(obj)
        else:
            for tsr in chain(obj.parameters(), obj.buffers()):
                torch._C._materialize_tensor(tsr)
    except ValueError as exc:
        raise ValueError(
            "The specified object was not constructed within the scope of a meta-init context "
            "or the meta-init cache has already been cleared. If you want to materialize more "
            "than one object, set the `keep_cache` parameter to `True`."
        ) from exc

    if not keep_cache:
        clear_meta_init_cache()


def clear_meta_init_cache() -> None:
    """Clears the meta-init cache used for materialization."""
    torch._C._clear_meta_init_cache()
