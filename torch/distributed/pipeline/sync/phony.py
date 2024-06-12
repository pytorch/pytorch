# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Provides phony for arbitrary dependency in a autograd graph."""
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .stream import default_stream, use_stream

__all__: List[str] = ["get_phony"]


_phonies: Dict[Tuple[torch.device, bool], Tensor] = {}


def get_phony(device: torch.device, *, requires_grad: bool) -> Tensor:
    """Get a phony. Phony is tensor without space.

    It is useful to make arbitrary dependency in a autograd graph because it doesn't require any
    gradient accumulation.

    .. note::

        Phonies for each device are cached. If an autograd function gets a phony
        internally, the phony must be detached to be returned. Otherwise, the
        autograd engine will mutate the cached phony in-place::

            class Phonify(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    phony = get_phony(input.device, requires_grad=False)
                    return phony.detach()  # detach() is necessary.

    """
    key = (device, requires_grad)

    try:
        phony = _phonies[key]
    except KeyError:
        with use_stream(default_stream(device)):
            phony = torch.empty(0, device=device, requires_grad=requires_grad)

        _phonies[key] = phony

    return phony
