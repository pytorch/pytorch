# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides phony for arbitrary dependency in a autograd graph."""
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .stream import default_stream, use_stream

__all__: List[str] = []


_phonies: Dict[Tuple[torch.device, bool], Tensor] = {}


def get_phony(device: torch.device, *, requires_grad: bool) -> Tensor:
    """Gets a phony. Phony is tensor without space. It is useful to make
    arbitrary dependency in a autograd graph because it doesn't require any
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
