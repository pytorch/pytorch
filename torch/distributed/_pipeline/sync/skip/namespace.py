# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Provides isolated namespace of skip tensors."""
import abc
from functools import total_ordering
from typing import Any
import uuid

__all__ = ["Namespace"]


@total_ordering
class Namespace(metaclass=abc.ABCMeta):
    """Namespace for isolating skip tensors used by :meth:`isolate()
    <torchpipe.skip.skippable.Skippable.isolate>`.
    """

    __slots__ = ("id",)

    def __init__(self) -> None:
        self.id = uuid.uuid4()

    def __repr__(self) -> str:
        return f"<Namespace '{self.id}'>"

    def __hash__(self) -> int:
        return hash(self.id)

    # Namespaces should support ordering, since SkipLayout will sort tuples
    # including a namespace. But actual order between namespaces is not
    # important. That's why they are ordered by version 4 UUID which generates
    # random numbers.
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id < other.id
        return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id == other.id
        return False


# 'None' is the default namespace,
# which means that 'isinstance(None, Namespace)' is 'True'.
Namespace.register(type(None))
