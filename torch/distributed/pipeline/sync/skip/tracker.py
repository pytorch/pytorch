# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Tracks skip tensors on a thread."""
from contextlib import contextmanager
import threading
from typing import Dict, Generator, List, Optional, Tuple

from torch import Tensor

from ..checkpoint import is_checkpointing
from ..dependency import fork, join
from ..microbatch import Batch
from ..stream import AbstractStream
from .layout import SkipLayout
from .namespace import Namespace
from .portal import Portal

__all__: List[str] = []


class SkipTracker:
    """Tracks saved skip tensors.

    It will update the given micro-batch in place. This is because when it
    manipulates the underlying skip tensors, the current micro-batch also has
    to be connected with the skip tensors.

    One thread has one skip tracker. Call :func:`current_skip_tracker` to get
    the skip tracker on the current thread.

    """

    def __init__(self) -> None:
        self.tensors: Dict[Tuple[Namespace, str], Optional[Tensor]] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        self.tensors[(ns, name)] = tensor

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        return self.tensors.pop((ns, name))

    def copy(
        self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str,
    ) -> None:
        raise TypeError("copy is not supported for non-portal skip tensors")


class SkipTrackerThroughPotals(SkipTracker):
    """Tracks saved skip tensors through portals. The skip tensors will be
    hidden in portals so that the autograd engine does not need to track them.

    This tracker is only used when the training or evaluating module is wrapped
    with :class:`torchpipe.Pipe`.

    """

    def __init__(self, skip_layout: SkipLayout) -> None:
        super().__init__()
        self.skip_layout = skip_layout
        self.portals: Dict[Tuple[Namespace, str], Portal] = {}

    def save(self, batch: Batch, ns: Namespace, name: str, tensor: Optional[Tensor]) -> None:
        """Saves the stashed skip tensor in a portal. The portal is then
        connected to the given micro-batch with :class:`Join`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            super().save(batch, ns, name, tensor)
            return

        # See [Tensor Life of Portal] at Portal.put_tensor() to understand the
        # below tensor_life values. Here are the selected events which retrieve
        # the tensor in portal:
        #
        #  1. [x] blue()
        #     ...
        #  6. [x]   PortalOrange.forward
        #     ...
        #  8. [x]   PortalOrange.forward (recomputed)
        #     ...
        # 11. [x] blue() (recomputed)
        #
        if (ns, name) not in self.portals:
            if is_checkpointing():
                # Under checkpointing, the tensor used by the first
                # PortalOrange should be alive in the portal. This tensor will
                # be used again by the second PortalOrange during the
                # recomputation.
                tensor_life = 3  # Delete at [8. PortalOrange.forward (recomputed)]
            else:
                tensor_life = 2  # Delete at [6. PortalOrange.forward]

            portal = Portal(tensor, tensor_life)
            self.portals[(ns, name)] = portal

        else:
            # Under recomputation, the portal already exists.
            portal = self.portals[(ns, name)]

            # The existing tensor life already became 0. It should be reset as
            # 1 to delete the tensor after the second PortalBlue immediately.
            tensor_life = 1  # Delete at [11. blue() (recomputed)]

            portal.put_tensor(tensor, tensor_life)

        phony = portal.blue()
        batch[0] = join(batch[0], phony)

    def load(self, batch: Batch, ns: Namespace, name: str) -> Optional[Tensor]:
        """Loads a skip tensor from the corresponding portal to pop. The given
        micro-batch is connected to the portal with :class:`Fork`.
        """
        if not self.skip_layout.requires_copy(ns, name):
            tensor = super().load(batch, ns, name)
            return tensor

        portal = self.portals[(ns, name)]
        batch[0], phony = fork(batch[0])
        tensor = portal.orange(phony)
        return tensor

    def copy(
        self, batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream, ns: Namespace, name: str,
    ) -> None:
        """Copies the skip tensor in the corresponding portal. The given
        micro-batch and the portal will be tied with :class:`Fork` and
        :class:`Join`.
        """
        assert self.skip_layout.requires_copy(ns, name)

        batch[0], phony = fork(batch[0])

        portal = self.portals[(ns, name)]
        phony = portal.copy(prev_stream, next_stream, phony)

        batch[0] = join(batch[0], phony)


class ThreadLocal(threading.local):
    def __init__(self) -> None:
        self.skip_tracker: Optional[SkipTracker] = None


thread_local = ThreadLocal()


@contextmanager
def use_skip_tracker(skip_tracker: SkipTracker) -> Generator[None, None, None]:
    """Registers the given skip tracker on the current thread within a
    context::

        with use_skip_tracker(my_skip_tracker):
            ...

    """
    orig = thread_local.skip_tracker

    thread_local.skip_tracker = skip_tracker

    try:
        yield
    finally:
        thread_local.skip_tracker = orig


def current_skip_tracker() -> SkipTracker:
    """Gets the skip tracker on the current thread."""
    skip_tracker = thread_local.skip_tracker

    if skip_tracker is None:
        skip_tracker = SkipTracker()
        thread_local.skip_tracker = skip_tracker

    return skip_tracker
