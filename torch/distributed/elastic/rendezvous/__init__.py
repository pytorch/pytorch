#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""" Rendezvous

In the context of torchelastic we use the term ``rendezvous`` to refer to
a particular functionality that combines a **distributed
synchronization** primitive with **peer discovery**.

It is used by torchelastic to gather participants of a training job
(i.e. workers) such that they all agree on the same list of participants
and everyone’s roles, as well as make a consistent collective decision
on when training can begin/resume.

Torchelastic Rendezvous provides the following critical functionalities:

**Barrier**:

Workers performing rendezvous will all block until the rendezvous is
considered complete - this happens when at least ``min`` total number of
workers have joined the rendezvous barrier (for the same job). This also
implies the barrier is not necessarily of fixed size.

There’s an additional small waiting time after reaching ``min`` number
of workers - this is used to ensure the rendezvous is not completed “too
quickly” (which could potentially exclude additional workers attempting
to join at approximately the same time).

If ``max`` number of workers is gathered at the barrier, the rendezvous
is completed immediately.

There’s also an overall timeout which causes the rendezvous to fail if
``min`` number of workers is never reached – this is meant to be a
simple fail-safe to help release partially allocated job resources, in
case there’s a problem with the resource manger, and is meant to be
interpreted as non-retryable.

**Exclusivity**:

A simple distributed barrier would not be sufficient, as we also need to
ensure that only one group of workers exists at any given time (for a
given job). In other words, new workers (i.e. joining late) should not
be able to form a parallel independent group of workers for the same
job.

Torchelastic rendezvous ensures that if a group of workers has already
completed a rendezvous (and hence might already be training), then
additional “late” workers attempting to rendezvous will only announce
themselves as waiting, and will have to wait until the (previously
completed) existing rendezvous is destroyed first.

**Consistency**:


When a rendezvous is completed, all its members will agree on the job
membership and everyone’s role in it. This role is represented using an
integer, called rank, that is between between 0 and world size.

Note that ranks are *not stable*, in the sense that the same worker
process can be assigned a different rank in the next (re-)rendezvous.

**Fault-tolerance**:

Torchelastic rendezvous is designed to tolerate worker failures during
the rendezvous process. Should a process crash (or lose network
connectivity, etc), between joining the rendezvous and it being
completed, then a re-rendezvous with remaining healthy workers will
happen automatically.

A worker can also fail *after* it has completed (or *has been
observered* by other workers to have completed) the rendezvous - this
scenario will be handled by the torchelastic ``train_loop`` instead
(where it will also trigger a re-rendezvous).

**Shared key-value store**:

When the rendezvous is completed, a shared key-value store is created
and returned. This store implements a ``torch.distributed.Store`` API
(see `distributed communication
docs <https://pytorch.org/docs/stable/distributed.html>`__).

This store is only shared by the members of the completed rendezvous. It
is intended to be used by torchelastic to exchange information necessary
to initialize job control and data-planes.

**Waiting workers and rendezvous closing**:

Torchelastic rendezvous handler object provides additional
functionalities, which are technically not part of the rendezvous
process:

1. Querying how many workers arrived late at the barrier, who
   can participate in *next* rendezvous.

2. Setting the rendezvous *closed* to signal all workers not
   to participate in next rendezvous.
"""

from .api import (  # noqa: F401
    RendezvousClosedException,
    RendezvousException,
    RendezvousHandler,
    RendezvousHandlerFactory,
    RendezvousNonRetryableError,
    RendezvousParameters,
    RendezvousTimeoutException,
)
