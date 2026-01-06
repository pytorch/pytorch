# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
In the context of Torch Distributed Elastic we use the term *rendezvous* to
refer to a particular functionality that combines a **distributed
synchronization** primitive with **peer discovery**.

It is used by Torch Distributed Elastic to gather participants of a training
job (i.e. nodes) such that they all agree on the same list of participants and
everyone's roles, as well as make a consistent collective decision on when
training can begin/resume.

Torch Distributed Elastic rendezvous provides the following critical
functionalities:

**Barrier**:

Nodes performing rendezvous will all block until the rendezvous is considered
complete - this happens when at least ``min`` total number of nodes have joined
the rendezvous barrier (for the same job). This also implies the barrier is not
necessarily of fixed size.

There's an additional small waiting time after reaching ``min`` number of
nodes - this is used to ensure the rendezvous is not completed "too quickly"
(which could potentially exclude additional nodes attempting to join at
approximately the same time).

If ``max`` number of nodes is gathered at the barrier, the rendezvous is
completed immediately.

There's also an overall timeout which causes the rendezvous to fail if ``min``
number of nodes is never reached - this is meant to be a simple fail-safe to
help release partially allocated job resources, in case there's a problem with
the resource manager, and is meant to be interpreted as non-retryable.

**Exclusivity**:

A simple distributed barrier would not be sufficient, as we also need to ensure
that only one group of nodes exists at any given time (for a given job). In
other words, new nodes (i.e. joining late) should not be able to form a parallel
independent group of workers for the same job.

Torch Distributed Elastic rendezvous ensures that if a group of nodes has
already completed a rendezvous (and hence might already be training), then
additional "late" nodes attempting to rendezvous will only announce themselves
as waiting, and will have to wait until the (previously completed) existing
rendezvous is destroyed first.

**Consistency**:

When a rendezvous is completed, all its members will agree on the job membership
and everyone's role in it. This role is represented using an integer, called
rank, that is between between 0 and world size.

Note that ranks are *not stable*, in the sense that the same node can be
assigned a different rank in the next (re-)rendezvous.

**Fault-tolerance**:

Torch Distributed Elastic rendezvous is designed to tolerate node failures
during the rendezvous process. Should a process crash (or lose network
connectivity, etc), between joining the rendezvous and it being completed, then
a re-rendezvous with remaining healthy nodes will happen automatically.

A node can also fail *after* it has completed (or *has been observed* by other
nodes to have completed) the rendezvous - this scenario will be handled by the
Torch Distributed Elastic ``train_loop`` instead (where it will also trigger a
re-rendezvous).

**Shared key-value store**:

When the rendezvous is completed, a shared key-value store is created and
returned. This store implements a ``torch.distributed.Store`` API (see
`distributed communication docs
<https://pytorch.org/docs/stable/distributed.html>`__).

This store is only shared by the members of the completed rendezvous. It
is intended to be used by Torch Distributed Elastic to exchange information
necessary to initialize job control and data-planes.

**Waiting workers and rendezvous closing**:

Torch Distributed Elastic rendezvous handler object provides additional
functionalities, which are technically not part of the rendezvous process:

1. Querying how many workers arrived late at the barrier, who can participate in
   *next* rendezvous.

2. Setting the rendezvous *closed* to signal all nodes not to participate in
   next rendezvous.

**DynamicRendezvousHandler**:

Torch Distributed Elastic comes with the :py:class:`.DynamicRendezvousHandler`
class that implements the rendezvous mechanism described above. It is a backend-
agnostic type that expects a particular :py:class:`.RendezvousBackend` instance
to be specified during construction.

Torch distributed users can either implement their own backend type or use one
of the following implementations that come with PyTorch:

- :py:class:`.C10dRendezvousBackend`: Uses a C10d store (by default
  ``TCPStore``) as the rendezvous backend. The main advantage of using a C10d
  store is that it requires no 3rd-party dependency (such as etcd) to establish
  a rendezvous.
- :py:class:`.EtcdRendezvousBackend`: Supersedes the legacy
  :py:class:`.EtcdRendezvousHandler` class. Passing an
  :py:class:`.EtcdRendezvousBackend` instance to
  :py:class:`.DynamicRendezvousHandler` is functionally equivalent to
  instantiating an :py:class:`.EtcdRendezvousHandler`.

  ::

     store = TCPStore("localhost")

     backend = C10dRendezvousBackend(store, "my_run_id")

     rdzv_handler = DynamicRendezvousHandler.from_backend(
         run_id="my_run_id", store=store, backend=backend, min_nodes=2, max_nodes=4
     )
"""

from .api import (
    rendezvous_handler_registry,
    RendezvousClosedError,
    RendezvousConnectionError,
    RendezvousError,
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousHandlerCreator,
    RendezvousHandlerRegistry,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)
from .registry import _register_default_handlers, _register_out_of_tree_handlers


_register_default_handlers()
_register_out_of_tree_handlers()


__all__ = [
    "RendezvousClosedError",
    "RendezvousConnectionError",
    "RendezvousError",
    "RendezvousGracefulExitError",
    "RendezvousHandler",
    "RendezvousHandlerCreator",
    "RendezvousHandlerRegistry",
    "RendezvousInfo",
    "RendezvousParameters",
    "RendezvousStateError",
    "RendezvousStoreInfo",
    "RendezvousTimeoutError",
    "rendezvous_handler_registry",
]
