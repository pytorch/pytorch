#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

Torchelastic agent and user worker failover contract:

**TL;DR;**:

* TE(torchelastic) expects user workers to finish with the 5 minutes drift
* It is better to design DDP app to fail for all workers, rather than a single one.
* TE does not synchronize number of restarts between agents
* TE re-rendezvous does not trigger restart decrease
* When a single agent finishes its job(successfully or not), it will close rendezvous.
  If other agents still have workers in progress, they will be terminated.
* Based on above, scale down does not work if at least single agent finishes the job.
* When Scale up is detected by agents, it will not decrease ``max_restarts``


In general TE(torchelastic) can launch arbitrary user code, but there is some
clarifications need to be done around what failover mechanism torchelastic
provides and what failover mechanism it expects from user workers.

Torchelastic currently supports DDP style applications.  That means that
TE expects *ALL* workers finish approximately at the same time. In practice,
it is nearly to impossible to guarantee that all workers in arbitrary
DDP application finish at the time, so TE provides a finalization barrier
that waits for TIMEOUT(5 minutes) for worker finalization.

**Worker Failure**

When worker fails, TE will check the number of restarts
available, if there is more than 0 restarts, TE will start a new rendezvous
round and restart the worker process. New rendezvous round will other
TE agents to terminate their workers.

.. note:: The TE agent does not synchronize restarts between themselves.
          When a single agent performs restart, it will trigger a local ``max_restarts``
          decrease, other agent will not decrease their ``max_restarts``.
          the user to run the distributed application locally on a dev host.

A single worker failure can cause the whole cluster to fail:
If a single worker is constantly failing, it will cause the TE agent
``max_restarts``  to go to zero. This will cause an agent to finish its
work and close rendezvous. If there are any other workers on different
agents, they will be terminated.


**Re-Rendezvous**

Re-rendezvous occurs when TE agents detect a new node
trying to joint a cluster. TE will not decrease ``max_restarts``. TE agents
will terminate its workers and start a new rendezvous round.

Note about DynamicRendezvous(etcd-v2, c10d-experimental): If the rendezvous
has already max_nodes, the new node won't be added to the wait list right
away since there is no need to tear down a rendezvous that is already fully
utilized. The new node will wait until its timeout (600 secs by default)
and periodically check the number of participants. If the number becomes
less than max_nodes, it will be added to the wait list; otherwise, it will time out after 600 secs.

*Scale up event*. When scale up event happens, torchelastic rendezvous
will detect that there are new nodes trying to join. Torchelastic agent
will stop all workers and perform re-rendezvous. Note: when scale up event
happens, *``max_restarts``* will *not* decrease.

*Scale down event*. When scale down event happens, rendezvous will not
notify the torchelastic agent about it. If TE agent launched with ``max_restarts=0`` ,
it relies on the underlying scheduler to handle job restart. If the ``max_restarts>0`` ,
TE agent will terminate workers and start a new rdzv round, which is a *Scale up event*.

"""
