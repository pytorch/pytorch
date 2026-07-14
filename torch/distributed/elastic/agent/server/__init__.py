#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The elastic agent is the control plane of torchelastic.

It is a process that launches and manages underlying worker processes.
The agent is responsible for:

1. Working with distributed torch: the workers are started with all the
   necessary information to successfully and trivially call
   ``torch.distributed.init_process_group()``.

2. Fault tolerance: monitors workers and upon detecting worker failures
   or unhealthiness, tears down all workers and restarts everyone.

3. Elasticity: Reacts to membership changes and restarts workers with the new
   members.

The simplest agents are deployed per node and works with local processes.
A more advanced agent can launch and manage workers remotely. Agents can
be completely decentralized, making decisions based on the workers it manages.
Or can be coordinated, communicating to other agents (that manage workers
in the same job) to make a collective decision.
"""

from .api import (  # noqa: F401
    ElasticAgent,
    RunResult,
    SimpleElasticAgent,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from .local_elastic_agent import TORCHELASTIC_ENABLE_FILE_TIMER, TORCHELASTIC_TIMER_FILE
