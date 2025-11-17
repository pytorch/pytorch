#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module ``torch.distributed.run``.

``torch.distributed.run`` is a module that spawns up multiple distributed
training processes on each of the training nodes.

``torchrun`` is a python
`console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
to the main module
`torch.distributed.run <https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py>`_
declared in the ``entry_points`` configuration in
`setup.py <https://github.com/pytorch/pytorch/blob/master/setup.py>`_.
It is equivalent to invoking ``python -m torch.distributed.run``.

``torchrun`` can be used for single-node distributed training, in which one or
more processes per node will be spawned. It can be used for either
CPU training or GPU training. If it is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. ``torchrun`` can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, ``torchrun`` will launch the given number of processes per node
(``--nproc-per-node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

.. versionchanged:: 2.0.0

    ``torchrun`` will pass the ``--local-rank=<rank>`` argument to your script.
    From PyTorch 2.0.0 onwards, the dashed ``--local-rank`` is preferred over the
    previously used underscored ``--local_rank``.

    For backward compatibility, it may be necessary for users to handle both
    cases in their argument parsing code. This means including both ``"--local-rank"``
    and ``"--local_rank"`` in the argument parser. If only ``"--local_rank"`` is
    provided, ``torchrun`` will trigger an error: "error: unrecognized arguments:
    --local-rank=<rank>". For training code that only supports PyTorch 2.0.0+,
    including ``"--local-rank"`` should be sufficient.

    ::

        >>> # xdoctest: +SKIP
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--local-rank", "--local_rank", type=int)
        >>> args = parser.parse_args()

Usage
-----

Single-node multi-worker
++++++++++++++++++++++++

::

    torchrun
        --standalone
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

.. note:: ``--nproc-per-node`` may be
          ``"gpu"`` (spawn one process per GPU),
          ``"cpu"`` (spawn one process per CPU),
          ``"xpu"`` (spawn one process per XPU),
          ``"auto"`` (equivalent to ``"gpu"`` if CUDA is available,
          else equivalent to ``"xpu"`` if XPU is available,
          else equivalent to ``"cpu"``),
          or an integer specifying the number of processes.
          See `torch.distributed.run.determine_local_world_size
          <https://github.com/pytorch/pytorch/blob/0a94bb432ed75cc2d950d81b2921363218a7e459/torch/distributed/run.py#L673-L716>`_
          for more details.

Stacked single-node multi-worker
++++++++++++++++++++++++++++++++

To run multiple instances (separate jobs) of single-node, multi-worker on the
same host, we need to make sure that each instance (job) is
setup on different ports to avoid port conflicts (or worse, two jobs being merged
as a single job). To do this you have to run with ``--rdzv-backend=c10d``
and specify a different port by setting ``--rdzv-endpoint=localhost:$PORT_k``.
For ``--nodes=1``, its often convenient to let ``torchrun`` pick a free random
port automatically instead of manually assigning different ports for each run.

::

    torchrun
        --rdzv-backend=c10d
        --rdzv-endpoint=localhost:0
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=$NUM_NODES
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership changes or failures)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=1:4
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Note on rendezvous backend
--------------------------

For multi-node training you need to specify:

1. ``--rdzv-id``: A unique job id (shared by all nodes participating in the job)
2. ``--rdzv-backend``: An implementation of
   :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler`
3. ``--rdzv-endpoint``: The endpoint where the rendezvous backend is running; usually in form
   ``host:port``.

Currently ``c10d`` (recommended), ``etcd-v2``, and ``etcd`` (legacy)  rendezvous backends are
supported out of the box. To use ``etcd-v2`` or ``etcd``, setup an etcd server with the ``v2`` api
enabled (e.g. ``--enable-v2``).

.. warning::
   ``etcd-v2`` and ``etcd`` rendezvous use etcd API v2. You MUST enable the v2 API on the etcd
   server. Our tests use etcd v3.4.3.

.. warning::
   For etcd-based rendezvous we recommend using ``etcd-v2`` over ``etcd`` which is functionally
   equivalent, but uses a revised implementation. ``etcd`` is in maintenance mode and will be
   removed in a future version.

Definitions
-----------

1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.

2. ``Worker`` - A worker in the context of distributed training.

3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).

4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.

5. ``RANK`` - The rank of the worker within a worker group.

6. ``WORLD_SIZE`` - The total number of workers in a worker group.

7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.

8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.

9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.

9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.

10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.

Environment Variables
---------------------

The following environment variables are made available to you in your script:

1. ``LOCAL_RANK`` -  The local rank.

2. ``RANK`` -  The global rank.

3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
   running a single worker group per node, this is the rank of the node.

4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
   of the worker is specified in the ``WorkerSpec``.

5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
   ``--nproc-per-node`` specified on ``torchrun``.

6. ``WORLD_SIZE`` - The world size (total number of workers in the job).

7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
   in ``WorkerSpec``.

8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
   the Torch Distributed backend.

9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.

10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.

11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.

12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).

13. ``PYTHON_EXEC`` - System executable override. If provided, the python user script will
    use the value of ``PYTHON_EXEC`` as executable. The `sys.executable` is used by default.

Deployment
----------

1. (Not needed for the C10d backend) Start the rendezvous backend server and get the endpoint (to be
   passed as ``--rdzv-endpoint`` to ``torchrun``)

2. Single-node multi-worker: Start ``torchrun`` on the host to start the agent process which
   creates and monitors a local worker group.

3. Multi-node multi-worker: Start ``torchrun`` with the same arguments on all the nodes
   participating in training.

When using a job/cluster manager, the entry point command to the multi-node job should be ``torchrun``.

Failure Modes
-------------

1. Worker failure: For a training job with ``n`` workers, if ``k<=n`` workers fail all workers
   are stopped and restarted up to ``max_restarts``.

2. Agent failure: An agent failure results in a local worker group failure. It is up to the job
   manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors
   are supported by the agent.

3. Node failure: Same as agent failure.

Membership Changes
------------------

1. Node departure (scale-down): The agent is notified of the departure, all existing workers are
   stopped, a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped,
   a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

Important Notices
-----------------

1. This utility and multi-process distributed (single-node or
   multi-node) GPU training currently only achieves the best performance using
   the NCCL distributed backend. Thus NCCL backend is the recommended backend to
   use for GPU training.

2. The environment variables necessary to initialize a Torch process group are provided to you by
   this module, no need for you to pass ``RANK`` manually.  To initialize a process group in your
   training script, simply run:

::

    >>> # xdoctest: +SKIP("stub")
    >>> import torch.distributed as dist
    >>> dist.init_process_group(backend="gloo|nccl")

3. In your training program, you can either use regular distributed functions
   or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
   training program uses GPUs for training and you would like to use
   :func:`torch.nn.parallel.DistributedDataParallel` module,
   here is how to configure it.

::

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[int(os.environ("LOCAL_RANK"))]``,
and ``output_device`` needs to be ``int(os.environ("LOCAL_RANK"))`` in order to use this
utility


4. On failures or membership changes ALL surviving workers are killed immediately. Make sure to
   checkpoint your progress. The frequency of checkpoints should depend on your job's tolerance
   for lost work.

5. This module only supports homogeneous ``LOCAL_WORLD_SIZE``. That is, it is assumed that all
   nodes run the same number of local workers (per role).

6. ``RANK`` is NOT stable. Between restarts, the local workers on a node can be assigned a
   different range of ranks than before. NEVER hard code any assumptions about the stable-ness of
   ranks or some correlation between ``RANK`` and ``LOCAL_RANK``.

7. When using elasticity (``min_size!=max_size``) DO NOT hard code assumptions about
   ``WORLD_SIZE`` as the world size can change as nodes are allowed to leave and join.

8. It is recommended for your script to have the following structure:

::

    def main():
        load_checkpoint(checkpoint_path)
        initialize()
        train()


    def train():
        for batch in iter(dataset):
            train_step(batch)

            if should_checkpoint:
                save_checkpoint(checkpoint_path)

9. (Recommended) On worker errors, this tool will summarize the details of the error
   (e.g. time, rank, host, pid, traceback, etc). On each node, the first error (by timestamp)
   is heuristically reported as the "Root Cause" error. To get tracebacks as part of this
   error summary print out, you must decorate your main entrypoint function in your
   training script as shown in the example below. If not decorated, then the summary
   will not include the traceback of the exception and will only contain the exitcode.
   For details on torchelastic error handling see: https://pytorch.org/docs/stable/elastic/errors.html

::

    from torch.distributed.elastic.multiprocessing.errors import record


    @record
    def main():
        # do train
        pass


    if __name__ == "__main__":
        main()
"""  # noqa: E501

import os
import sys
import uuid
from argparse import ArgumentParser, REMAINDER
from collections.abc import Callable
from importlib import metadata
from typing import Optional, Union

import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, LogsSpecs, Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.numa.binding import (
    AffinityMode as _AffinityMode,  # Signify as private with _
    NumaOptions as _NumaOptions,
)
from torch.utils.backend_registration import _get_custom_mod_func


logger = get_logger(__name__)


def get_args_parser() -> ArgumentParser:
    """Parse the command line options."""
    parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")

    #
    # Worker/node size related arguments.
    #

    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, xpu, int].",
    )

    #
    # Rendezvous related arguments
    #

    parser.add_argument(
        "--rdzv-backend",
        "--rdzv_backend",
        action=env,
        type=str,
        default="static",
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv-id",
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv-conf",
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on a free port. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max-restarts",
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        action=env,
        type=float,
        default=0.1,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start-method",
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--event-log-handler",
        "--event_log_handler",
        action=env,
        type=str,
        default="null",
        help="name of a registered event logging handler (see: https://docs.pytorch.org/docs/stable/elastic/events.html)",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run-path",
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no-python.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is reused for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )

    parser.add_argument(
        "--local-ranks-filter",
        "--local_ranks_filter",
        action=env,
        type=str,
        default="",
        help="Only show logs from specified ranks in console (e.g. [--local_ranks_filter=0,1,2] will "
        "only show logs from rank 0, 1 and 2). This will only apply to stdout and stderr, not to"
        "log files saved via --redirect or --tee",
    )

    #
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0) that only used for static rendezvous. It should "
        "be either the IP address or the hostname of rank 0. For single node multi-proc training "
        "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
        "`[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous.",
    )
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )

    parser.add_argument(
        "--logs-specs",
        "--logs_specs",
        default=None,
        type=str,
        help="torchrun.logs_specs group entrypoint name, value must be type of LogsSpecs. "
        "Can be used to override custom logging behavior.",
    )

    parser.add_argument(
        "--numa-binding",
        "--numa_binding",
        type=str,
        choices=[mode.value for mode in _AffinityMode],
        default=None,
        help="""
        If provided, we will affinitize the worker processes based on NUMA nodes
        for better performance. (E.g., preferring to allocate memory locally and run on CPUs on the
        same NUMA node.)

        NOTE: This is currently only supported for GPUs, and we assume
        that the LOCAL_RANK process corresponds to the GPU with index LOCAL_RANK. If this is not
        accurate for your workload, this feature may be a pessimization.

        Available options are:
          - node: Processes are bound to cpu cores within a NUMA node. This is a good starting point,
          but other options may perform even slightly better in some cases.
          - socket: Processes are bound to cpu cores within a socket.
          - exclusive: Processes are bound to exclusive sets of cpu cores within a NUMA node.
          - core-complex: Processes are bound to cpu cores in a core-complex.
          NOTE: The core-complex option might not achieve optimal performance on architectures
          featuring a single L3 cache per socket.""",
    )

    parser.add_argument(
        "--signals-to-handle",
        "--signals_to_handle",
        action=env,
        type=str,
        default="SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        help="Comma-separated list of signals to handle and forward to subprocesses. "
        "Default: SIGTERM,SIGINT,SIGHUP,SIGQUIT. "
        "Common additional signals: SIGUSR1,SIGUSR2 (used in SLURM environments).",
    )

    #
    # Positional arguments.
    #

    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )

    # Rest from the training program.
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser


def parse_args(args):
    parser = get_args_parser()
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')  # noqa: E231

    return min_nodes, max_nodes


def determine_local_world_size(nproc_per_node: str):
    try:
        logger.info("Using nproc_per_node=%s.", nproc_per_node)
        return int(nproc_per_node)
    except ValueError as e:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.") from e
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "xpu":
            if not torch.xpu.is_available():
                raise ValueError("Xpu is not available.") from e
            device_type = "xpu"
            num_proc = torch.xpu.device_count()
        elif nproc_per_node == torch._C._get_privateuse1_backend_name():
            if not _get_custom_mod_func("is_available")():
                raise ValueError(f"{nproc_per_node} is not available.") from e
            device_type = nproc_per_node
            num_proc = _get_custom_mod_func("device_count")()
        elif nproc_per_node == "auto":
            if torch.accelerator.is_available():
                num_proc = torch.accelerator.device_count()
                device_type = torch.accelerator.current_accelerator().type  # type: ignore[union-attr]
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(
                f"Unsupported nproc_per_node value: {nproc_per_node}"
            ) from e

        logger.info(
            "Using nproc_per_node=%s, setting nproc_per_node to %s since the instance has %s %s",
            nproc_per_node,
            num_proc,
            num_proc,
            device_type,
        )
        return num_proc


def get_rdzv_endpoint(args):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        return f"{args.master_addr}:{args.master_port}"  # noqa: E231
    return args.rdzv_endpoint


def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.

    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    if not hasattr(args, "use_env"):
        return True
    return args.use_env


def _get_logs_specs_class(logs_specs_name: Optional[str]) -> type[LogsSpecs]:
    """
    Attempts to load `torchrun.logs_spec` entrypoint with key of `logs_specs_name` param.
    Provides plugin mechanism to provide custom implementation of LogsSpecs.

    Returns `DefaultLogsSpecs` when logs_spec_name is None.
    Raises ValueError when entrypoint for `logs_spec_name` can't be found in entrypoints.
    """
    logs_specs_cls = None
    if logs_specs_name is not None:
        eps = metadata.entry_points()
        group = eps.select(group="torchrun.logs_specs")
        if group.select(name=logs_specs_name):
            logs_specs_cls = group[logs_specs_name].load()

        if logs_specs_cls is None:
            raise ValueError(
                f"Could not find entrypoint under 'torchrun.logs_specs[{logs_specs_name}]' key"
            )

        logger.info(
            "Using logs_spec '%s' mapped to %s", logs_specs_name, str(logs_specs_cls)
        )
    else:
        logs_specs_cls = DefaultLogsSpecs

    return logs_specs_cls


def config_from_args(args) -> tuple[LaunchConfig, Union[Callable, str], list[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0

    if (
        hasattr(args, "master_addr")
        and args.rdzv_backend != "static"
        and not args.rdzv_endpoint
    ):
        logger.warning(
            "master_addr is only used for static rdzv_backend and when rdzv_endpoint "
            "is not specified."
        )

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        logger.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    log_line_prefix_template = os.getenv("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE")

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args)

    ranks: Optional[set[int]] = None
    if args.local_ranks_filter:
        try:
            ranks = set(map(int, args.local_ranks_filter.split(",")))
            assert ranks
        except Exception as e:
            raise ValueError(
                "--local_ranks_filter must be a comma-separated list of integers e.g. --local_ranks_filter=0,1,2"
            ) from e

    logs_specs_cls: type[LogsSpecs] = _get_logs_specs_class(args.logs_specs)
    logs_specs = logs_specs_cls(
        log_dir=args.log_dir,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        local_ranks_filter=ranks,
    )
    numa_options = (
        None
        if args.numa_binding is None
        else _NumaOptions(affinity_mode=_AffinityMode(args.numa_binding))
    )

    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        log_line_prefix_template=log_line_prefix_template,
        local_addr=args.local_addr,
        logs_specs=logs_specs,
        event_log_handler=args.event_log_handler,
        numa_options=numa_options,
        signals_to_handle=args.signals_to_handle,
    )

    with_python = not args.no_python
    cmd: Union[Callable, str]
    cmd_args = []
    use_env = get_use_env(args)
    if args.run_path:
        cmd = run_script_path
        cmd_args.append(args.training_script)
    else:
        if with_python:
            cmd = os.getenv("PYTHON_EXEC", sys.executable)
            cmd_args.append("-u")
            if args.module:
                cmd_args.append("-m")
            cmd_args.append(args.training_script)
        else:
            if args.module:
                raise ValueError(
                    "Don't use both the '--no-python' flag"
                    " and the '--module' flag at the same time."
                )
            cmd = args.training_script
    if not use_env:
        cmd_args.append(f"--local-rank={macros.local_rank}")
    cmd_args.extend(args.training_script_args)

    return config, cmd, cmd_args


def run_script_path(training_script: str, *training_script_args: str):
    """
    Run the provided `training_script` from within this interpreter.

    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    """
    import runpy
    import sys

    sys.argv = [training_script] + [*training_script_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


def run(args):
    torch.multiprocessing._set_thread_name("pt_elastic")

    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:0"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            "\n**************************************\n"
            "Rendezvous info:\n"
            "--rdzv-backend=%s "
            "--rdzv-endpoint=%s "
            "--rdzv-id=%s\n"
            "**************************************\n",
            args.rdzv_backend,
            args.rdzv_endpoint,
            args.rdzv_id,
        )

    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


@record
def main(args=None):
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
