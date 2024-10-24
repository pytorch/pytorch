

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import os
import signal
import socket
import time
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import ProcessFailure, SignalException
from torch.distributed.elastic.rendezvous import RendezvousGracefulExitError
from torch.distributed.elastic.utils.logging import get_logger


__all__ = [
    "WorkerSpec",
    "Worker",
    "WorkerState",
    "WorkerGroup",
    "RunResult",
    "ElasticAgent",
    "SimpleElasticAgent",
]
_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"

DEFAULT_ROLE = "default"
logger = get_logger(__name__)


@dataclass
class WorkerSpec:
    """Blueprint information about a particular type of worker.

    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogeneous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.

    Args:
        role: user-defined role for the workers with this spec
        local_world_size: number local workers to run
        fn: (deprecated use entrypoint instead)
        entrypoint: worker function or command
        args: arguments to pass to ``entrypoint``
        rdzv_handler: handles rdzv for this set of workers
        max_restarts: number of max retries for the workers
        monitor_interval: monitor status of workers every ``n`` seconds
        master_port: fixed port to run the c10d store on rank 0
                     if not specified then will chose a random free port
        master_addr: fixed master_addr to run the c10d store on rank 0
                     if not specified then will chose hostname on agent rank 0
        redirects: redirect std streams to a file,
                   selectively redirect for a particular
                   local rank by passing a map
        tee: tees the specified std stream(s) to console + file,
             selectively tee for a particular local rank by passing a map,
             takes precedence over ``redirects`` settings.

    """

    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Optional[Callable] = None
    # TODO @kiuk - make entrypoint a required field
    entrypoint: Union[Callable, str, None] = None
    args: Tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 0.1
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    local_addr: Optional[str] = None

    def __post_init__(self):
        assert self.local_world_size > 0
        assert self.monitor_interval > 0

        if self.fn:
            warnings.warn(
                "WorkerSpec.fn will be deprecated,"
                " please use WorkerSpec.entrypoint instead",
                category=DeprecationWarning,
            )
            self.entrypoint = self.fn
        assert self.entrypoint

    def get_entrypoint_name(self):
        """Get the entry point name.

        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        else:
            assert self.entrypoint is not None
            return self.entrypoint.__qualname__


class Worker:
    """A worker instance.

    Contrast this with ``WorkerSpec`` that represents the specifications of a
    worker. A ``Worker`` is created from a ``WorkerSpec``. A ``Worker`` is to
    a ``WorkerSpec`` as an object is to a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Args:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """

    __slots__ = [
        "id",
        "local_rank",
        "global_rank",
        "role_rank",
        "world_size",
        "role_world_size",
    ]

    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        # unique identifier for this worker
        self.id: Any = None

        # rank of the worker among workers with the same role being monitored
        # by the same ``agent`` instance.
        self.local_rank: int = local_rank

        #  rank of the worker among all the workers across all roles
        #  across all ``agent`` instances.
        #  Global rank is not stable between re-rendezvous.
        self.global_rank: int = global_rank

        #  rank of the worker among all the workers with the same role
        #  across all ``agent`` instances.
        #  Role rank is not stable between re-rendezvous.
        self.role_rank: int = role_rank

        # total number of workers (globally). Due to elasticity
        # the world size may change between re-rendezvous.
        self.world_size: int = world_size

        # total number of workers that share the same role. Due to elasticity
        # the role world size may change between re-rendezvous.
        self.role_world_size: int = role_world_size

    def __str__(self):
        return (
            f"local_rank={self.local_rank},global_rank={self.global_rank}"
            f",role_rank={self.role_rank},world_size={self.world_size}"
            f",role_world_size={self.role_world_size}"
        )

    def __repr__(self):
        return str(self)


class WorkerState(str, Enum):
    """A state of the ``WorkerGroup``.

    Workers in a worker group change state as a unit. If a single worker
    in a worker group fails the entire set is considered failed::

      UNKNOWN - agent lost track of worker group state, unrecoverable
      INIT - worker group object created not yet started
      HEALTHY - workers running and healthy
      UNHEALTHY - workers running and unhealthy
      STOPPED - workers stopped (interrupted) by the agent
      SUCCEEDED - workers finished running (exit 0)
      FAILED - workers failed to successfully finish (exit !0)


    A worker group starts from an initial ``INIT`` state,
    then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
    and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.

    Worker groups can be interrupted and temporarily put into ``STOPPED`` state
    by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
    in the near future by the agent. Some examples of workers being put into
    ``STOPPED`` state are:

    1. Worker group failure|unhealthy observed
    2. Membership change detected

    When actions (start, stop, rdzv, retry, etc) on worker group fails
    and results in the action being partially applied to the worker group
    the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
    exceptions during state change events on the agent. The agent is not
    expected to recover worker groups in ``UNKNOWN`` state and is better off
    self terminating and allowing the job manager to retry the node.
    """

    UNKNOWN = "UNKNOWN"
    INIT = "INIT"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        """Return the state of the Worker.

        Returns:
             True if the worker state represents workers still running
             (e.g. that the process exists but not necessarily healthy).
        """
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}


class WorkerGroup:
    """A set of ``Worker`` instances.

    The class defines a set of ``Worker`` instances for the given ``WorkerSpec`` managed by ``ElasticAgent``. Whether the worker
    group contains cross instance workers or not depends on the implementation of the agent.
    """

    __slots__ = [
        "spec",
        "workers",
        "store",
        "group_rank",
        "group_world_size",
        "state",
        "master_addr",
        "master_port",
    ]

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None
        self.group_rank = None
        self.group_world_size = None
        self.master_addr = None
        self.master_port = None

        self.state = WorkerState.INIT


class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)


@dataclass
class RunResult:
    """Return results of the worker executions.

    Run results follow an "all-or-nothing" policy where the run is successful if and
    only if ALL local workers managed by this agent complete successfully.

    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.

    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.

    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.

    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker's final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent's restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    """

    state: WorkerState
    return_values: Dict[int, Any] = field(default_factory=dict)
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return self.state == WorkerState.FAILED


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class ElasticAgent(abc.ABC):
    """An agent process responsible for managing one or more worker processes.

    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        logger.exception("worker 0 failed with exit code : %s", failure.exit_code)
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the agent.

        Supports retrying the worker group on failures up to ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """Return the ``WorkerGroup`` for the given ``role``.

        Note that the worker group is a mutable object and hence in a
        multi-threaded/process environment it may change state.
        Implementors are encouraged (but not required) to return
        a defensive read-only copy.
        """
        raise NotImplementedError


class SimpleElasticAgent(ElasticAgent):
    """An ``ElasticAgent`` that manages one particular type of worker role.

    An ``ElasticAgent`` that manages workers (``WorkerGroup``) for a single ``WorkerSpec``
    such as one particular type of worker role.
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._total_execution_time = 0

    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        return self._worker_group

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        r"""Start ``worker_group.spec.local_world_size`` number of workers.

        This is according to worker spec for the worker group .
        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _stop_workers(
        self, worker_group: WorkerGroup, is_restart: bool = False
    ) -> None:
        r"""Stop all workers in the given worker group.

        Implementors must deal with workers in all states defined by
        ``WorkerState``. That is, it must gracefully handle stopping
        non-existent workers, unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        r"""Check on the workers for the ``worker_group``.

        This function also returns the new state of the worker group.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, is_restart: bool = False
    ) -> None:
        """Clean up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
        """
        raise NotImplementedError

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
        spec = worker_group.spec

        with self.record_duration("RENDEZVOUS"):
            rdzv_info = spec.rdzv_handler.next_rendezvous()
        store = rdzv_info.store
        group_rank = rdzv_info.rank
        group_world_size = rdzv_info.world_size

        # master_addr/master_port could be explicitly overriden
        # TODO: BC - specific to static rdzv and can be simplifed further
        master_addr = spec.master_addr or rdzv_info.bootstrap_store_info.master_addr
        master_port = spec.master_port or rdzv_info.bootstrap_store_info.master_port

        self._store = store

        with self.record_duration("ASSIGN_WORKER_RANKS"):
            workers = self._assign_worker_ranks(
                store, group_rank, group_world_size, spec
            )
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size
        worker_group.master_addr = master_addr
        worker_group.master_port = master_port

        restart_count = spec.max_restarts - self._remaining_restarts

        logger.info(
            "[%(role)s] Rendezvous complete for workers. Result:\n"
            "  restart_count=%(restart_count)s\n"
            "  master_addr=%(master_addr)s\n"
            "  master_port=%(master_port)s\n"
            "  group_rank=%(group_rank)s\n"
            "  group_world_size=%(group_world_size)s\n"
            "  local_ranks=%(local_ranks)s\n"
            "  role_ranks=%(role_ranks)s\n"
            "  global_ranks=%(global_ranks)s\n"
            "  role_world_sizes=%(role_world_sizes)s\n"
            "  global_world_sizes=%(global_world_sizes)s\n",
            {
                "role": spec.role,
                "restart_count": restart_count,
                "master_addr": master_addr,
                "master_port": master_port,
                "group_rank": group_rank,
                "group_world_size": group_world_size,
                "local_ranks": [worker.local_rank for worker in workers],
                "role_ranks": [worker.role_rank for worker in workers],
                "global_ranks": [worker.global_rank for worker in workers],
                "role_world_sizes": [worker.role_world_size for worker in workers],
                "global_world_sizes": [worker.world_size for worker in workers],
            },
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> List[Worker]:
        """Determine proper ranks for worker processes.

        Fast Path: when all workers have the same role and world size. We calculate
        the global rank to be group_rank * group_world_size + local_rank. And the
        `role_world_size` is the same as `global_world_size`. No TCP store is used in
        this case. This is only enabled when users set the environment variable
        `TORCH_ELASTIC_WORKER_IDENTICAL` to 1.

        Time complexity: each worker O(1), overall O(1)

        Slow Path: when workers have different roles and world sizes. We use the
        the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. The rank 0 agent reads all the role_info from the store and
           determines each agents worker ranks.
        3. Determine the global rank: the global rank of the workers is computed
           by cumulative sum of the local_world_size for all workers in front of it.
           For efficiency reasons each worker is assigned a base global rank
           such that it's workers are in the range [base_global_rank,
           base_global_rank + local_world_size).
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the ranks are calculated with
           respect to the role name.
        5. The rank 0 agent writes the assigned ranks to the store.
        6. Each agent reads the assigned ranks from the store.

        Time complexity: each worker O(1), rank0 O(n), overall O(n)
        """

        if os.environ.get("TORCH_ELASTIC_WORKER_IDENTICAL", "0") == "1":
            global_world_size = group_world_size * spec.local_world_size
            base_global_rank = group_rank * spec.local_world_size
            base_role_rank = base_global_rank
            role_world_size = global_world_size
        else:
            ROLE_INFO_PREFIX = "torchelastic/role_info/"
            ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

            agent_role_info = _RoleInstanceInfo(
                spec.role, group_rank, spec.local_world_size
            )
            store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())

            # tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
            if group_rank == 0:
                role_infos_bytes = store.multi_get(
                    [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
                )
                role_infos = [
                    _RoleInstanceInfo.deserialize(info_bytes)
                    for info_bytes in role_infos_bytes
                ]

                role_sizes = defaultdict(lambda: 0)
                global_size = 0
                for role_info in role_infos:
                    role_sizes[role_info.role] += role_info.local_world_size
                    global_size += role_info.local_world_size

                base_global_rank = 0
                role_ranks = defaultdict(lambda: 0)

                keys = []
                values = []
                for i, role_info in enumerate(role_infos):
                    keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
                    values.append(
                        json.dumps(
                            [
                                base_global_rank,
                                global_size,
                                role_ranks[role_info.role],
                                role_sizes[role_info.role],
                            ]
                        )
                    )

                    base_global_rank += role_info.local_world_size
                    role_ranks[role_info.role] += role_info.local_world_size

                store.multi_set(keys, values)

            # get will block until the data is available in the store.
            (
                base_global_rank,
                global_world_size,
                base_role_rank,
                role_world_size,
            ) = json.loads(store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}"))

        workers = []
        for local_rank in range(spec.local_world_size):
            worker = Worker(
                local_rank=local_rank,
                global_rank=base_global_rank + local_rank,
                role_rank=base_role_rank + local_rank,
                world_size=global_world_size,
                role_world_size=role_world_size,
            )
            workers.append(worker)
        return workers

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        logger.info("[%s] Rendezvous'ing worker group", role)

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group)

        logger.info("[%s] Starting worker group", role)
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id

        worker_group.state = WorkerState.HEALTHY

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Restart (stops, rendezvous, starts) all local workers in the group."""
        role = worker_group.spec.role
        logger.info("[%s] Stopping worker group", role)
        self._stop_workers(worker_group, is_restart=True)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except RendezvousGracefulExitError as e:
            logger.info("Rendezvous gracefully exited: %s", e)
        except SignalException as e:
            logger.warning("Received %s death signal, shutting down workers", e.sigval)
            self._shutdown(e.sigval)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown()
            # record the execution time in case there were any exceptions during run.
            self._total_execution_time = int(time.monotonic() - start_time)

    def get_event_failed(self) -> Event:
        return self._construct_event(
            state="FAILED",
            source=EventSource.AGENT,
            raw_error=traceback.format_exc(),
        )

    def get_event_succeeded(self) -> Event:
        return self._construct_event(
            state="SUCCEEDED",
            source=EventSource.AGENT,
        )

    def _record_worker_events(self, result: RunResult) -> None:
        for worker in self._worker_group.workers:
            failure = result.failures.get(worker.global_rank)
            state: str = self._get_worker_state(worker, result)
            raw_error = json.dumps(failure.error_file_data) if failure else None
            record(self._construct_event(state, EventSource.WORKER, worker, raw_error))

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        failure = result.failures.get(worker.global_rank)
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and not failure:
            # The worker got terminated by the torchelastic agent via SIGTERM signal
            return "TERMINATED"
        elif failure or worker.global_rank in result.return_values:
            return result.state.value
        else:
            raise ValueError(f"Unknown worker: {worker.global_rank}")

    @contextmanager
    def record_duration(self, state: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            record(
                self._construct_event(
                    state=state, source=EventSource.AGENT, duration_ms=duration_ms
                )
            )

    def _construct_event(
        self,
        state: str,
        source: EventSource,
        worker: Optional[Worker] = None,
        raw_error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Event:
        wg = self._worker_group
        spec = wg.spec
        md = {
            "group_world_size": wg.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
        }
        if worker:
            md["local_rank"] = (worker.local_rank,)
            md["role_rank"] = (worker.role_rank,)
            md["role_world_size"] = (worker.role_world_size,)
            global_rank = worker.global_rank
            worker_id = str(worker.id)
        else:
            global_rank = None
            worker_id = None
        md_str = json.dumps(md)
        metadata = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": global_rank,
            "group_rank": wg.group_rank,
            "worker_id": worker_id,
            "role": spec.role,
            "hostname": _get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": raw_error,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
            "duration_ms": duration_ms,
        }
        return Event(
            f"torchelastic.worker.status.{state}", source=source, metadata=metadata
        )

    def _record_metrics(self, group_results: RunResult):
        is_failed = group_results.is_failed()
        self._record_flakiness_metric(is_failed)
        spec = self._worker_group.spec
        restarts_happened = self._remaining_restarts != spec.max_restarts
        put_metric(f"workers.{spec.role}.run_total", 1)
        self._record_metric_with_condition(
            "run_success_with_retries", not is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not is_failed and not restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_with_retries", is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", is_failed and not restarts_happened
        )

    def _record_metric_with_condition(self, metric_name, condition):
        spec = self._worker_group.spec
        if condition:
            put_metric(f"workers.{spec.role}.{metric_name}", 1)
        else:
            put_metric(f"workers.{spec.role}.{metric_name}", 0)

    def _record_flakiness_metric(self, is_failed: bool = False):
        if is_failed:
            flakiness = 100.0
        else:
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        spec = self._worker_group.spec

        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            "[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name()
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                logger.info(
                    "[%s] worker group successfully finished."
                    " Waiting %s seconds for other agents to finish.",
                    role,
                    self._exit_barrier_timeout,
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    logger.info(
                        "[%s] Worker group %s. "
                        "%s/%s attempts left;"
                        " will restart worker group",
                        role,
                        state.name,
                        self._remaining_restarts,
                        spec.max_restarts,
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    logger.info(
                        "[%s] Detected %s "
                        "new nodes from group_rank=%s; "
                        "will restart worker group",
                        role,
                        num_nodes_waiting,
                        group_rank,
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(  # noqa: TRY002
                    f"[{role}] Worker group in {state.name} state"
                )

    def _exit_barrier(self):
        """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
        logger.info(
            "Local worker group finished (%s). "
            "Waiting %s seconds for other agents to finish",
            self._worker_group.state,
            self._exit_barrier_timeout,
        )
        start = time.time()
        try:
            store_util.barrier(
                store=self._store,
                world_size=self._worker_group.group_world_size,
                key_prefix=_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            logger.info(
                "Done waiting for other agents. Elapsed: %s seconds",
                time.time() - start,
            )
        except SignalException as e:
            logger.warning("Got termination signal: %s", e.sigval)
            raise
        except Exception:
            logger.exception(
                "Error waiting on exit barrier. Elapsed: %s seconds",
                time.time() - start,
            )
