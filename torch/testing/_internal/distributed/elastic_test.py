#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerSpec,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
from torch.distributed.elastic.rendezvous import RendezvousParameters


@dataclass
class Conf:
    """
    Holds arguments to launch an agent (e.g. simulates an agent run on a node).

    """

    entrypoint: Callable
    local_world_size: int
    args: Tuple = ()
    role: str = "default"
    redirects: Std = Std.NONE
    tee: Std = Std.NONE
    # c10d-experimental rdzv specific params
    is_host: bool = False


def get_etcd_rdzv_params(
    endpoint: str,
    run_id: str,
    min_nodes: int = 1,
    max_nodes: int = 1,
    timeout: int = 10,
) -> RendezvousParameters:
    return RendezvousParameters(
        backend="etcd",
        endpoint=endpoint,
        run_id=run_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )


def get_dynamic_rdzv_params(
    endpoint: str,
    run_id: str,
    is_host: bool = False,
    min_nodes: int = 1,
    max_nodes: int = 1,
    timeout: int = 10,
) -> RendezvousParameters:
    return RendezvousParameters(
        backend="c10d-experimental",
        endpoint=endpoint,
        run_id=run_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        is_host=is_host,
    )


def get_worker_spec(
    rdzv_params: RendezvousParameters,
    node_config: Conf,
    max_restarts=0,
    monitor_interval=0.01,
    master_addr_override: Optional[str] = None,
    master_port_override: Optional[int] = None,
):
    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_params)
    return WorkerSpec(
        role=node_config.role,
        local_world_size=node_config.local_world_size,
        entrypoint=node_config.entrypoint,
        args=node_config.args,
        rdzv_handler=rdzv_handler,
        max_restarts=max_restarts,
        monitor_interval=monitor_interval,
        redirects=node_config.redirects,
        tee=node_config.tee,
        master_addr=master_addr_override,
        master_port=master_port_override,
    )


def get_agent(
    spec: WorkerSpec,
    log_dir: str,
    start_method: str = "spawn",
    exit_barrier_timeout=5,
) -> LocalElasticAgent:
    return LocalElasticAgent(
        spec,
        start_method=start_method,
        exit_barrier_timeout=exit_barrier_timeout,
        log_dir=log_dir,
    )


# pyre-fixme[56]: Pyre was not able to infer the type of the decorator
#  `torch.distributed.elastic.multiprocessing.errors.record`.
@record
def run_agent(
    rdzv_params: RendezvousParameters,
    conf: Conf,
    log_dir: str,
    agent_results: Optional[mp.Queue] = None,  # (role, agent_result)
    start_method: str = "spawn",
    max_restarts: int = 0,
    exit_barrier_timeout=5,
    master_addr_override: Optional[str] = None,
    master_port_override: Optional[int] = None,
) -> Optional[RunResult]:

    rdzv_params.config["is_host"] = conf.is_host
    spec = get_worker_spec(
        rdzv_params=rdzv_params,
        node_config=conf,
        max_restarts=max_restarts,
        master_addr_override=master_addr_override,
        master_port_override=master_port_override,
    )
    agent = get_agent(
        spec=spec,
        log_dir=log_dir,
        start_method=start_method,
        exit_barrier_timeout=exit_barrier_timeout,
    )
    result = agent.run()
    if agent_results:
        agent_results.put((conf.role, result))

    if result.is_failed():
        raise ChildFailedError(spec.get_entrypoint_name(), result.failures)
    else:
        if not agent_results:
            return result


def run_job_async(
    rdzv_params: RendezvousParameters,
    log_dir: str,
    node_configs: List[Conf],
    exit_barrier_timeout: int = 5,
):
    """
    Simulates running a distributed job by running multiple agents
    (one on each process). Agent 0 is run on the main process for
    test coverage and ease of debugging
    """

    # each element in this queue holds a tuple (role, RunResult) for each agent
    agent_results = mp.Queue()

    # run first agent of first config on main process for test coverage + ease of debugging
    # it is important we loop in reverse order b/c running fn on the main process blocks
    procs = []
    for node_idx in range(len(node_configs)):
        conf = node_configs[node_idx]
        run_agent_args = {
            "rdzv_params": rdzv_params,
            "log_dir": log_dir,
            "conf": conf,
            "agent_results": agent_results,
            "start_method": "spawn",
            "max_restarts": 0,
            "exit_barrier_timeout": exit_barrier_timeout,
        }
        p = mp.Process(target=run_agent, kwargs=run_agent_args)
        procs.append(p)
        p.start()
    return procs, agent_results


def run_job(
    rdzv_params: RendezvousParameters,
    log_dir: str,
    node_configs: List[Conf],
    exit_barrier_timeout: int = 5,
) -> Dict[str, List[RunResult]]:
    procs, agent_results = run_job_async(
        rdzv_params, log_dir, node_configs, exit_barrier_timeout
    )
    for p in procs:
        p.join()

    results: Dict[str, List[RunResult]] = {}
    while not agent_results.empty():
        role, run_result = agent_results.get()
        results.setdefault(role, []).append(run_result)
    return results
