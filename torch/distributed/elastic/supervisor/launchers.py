# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata as metadata
import io
import os
import signal
import socket
import subprocess
import sys
import traceback

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch.distributed.elastic.supervisor.hostmanager as hostmanager

from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.elastic.utils.logging import get_logger

from . import as_completed, Context, Future, Host, Process, wait_on


PORT = 55555
REGISTRY_LAUNCHERS_METADATA_GROUP_KEY = "torchrun.supervisor.launchers"
REGISTRY_POLICES_METADATA_GROUP_KEY = "torchrun.supervisor.policies"

logger = get_logger(__name__)


from dataclasses import dataclass


@dataclass
class ProcessConfig:
    """Process configuration

    Args:
        run_path: path to the script to run
        no_python: if True, run the script directly without python
        module: if True, run the script as a python module
    """

    run_path: Optional[str]
    no_python: Optional[bool]
    module: Optional[bool]


@dataclass
class TimeoutConfig:
    """Timeout configuration

    Args:
        join_timeout: timeout for hostmanagers to join
        close_timeout: timeout for graceful process exit
    """

    join_timeout: float = 300
    close_timeout: float = 60


@dataclass
class JobConfig:
    """Job configuration

    Args:
        min_nodes: minimum number of nodes to allocate
        max_nodes: maximum number of nodes to allocate
        nproc_per_node: number of processes per node
        max_restarts: maximum number of restarts
        timeouts: timeout configuration
        training_script: path to the training script
        training_script_args: arguments to the training script

    """

    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    max_restarts: int
    timeouts: TimeoutConfig
    training_script: str
    training_script_args: List[str]


@dataclass
class SupervisorConfig:
    """Supervisor configuration

    Args:
        proc: Process configuration/behavior for process launching on the hosts
        job: Job configuration
        run_id: run id
        port: port that supervisor listens on
        host: host that supervisor runs on
        root: if True, run as supervision root (not a hostmanager)

    """

    proc: ProcessConfig
    job: JobConfig

    run_id: Optional[str] = None
    port: Optional[int] = None
    host: Optional[str] = None
    root: Optional[bool] = None


Policy = Callable[[Context, SupervisorConfig], None]
Launcher = Callable[[Policy, SupervisorConfig], None]


class ProcessFailure(Exception):
    def __init__(self, blamed: Process):
        super().__init__()
        self.blamed = blamed


def _allocate_hosts(ctx: Context, conf: SupervisorConfig) -> List[Host]:
    hosts = ctx.request_hosts(conf.job.max_nodes)
    hosts_with_hostname: Sequence[Tuple[Host, Future[str]]] = wait_on(
        hosts, lambda host: host.hostname(), conf.job.timeouts.join_timeout
    )
    connected_hosts = [host for host, _ in hosts_with_hostname]
    return connected_hosts


# TODO: extract Context interface as a public API
def _torchrun_policy(ctx: Context, conf: SupervisorConfig) -> None:
    """Policy that supervises the training process that intents to match
    torchrun/torchelastic behavior.

    Args:
        ctx: Supervisor context
        conf: configuration
    """
    logger.info("Running torchrun supervisor policy, conf: %s", conf)

    assert 0 < conf.job.min_nodes <= conf.job.max_nodes
    assert conf.job.max_restarts >= 0
    if conf.proc.run_path:
        raise NotImplementedError(
            "supervisor based torchun does not support run_path option"
        )

    connected_hosts = _allocate_hosts(ctx, conf)
    if len(connected_hosts) < conf.job.min_nodes:
        raise TimeoutError(
            "Only %s out of required %s hosts were allocated",
            len(connected_hosts),
            conf.job.min_nodes,
        )

    failure: Optional[ProcessFailure] = None
    for attempt in range(conf.job.max_restarts + 1):
        logger.info(
            "Starting attempt idx %s on hosts: %s",
            attempt,
            ",".join([h.hostname().result() for h in connected_hosts]),
        )

        scripts_path = str(Path(__file__).parent.absolute())

        proc = [
            sys.executable,
            f"{scripts_path}/scripts/supervisor_prefix.py",
            "-p",
            "[trainer{LOCAL_RANK}]: ",
        ]
        proc = []
        if not conf.proc.no_python:
            proc = [os.getenv("PYTHON_EXEC", sys.executable), "-u"]
            if conf.proc.module:
                proc.append("-m")

        proc += [
            conf.job.training_script,
            *conf.job.training_script_args,
        ]

        pg: Sequence[Process] = ctx.create_process_group(
            list(connected_hosts),
            args=proc,
            processes_per_host=int(conf.job.nproc_per_node),
            env={
                **os.environ,
                "TORCHELASTIC_RESTART_COUNT": str(attempt),
                "TORCHELASTIC_MAX_RESTARTS": str(conf.job.max_restarts),
                "MASTER_ADDR": socket.getfqdn(),
                "MASTER_PORT": str(get_free_port()),
                "TORCHELASTIC_RUN_ID": f"{conf.run_id}",
                "TORCHELASTIC_ERROR_FILE": "/var/logs/templog",  # TODO
            },
            name=f"train/{str(attempt).zfill(3)}/{{rank}}",
        )

        failure = None
        proc_table: Dict[Future, Process] = {p.returncode(): p for p in pg}
        for returncode_fut in as_completed(proc_table.keys()):  # type: ignore[arg-type]
            try:
                r = returncode_fut.result()
                if r != 0:
                    failure = ProcessFailure(proc_table[returncode_fut])
                    break
            except Exception as e:
                failure = ProcessFailure(proc_table[returncode_fut])
                break

        if failure:
            for p in pg:
                if not p.returncode().done():
                    p.signal()

            # wait for pg to clean-up within a grace period
            logger.warning("Stopping other processes due to failure %s", attempt)
            as_completed(
                [p.returncode() for p in pg], timeout=conf.job.timeouts.close_timeout  # type: ignore[misc]
            )
        else:
            logger.info("all the local ranks are done")
            break

    if failure:
        raise failure
    else:
        logger.info("torchrun policy completed")


def _default_launcher(
    supervise: Callable[[Context, SupervisorConfig], None], conf: SupervisorConfig
) -> None:
    """Default launcher inteds to match torchrun's behavior"""
    logger.info("Running as supervisor based implementation with config %s", conf)

    host = conf.host or socket.getfqdn()
    port = conf.port or PORT
    addr = f"tcp://{host}:{port}"

    host_process: Optional[subprocess.Popen] = None
    if conf.root:
        logger.info("root node, starting supervisor with addr %s", addr)
        try:
            host_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.elastic.supervisor.hostmanager",
                    f"{addr}",
                ]
            )

            ctx = Context(port=port)
            logger.info(ctx)
            supervise(ctx, conf)
            ctx.shutdown()  # TODO: clean up on failures
            logger.info("Supervisor shutdown complete.")
        # TODO catch ProcessFailure and improve attribution
        except Exception:
            ty, e, st = sys.exc_info()
            s = io.StringIO()
            traceback.print_tb(st, file=s)
            msg = f"{ty.__name__}: {str(e)}\n{s.getvalue()}"  # type: ignore[union-attr]
            logger.error(msg)
            if host_process:
                host_process.send_signal(signal.SIGINT)
            raise
        return_code = host_process.wait(timeout=30)
        if return_code != 0:
            logger.warning(
                f"Host manager on supervisor returned non-zero code: {return_code}."  # noqa: G004
            )
            sys.exit(return_code)
    else:
        logger.info("non-root node, starting just a hostmanager")
        hostmanager.main(addr)

    logger.info("Running supervor using torchrun policy")


def _test_launcher(supervise: Policy, conf: SupervisorConfig) -> None:
    # basic launcher used for testing purposes, requires:
    # - `HOSTNAMES`` env variable, where the first one is the supervisor root node address
    # - `TORCH_ELASTIC_SUPERVISOR` env variable to indicate if this node is root node
    host = os.environ["HOSTNAMES"].split(",")[0]
    addr = f"tcp://{host}:{PORT}"
    if os.environ["TORCH_ELASTIC_SUPERVISOR"].lower() == "true":
        popen = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.elastic.supervisor.hostmanager",
                f"{addr}",
            ]
        )
        ctx = Context(port=PORT)
        try:
            supervise(ctx, conf)
        finally:
            ctx.shutdown()
            if popen.poll() is None:
                popen.kill()
    else:
        hostmanager.main(addr)


launcher_registry: Dict[str, Launcher] = {"default": _default_launcher}
policy_registry: Dict[str, Policy] = {"default": _torchrun_policy}


def _update_registry(
    group: str, registry: Union[Dict[str, Launcher], Dict[str, Policy]]
) -> None:
    all_entrypoints = metadata.entry_points()
    # >= python 3.8 and < 3.10
    if type(all_entrypoints) == dict:
        entrypoints = all_entrypoints.get(group, [])  # type: ignore[var-annotated]
    else:
        entrypoints = metadata.entry_points().select(group=group)  # type: ignore[var-annotated]
    for ep in entrypoints:
        logger.debug("Adding %s to %s registry", ep.name, group)
        registry[ep.name] = ep.load()


_update_registry(REGISTRY_LAUNCHERS_METADATA_GROUP_KEY, launcher_registry)
_update_registry(REGISTRY_POLICES_METADATA_GROUP_KEY, policy_registry)


__all__ = [
    "REGISTRY_LAUNCHERS_METADATA_GROUP_KEY",
    "REGISTRY_POLICES_METADATA_GROUP_KEY",
    "launcher_registry",
    "policy_registry",
    "SupervisorConfig",
    "ProcessConfig",
    "TimeoutConfig",
    "JobConfig",
    "ProcessFailure",
]
