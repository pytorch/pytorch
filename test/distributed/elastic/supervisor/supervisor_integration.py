import itertools
import logging

import os
import signal
import sys
import time
from typing import Any, Dict, List, Set

import torch.distributed.elastic.supervisor.launchers as launchers

from torch.distributed.elastic.supervisor import (
    as_completed,
    Context,
    Future,
    get_message_queue,
    Host,
    Process,
    wait,
)
from torch.distributed.elastic.supervisor.launchers import SupervisorConfig

logger = logging.getLogger(__name__)


def start_training(
    ctx, N: int, hosts: Set[Host], npp: int, run_fraction, rank_fraction, attempt
):
    # we will use `run_fraction` of machines as training machines.
    # the rest will be used as failover machines.
    desired_run_size: int = int(run_fraction * N)

    # The supervisor can now create processes on hosts.
    # We will start by running a health check on all of our machines
    # to find the top `rank_fraction` of machines and exclude the remaining stragglers.

    logger.info("starting health checks host %s hosts", len(hosts))
    env = {"ATTEMPT": str(attempt)}
    pg: List[Process] = ctx.create_process_group(
        hosts,
        args=[sys.executable, __file__, "--health", sys.argv[2]],
        env=env,
        processes_per_host=1,
        name="health_check",
    )

    health_responses: Dict[Future[Any], Process] = {p.recv(): p for p in pg}

    # as_completed returns messages as we receive them, avoiding waiting for stragglers.
    # if we do not hear from enough machines in 5 minutes, we assume
    # something about the cluster is unhealthy and then bail out entirely
    TIMEOUT = 60 * 5
    working_machines = as_completed(health_responses.keys(), timeout=TIMEOUT)

    # But we don't have to hear from all of our machines before starting,
    # some might have major health check issues that will cause them to hang
    # for awhile, let's only sort the first 99% of machines by using zip
    # to stop iteration after to_sort machines have been received
    to_sort = int(rank_fraction * N)
    working_machines = zip(range(to_sort), working_machines)

    to_rank = [(f.result(), health_responses.pop(f)) for _, f in working_machines]

    logger.info("Found %s hosts that passed health checks, ranking...", len(to_rank))
    to_rank = sorted(to_rank, key=lambda x: x[0])

    if len(to_rank) < desired_run_size:
        raise RuntimeError("Not enough healthy hosts")

    good_hosts = [p.host for score, p in to_rank[:desired_run_size]]
    logger.info("Chose hosts: %s", good_hosts)

    # Let's get training started.
    logger.info("Launching %s processes", npp * desired_run_size)

    return ctx.create_process_group(
        good_hosts,
        args=[sys.executable, __file__, "--train", sys.argv[2]],
        env=env,
        processes_per_host=npp,
        name="train",
    )


def wait_for_exit(process_group):
    for f in as_completed([f.returncode() for f in process_group]):
        if f.exception() is not None or f.result() != 0:
            for p in process_group:
                p.signal(
                    signal.SIGTERM
                )  # TODO: maybe have a broadcasting signal function.
            return False
    return True


def supervise(ctx: Context, conf: SupervisorConfig):
    N = config["N"]
    hosts: Set[Host] = set(ctx.request_hosts(n=config["N"]))
    # wait for hosts to be ready
    wait([h.hostname() for h in hosts], timeout=30)
    npp = 2
    for i in itertools.count():
        if i >= len(config["train"]):
            raise RuntimeError("Too many attempts")
        logger.info(
            "Starting training attemp %s with %s hosts, %s processes per host.",
            i,
            N,
            npp,
        )
        process_group = start_training(
            ctx, N, hosts, npp, config["run_fraction"], config["rank_fraction"], i
        )
        if wait_for_exit(process_group):
            break
        logger.info("Training has failed, attempting restart...")
        disconnected = [h for h in hosts if h.connection_lost()]
        hosts.difference_update(disconnected)
        hosts.update(ctx.replace_hosts(disconnected))
    logger.info("Training exited successfully.")


def train():
    rank = int(os.environ["RANK"])
    attempt = int(os.environ["ATTEMPT"])
    action = config["train"][attempt][rank]
    if action == "F":
        raise RuntimeError("Failed!")
    elif action == ".":
        pass
    elif action == "E":
        os.kill(os.getppid(), signal.SIGKILL)


def health():
    rank = int(os.environ["RANK"])
    attempt = int(os.environ["ATTEMPT"])
    q = get_message_queue()
    health = config["health"][attempt][rank]
    if health == "hang":
        while True:
            time.sleep(1)
    q.send_pyobj(health)


if __name__ == "__main__":
    config = eval(sys.argv[2])
    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--health":
        health()
    else:
        assert sys.argv[1] == "--supervise"
        # defined as entrypoint under test_launcher.dist-info
        # we launch the process with PYTHONPATH pointing to the parent dir
        launcher = launchers.launcher_registry["test_launcher"]
        launcher(supervise, None)
