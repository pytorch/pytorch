import argparse
import logging
import os

import torch
import torch.distributed as c10d


FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

log = logging.getLogger("log")
log.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(FORMAT)
handler.setFormatter(formatter)

log.addHandler(handler)
log.propagate = False  # Prevent log duplication

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple script to simulate NCCL errors. The script is "
        "supposed to be run on multiple different nodes simultaneously with "
        "appropriate rank and world_size. The script run an allreduce() on "
        "the rank 0 node and aborts all the other nodes to simulate an error "
        "in NCCL"
    )
    parser.add_argument("addr", help="address of the master node to connect to.")
    parser.add_argument("port", help="port of the master node to connect to.")
    parser.add_argument("rank", help="rank of this node")
    parser.add_argument("world_size", help="number of nodes in process group")
    args = parser.parse_args()
    rank = int(args.rank)
    world_size = int(args.world_size)
    port = int(args.port)

    store = c10d.TCPStore(args.addr, port, world_size, rank == 0)
    process_group = c10d.ProcessGroupNCCL(store, rank, world_size)
    log.info("Running first allreduce")
    process_group.allreduce(torch.rand(10).cuda(rank)).wait()
    if rank == 0:
        log.info("Running second allreduce only on rank 0")
        work = process_group.allreduce(torch.rand(10).cuda(rank))
        log.info("Waiting for allreduce to complete...")
        work.wait()
        log.info("Second allreduce successful: %s", work.is_success())
    else:
        log.info("Aborting all other ranks.")
        os.abort()
