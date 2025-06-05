"""
Example to run benchmarks for the pg transport class

torchrun --nproc_per_node=2 -m torch.distributed.checkpoint._pg_transport_example -- --device cuda
"""

import logging
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.checkpoint._pg_transport import PGTransport


class Timer:
    def __init__(self, description=""):
        self.description = description

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(
            f"[Rank {dist.get_rank()}] {self.description} - {self.elapsed_time:.4f} seconds"
        )


logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=3_000_000)  # 3MB
    parser.add_argument("--total-size", type=int, default=12_000_000_000)  # 12GB
    args = parser.parse_args(argv)

    CHUNK_SIZE: int = args.chunk_size
    TOTAL_SIZE: int = args.total_size
    INPLACE: bool = args.inplace
    DEVICE: str = args.device

    # Initialize process group with torchrun
    dist.init_process_group(backend="nccl" if DEVICE == "cuda" else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    timeout: timedelta = timedelta(seconds=10)

    if DEVICE == "cuda":
        torch.cuda.set_device(rank)

    device = torch.device(DEVICE)

    with Timer("create state_dict"):
        state_dict: dict[str, torch.Tensor] = {}
        for i in range(0, TOTAL_SIZE, CHUNK_SIZE):
            state_dict[f"chunk/{i}"] = torch.zeros(
                CHUNK_SIZE // 4, dtype=torch.float32, device=device
            )

    def get_state_dict() -> object:
        return state_dict

    assert dist.group.WORLD is not None
    transport = PGTransport(
        pg=dist.group.WORLD,
        timeout=timeout,
        device=device,
        state_dict=get_state_dict if INPLACE else None,
    )

    if rank == 0:
        with Timer("send_checkpoint"):
            transport.send_checkpoint(
                dst_ranks=[1],
                state_dict=state_dict,
            )
    elif rank == 1:
        with Timer("recv_checkpoint"):
            transport.recv_checkpoint(src_rank=0)

    # Clean up
    logger.info(f"[Rank {dist.get_rank()}] Finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
