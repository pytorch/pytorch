import argparse
import random
import time
from abc import abstractmethod
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

import torch


class BenchmarkRunner:
    """
    BenchmarkRunner is a base class for all benchmark runners. It provides an interface to run benchmarks in order to
    collect data with AutoHeuristic.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self) -> None:
        self.parser.add_argument(
            "--device",
            type=int,
            default=None,
            help="torch.cuda.set_device(device) will be used",
        )
        self.parser.add_argument(
            "--use-heuristic",
            action="store_true",
            help="Use learned heuristic instead of collecting data.",
        )
        self.parser.add_argument(
            "-o",
            type=str,
            default="ah_data.txt",
            help="Path to file where AutoHeuristic will log results.",
        )
        self.parser.add_argument(
            "--num-samples",
            type=int,
            default=1000,
            help="Number of samples to collect.",
        )
        self.parser.add_argument(
            "--num-reps",
            type=int,
            default=3,
            help="Number of measurements to collect for each input.",
        )

    def run(self) -> None:
        torch.set_default_device("cuda")
        args = self.parser.parse_args()
        if args.use_heuristic:
            torch._inductor.config.autoheuristic_use = self.name
            torch._inductor.config.autoheuristic_collect = ""
        else:
            torch._inductor.config.autoheuristic_use = ""
            torch._inductor.config.autoheuristic_collect = self.name
        torch._inductor.config.autoheuristic_log_path = args.o
        if args.device is not None:
            torch.cuda.set_device(args.device)
        random.seed(time.time())
        self.main(args.num_samples, args.num_reps)

    @abstractmethod
    def run_benchmark(self, *args: Any) -> None: ...

    @abstractmethod
    def create_input(self) -> tuple[Any, ...]: ...

    def main(self, num_samples: int, num_reps: int) -> None:
        for _ in tqdm(range(num_samples)):
            input = self.create_input()
            for _ in range(num_reps):
                self.run_benchmark(*input)
