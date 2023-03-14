import dataclasses
import queue
import time
import warnings
from typing import List, Optional

import torch
from torch import multiprocessing, Tensor

from torch._inductor import select_algorithm
from torch._inductor.select_algorithm import ChoiceCaller


@dataclasses.dataclass
class BenchmarkRequest:
    choice: ChoiceCaller
    inputs: List[Tensor]
    output: Tensor
    expected_output: Optional[Tensor]

    def benchmark(self) -> float:
        result = self.choice.benchmark(*self.inputs, out=self.output)
        if self.expected_output is not None:
            torch.testing.assert_close(self.output, self.expected_output)

        choice = inputs = output = expected_output = None
        return result[0]


@dataclasses.dataclass
class SyncTemplateKernelRequest:
    template_kernels: select_algorithm.KernelNamespace


EXIT_HANDLER_REGISTERED = False


@dataclasses.dataclass
class TuningProcess:
    process: multiprocessing.Process = None
    request_queue: multiprocessing.Queue = None
    response_queue: multiprocessing.Queue = None

    # record the nubmer of template kernels synced to child process. When more
    # are registered in the parent process, we need sync the extra to the child
    # process
    n_template_kernel_synced: int = 0

    @staticmethod
    def process_main(request_queue, response_queue):
        print("enter child process main")
        while True:
            obj = request_queue.get()

            if obj is None:
                break  # None is a sentinel for the child to terminate
            elif isinstance(obj, BenchmarkRequest):
                response_queue.put(obj.benchmark())
            elif isinstance(obj, SyncTemplateKernelRequest):
                select_algorithm.template_kernels = obj.template_kernels
                response_queue.put(
                    True
                )  # give a response so the parent can wait for the completion of the sync

    def valid(self):
        return (
            self.process is not None
            and self.request_queue is not None
            and self.response_queue is not None
        )

    @staticmethod
    def exit_handler():
        tuning_process.terminate()

    def initialize(self):
        """
        Create child process, request/response queues and do the warm up.
        """
        if self.valid():
            return

        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
            ),
        )
        self.process.start()
        self.n_template_kernel_synced = 0
        self.sync_template_kernels()

        # register the exit handler for the parent process so it will terminate
        # the child processes
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            atexit.register(self.exit_handler)

    def sync_template_kernels(self):
        """
        warmup the process with the registered template kernels. Spawn'ed child
        process does not inheirit them automatically
        """
        if self.n_template_kernel_synced < len(
            select_algorithm.template_kernels.__dict__
        ):
            start_ts = time.time()
            # TODO: only send the incremental part
            self.request_queue.put(
                SyncTemplateKernelRequest(select_algorithm.template_kernels)
            )
            assert self.response_queue.get() is True
            self.n_template_kernel_synced = len(
                select_algorithm.template_kernels.__dict__
            )
            elapse = time.time() - start_ts
            print(
                f"Sync {self.n_template_kernel_synced} template kernels took {elapse} seconds"
            )

    def terminate(self):
        if self.valid():
            self.request_queue.put(None)
            self.process.join()


tuning_process = TuningProcess()


def autotune(
    choice: ChoiceCaller,
    inputs: List[Tensor],
    output: Tensor,
    expected_output: Optional[Tensor],
) -> float:
    """
    Do autotuning and return the perf number (latency).
    """
    tuning_process.initialize()
    assert tuning_process.valid()
    tuning_process.sync_template_kernels()

    bmreq = BenchmarkRequest(
        choice,
        inputs,
        output,
        expected_output,
    )

    tuning_process.request_queue.put(bmreq)

    while True:
        try:
            timing = tuning_process.response_queue.get(timeout=1.0)
        except queue.Empty:
            status = tuning_process.process.exitcode
            if status is None:
                # child process is still running
                continue
            # child process fail
            assert status != 0

            warnings.warn(
                f"Fail to benchmark choice '{choice}'. It will be ignored. Please debug the root cause in case the choice can bring perf gains."  # noqa: B950 line too long
            )
            # return a large value to this choice will be ignored
            return 1e10

        return timing
