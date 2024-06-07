import argparse

import asyncio
import os.path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp


class FrontendWorker(mp.Process):
    """
    This worker will send requests to a backend process, and measure the
    throughput and latency of those requests as well as GPU utilization.
    """

    def __init__(
        self,
        metrics_dict,
        request_queue,
        response_queue,
        read_requests_event,
        batch_size,
        num_iters=10,
    ):
        super().__init__()
        self.metrics_dict = metrics_dict
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.read_requests_event = read_requests_event
        self.warmup_event = mp.Event()
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.poll_gpu = True
        self.start_send_time = None
        self.end_recv_time = None

    def _run_metrics(self, metrics_lock):
        """
        This function will poll the response queue until it has received all
        responses. It records the startup latency, the average, max, min latency
        as well as througput of requests.
        """
        warmup_response_time = None
        response_times = []

        for i in range(self.num_iters + 1):
            response, request_time = self.response_queue.get()
            if warmup_response_time is None:
                self.warmup_event.set()
                warmup_response_time = time.time() - request_time
            else:
                response_times.append(time.time() - request_time)

        self.end_recv_time = time.time()
        self.poll_gpu = False

        response_times = np.array(response_times)
        with metrics_lock:
            self.metrics_dict["warmup_latency"] = warmup_response_time
            self.metrics_dict["average_latency"] = response_times.mean()
            self.metrics_dict["max_latency"] = response_times.max()
            self.metrics_dict["min_latency"] = response_times.min()
            self.metrics_dict["throughput"] = (self.num_iters * self.batch_size) / (
                self.end_recv_time - self.start_send_time
            )

    def _run_gpu_utilization(self, metrics_lock):
        """
        This function will poll nvidia-smi for GPU utilization every 100ms to
        record the average GPU utilization.
        """

        def get_gpu_utilization():
            try:
                nvidia_smi_output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--id=0",
                        "--format=csv,noheader,nounits",
                    ]
                )
                gpu_utilization = nvidia_smi_output.decode().strip()
                return gpu_utilization
            except subprocess.CalledProcessError:
                return "N/A"

        gpu_utilizations = []

        while self.poll_gpu:
            gpu_utilization = get_gpu_utilization()
            if gpu_utilization != "N/A":
                gpu_utilizations.append(float(gpu_utilization))

        with metrics_lock:
            self.metrics_dict["gpu_util"] = torch.tensor(gpu_utilizations).mean().item()

    def _send_requests(self):
        """
        This function will send one warmup request, and then num_iters requests
        to the backend process.
        """

        fake_data = torch.randn(self.batch_size, 3, 250, 250, requires_grad=False)
        other_data = [
            torch.randn(self.batch_size, 3, 250, 250, requires_grad=False)
            for i in range(self.num_iters)
        ]

        # Send one batch of warmup data
        self.request_queue.put((fake_data, time.time()))
        # Tell backend to poll queue for warmup request
        self.read_requests_event.set()
        self.warmup_event.wait()
        # Tell backend to poll queue for rest of requests
        self.read_requests_event.set()

        # Send fake data
        self.start_send_time = time.time()
        for i in range(self.num_iters):
            self.request_queue.put((other_data[i], time.time()))

    def run(self):
        # Lock for writing to metrics_dict
        metrics_lock = threading.Lock()
        requests_thread = threading.Thread(target=self._send_requests)
        metrics_thread = threading.Thread(
            target=self._run_metrics, args=(metrics_lock,)
        )
        gpu_utilization_thread = threading.Thread(
            target=self._run_gpu_utilization, args=(metrics_lock,)
        )

        requests_thread.start()
        metrics_thread.start()

        # only start polling GPU utilization after the warmup request is complete
        self.warmup_event.wait()
        gpu_utilization_thread.start()

        requests_thread.join()
        metrics_thread.join()
        gpu_utilization_thread.join()


class BackendWorker:
    """
    This worker will take tensors from the request queue, do some computation,
    and then return the result back in the response queue.
    """

    def __init__(
        self,
        metrics_dict,
        request_queue,
        response_queue,
        read_requests_event,
        batch_size,
        num_workers,
        model_dir=".",
        compile_model=True,
    ):
        super().__init__()
        self.device = "cuda:0"
        self.metrics_dict = metrics_dict
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.read_requests_event = read_requests_event
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_dir = model_dir
        self.compile_model = compile_model
        self._setup_complete = False
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()
        # maps thread_id to the cuda.Stream associated with that worker thread
        self.stream_map = dict()

    def _setup(self):
        import time

        from torchvision.models.resnet import BasicBlock, ResNet

        import torch

        # Create ResNet18 on meta device
        with torch.device("meta"):
            m = ResNet(BasicBlock, [2, 2, 2, 2])

        # Load pretrained weights
        start_load_time = time.time()
        state_dict = torch.load(
            f"{self.model_dir}/resnet18-f37072fd.pth",
            mmap=True,
            map_location=self.device,
        )
        self.metrics_dict["torch_load_time"] = time.time() - start_load_time
        m.load_state_dict(state_dict, assign=True)
        m.eval()

        if self.compile_model:
            start_compile_time = time.time()
            m.compile()
            end_compile_time = time.time()
            self.metrics_dict["m_compile_time"] = end_compile_time - start_compile_time
        return m

    def model_predict(
        self,
        model,
        input_buffer,
        copy_event,
        compute_event,
        copy_sem,
        compute_sem,
        response_list,
        request_time,
    ):
        # copy_sem makes sure copy_event has been recorded in the data copying thread
        copy_sem.acquire()
        self.stream_map[threading.get_native_id()].wait_event(copy_event)
        with torch.cuda.stream(self.stream_map[threading.get_native_id()]):
            with torch.no_grad():
                response_list.append(model(input_buffer))
                compute_event.record()
                compute_sem.release()
        del input_buffer

    def copy_data(self, input_buffer, data, copy_event, copy_sem):
        data = data.pin_memory()
        with torch.cuda.stream(self.h2d_stream):
            input_buffer.copy_(data, non_blocking=True)
            copy_event.record()
            copy_sem.release()

    def respond(self, compute_event, compute_sem, response_list, request_time):
        # compute_sem makes sure compute_event has been recorded in the model_predict thread
        compute_sem.acquire()
        self.d2h_stream.wait_event(compute_event)
        with torch.cuda.stream(self.d2h_stream):
            self.response_queue.put((response_list[0].cpu(), request_time))

    async def run(self):
        def worker_initializer():
            self.stream_map[threading.get_native_id()] = torch.cuda.Stream()

        worker_pool = ThreadPoolExecutor(
            max_workers=self.num_workers, initializer=worker_initializer
        )
        h2d_pool = ThreadPoolExecutor(max_workers=1)
        d2h_pool = ThreadPoolExecutor(max_workers=1)

        self.read_requests_event.wait()
        # Clear as we will wait for this event again before continuing to
        # poll the request_queue for the non-warmup requests
        self.read_requests_event.clear()
        while True:
            try:
                data, request_time = self.request_queue.get(timeout=5)
            except Empty:
                break

            if not self._setup_complete:
                model = self._setup()

            copy_sem = threading.Semaphore(0)
            compute_sem = threading.Semaphore(0)
            copy_event = torch.cuda.Event()
            compute_event = torch.cuda.Event()
            response_list = []
            input_buffer = torch.empty(
                [self.batch_size, 3, 250, 250], dtype=torch.float32, device="cuda"
            )
            asyncio.get_running_loop().run_in_executor(
                h2d_pool,
                self.copy_data,
                input_buffer,
                data,
                copy_event,
                copy_sem,
            )
            asyncio.get_running_loop().run_in_executor(
                worker_pool,
                self.model_predict,
                model,
                input_buffer,
                copy_event,
                compute_event,
                copy_sem,
                compute_sem,
                response_list,
                request_time,
            )
            asyncio.get_running_loop().run_in_executor(
                d2h_pool,
                self.respond,
                compute_event,
                compute_sem,
                response_list,
                request_time,
            )

            if not self._setup_complete:
                self.read_requests_event.wait()
                self._setup_complete = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument(
        "--compile", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--output_file", type=str, default="output.csv")
    parser.add_argument(
        "--profile", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    downloaded_checkpoint = False
    if not os.path.isfile(f"{args.model_dir}/resnet18-f37072fd.pth"):
        p = subprocess.run(
            [
                "wget",
                "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            ]
        )
        if p.returncode == 0:
            downloaded_checkpoint = True
        else:
            raise RuntimeError("Failed to download checkpoint")

    try:
        mp.set_start_method("forkserver")
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        read_requests_event = mp.Event()

        manager = mp.Manager()
        metrics_dict = manager.dict()
        metrics_dict["batch_size"] = args.batch_size
        metrics_dict["compile"] = args.compile

        frontend = FrontendWorker(
            metrics_dict,
            request_queue,
            response_queue,
            read_requests_event,
            args.batch_size,
            num_iters=args.num_iters,
        )
        backend = BackendWorker(
            metrics_dict,
            request_queue,
            response_queue,
            read_requests_event,
            args.batch_size,
            args.num_workers,
            args.model_dir,
            args.compile,
        )

        frontend.start()

        if args.profile:

            def trace_handler(prof):
                prof.export_chrome_trace("trace.json")

            with torch.profiler.profile(on_trace_ready=trace_handler) as prof:
                asyncio.run(backend.run())
        else:
            asyncio.run(backend.run())

        frontend.join()

        metrics_dict = {k: [v] for k, v in metrics_dict._getvalue().items()}
        output = pd.DataFrame.from_dict(metrics_dict, orient="columns")
        output_file = "./results/" + args.output_file
        is_empty = not os.path.isfile(output_file)

        with open(output_file, "a+", newline="") as file:
            output.to_csv(file, header=is_empty, index=False)

    finally:
        # Cleanup checkpoint file if we downloaded it
        if downloaded_checkpoint:
            os.remove(f"{args.model_dir}/resnet18-f37072fd.pth")
