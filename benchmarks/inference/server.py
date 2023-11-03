import argparse
import os.path
import subprocess
import time
from queue import Empty

import numpy as np

import torch
import torch.multiprocessing as mp


class FrontendWorker(mp.Process):
    """
    This worker will send requests to a backend process, and measure the
    throughput and latency of those requests as well as GPU utilization.
    """

    def __init__(self, request_queue, response_queue, batch_size, num_iters=10):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.warmup_event = mp.Event()
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.poll_gpu = True

    def _run_metrics(self):
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

        self.poll_gpu = False

        response_times = np.array(response_times)
        print(f"Warmup latency: {warmup_response_time:.5f} s")
        print(
            f"Average latency (exclude warmup): {response_times.mean():.5f} +/- {response_times.std():.5f} s"
        )
        print(f"Max latency: {response_times.max():.5f} s")
        print(f"Min latency: {response_times.min():.5f} s")
        print(
            "Throughput (exclude warmup): "
            f"{(self.num_iters * self.batch_size) / response_times.sum():.5f} samples per second"
        )

    def _run_gpu_utilization(self):
        """
        This function will poll nvidi-smi for GPU utilization every 100ms to
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
            time.sleep(0.1)
        print(f"Average GPU utilization: {np.array(gpu_utilizations).mean():.5f}")

    def _send_requests(self):
        """
        This function will send one warmup request, and then num_iters requests
        to the backend process.
        """
        # Send one batch of warmup data
        fake_data = torch.randn(
            self.batch_size, 3, 250, 250, requires_grad=False, pin_memory=True
        )
        self.request_queue.put((fake_data, time.time()))
        self.warmup_event.wait()

        # Send fake data
        for i in range(self.num_iters):
            fake_data = torch.randn(
                self.batch_size, 3, 250, 250, requires_grad=False, pin_memory=True
            )
            self.request_queue.put((fake_data, time.time()))

    def run(self):
        import threading

        requests_thread = threading.Thread(target=self._send_requests)
        metrics_thread = threading.Thread(target=self._run_metrics)
        gpu_utilization_thread = threading.Thread(target=self._run_gpu_utilization)

        requests_thread.start()
        metrics_thread.start()
        gpu_utilization_thread.start()

        requests_thread.join()
        metrics_thread.join()
        gpu_utilization_thread.join()


class BackendWorker(mp.Process):
    """
    This worker will take tensors from the request queue, do some computation,
    and then return the result back in the response queue.
    """

    def __init__(
        self, request_queue, response_queue, model_dir=".", compile_model=True
    ):
        super().__init__()
        self.device = "cuda:0"
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.model_dir = model_dir
        self.compile_model = compile_model
        self._setup_complete = False

    def _setup(self):
        import time

        import torch
        from torchvision.models.resnet import BasicBlock, ResNet

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
        print(f"torch.load() time: {time.time() - start_load_time:.5f} s")
        m.load_state_dict(state_dict, assign=True)
        m.eval()

        if self.compile_model:
            start_compile_time = time.time()
            m.compile()
            end_compile_time = time.time()
            print(
                f"m.compile() time (not actual first compilation): {end_compile_time - start_compile_time:.5f} s"
            )
        return m

    def run(self):
        while True:
            try:
                data, request_time = self.request_queue.get(timeout=10)
            except Empty:
                break

            if not self._setup_complete:
                model = self._setup()
                self._setup_complete = True

            with torch.no_grad():
                data = data.to(self.device, non_blocking=True)
                out = model(data)
            self.response_queue.put((out, request_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument("--compile", type=bool, default=True)
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

        frontend = FrontendWorker(
            request_queue, response_queue, args.batch_size, num_iters=args.num_iters
        )
        backend = BackendWorker(
            request_queue, response_queue, args.model_dir, args.compile
        )

        frontend.start()
        backend.start()

        frontend.join()
        backend.join()

    finally:
        # Cleanup checkpoint file if we downloaded it
        if downloaded_checkpoint:
            os.remove(f"{args.model_dir}/resnet18-f37072fd.pth")
