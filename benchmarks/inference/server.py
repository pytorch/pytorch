import argparse
import os.path
import subprocess
import time
from queue import Empty

import torch
import torch.multiprocessing as mp


class FrontendWorker(mp.Process):
    """
    This worker will collect metrics about the response latency of the model
    """

    def __init__(self, response_queue, warmup_event, num_iters=10):
        super().__init__()
        self.response_queue = response_queue
        self.warmup_event = warmup_event
        self.num_iters = num_iters
        self.response_times = []
        self.warmup_response_time = None

    def run(self):
        import time

        import numpy as np

        for i in range(self.num_iters):
            response, request_time = self.response_queue.get()
            if self.warmup_response_time is None:
                self.warmup_event.set()
                self.warmup_response_time = time.time() - request_time
            else:
                self.response_times.append(time.time() - request_time)

        response_times = np.array(self.response_times)

        print(f"Warmup latency: {self.warmup_response_time:.5f} seconds")
        print(
            f"Average latency (exclude warmup): {response_times.mean():.5f} +/- {response_times.std():.5f} seconds, "
            f"max {response_times.max():.5f} seconds, min {response_times.min():.5f} seconds"
        )
        print(
            f"Throughput (exclude warmup): {self.num_iters / response_times.sum()} batches per second"
        )


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
        print(f"Load time: {time.time() - start_load_time:.5f} seconds")
        m.load_state_dict(state_dict, assign=True)
        m.eval()

        if self.compile_model:
            start_compile_time = time.time()
            m.compile()
            end_compile_time = time.time()
            print(f"Compile time: {end_compile_time - start_compile_time:.5f} seconds")
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
        warmup_event = mp.Event()

        frontend = FrontendWorker(
            response_queue, warmup_event, num_iters=args.num_iters
        )
        backend = BackendWorker(
            request_queue, response_queue, args.model_dir, args.compile
        )

        frontend.start()
        backend.start()

        # Send one batch of warmup data
        fake_data = torch.randn(
            args.batch_size, 3, 250, 250, requires_grad=False, pin_memory=True
        )
        request_queue.put((fake_data, time.time()))
        warmup_event.wait()

        # Send fake data
        for i in range(args.num_iters):
            fake_data = torch.randn(
                args.batch_size, 3, 250, 250, requires_grad=False, pin_memory=True
            )
            request_queue.put((fake_data, time.time()))

        frontend.join()
        backend.join()

    finally:
        # Cleanup checkpoint file if we downloaded it
        if downloaded_checkpoint:
            os.remove(f"{args.model_dir}/resnet18-f37072fd.pth")
