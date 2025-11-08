# Owner(s): ["oncall: distributed"]

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch.distributed as dist
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.testing._internal.common_utils import run_tests, TestCase


session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


class TestDebug(TestCase):
    def test_basics(self) -> None:
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25999

        def fetch(path: str) -> str:
            resp = session.get(f"http://localhost:{port}{path}")
            resp.raise_for_status()
            return resp.text

        print("starting!")

        start_debug_server(port=port)

        self.assertIn("torch profiler", fetch("/"))
        self.assertIn("View 0", fetch("/profile?duration=0.01"))
        self.assertIn("test_basics", fetch("/stacks"))
        self.assertIn("pg_status", fetch("/fr_trace"))
        self.assertIn("pg_status", fetch("/fr_trace_nccl"))

        stop_debug_server()


if __name__ == "__main__":
    run_tests()
