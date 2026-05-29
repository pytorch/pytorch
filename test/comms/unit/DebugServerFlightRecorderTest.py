# Copyright (c) Meta Platforms, Inc. and affiliates.
# Owner(s): ["oncall: distributed"]
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Tests for the TorchComms FlightRecorder debug server endpoints.

This test validates that the /torchcomms_fr_trace and /torchcomms_fr_trace_json
endpoints from the debug server properly return flight recorder data after
running collective operations through TorchComms.
"""

import os
import socket
import time
from datetime import timedelta
from unittest.mock import patch

import requests
from requests.adapters import HTTPAdapter

# pyre-fixme[21]: Could not find module `urllib3.util.retry`.
from urllib3.util.retry import Retry

import torch
import torch.comms
import torch.distributed as dist
import torch.distributed.debug as debug_module
from torch.comms.hooks import FlightRecorderHook
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.testing._internal.common_utils import run_tests, TestCase


session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def _reset_debug_server_state() -> None:
    debug_module._WORKER_SERVER = None
    debug_module._DEBUG_SERVER_PROC = None


class TestDebugServerFlightRecorder(TestCase):
    """Test that TorchComms FlightRecorder endpoints work with the debug server.

    This test validates the end-to-end flow:
    1. Start the debug server
    2. Initialize TorchComms communicator with FlightRecorderHook
    3. Run some collective operations
    4. Verify that /torchcomms_fr_trace and /torchcomms_fr_trace_json endpoints
       return the expected data with the collectives
    """

    def setUp(self) -> None:
        super().setUp()
        _reset_debug_server_state()

    def tearDown(self) -> None:
        super().tearDown()
        if (
            debug_module._DEBUG_SERVER_PROC is not None
            or debug_module._WORKER_SERVER is not None
        ):
            try:
                stop_debug_server()
            except Exception:
                pass
        _reset_debug_server_state()

    def test_torchcomms_fr_trace_endpoints(self) -> None:
        """Test that /torchcomms_fr_trace and /torchcomms_fr_trace_json work correctly.

        This test:
        1. Starts the debug server
        2. Creates a TorchComms communicator
        3. Registers FlightRecorderHook
        4. Runs collective operations (all_reduce, broadcast)
        5. Verifies the endpoints return data with the collectives
        """
        # Mock socket.gethostname to return localhost so the debug server
        # registers worker addresses that are reachable in test environments
        with patch.object(socket, "gethostname", return_value="localhost"):
            # Create our own TCPStore since the debug server needs one.
            store = dist.TCPStore(
                "localhost", 0, 1, is_master=True, wait_for_workers=False
            )
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(store.port)
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            # Use a dynamically allocated free port for the debug server
            port = 25998

            def fetch(path: str) -> requests.Response:
                resp = session.get(f"http://localhost:{port}{path}")
                return resp

            start_debug_server(port=port)

            backend = os.environ["TEST_BACKEND"]
            device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
            comm = torch.comms.new_comm(
                backend=backend,
                device=device,
                name="test_debug_server_comm",
                timeout=timedelta(seconds=300),
                store=store,
            )

            recorder = FlightRecorderHook(max_entries=100, isolated=True)
            recorder.register_with_comm(comm)

            t = torch.rand(10, 10, device=device)

            num_all_reduce = 2
            for _ in range(num_all_reduce):
                comm.all_reduce(t, op=torch.comms.ReduceOp.SUM, async_op=False)
            comm.broadcast(t, root=0, async_op=False)

            time.sleep(0.5)

            with self.subTest("index_page_nav_links"):
                resp = fetch("/")
                self.assertEqual(resp.status_code, 200)
                html_content = resp.text
                self.assertIn("TorchComms FR", html_content)

            with self.subTest("torchcomms_fr_trace_html"):
                resp = fetch("/torchcomms_fr_trace")
                self.assertEqual(resp.status_code, 200)
                html_content = resp.text
                self.assertIn("Memberships", html_content)

            with self.subTest("torchcomms_fr_trace_json"):
                resp = fetch("/torchcomms_fr_trace_json")
                self.assertEqual(resp.status_code, 200)
                # The endpoint returns HTML with JSON content wrapped in <pre> tags
                # We verify the HTML page contains the expected JSON structure
                html_content = resp.text
                self.assertIn("pg_status", html_content)

            comm.finalize()
            stop_debug_server()


if __name__ == "__main__":
    run_tests()
