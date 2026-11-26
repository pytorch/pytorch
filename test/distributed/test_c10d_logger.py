# Owner(s): ["oncall: distributed"]

import json
import logging
import re
import sys

import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _c10d_logger, _exception_logger


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_distributed import DistributedTestBase, with_comms
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def _nccl_expected_version() -> str:
    return ".".join(str(v) for v in torch.cuda.nccl.version())


class C10dErrorLoggerTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_get_or_create_logger(self):
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)


class C10dErrorLoggerTestDevice(DistributedTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @property
    def world_size(self):
        # Concurrency budget of the current environment, not a per-variant
        # device source.
        if torch.accelerator.is_available():
            return min(4, max(2, torch.accelerator.device_count()))
        return 2

    @property
    def process_group(self):
        return dist.group.WORLD

    @_exception_logger
    def _failed_broadcast_raise_exception(self, rank_device):
        tensor = torch.arange(2, dtype=torch.int64, device=rank_device)
        dist.broadcast(tensor, self.world_size + 1)

    @_exception_logger
    def _failed_broadcast_not_raise_exception(self, rank_device):
        try:
            tensor = torch.arange(2, dtype=torch.int64, device=rank_device)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception:
            pass

    @onlyAccelerator
    @with_comms(barrier_on_success=True)
    def test_exception_logger(self, device) -> None:
        device_type = torch.device(device).type
        rank_device = (
            torch.device("cpu")
            if device_type == "cpu"
            else torch.device(device_type, self.rank)
        )

        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception(rank_device)

        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._failed_broadcast_not_raise_exception(rank_device)
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            # NCCL adds additional nccl_version data to the error_msg_dict
            expected_backend = self.backend(device_type)
            if expected_backend == dist.Backend.NCCL:
                self.assertEqual(len(error_msg_dict), 9)
            else:
                self.assertEqual(len(error_msg_dict), 8)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual(expected_backend, error_msg_dict["backend"])

            if expected_backend == dist.Backend.NCCL:
                self.assertIn("nccl_version", error_msg_dict.keys())
                self.assertEqual(
                    _nccl_expected_version(), error_msg_dict["nccl_version"]
                )

            # In this test case, group_size = world_size, since we don't have multiple processes on one node.
            self.assertIn("group_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["group_size"])

            self.assertIn("world_size", error_msg_dict.keys())
            self.assertEqual(str(self.world_size), error_msg_dict["world_size"])

            self.assertIn("global_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["global_rank"])

            # In this test case, local_rank = global_rank, since we don't have multiple processes on one node.
            self.assertIn("local_rank", error_msg_dict.keys())
            self.assertIn(str(dist.get_rank()), error_msg_dict["local_rank"])


instantiate_device_type_tests(C10dErrorLoggerTestDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
