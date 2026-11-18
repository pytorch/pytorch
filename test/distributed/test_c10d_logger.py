# Owner(s): ["oncall: distributed"]

import json
import logging
import re
import sys
import unittest

import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _c10d_logger, _exception_logger
from torch.distributed.distributed_c10d import is_backend_available


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    requires_nccl,
    TEST_SKIPS,
)
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


class C10dErrorLoggerTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_get_or_create_logger(self):
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)


class _C10dErrorLoggerBase(DistributedTestBase):
    @property
    def world_size(self):
        # Concurrency budget of the current environment, not a per-variant
        # device source: CPU variants run this many gloo ranks, accelerator
        # variants spread the same budget over their own device module.
        if torch.accelerator.is_available():
            return min(4, max(2, torch.accelerator.device_count()))
        return 2

    @property
    def process_group(self):
        return dist.group.WORLD

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def _init_comms(self, device, *, backend=None):
        device_type = torch.device(device).type
        if backend is None:
            backend = self.backend(device_type)
        if device_type != "cpu" and (
            torch.get_device_module(device_type).device_count() < self.world_size
        ):
            sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)
        if not is_backend_available(backend):
            raise unittest.SkipTest(f"Backend '{backend}' is not available")
        self.create_pg(device_type, backend=backend)

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


@requires_nccl()
@unittest.skipIf(not TEST_CUDA, "CUDA is not available")
class C10dErrorLoggerTestCUDA(_C10dErrorLoggerBase):
    hw_classification = HardwareClassification.CUDA

    def test_exception_logger_nccl_version(self) -> None:
        if torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)
        self.create_pg("cuda", backend="nccl")

        rank_device = torch.device("cuda", self.rank)
        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception(rank_device)

        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._failed_broadcast_not_raise_exception(rank_device)
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            self.assertEqual(len(error_msg_dict), 9)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual(dist.Backend.NCCL, error_msg_dict["backend"])

            self.assertIn("nccl_version", error_msg_dict.keys())
            nccl_ver = torch.cuda.nccl.version()
            self.assertEqual(
                ".".join(str(v) for v in nccl_ver), error_msg_dict["nccl_version"]
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

        self.destroy_comms()


class C10dErrorLoggerTestDevice(_C10dErrorLoggerBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_exception_logger(self, device) -> None:
        device_type = torch.device(device).type
        rank_device = (
            torch.device("cpu")
            if device_type == "cpu"
            else torch.device(device_type, self.rank)
        )
        self._init_comms(device)

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
                self.assertIn("nccl_version", error_msg_dict.keys())
            else:
                self.assertEqual(len(error_msg_dict), 8)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual(expected_backend, error_msg_dict["backend"])

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

        self.destroy_comms()


instantiate_device_type_tests(C10dErrorLoggerTestDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
