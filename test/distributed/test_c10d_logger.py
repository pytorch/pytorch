# Owner(s): ["oncall: distributed"]

import json
import logging
import os
import re
import sys
import time
from functools import partial, wraps

import torch
import torch.distributed as dist

from torch.distributed.c10d_logger import _c10d_logger, _exception_logger, _time_logger

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

BACKEND = dist.Backend.NCCL
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))


def with_comms(func=None):
    if func is None:
        return partial(
            with_comms,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.dist_init()
        func(self)
        self.destroy_comms()

    return wrapper


class C10dErrorLoggerTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        self._spawn_processes()

    @property
    def device(self):
        return (
            torch.device(self.rank)
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        return WORLD_SIZE

    @property
    def process_group(self):
        return dist.group.WORLD

    def destroy_comms(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def dist_init(self):
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if BACKEND == "nccl":
            torch.cuda.set_device(self.rank)

    def test_get_or_create_logger(self):
        self.assertIsNotNone(_c10d_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))
        self.assertIsInstance(_c10d_logger.handlers[0], logging.NullHandler)

    @_exception_logger
    def _failed_broadcast_raise_exception(self):
        tensor = torch.arange(2, dtype=torch.int64)
        dist.broadcast(tensor, self.world_size + 1)

    @_exception_logger
    def _failed_broadcast_not_raise_exception(self):
        try:
            tensor = torch.arange(2, dtype=torch.int64)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception:
            pass

    @with_comms
    def test_exception_logger(self) -> None:
        with self.assertRaises(Exception):
            self._failed_broadcast_raise_exception()

        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._failed_broadcast_not_raise_exception()
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )

            self.assertEqual(len(error_msg_dict), 10)

            self.assertIn("pg_name", error_msg_dict.keys())
            self.assertEqual("None", error_msg_dict["pg_name"])

            self.assertIn("func_name", error_msg_dict.keys())
            self.assertEqual("broadcast", error_msg_dict["func_name"])

            self.assertIn("args", error_msg_dict.keys())

            self.assertIn("backend", error_msg_dict.keys())
            self.assertEqual("nccl", error_msg_dict["backend"])

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

    @_time_logger
    def _dummy_sleep(self):
        time.sleep(5)

    @with_comms
    def test_time_logger(self) -> None:
        with self.assertLogs(_c10d_logger, level="DEBUG") as captured:
            self._dummy_sleep()
            msg_dict = json.loads(
                re.search("({.+})", captured.output[0]).group(0).replace("'", '"')
            )
            self.assertEqual(len(msg_dict), 10)

            self.assertIn("pg_name", msg_dict.keys())
            self.assertEqual("None", msg_dict["pg_name"])

            self.assertIn("func_name", msg_dict.keys())
            self.assertEqual("_dummy_sleep", msg_dict["func_name"])

            self.assertIn("args", msg_dict.keys())

            self.assertIn("backend", msg_dict.keys())
            self.assertEqual("nccl", msg_dict["backend"])

            self.assertIn("nccl_version", msg_dict.keys())
            nccl_ver = torch.cuda.nccl.version()
            self.assertEqual(
                ".".join(str(v) for v in nccl_ver), msg_dict["nccl_version"]
            )

            # In this test case, group_size = world_size, since we don't have multiple processes on one node.
            self.assertIn("group_size", msg_dict.keys())
            self.assertEqual(str(self.world_size), msg_dict["group_size"])

            self.assertIn("world_size", msg_dict.keys())
            self.assertEqual(str(self.world_size), msg_dict["world_size"])

            self.assertIn("global_rank", msg_dict.keys())
            self.assertIn(str(dist.get_rank()), msg_dict["global_rank"])

            # In this test case, local_rank = global_rank, since we don't have multiple processes on one node.
            self.assertIn("local_rank", msg_dict.keys())
            self.assertIn(str(dist.get_rank()), msg_dict["local_rank"])

            self.assertIn("time_spent", msg_dict.keys())
            time_ns = re.findall(r"\d+", msg_dict["time_spent"])[0]
            self.assertEqual(5, int(float(time_ns) / pow(10, 9)))


if __name__ == "__main__":
    run_tests()
