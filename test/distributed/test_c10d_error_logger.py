# Owner(s): ["oncall: distributed"]

import json
import logging
import re
import sys

import torch
import torch.distributed as dist

from torch.distributed.c10d_error_logger import _get_or_create_logger
from torch.distributed.distributed_c10d import exception_handler

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import TestCase, run_tests


class C10dErrorLoggerTest(TestCase):
    def test_get_or_create_logger(self):
        logger = _get_or_create_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)


class C10dExceptionHandlerTest(MultiThreadedTestCase):
    @property
    def world_size(self):
        # return 1 # runs without problem with world_size = 1
        return 2

    @exception_handler
    def failed_broadcast_raise_exception(self):
        tensor = torch.arange(2, dtype=torch.int64)
        # broadcast to world size out of bound to raise exception
        dist.broadcast(tensor, self.world_size + 1)

    @exception_handler
    def failed_broadcast_not_raise_exception(self):
        try:
            tensor = torch.arange(2, dtype=torch.int64)
            dist.broadcast(tensor, self.world_size + 1)
        except Exception as exception:
            pass

    def test_exception_handler_with_dist(self) -> None:
        print(dist.get_rank())
        with self.assertRaises(Exception) as exception:
            self.failed_broadcast_raise_exception()

        with self.assertLogs(
            dist._c10d_error_logger, level="DEBUG"
        ) as captured:
            self.failed_broadcast_not_raise_exception()
            error_msg_dict = json.loads(
                re.search("({.+})", captured.output[0])
                .group(0)
                .replace("'", '"')
            )
            print(error_msg_dict)
            self.assertEqual(len(error_msg_dict), 7)
            self.assertEqual("broadcast", error_msg_dict["func_name"])
            self.assertEqual(f"(tensor([0, 1]), {self.world_size+1}), {{}}", error_msg_dict["args"])
            self.assertEqual("threaded", error_msg_dict["backend"])
            self.assertEqual(str(self.world_size), error_msg_dict["world_size"])
            self.assertEqual(str(dist.get_rank()), error_msg_dict["global_rank"])
            # In this test case, local_rank = global_rank, since we don't have multiple processes on one node.
            self.assertEqual(str(dist.get_rank()), error_msg_dict["local_rank"])
            print("print after all assert")
            # timeout on rank 1 if world_size = 2


if __name__ == "__main__":
    run_tests()
