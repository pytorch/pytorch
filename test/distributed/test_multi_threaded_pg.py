# Owner(s): ["oncall: distributed"]

import sys
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    spawn_threads_and_init_comms,
    MultiThreadedTestCase

)
from torch.testing._internal.common_utils import TestCase, run_tests

DEFAULT_WORLD_SIZE = 4

class TestObjectCollectivesWithWrapper(TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_broadcast_object_list(self):
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

class TestObjectCollectivesWithBaseClass(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def test_broadcast_object_list(self):
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()
        print(f"{dist.get_rank()} -> {dist.get_world_size()}")

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

    def test_something_else(self):
        pass

if __name__ == "__main__":
    run_tests()
