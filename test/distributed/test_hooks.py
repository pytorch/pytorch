# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.hooks as dhooks

if not dist.is_available():
    print("torch.distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


from torch.testing._internal.common_distributed import MultiProcessTestCase

from torch.testing._internal.common_utils import run_tests


class TestMultiThreadedWait(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def test_pg_hook(self):
        pgs = []

        def pg_hook(pg, pg_name):
            pgs.append((pg, pg_name))

        dhooks.register_process_group_hook(pg_hook)
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )
        self.assertEqual(len(pgs), 1)
        self.assertEqual(pgs[0][0], dist.group.WORLD)

        # create two partial world PGs
        pg0 = dist.new_group(ranks=[0, 1])
        pg1 = dist.new_group(ranks=[2, 3])

        self.assertEqual(len(pgs), 2)
        self.assertEqual(pgs[1][0], pg0 if self.rank < 2 else pg1)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
