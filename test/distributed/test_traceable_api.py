# Owner(s): ["oncall: distributed"]

import sys
import torch
import torch.distributed as dist
import torch.distributed.traceable_collectives as tr_c
import torch.distributed._tensor as dt

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import (
    run_tests,
)

DEFAULT_WORLD_SIZE = 4

class TestExpand(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def test_expand_1d_rank_list(self):
        tag, rankset, stride = tr_c._expand_group([0,1,2,3])
        self.assertEqual("", tag)
        self.assertEqual([0,1,2,3], rankset)
        self.assertEqual(4, stride)

        tag, rankset, stride = tr_c._expand_group([0,1,2,3], "bla")
        self.assertEqual("bla", tag)

    def test_expand_2d_rank_list(self):
        tag, rankset, stride = tr_c._expand_group([[0,1],[2,3]])
        self.assertEqual("", tag)
        self.assertEqual([0,1,2,3], rankset)
        self.assertEqual(2, stride)

        tag, rankset, stride = tr_c._expand_group([[0,1],[2,3]], "blu")
        self.assertEqual("blu", tag)

        with self.assertRaisesRegex(ValueError, "group sizes must be identical"):
            tr_c._expand_group([[0],[1, 2, 3]])

    def test_expand_process_group(self):
        tag, rankset, stride = tr_c._expand_group(dist.group.WORLD)
        self.assertEqual("", tag)
        self.assertEqual([0,1,2,3], rankset)
        self.assertEqual(4, stride)

        tag, rankset, stride = tr_c._expand_group(dist.group.WORLD, "bla")
        self.assertEqual("bla", tag)

        my_pg, others = dist.new_subgroups(group_size=2)
        tag, rankset, stride = tr_c._expand_group(my_pg)
        self.assertEqual("", tag)
        self.assertEqual(dist.get_process_group_ranks(my_pg), rankset)
        self.assertEqual(2, stride)

        my_pg = None
        for i in range(dist.get_world_size()):
            group = dist.new_group([i], pg_tag="my_pg")
            if i == dist.get_rank():
                my_pg = group
        tag, rankset, stride = tr_c._expand_group(my_pg)
        self.assertEqual("my_pg", tag)
        self.assertEqual([dist.get_rank()], rankset)
        self.assertEqual(1, stride)

        tag, rankset, stride = tr_c._expand_group(my_pg, "bla")
        self.assertEqual("bla", tag)

    def test_expand_device_mesh(self):
        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, stride = tr_c._expand_group(mesh)
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, stride)

        mesh = dt.DeviceMesh("cpu", torch.arange(4))
        tag, rankset, stride = tr_c._expand_group(mesh)
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(4, stride)

    def test_expand_device_mesh_tuple(self):
        mesh = dt.DeviceMesh("cpu", torch.arange(4).view(2,2))
        tag, rankset, stride = tr_c._expand_group((mesh, 0))
        self.assertEqual("", tag)
        self.assertEqual([0, 2, 1, 3], rankset)
        self.assertEqual(2, stride)

        tag, rankset, stride = tr_c._expand_group((mesh, 1))
        self.assertEqual("", tag)
        self.assertEqual([0, 1, 2, 3], rankset)
        self.assertEqual(2, stride)


if __name__ == "__main__":
    run_tests()
