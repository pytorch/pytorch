import unittest
from unittest import skipIf
import torch
from torch._dynamo.test_case import TestCase as DynamoTestCase
import torch.distributed as dist
from torch._dynamo.source import LocalSource
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed._functional_collectives import all_to_all_single_autograd, all_to_all_single, wait_tensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@skipIf(not dist.is_available(), "requires distributed")
class TestFakeDistributed(DynamoTestCase):
    def setUp(self):
        # Use FakeProcessGroup to run tests on a single process
        self.store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=self.store)

    def tearDown(self):
        dist.destroy_process_group()

    @parametrize("autograd", [True, False])
    def test_all_to_all_single(self, autograd):
        comm = all_to_all_single_autograd if autograd else all_to_all_single

        @torch.compile(fullgraph=True)
        def fn(x):
            return comm(
                x,
                None, # Will use equal splits
                None, # Will use equal splits
                group=dist.group.WORLD)

        x = torch.randn(8, 8, requires_grad=autograd)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        wait_tensor(fn(x))

        x = torch.randn(8, 8, requires_grad=autograd)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        fn(x)

instantiate_parametrized_tests(TestFakeDistributed)

if __name__ == '__main__':
    unittest.main()
