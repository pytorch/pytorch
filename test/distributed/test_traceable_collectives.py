# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch.testing._internal.common_distributed import (
    requires_nccl,
    DynamoDistributedSingleProcTestCase,
)

@requires_nccl()
class TestDynamoSingleProc(DynamoDistributedSingleProcTestCase):
    """
    Single proc test case runner that resets dynamo between tests
    """

    # TODO test backwards..

    def test_meta(self):
        x = torch.rand((2, 3, 4), device="meta")
        pg_id = 0  # hack since pg api isn't landed
        out = torch.ops.aten.all_reduce(x, group_id=pg_id, reduce_op="sum")
        assert x.size() == out.size()

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
