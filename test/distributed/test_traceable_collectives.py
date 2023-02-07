# Owner(s): ["module: dynamo"]
import torch
from torch._dispatch.python import enable_python_dispatcher
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    requires_nccl
)
import torch._dynamo.logging

# LOL if you don't remember to import this, then the op isn't registered and it hits
# the no-op C++ kernel that i am forced to implement despite not using it
import torch.distributed.traceable_collectives


@requires_nccl()
class TestCollectives(DynamoDistributedSingleProcTestCase):
    def get_world_trs(self, world_size=1):
        return {
            "tag": "",
            "ranks": list(range(world_size)),
            "stride": world_size,
        }

    def test_backwards(self):
        """
        It's probably not that common to need backwards support for collectives.

        However, I wanted to at least see if it was possible to support it as a design goal.
        """
        def func(inp, *, tag, ranks, stride):
            ar = torch.ops.aten.all_reduce(inp, "sum", tag, ranks, stride)
            return ar

        input = torch.ones(4, 4, device="cuda", requires_grad=True)
        with enable_python_dispatcher():
            compiled = torch.compile(func, backend="aot_eager")  # inductor bug with single-op allreduce graph
            out = compiled(input, **self.get_world_trs())
            out.sum().backward()

            correct_input = input.clone().detach().requires_grad_()
            correct = func(correct_input, **self.get_world_trs())
            correct.sum().backward()
            assert same(out, correct)
            assert same(input.grad, correct_input.grad)

    def test_meta(self):
        x = torch.rand((2, 3, 4), device="meta")
        out = torch.ops.aten.all_reduce(x, "sum", **self.get_world_trs())
        assert x.size() == out.size()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
