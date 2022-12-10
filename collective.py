import torch
import torch.distributed as dist
import torch._dynamo
from contextlib import contextmanager
import os
import logging
torch._dynamo.config.log_level = logging.DEBUG
@contextmanager
def _per_rank_init(rank, world_size, backend="nccl"):
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16789'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    dist.destroy_process_group()

class Tester():
    def __init__(self):
        self.rank = 0
        self.world_size = 1
    def test_trace_allreduce(self):
        """
        Confirm allreduce op shows up in the graph, and ProcessGroup guards work
        """
        class CheckOpCompiler:
            """
            Records how many times the compiler is called,
            and if the graph contains a particular op
            """
            def __init__(self, op_target):
                self.compiler_called = 0
                self.graphs_contain_op = 0
                self.op_target = op_target

            def compile_fn(self, gm, example_inputs):
                self.compiler_called += 1
                for node in gm.graph.nodes:
                    if node.target == self.op_target:
                        self.graphs_contain_op += 1
                        break
                return gm

        input = torch.rand(20, 20).cuda(self.rank)
        check_compiler = CheckOpCompiler(dist.all_reduce)

        def test_fn(input: torch.Tensor) -> int:
            tensor = input / (self.rank + 1)
            dist.all_reduce(tensor, group=dist.group.WORLD)
            return tensor

        opt_fn = torch._dynamo.optimize(check_compiler.compile_fn)(test_fn)
        with _per_rank_init(self.rank, self.world_size, backend="gloo"):
            ref_out = test_fn(input)
            # repeated calls with the same PG reuse the compiled code
            for _ in range(2):
                opt_out = opt_fn(input)
                # self.assertTrue(same(ref_out, opt_out))
                # self.assertEqual(1, check_compiler.compiler_called)
                # self.assertEqual(1, check_compiler.graphs_contain_op)

        # with _per_rank_init(self.rank, self.world_size, backend="nccl"):
        #     ref_out = test_fn(input, dist.group.WORLD)
        #     # a new PG causes one recompilation
        #     for _ in range(2):
        #         opt_out = opt_fn(input, dist.group.WORLD)
        #         self.assertTrue(same(ref_out, opt_out))
        #         self.assertEqual(2, check_compiler.compiler_called)
        #         self.assertEqual(2, check_compiler.graphs_contain_op)

    def test_meta(self):
        x = torch.empty((2,3,4), device="meta")
        with _per_rank_init(self.rank, self.world_size, backend="gloo"):
            dist.all_reduce(x, group=dist.group.WORLD, async_op=True)

t = Tester()
# t.test_trace_allreduce()
t.test_meta()