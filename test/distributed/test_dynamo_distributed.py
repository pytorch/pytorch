# Owner(s): ["module: dynamo"]
import os
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.distributed as dist
from torch import nn
from torch._dynamo import config
from torch._dynamo.utils import same
from torch._inductor.utils import has_triton
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import requires_nccl

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ToyModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, num_hidden=2, out_feat=5):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()] * num_hidden
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )

    def forward(self, inputs):
        return self.net(inputs)


class CheckSplitsCompiler:
    def __init__(self):
        self.compiler_called = 0

    def compile_fn(self, gm, example_inputs):
        self.compiler_called += 1
        return gm


@requires_nccl()
class TestDistributed(torch._dynamo.test_case.TestCase):
    """
    Test harness initializes dist process group
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # _exit_stack is set up in TestCase
        cls._exit_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                },
            )
        )
        cls.rank = 0
        cls.device = f"cuda:{cls.rank}"
        cls.device_ids = None if "cuda" in cls.device else [cls.rank]
        dist.init_process_group("nccl", rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        super().tearDownClass()

    def get_model(self, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5):
        m = ToyModel(in_feat=in_feat, hidden_feat=hidden_feat, out_feat=out_feat).to(self.device)
        m.apply(init_weights)
        inputs = torch.rand(bsz, in_feat).to(self.device)
        outputs = m(inputs)
        return m, inputs, outputs

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager(self):
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("aot_eager")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_inductor(self):
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("inductor")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    # TODO(whc) move these tests to 'distributed' shard to get nccl, or see if it's available already in pytorch CI?
    @unittest.skip(
        "can't run with gloo (no support for _allgather_base) and nccl not available in CI"
    )
    @patch.object(config, "optimize_ddp", False)
    def test_fsdp_baseline_aot_eager(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        m, inputs, correct_outputs = self.get_model()
        fsdp_m = FSDP(m, device_id=self.device_ids[0] if self.device_ids else None)
        fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
        outputs = fsdp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @unittest.skip("hangs/crashes with inductor currently")
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", False)
    def test_fsdp_baseline_inductor(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        m, inputs, correct_outputs = self.get_model()
        fsdp_m = FSDP(m, device_id=self.device_ids[0] if self.device_ids else None)
        fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
        outputs = fsdp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split(self):
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor(self):
        """
        Same as above, but using inductor backend.
        We observed issues with inductor/fx interface in the past.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("inductor")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        """
        Ensures the DDPOptimizer returns a correct, compiled module without
        introducing graph splits. (Based on model parmeters fitting in the bucket)
        """
        # DDP will always do a 'first bucket' with a really small size;  so only a tiny model will escape this
        m, inputs, correct_outputs = self.get_model(hidden_feat=5)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=250)
        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 1)

    @patch.object(config, "optimize_ddp", True)
    def test_aot_autograd(self):
        """
        Explicitly check AotAutograd family of compilers work,
        since they require example inputs propagated between graph splits.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("aot_eager")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        opt_outputs.sum().backward()
        self.assertTrue(same(correct_outputs, opt_outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_custom_layer(self):
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """

        class MyCustomLinear(torch.nn.Module):
            def __init__(self):
                super(MyCustomLinear, self).__init__()
                self.weight = nn.Parameter(torch.randn(512, 512))

            def forward(self, x):
                return torch.mm(x, self.weight.t())

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super(MyLinear, self).__init__()
                self.linear = torch.nn.Linear(512, 512)

            def forward(self, x):
                return self.linear(x)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                mods = [
                    (MyLinear(), torch.nn.ReLU()),
                    # sandwitch the custom in the middle so it comes before and after
                    (MyCustomLinear(), torch.nn.ReLU()),
                    (MyLinear(), torch.nn.ReLU()),
                ]
                self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

            def forward(self, x):
                return self.seq(x)

        m = MyModule().to(self.device)
        m.apply(init_weights)
        inputs = torch.rand((512, 512)).to(self.device)
        correct_outputs = m(inputs)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=1)

        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_empty_graph_inductor(self):
        def fn():
            get_world_size = torch.distributed.distributed_c10d.get_world_size()
            return (get_world_size,)

        opt_fn = torch._dynamo.optimize("inductor")(fn)
        res = None
        try:
            res = opt_fn()[0]
        except Exception:
            pass
        self.assertEqual(res, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
