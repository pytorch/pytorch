#!/usr/bin/env pytest
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.dynamo
from torch.dynamo import config
from torch.dynamo.testing import same


class ToyModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, num_hidden=2, out_feat=5):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(5000, 5000), nn.ReLU()] * num_hidden
            + [nn.Linear(5000, 5), nn.ReLU()]
        )

    def forward(self, inputs):
        return self.net(inputs)


class CheckSplitsCompiler:
    def __init__(self):
        self.compiler_called = 0

    def compile_fn(self, gm, example_inputs):
        self.compiler_called += 1
        return gm


@pytest.mark.skip("Module hangs in PyTorch CI")
class TestDistributed(torch.dynamo.testing.TestCase):
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
        cls.device = f"cpu:{cls.rank}"
        cls.device_ids = None if "cpu" in cls.device else [cls.rank]
        dist.init_process_group("gloo", rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        super().tearDownClass()

    def get_model(self):
        m = ToyModel().to(self.device)
        inputs = torch.randn(20, 10).to(self.device)
        outputs = m(inputs)
        return m, inputs, outputs

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager(self):
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch.dynamo.optimize("aot_eager")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_inductor(self):
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch.dynamo.optimize("inductor")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    # can't run with gloo (no support for _allgather_base) and nccl not available in CI
    @pytest.mark.xfail
    @patch.object(config, "optimize_ddp", False)
    def test_fsdp_baseline_aot_eager(self):
        m, inputs, correct_outputs = self.get_model()
        fsdp_m = FSDP(m, device_id=self.device_ids[0] if self.device_ids else None)
        fsdp_m = torch.dynamo.optimize("aot_eager")(fsdp_m)
        outputs = fsdp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    # hangs/crashes with inductor currently
    @pytest.mark.skip
    @patch.object(config, "optimize_ddp", False)
    def test_fsdp_baseline_inductor(self):
        m, inputs, correct_outputs = self.get_model()
        fsdp_m = FSDP(m, device_id=self.device_ids[0] if self.device_ids else None)
        fsdp_m = torch.dynamo.optimize("inductor")(fsdp_m)
        outputs = fsdp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @pytest.mark.skipif(
        not hasattr(DDP, "_get_active_ddp_module"),
        reason="requires pytorch landing in parallel",
    )
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

        @torch.dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @pytest.mark.skipif(
        not hasattr(DDP, "_get_active_ddp_module"),
        reason="requires pytorch landing in parallel",
    )
    @patch.object(config, "optimize_ddp", True)
    def test_graph_split_inductor(self):
        """
        Same as above, but using inductor backend.
        We observed issues with inductor/fx interface in the past.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch.dynamo.optimize("inductor")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))

    @pytest.mark.skipif(
        not hasattr(DDP, "_get_active_ddp_module"),
        reason="requires pytorch landing in parallel",
    )
    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        """
        Ensures the DDPOptimizer returns a correct, compiled module without
        introducing graph splits. (Based on model parmeters fitting in the bucket)
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=250)

        check_splits_compiler = CheckSplitsCompiler()

        @torch.dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 1)

    @pytest.mark.skipif(
        not hasattr(DDP, "_get_active_ddp_module"),
        reason="requires pytorch landing in parallel",
    )
    @patch.object(config, "optimize_ddp", True)
    def test_aot_autograd(self):
        """
        Explicitly check AotAutograd family of compilers work,
        since they require example inputs propagated between graph splits.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch.dynamo.optimize("aot_nvfuser")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        opt_outputs.sum().backward()
        self.assertTrue(same(correct_outputs, opt_outputs))
