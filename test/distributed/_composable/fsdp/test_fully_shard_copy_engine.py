# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    CopyEngineAllGather,
    DefaultAllGather,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import MLP
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


@requires_nccl_version((2, 28), "Need NCCL 2.28+ for CE collectives")
@requires_cuda_p2p_access()
class TestCopyEngineAllGather(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> Optional[str]:
        return "nccl"

    @classmethod
    def opts(cls) -> Optional[dist.ProcessGroupNCCL.Options]:
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        return opts

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @skip_if_lt_x_gpu(2)
    def test_copy_engine_all_gather_fsdp(self):
        torch.cuda.set_device(self.rank)
        symm_mem.set_backend("NCCL")
        # Warmup NCCL communicator
        dist.all_reduce(torch.ones(1, device=self.device))

        torch.manual_seed(42)
        ref_model = MLP(16, device=self.device)
        ref_model = copy.deepcopy(ref_model)

        torch.manual_seed(42)
        model = MLP(16, device=self.device)
        fully_shard(model)
        model.set_copy_engine_all_gather(True)

        torch.manual_seed(self.rank)
        inp = torch.randn(4, 16, device=self.device)

        ref_out = ref_model(inp)
        out = model(inp)
        torch.testing.assert_close(out, ref_out)

        ref_out.sum().backward()
        out.sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_copy_engine_all_gather_toggle(self):
        torch.cuda.set_device(self.rank)
        symm_mem.set_backend("NCCL")
        dist.all_reduce(torch.ones(1, device=self.device))

        torch.manual_seed(42)
        model = MLP(16, device=self.device)
        fully_shard(model)

        model.set_copy_engine_all_gather(True)
        state = model._get_fsdp_state()
        for pg in state._fsdp_param_groups:
            self.assertIsInstance(pg._all_gather_comm, CopyEngineAllGather)

        model.set_copy_engine_all_gather(False)
        for pg in state._fsdp_param_groups:
            self.assertIsInstance(pg._all_gather_comm, DefaultAllGather)


if __name__ == "__main__":
    run_tests()
