# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    NestedWrappedModule,
    FSDPInitMode,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFSDPMisc(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_diff_num_params(self):
        class DiffNumParams(nn.Module):
            def __init__(self, rank):
                super().__init__()
                # non-zero ranks have one more param.
                self.lin = torch.nn.Linear(10, 10, bias=False)
                if rank != 0:
                    self.lin2 = torch.nn.Linear(10, 10, bias=False)

        m = DiffNumParams(self.rank)
        with self.assertRaisesRegex(
            RuntimeError, "expects same model across all ranks"
        ):
            FSDP(m)
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_diff_param_shape(self):
        # create nccl group with small timeout
        class DiffShapeParams(nn.Module):
            def __init__(self, rank):
                # non-zero rank have diff embedding shape.
                super().__init__()
                dim = 10 if rank == 0 else 20
                self.e = nn.Embedding(10, dim)

        m = DiffShapeParams(self.rank)
        with self.assertRaisesRegex(RuntimeError, "appears not to match sizes"):
            FSDP(m)

        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_same_model_across_ranks(self):
        """
        FSDP broadcasts model from rank 0 to ensure it starts off with the same
        values.
        """
        class MyModel(nn.Module):
            def __init__(self, rank):
                super().__init__()
                torch.manual_seed(rank)
                torch.cuda.manual_seed(rank)
                self.lin = nn.Linear(10, 10, bias=False)

        m = MyModel(self.rank).cuda()

        def _validate(model, equal):
            tensor = next(model.parameters()).data
            tlist = [
                torch.empty_like(tensor) for _ in range(
                    dist.get_world_size(self.process_group)
                )
            ]
            dist.all_gather(tlist, tensor, group=self.process_group)
            rank0_tensor = tlist[0]
            assert_fn = self.assertEqual if equal else self.assertNotEqual
            for t in tlist[1:]:
                assert_fn(t, rank0_tensor)

        _validate(m, equal=False)
        # FSDP makes the model the same during init
        fsdp = FSDP(m)
        with fsdp.summon_full_params(fsdp):
            _validate(m, equal=True)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_init_stays_on_cpu(self):
        """
        Ensure that CPU model input stays on CPU
        after FSDP init even though sharding, flattening
        is run on GPU.
        """
        torch.cuda.set_device(self.rank)
        mod = NestedWrappedModule(
            group=self.process_group,
            wrap_fsdp=True,
            wrap_everything=True,
            fsdp_init_mode=FSDPInitMode.CUDA_NEVER,
        )
        regex = "Module is input on CPU"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            fsdp = FSDP(mod)
        devices = {p.device for p in fsdp.parameters()}
        self.assertEqual(1, len(devices))
        self.assertEqual(torch.device("cpu"), devices.pop())
        fsdp = fsdp.cuda()
        # Ensure fwd + backward can be performed after moving to CUDA.
        inp = mod.get_input(device=torch.cuda.current_device())
        fsdp(inp[0]).sum().backward()



if __name__ == "__main__":
    run_tests()
