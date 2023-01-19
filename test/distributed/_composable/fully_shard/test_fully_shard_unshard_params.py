# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed._composable.fully_shard import unshard_params
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.testing._internal.common_dist_composable import CompositeParamModel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestUnshardParams(FSDPTest):
    """Tests ``unshard_params`` for ``fully_shard``."""

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_unshard_params_param_data(self):
        """
        Tests that parameters are exposed correctly for ``recurse=True`` and
        all other argument configs for a non-FSDP root module.
        """
        self.run_subtests(
            {
                "rank0_only": [True, False],
                "offload_to_cpu": [True, False],
                "mixed_precision": [MixedPrecision(param_dtype=torch.float16), None],
                "cpu_offload": [
                    CPUOffload(offload_params=False),
                    CPUOffload(offload_params=True),
                ],
            },
            self._test_unshard_params_param_data,
        )

    def _test_unshard_params_param_data(
        self,
        rank0_only: bool,
        offload_to_cpu: bool,
        cpu_offload: CPUOffload,
        mixed_precision: Optional[MixedPrecision],
    ):
        model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(model)
        fully_shard_kwargs = {
            "cpu_offload": cpu_offload,
            "mixed_precision": mixed_precision,
        }
        fully_shard(composable_model.u1, **fully_shard_kwargs)
        fully_shard(composable_model.u2, **fully_shard_kwargs)
        non_fsdp_managed_param_names = {"l.weight", "l.bias", "p"}
        # TODO (awgu): How do we de-duplicate with wrapper FSDP unit test?
        with unshard_params(
            composable_model,
            rank0_only=rank0_only,
            writeback=not rank0_only,
            offload_to_cpu=offload_to_cpu,
        ):
            if not rank0_only or self.rank == 0:
                for p1, (n2, p2) in zip(
                    model.parameters(), composable_model.named_parameters()
                ):
                    if offload_to_cpu and n2 not in non_fsdp_managed_param_names:
                        self.assertEqual(torch.device("cpu"), p2.device)
                    else:
                        self.assertEqual(p1.device, p2.device)
                    self.assertEqual(p1.dtype, p2.dtype)
                    self.assertEqual(p1, p2)
                    self.assertTrue(isinstance(p2, nn.Parameter))
            else:
                # Check that each `FlatParameter` has the sharded size as a
                # proxy for it being resharded
                for handle in traversal_utils._get_fsdp_handles(composable_model):
                    if handle.uses_sharded_strategy:
                        self.assertEqual(
                            handle.flat_param.shape, handle.flat_param._sharded_size
                        )
                    else:
                        self.assertEqual(
                            handle.flat_param.shape,
                            handle.flat_param._unpadded_unsharded_size,
                        )


if __name__ == "__main__":
    run_tests()
