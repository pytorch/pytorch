# Owner(s): ["oncall: distributed"]

import copy

from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint, replicate

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeModel,
    CompositeParamModel,
)
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardCompose(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_replicate_and_fully_shard(self):
        self.run_subtests(
            {
                "config": [
                    "1fm,1r",
                    "1r,1fm",
                    "1r1fm,1fm",
                    "1fm1fm,1r1r,1fm",
                ]
            },
            self._test_replicate_and_fully_shard,
        )

    def _test_replicate_and_fully_shard(
        self,
        config: str,
    ):
        """
        To interpret the config, each comma delineates a level in the module
        tree ordered bottom-up; 'r' means ``replicate``; 'f' means
        ``fully_shard``; 'a' means auto wrap; and 'm' means manual wrap.

        TODO: We have not implemented auto wrapping yet, so we exclude those
        configs.
        """
        torch.manual_seed(42)
        if config == "1fm,1r":
            model = CompositeModel(device=torch.device("cpu"))
            ref_model = copy.deepcopy(model)
            fully_shard(model.l1)
            replicate(model.cuda())
        elif config == "1r,1fm":
            model = CompositeParamModel(torch.device("cpu"))
            ref_model = copy.deepcopy(model)
            replicate(model.u1.cuda())
            fully_shard(model)
        elif config == "1r1fm,1fm":
            model = CompositeParamModel(torch.device("cpu"))
            ref_model = copy.deepcopy(model)
            replicate(model.u1.cuda())
            fully_shard(model.u2)
            fully_shard(model)
        elif config == "1fm1fm,1r1r,1fm":
            model = CompositeParamModel(torch.device("cpu"))
            ref_model = copy.deepcopy(model)
            fully_shard(model.u1.seq)
            fully_shard(model.u2.seq)
            replicate(model.u1.cuda())
            replicate(model.u2.cuda())
            fully_shard(model)
        else:
            raise ValueError(f"Unknown config: {config}")
        replicate(ref_model.cuda())
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(self.rank + 42)
        inp = torch.randn((2, 100), device="cuda")
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_and_fully_shard(self):
        self.run_subtests(
            {
                "checkpoint_strict_submodule": [False, True],
            },
            self._test_checkpoint_and_fully_shard,
        )

    def _test_checkpoint_and_fully_shard(self, checkpoint_strict_submodule: bool):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        x = torch.zeros(2, 100, device="cuda")

        if checkpoint_strict_submodule:
            checkpoint(model.c2.l)
        else:
            checkpoint(model.c2)
        fully_shard(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        fully_shard(model)

        loss = model(x).sum()
        loss.backward()

        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        # Notably, check that the recomputed forward preserves the right dtype
        # (if not checkpointing the strict submodule)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)


if __name__ == "__main__":
    run_tests()
