# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp.wrap import _FSDPPolicy, ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
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


class TestFSDPSharedParams(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    class ModelWithSharedParams(nn.Module):
        d_vocab = 23
        d_model = 16

        def __init__(self) -> None:
            super().__init__()
            d_vocab = self.d_vocab
            d_model = self.d_model
            self.embed_tokens = nn.Embedding(d_vocab, d_model)
            self.lin = nn.Linear(d_model, d_model)
            self.seq = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
            )
            self.output_proj = nn.Linear(d_model, d_vocab)
            self.relu = nn.ReLU()
            self.output_proj.weight = self.embed_tokens.weight

        def forward(self, x: torch.Tensor):
            z = self.embed_tokens(x)
            z = self.relu(self.lin(z))
            z = self.relu(self.seq(z))
            return self.output_proj(z)

    @skip_if_lt_x_gpu(2)
    def test_sibling_shared_params(self):
        """
        Tests the case of sibling shared parameters for ``fully_shard``. The
        sibling shared parameters should be assigned to their lowest common
        ancestor module, which differs from either sibling, and training should
        work.
        """
        # Wrap the embedding and output projection, which are siblings
        self.run_subtests(
            {
                "policy": [
                    ModuleWrapPolicy({nn.Linear, nn.Embedding, nn.Sequential}),
                    ModuleWrapPolicy(
                        {nn.Linear, nn.Embedding, self.ModelWithSharedParams}
                    ),
                ]
            },
            self._test_shared_params,
        )

    @skip_if_lt_x_gpu(2)
    def test_parent_child_shared_params(self):
        """
        Tests the case of parent-child shared parameters for ``fully_shard``.
        The parent-child shared parameters should be assigned to their lowest
        common ancestor, which is the parent, and training should work.
        """
        # Wrap the embedding (child) and `ModelWithSharedParams` (parent)
        self._test_shared_params(
            policy=ModuleWrapPolicy({nn.Embedding, self.ModelWithSharedParams})
        )

    def _test_shared_params(self, policy: _FSDPPolicy):
        composable_module = nn.Sequential(
            self.ModelWithSharedParams(),
            nn.Linear(
                self.ModelWithSharedParams.d_vocab, self.ModelWithSharedParams.d_vocab
            ),
        )
        ddp_module = DDP(
            copy.deepcopy(composable_module).cuda(), device_ids=[self.rank]
        )
        fully_shard(
            composable_module,
            process_group=self.process_group,
            policy=policy,
            device_id=torch.cuda.current_device(),
        )

        # Check that the shared embedding/output projection weight is flattened
        # once, meaning that it appears in only one `FlatParameter`'s `_params`
        flattened_count = sum(
            param is composable_module[0].output_proj.weight
            or param is composable_module[0].embed_tokens.weight
            for handle in fully_shard.state(composable_module)._handles
            for param in handle.flat_param._params
        )
        self.assertEqual(flattened_count, 1)

        # Check that we can running a few training iterations without erroring
        ddp_optim = torch.optim.Adam(ddp_module.parameters(), lr=1e-2)
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=1e-2)
        for i in range(4):
            losses = []
            for (module, optim) in (
                (ddp_module, ddp_optim),
                (composable_module, composable_optim),
            ):
                optim.zero_grad(set_to_none=(i % 2 == 0))
                inp = torch.arange(12, device=torch.device("cuda")).view(6, 2)
                loss = module(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
