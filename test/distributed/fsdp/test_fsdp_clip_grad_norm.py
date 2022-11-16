# Owner(s): ["oncall: distributed"]

import itertools
import sys
from typing import Union

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


class TestClipGradNorm(FSDPTest):
    """Tests :meth:`FullyShardedDataParallel.clip_grad_norm_`."""

    @skip_if_lt_x_gpu(2)
    def test_non_root(self):
        """
        Tests that calling ``clip_grad_norm_()`` on a non-root FSDP instance
        raises an error.
        """

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5)
                self.lin2 = nn.Linear(5, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin2(self.lin1(x))

        model = Model().cuda()
        model.lin2 = FSDP(model.lin2)
        fsdp_model = FSDP(model)
        fsdp_model(torch.randn((2, 5), device=torch.device("cuda"))).sum().backward()
        error_regex = "should only be called on the root FSDP instance"
        with self.assertRaisesRegex(RuntimeError, error_regex):
            fsdp_model.lin2.clip_grad_norm_(max_norm=2)

    @skip_if_lt_x_gpu(2)
    def test_ddp_parity(self):
        """
        Tests FSDP with ``FullyShardedDataParallel.clip_grad_norm_()` against
        DDP with ``torch.nn.utils.clip_grad_norm_()`.
        """
        self.run_subtests(
            {
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                    "mixed_strategy",
                ],
                "use_orig_params": [False, True],
                "offload_params": [False, True],
            },
            self._test_ddp_parity,
        )

    def _test_ddp_parity(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        sharding_strategy: Union[ShardingStrategy, str],
        use_orig_params: bool,
        offload_params: bool,
    ):
        local_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(local_model, device_ids=[self.rank])
        fsdp_kwargs = {
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "use_orig_params": use_orig_params,
        }
        if sharding_strategy == "mixed_strategy":
            fsdp_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
            )
            # Apply `NO_SHARD` to the encoder
            fsdp_model.transformer.encoder = FSDP(
                fsdp_model.transformer.encoder,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                **fsdp_kwargs,
            )
            # Apply `FULL_SHARD` to the decoder
            fsdp_model.transformer.decoder = FSDP(
                fsdp_model.transformer.decoder,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                **fsdp_kwargs,
            )
            # TODO: FSDP's `clip_grad_norm_()` is not a static method, so we
            # must make the root module an FSDP instance
            fsdp_model = FSDP(
                fsdp_model, sharding_strategy=ShardingStrategy.FULL_SHARD, **fsdp_kwargs
            )
        else:
            fsdp_kwargs.update(
                {
                    "sharding_strategy": sharding_strategy,
                    "auto_wrap_policy": ModuleWrapPolicy(
                        {
                            TransformerEncoderLayer,
                            TransformerDecoderLayer,
                        }
                    ),
                }
            )
            fsdp_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
                fsdp_kwargs=fsdp_kwargs,
            )
        LR = 1e-2
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device("cuda")
        LARGE_FACTOR = 100
        inp = ddp_model.module.get_input(device)
        for model in (ddp_model, fsdp_model):
            out = model(*inp)
            if isinstance(model, (DDP, FSDP)):
                loss = model.module.get_loss(inp, out)
            else:
                loss = model.get_loss(inp, out)
            loss.backward()

        # Multiply gradients by a large factor to ensure that gradients will
        # actually be clipped
        for param in itertools.chain(ddp_model.parameters(), fsdp_model.parameters()):
            if (
                param.grad is not None
            ):  # gradients may be `None` for `use_orig_params=True`
                param.grad *= LARGE_FACTOR
        orig_ddp_grads = [
            param.grad.detach().clone() for param in ddp_model.parameters()
        ]
        orig_fsdp_grads = [
            param.grad.detach().clone() if param.grad is not None else None
            for param in fsdp_model.parameters()
        ]

        ddp_total_norm = torch.nn.utils.clip_grad_norm_(
            ddp_model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
        )
        fsdp_total_norm = fsdp_model.clip_grad_norm_(
            max_norm=max_norm, norm_type=norm_type
        )
        self.assertEqual(ddp_total_norm, fsdp_total_norm)

        # Check that the gradients were modified by `clip_grad_norm_()`
        for param, orig_grad in zip(ddp_model.parameters(), orig_ddp_grads):
            assert not torch.equal(param.grad, orig_grad)
        for param, orig_grad in zip(fsdp_model.parameters(), orig_fsdp_grads):
            if param.grad is None:
                self.assertEqual(param.grad, orig_grad)  # `None`
            else:
                assert not torch.equal(param.grad, orig_grad)

        # Run an optimizer step to ensure gradients matched after clipping
        ddp_optim.step()
        fsdp_optim.step()
        with FSDP.summon_full_params(fsdp_model):
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(),
                fsdp_model.named_parameters(),
            ):
                self.assertEqual(n1, n2)
                self.assertEqual(p1, p2)


instantiate_parametrized_tests(TestClipGradNorm)

if __name__ == "__main__":
    run_tests()
