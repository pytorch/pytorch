# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.algorithms._comm_hooks import allreduce_hook
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

class TestCommunicationHooks(FSDPTest):

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD
        ])
    def test_default_communication_hook_behaviour(
        self,
        sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's default communication hook's behaviour and correctness.
        Arguments:
            has_wrapping (bool): Configures wrapping of a module.
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """
        m = torch.nn.Linear(1, 2, bias=False)
        inpt = torch.tensor([self.rank]).float().cuda(self.rank)

        net_default_hook = FSDP(
            copy.deepcopy(m),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(self.rank)

        # Check that default hook is set to `all_reduce`
        for entry in FSDP.fsdp_modules(net_default_hook):
            self.assertEqual(entry.communication_hook.__qualname__, allreduce_hook.allreduce_hook.__qualname__)

        for _ in range(4):

            # Clear gradients manually.
            for entry in net_default_hook.parameters():
                if entry.grad is not None:
                    entry.grad.requires_grad_(False)
                    entry.grad.zero_()

            loss = net_default_hook(inpt).sum()
            loss.backward()

            # For each worker, the gradient on the weight should be worker_rank.
            grad = [param.grad for param in net_default_hook.parameters()]

            avg = copy.deepcopy(grad[0])
            expected_grad = (
                sum(i for i in range(dist.get_world_size())) / dist.get_world_size()
            )
            # Verify default hook produces expected gradients
            self.assertEqual(
                avg[0].item(),
                expected_grad,
                msg=f"Expected hook grad of {expected_grad} but got {avg[0].item()}")

    @skip_if_lt_x_gpu(2)
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP
        ])
    def test_default_communication_hook_initialization(
        self,
        has_wrapping: bool,
        sharding_strategy: Optional[ShardingStrategy]
    ):

        """
        Tests FSDP's communication hook interface behaviour.
        Arguments:
            has_wrapping (bool): Configures wrapping of a module.
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        class DummyState(object):

            __slots__ = [
                "process_group"
            ]

            def __init__(self, process_group):
                self.process_group = process_group

        def dummy_hook(state: DummyState, grad: torch.Tensor):
            pass


        class Net(nn.Module):

            def __init__(self, has_wrapping):
                # to ensure determinizm
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                super().__init__()

                if has_wrapping:
                    self.net = FSDP(nn.Sequential(
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        FSDP(
                            nn.Linear(16, 8),
                            device_id=torch.cuda.current_device(),
                            sharding_strategy=sharding_strategy
                        )
                    ),
                        device_id=torch.cuda.current_device(),
                        sharding_strategy=sharding_strategy
                    )
                else:
                    self.net = nn.Sequential(
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8)
                    )

                self.out = nn.Linear(8, 4)

            def forward(self, x):
                return self.out(F.relu(self.net(x)))

        # Initialize the model and inputs
        device = torch.device("cuda")
        fsdp_model_with_hook = FSDP(
            Net(has_wrapping),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(device)

        dummy_state = DummyState(process_group=None)

        # FSDP currently suports communication hooks for a NO_SHARD strategy
        # Check that a `NotImplementedError`` is raised for other strategies
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            # Check that default hook is set to None
            for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertIsNone(entry.communication_hook)
                self.assertIsNone(entry.communication_hook_state)

            with self.assertRaises(
                NotImplementedError,
                msg="Communication hooks are currently only available for a NO_SHARD strategy."
            ):
                fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)

        else:

            # Check that default hook is set to `all_reduce`
            for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertEqual(entry.communication_hook.__qualname__, allreduce_hook.allreduce_hook.__qualname__)

            dummy_state = DummyState(process_group=None)
            fsdp_model_with_hook.register_comm_hook(
                dummy_state,
                dummy_hook
            )
            with self.assertRaises(AssertionError):
                fsdp_model_with_hook.register_comm_hook(
                    dummy_state,
                    dummy_hook
                )

            # Check dummy hook was registered for the root and all submodules if any
            for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertEqual(
                    entry.communication_hook.__qualname__,
                    dummy_hook.__qualname__
                )
                self.assertEqual(
                    entry.communication_hook_state,
                    dummy_state
                )


instantiate_parametrized_tests(TestCommunicationHooks)

if __name__ == "__main__":
    run_tests()
