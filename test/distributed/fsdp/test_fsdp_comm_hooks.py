# Owner(s): ["oncall: distributed"]

import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import (
    requires_nccl,
    requires_nccl_version,
    skip_but_pass_in_sandcastle_if,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# bfloat16 is only supported by CUDA 11+
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    (torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 11)
    or torch.version.hip is not None
)


class Net(nn.Module):
    def __init__(self, has_wrapping, sharding_strategy, mixed_precision=None):
        # to ensure determinism
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()

        if has_wrapping:
            self.net = FSDP(
                nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    FSDP(
                        nn.Linear(16, 8),
                        device_id=torch.cuda.current_device(),
                        sharding_strategy=sharding_strategy,
                        mixed_precision=mixed_precision,
                    ),
                ),
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
            )
        else:
            self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(F.relu(self.net(x)))


class DummyState:
    __slots__ = ["process_group", "noise"]

    def __init__(self, process_group: dist.ProcessGroup, noise: int):
        self.process_group = process_group
        self.noise = noise


class DummyHook:
    def dummy_hook_for_no_shard_fsdp(self, state: DummyState, grad: torch.Tensor):
        """
        This communication hook is for illustration and testing purpose only.
        This communication hook is used during FSDP ``NO_SHARD`` training. It adds some noise to
        the provided ``grad`` parameter and uses ``all_reduce`` to communicate full, flattened,
        unsharded gradient.
        """
        grad.add_(state.noise)
        dist.all_reduce(grad, group=state.process_group)

    def custom_reduce_scatter(self, output, input, group=None):
        """
        This function is for illustrative purpose only.
        It is meant to implement a custom reduce-scatter
        of a flattened tensor to all processes in a group.
        Currently a no-op.
        """
        pass

    def dummy_hook_for_sharded_fsdp(
        self, state: DummyState, grad: torch.Tensor, output: torch.Tensor
    ):
        """
        This communication hook is for illustration and testing purposes only.
        This communication hook is used during FSDP ``FULL_SHARD`` or ``SHARD_GRAD_OP`` training.
        It adds some noise to the provided ``grad`` parameter, uses
        ``reduce_scatter`` for gradient communication and stores a sharded gradient in ``output``.
        """
        grad.add_(state.noise)
        self.custom_reduce_scatter(output, grad, group=state.process_group)


class TestCommunicationHooks(FSDPTest):
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_default_communication_hook_behavior(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's default communication hook's behavior and correctness.
        This test creates a simple linear net with weight shape  ``1 X N``,
        where ``N`` is the number of workers.
        For sharded cases, each worker gets 1 element of the weight parameter. This test
        checks that after backward, each worker has a proper value in its chunk of
        the gradient, or the whole gradient on every worker is equal to an expected value.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """
        out_dim = self.world_size
        net = torch.nn.Linear(1, out_dim, bias=False)
        inpt = torch.tensor([self.rank]).float().cuda(self.rank)

        net_default_hook = FSDP(
            net,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
        ).to(self.rank)

        # Check that by default, `_comm_hook` is None
        for entry in FSDP.fsdp_modules(net_default_hook):
            self.assertEqual(entry._comm_hook, None)

        for _ in range(4):
            # Clear gradients
            net_default_hook.zero_grad()
            loss = net_default_hook(inpt).sum()
            loss.backward()

            # For each worker, the gradient on the weight should be worker_rank.
            grad = net_default_hook.params[0].grad
            expected_grad = (
                sum(i for i in range(dist.get_world_size())) / dist.get_world_size()
            )
            # Verify default hook produces expected gradients
            self.assertEqual(
                grad[0].item(),
                expected_grad,
                msg=f"Expected hook grad of {expected_grad} but got {grad[0].item()}",
            )

    def _get_submodules(self, fsdp_net):
        return [
            submodule
            for submodule in FSDP.fsdp_modules(fsdp_net)
            if not submodule.check_is_root()
        ]

    def _init_model(self, core, sharding_strategy, mixed_precision=None):
        device = torch.device("cuda")
        return FSDP(
            core,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
        ).to(device)

    @skip_if_lt_x_gpu(2)
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_default_communication_hook_initialization(
        self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's communication hook interface behavior.

        Arguments:
            has_wrapping (bool): Configures wrapping of a module.
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        # Initialize a model
        fsdp_model_with_hook = self._init_model(
            Net(has_wrapping=has_wrapping, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )

        # Check that by default, `_comm_hook` is None
        for fsdp_module in FSDP.fsdp_modules(fsdp_model_with_hook):
            self.assertEqual(fsdp_module._comm_hook, None)

        dummy_state = DummyState(process_group=None, noise=1234)
        dummy_hook = (
            DummyHook.dummy_hook_for_no_shard_fsdp
            if sharding_strategy != ShardingStrategy.NO_SHARD
            else DummyHook.dummy_hook_for_sharded_fsdp
        )

        fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)

        # Check that we can't register comm hook twice
        with self.assertRaisesRegex(
            AssertionError, "^A communication hook is already registered$"
        ):
            fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)

        # Check dummy hook was registered for the root and all submodules if any
        for fsdp_module in FSDP.fsdp_modules(fsdp_model_with_hook):
            self.assertEqual(fsdp_module._comm_hook, dummy_hook)
            self.assertEqual(fsdp_module._comm_hook_state, dummy_state)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_registering_hook_non_root(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's communication hook registering for submodules.
        Make sure it can't be registered for non-root submodules.
        Currently tests only ``NO_SHARD`` strategy.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        fsdp_model_with_hook = self._init_model(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        dummy_state = DummyState(process_group=None, noise=1234)
        dummy_hook = (
            DummyHook.dummy_hook_for_no_shard_fsdp
            if sharding_strategy != ShardingStrategy.NO_SHARD
            else DummyHook.dummy_hook_for_sharded_fsdp
        )
        # Creating a list of non-root submodules to test
        submodules = self._get_submodules(fsdp_model_with_hook)
        # Check that assertion is raised for registering a comm hook on a non-root
        with self.assertRaisesRegex(
            AssertionError,
            "^register_comm_hook can only be called on a root instance.$",
        ):
            submodules[1].register_comm_hook(dummy_state, dummy_hook)

    @skip_if_lt_x_gpu(2)
    def test_registering_hook_hybrid_strategy(self):
        for sharding_strategy in (
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ):
            model = Net(False, None, None).cuda()
            fsdp_model = FSDP(
                model,
                auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                sharding_strategy=sharding_strategy,
            )
            dummy_state = DummyState(process_group=None, noise=1234)
            dummy_hook = DummyHook.dummy_hook_for_sharded_fsdp
            with self.assertRaisesRegex(
                AssertionError,
                "Communication hook is not supported for hybrid strategies",
            ):
                fsdp_model.register_comm_hook(dummy_state, dummy_hook)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_registering_hook_submodules(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        """
        Tests FSDP's communication hook registering for submodules.
        Checks behavior if a hook was registered for a non-root submodule
        Currently tests only ``NO_SHARD`` strategy.

        Arguments:
            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        fsdp_model_with_hook = self._init_model(
            Net(has_wrapping=True, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        dummy_state = DummyState(process_group=None, noise=1234)
        dummy_hook = (
            DummyHook.dummy_hook_for_no_shard_fsdp
            if sharding_strategy != ShardingStrategy.NO_SHARD
            else DummyHook.dummy_hook_for_sharded_fsdp
        )
        submodules = self._get_submodules(fsdp_model_with_hook)

        # Simulate a registration of a hook on a submodule
        submodules[1]._comm_hook = dummy_hook
        # Check that an error is raised when some of submodules have a non-default hook assigned
        with self.assertRaisesRegex(
            AssertionError, "^A communication hook is already registered$"
        ):
            fsdp_model_with_hook.register_comm_hook(dummy_state, dummy_hook)

    def _check_low_precision_hook(
        self, state, hook, sharding_strategy, dtype, has_wrapping
    ):
        # keep everything deterministic for input data
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        fsdp_with_hook = self._init_model(
            Net(has_wrapping=has_wrapping, sharding_strategy=sharding_strategy),
            sharding_strategy=sharding_strategy,
        )
        fsdp_with_hook.register_comm_hook(state, hook)

        mp_only_grad = MixedPrecision(reduce_dtype=dtype)
        fsdp_with_mp = self._init_model(
            Net(
                has_wrapping=has_wrapping,
                sharding_strategy=sharding_strategy,
                mixed_precision=mp_only_grad,
            ),
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_only_grad,
        )

        optim_hook = torch.optim.SGD(fsdp_with_hook.parameters(), lr=0.1)
        optim_mp = torch.optim.SGD(fsdp_with_mp.parameters(), lr=0.1)

        in_data = torch.rand(16, 8).cuda()
        fsdp_with_hook.train()
        fsdp_with_mp.train()
        loss_hook = fsdp_with_hook(in_data).sum()
        loss_mp = fsdp_with_mp(in_data).sum()
        loss_hook.backward()
        # Make sure grads were cast to the parameter's precision
        self.assertEqual(fsdp_with_hook.params[0].grad.dtype, state.parameter_type)
        loss_mp.backward()
        optim_hook.step()
        optim_mp.step()

        dist.barrier()

        for hook_param, mp_param in zip(
            fsdp_with_hook.parameters(), fsdp_with_mp.parameters()
        ):
            self.assertEqual(hook_param.grad, mp_param.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_fp16_hook(
        self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
    ):
        state = default_hooks.LowPrecisionState(process_group=_get_default_group())
        hook = default_hooks.fp16_compress_hook

        self._check_low_precision_hook(
            state, hook, sharding_strategy, torch.float16, has_wrapping
        )

    @requires_nccl()
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ],
    )
    def test_bf16_hook(
        self, has_wrapping: bool, sharding_strategy: Optional[ShardingStrategy]
    ):
        state = default_hooks.LowPrecisionState(process_group=_get_default_group())
        hook = default_hooks.bf16_compress_hook

        self._check_low_precision_hook(
            state, hook, sharding_strategy, torch.bfloat16, has_wrapping
        )


instantiate_parametrized_tests(TestCommunicationHooks)

if __name__ == "__main__":
    run_tests()
