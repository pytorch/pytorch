# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import sys
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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


@dataclass
class _GradAccConfig:
    """
    This configures how gradients are accumulated in :meth:`_test_grad_acc`.
    Each instance of this class represents ``num_iters``-many consecutive
    iterations, where the ``no_sync()`` context manager is used or not as given
    by ``use_no_sync``.

    Attributes:
        use_no_sync (bool): Indicates whether to use the ``no_sync()`` context
            manager as the way to accumulate gradients.
        num_iters (int): Number of iterations to accumulate gradients.
    """

    use_no_sync: bool
    num_iters: int

    def __repr__(self) -> str:
        # Override to remove any spaces in the string to appease the internal
        # build's test name parser
        return f"(use_no_sync={self.use_no_sync},num_iters={self.num_iters})"


@dataclass
class _GradAccConfigs:
    """
    This wraps a :class:`list` of :class:`_GradAccConfig` instances with the
    sole purpose of overriding :meth:`__repr__` to remove spaces.
    """

    configs: list[_GradAccConfig]

    def __repr__(self) -> str:
        # Override to remove any spaces in the string to appease the internal
        # build's test name parser
        return "[" + ",".join(config.__repr__() for config in self.configs) + "]"


class TestGradAcc(FSDPTestContinuous):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via both its
    ``no_sync()`` context manager and without the context manager."""

    @property
    def world_size(self) -> int:
        return 2

    def _test_grad_acc(
        self,
        batch_dim: int,
        configs: list[_GradAccConfig],
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        """
        Tests gradient accumulation by comparing a run that trains sequentially
        through some batches while accumulating gradients with a run that
        trains on the concatenation of those batches in a single iteration.

        The last iteration always synchronizes gradients regardless of what is
        specified by the last element of ``configs``.

        Arguments:
            batch_dim (int): Batch dimension in the input tensor to be passed
                into the model for the forward pass.
            configs (List[_GradAccConfig]): :class:`list` of configurations
                specifying how gradients are accumulated; for example, a list
                corresponding to [(False, 2), (True, 2), (False, 2)] indicates
                to accumulate over 2 + 2 + 2 = 6 total iterations, where the
                first two do not use ``no_sync()``, the middle two do use
                ``no_sync()``, and the final two again do not use
                ``no_sync()``.
            cpu_offload (CPUOffload): Configures CPU offloading.
            backward_prefetch (Optional[BackwardPrefetch]): Specifies at which
                point to prefetch the next layer's full parameters during the
                backward pass, if at all.
        """
        # Initialize the FSDP model and optimizer
        fsdp_kwargs = {
            "cpu_offload": cpu_offload,
            "backward_prefetch": backward_prefetch,
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
        }
        fsdp_model: FSDP = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs,
            deterministic=True,
            add_bn=False,  # disable BN since the test uses varying batch sizes
        )
        device = torch.device("cuda")
        optim = torch.optim.SGD(
            fsdp_model.parameters(),
            lr=0.01,
            momentum=0.9,
        )

        # Generate the sequence of batches, each containing the same data
        # but permuted
        def permute_tensor(x: torch.Tensor):
            return x.view(-1)[torch.randperm(x.numel())].view_as(x)

        batch: tuple[torch.Tensor, ...] = fsdp_model.module.get_input(device)
        batches: list[tuple[torch.Tensor, ...]] = [batch]
        num_iters_to_acc = sum(config.num_iters for config in configs)
        for _ in range(num_iters_to_acc - 1):
            batches.append(tuple(permute_tensor(t) for t in batch))
        for batch1, batch2 in itertools.combinations(batches, r=2):
            for t1, t2 in zip(batch1, batch2):
                if torch.all(t1 == t2):
                    raise AssertionError(
                        "Check the test to make sure that batches are distinct"
                    )

        # Concatenate the batches along the given batch dimension
        concat_batch: tuple[torch.Tensor, ...] = tuple(
            torch.cat(ts, dim=batch_dim) for ts in zip(*batches)
        )

        # Establish reference gradients using the concatenated batch
        fsdp_model.zero_grad()
        output = fsdp_model(*concat_batch)
        ref_loss = fsdp_model.module.get_loss(concat_batch, output)
        ref_loss.backward()
        ref_grads = [
            p.grad.detach().clone()
            for p in fsdp_model.parameters()
            if p.grad is not None
        ]

        # Compute and accumulate the gradients
        fsdp_model.zero_grad()
        losses = []
        batch_idx = 0
        for config in configs:
            sync_context = (
                fsdp_model.no_sync() if config.use_no_sync else contextlib.nullcontext()
            )
            with sync_context:
                for _ in range(config.num_iters):
                    if batch_idx == num_iters_to_acc - 1:
                        break  # always sync on the last iteration
                    batch = batches[batch_idx]
                    batch_idx += 1
                    output = fsdp_model(*batch)
                    loss = fsdp_model.module.get_loss(batch, output)
                    loss.backward()
                    losses.append(loss)
        output = fsdp_model(*batches[-1])
        loss = fsdp_model.module.get_loss(batches[-1], output)
        loss.backward()
        losses.append(loss)
        acc_loss = sum(losses)
        acc_grads = [
            p.grad.detach().clone()
            for p in fsdp_model.parameters()
            if p.grad is not None
        ]

        # Compare the losses and gradients
        torch.testing.assert_close(ref_loss, acc_loss)
        self.assertEqual(len(ref_grads), len(acc_grads))
        for ref_grad, acc_grad in zip(ref_grads, acc_grads):
            self.assertEqual(ref_grad.device, acc_grad.device)
            self.assertEqual(ref_grad.size(), acc_grad.size())
            self.assertEqual(ref_grad.dtype, acc_grad.dtype)
            torch.testing.assert_close(ref_grad, acc_grad)

        # Check that the optimizer step does not error
        optim.step()

    def _get_subtest_config(self) -> dict[str, list[Any]]:
        """Returns a subtest configuration that subtests prefetching."""
        return {
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
            "sharding_strategy": [
                ShardingStrategy.FULL_SHARD,
                ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.NO_SHARD,
            ],
        }

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "configs",
        [
            _GradAccConfigs(
                [
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                ]
            ),
            _GradAccConfigs(
                [
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                ]
            ),
        ],
    )
    @parametrize("use_orig_params", [False, True])
    def test_grad_acc(
        self,
        configs: _GradAccConfigs,
        use_orig_params: bool,
    ):
        """
        Tests gradient accumulation without parameter CPU offloading.

        This exercises gradient accumulation inside and outside the
        ``no_sync()`` context manager, in particular by interleaving the two.
        It tests both interleaving starting with (and ending with, resp.)
        inside versus outside ``no_sync()`` to ensure that initial conditions
        (and final conditions, resp.) do not affect the correctness.
        """
        subtest_config = self._get_subtest_config()
        subtest_config["cpu_offload"] = [CPUOffload(offload_params=False)]
        self.run_subtests(
            subtest_config,
            self._test_grad_acc,
            batch_dim=1,
            configs=configs.configs,
            use_orig_params=use_orig_params,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    def test_grad_acc_cpu_offload(
        self,
        use_orig_params: bool,
    ):
        """
        Tests gradient accumulation with parameter CPU offloading.

        NOTE: Gradient accumulation without using the ``no_sync()`` context
        manager is not currently compatible with CPU offloading.
        """
        # Only test `no_sync` since outside `no_sync()` is not supported with
        # parameter CPU offloading
        configs = _GradAccConfigs([_GradAccConfig(use_no_sync=True, num_iters=3)])
        subtest_config = self._get_subtest_config()
        subtest_config["cpu_offload"] = [CPUOffload(offload_params=True)]
        self.run_subtests(
            subtest_config,
            self._test_grad_acc,
            batch_dim=1,
            configs=configs.configs,
            use_orig_params=use_orig_params,
        )


instantiate_parametrized_tests(TestGradAcc)

if __name__ == "__main__":
    run_tests()
