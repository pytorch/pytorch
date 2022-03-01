# Owner(s): ["oncall: distributed"]

import contextlib
import itertools
import sys
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import distributed as dist
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
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


class _GradAccConfig(NamedTuple):
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


class TestGradAcc(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via both its
    ``no_sync()`` context manager and without the context manager."""

    def _test_grad_acc(
        self,
        batch_dim: int,
        configs: List[_GradAccConfig],
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
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
        # Gradient accumulation without `no_sync()` is not currently compatible
        # with CPU offloading
        if cpu_offload.offload_params and \
                any(not config.use_no_sync for config in configs):
            return
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            # Disable TF32 to prevent floating point drift
            torch.backends.cuda.matmul.allow_tf32 = False

            # Initialize the FSDP model and optimizer
            group = dist.distributed_c10d._get_default_group()
            fsdp_model: FSDP = self._get_wrapped_model(
                group, cuda_first=False, add_bn=False,
                config={
                    "cpu_offload": cpu_offload,
                    "backward_prefetch": backward_prefetch
                },
            )  # disable BN since the test uses varying batch sizes
            fsdp_model.eval()  # disable dropout
            device = torch.device("cuda")
            optim = torch.optim.SGD(
                fsdp_model.parameters(), lr=0.01, momentum=0.9,
            )

            # Generate the sequence of batches, each containing the same data
            # but permuted
            def permute_tensor(x: torch.Tensor):
                return x.view(-1)[torch.randperm(x.numel())].view_as(x)

            batch: Tuple[torch.Tensor, ...] = \
                fsdp_model.module.get_input(device)
            batches: List[Tuple[torch.Tensor, ...]] = [batch]
            num_iters_to_acc = sum(config.num_iters for config in configs)
            for _ in range(num_iters_to_acc - 1):
                batches.append(tuple(permute_tensor(t) for t in batch))
            for (batch1, batch2) in itertools.combinations(batches, r=2):
                for t1, t2 in zip(batch1, batch2):
                    assert not torch.all(t1 == t2)

            # Concatenate the batches along the given batch dimension
            concat_batch: Tuple[torch.Tensor, ...] = tuple(
                torch.cat(ts, dim=batch_dim) for ts in zip(*batches)
            )

            # Run twice to check that everything works even after the first
            # gradient synchronization
            for _ in range(2):
                # Establish reference gradients using the concatenated batch
                fsdp_model.zero_grad()
                output = fsdp_model(*concat_batch)
                ref_loss = fsdp_model.module.get_loss(concat_batch, output)
                ref_loss.backward()
                ref_grads = [
                    p.grad.detach().clone() for p in fsdp_model.parameters()
                ]

                # Compute and accumulate the gradients
                fsdp_model.zero_grad()
                losses = []
                batch_idx = 0
                for config in configs:
                    context = fsdp_model.no_sync() if config.use_no_sync \
                        else contextlib.suppress()
                    with context:
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
                    p.grad.detach().clone() for p in fsdp_model.parameters()
                ]

                # Compare the losses and gradients
                torch.testing.assert_close(ref_loss, acc_loss)
                assert len(ref_grads) == len(acc_grads)
                for ref_grad, acc_grad in zip(ref_grads, acc_grads):
                    assert ref_grad.device == acc_grad.device
                    assert ref_grad.size() == acc_grad.size()
                    assert ref_grad.dtype == acc_grad.dtype
                    torch.testing.assert_close(ref_grad, acc_grad)

                # Check that the optimizer step does not error
                optim.step()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "configs",
        [
            [_GradAccConfig(use_no_sync=True, num_iters=4)],
            [_GradAccConfig(use_no_sync=False, num_iters=4)],
            [
                _GradAccConfig(use_no_sync=True, num_iters=2),
                _GradAccConfig(use_no_sync=False, num_iters=2),
                _GradAccConfig(use_no_sync=True, num_iters=2),
            ],
            [
                _GradAccConfig(use_no_sync=False, num_iters=2),
                _GradAccConfig(use_no_sync=True, num_iters=2),
                _GradAccConfig(use_no_sync=False, num_iters=2),
            ],
        ]
    )
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize(
        "backward_prefetch",
        [BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST, None]
    )
    def test_grad_acc(
        self,
        configs: List[_GradAccConfig],
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
    ):
        """
        Tests gradient accumulation.

        This exercises gradient accumulation using the ``no_sync()`` context
        manager, without using the ``no_sync()`` context manager, and
        interleaving using and not using the ``no_sync()`` context manager. For
        the interleaving, we test the case where the final iteration before the
        gradient synchronization is in the ``no_sync()`` context and the case
        where it is not, which is why we have four elements in the ``configs``
        list. This test also checks for compatibility with the CPU offload and
        backward prefetch options.

        NOTE: Gradient accumulation without using the ``no_sync()`` context
        manager is not currently compatible with CPU offloading, so those tests
        are skipped.
        """
        self._test_grad_acc(
            batch_dim=1,
            configs=configs,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
        )


instantiate_parametrized_tests(TestGradAcc)

if __name__ == "__main__":
    run_tests()
