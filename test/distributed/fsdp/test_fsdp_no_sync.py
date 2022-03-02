# Owner(s): ["oncall: distributed"]

import itertools
import sys
from typing import List, Optional, Tuple

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


class TestNoSync(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via its
    ``no_sync()`` context manager."""

    def _test_no_sync(
        self,
        batch_dim: int,
        num_iters_to_acc: int,
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
    ):
        """
        Tests ``no_sync()`` by comparing a run that trains sequentially through
        some batches while accumulating gradients with a run that trains on the
        concatenation of those batches in a single iteration. The number of
        batches, i.e. the number of iterations for which to accumulate
        gradients, is given by ``num_iters_to_acc``.

        Arguments:
            batch_dim (int): Batch dimension in the input tensor to be passed
                into the model for the forward pass.
            num_iters_to_acc (int): Number of iterations for which to
                accumulate gradients; all but the last iteration are run using
                the ``no_sync()`` context manager so that gradients are not
                synchronized until the final iteration.
            cpu_offload (CPUOffload): Configures CPU offloading.
            backward_prefetch (Optional[BackwardPrefetch]): Specifies at which
                point to prefetch the next layer's full parameters during the
                backward pass, if at all.
        """
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            # Disable TF32 to prevent floating point drift
            torch.backends.cuda.matmul.allow_tf32 = False

            # Initialize the FSDP model and optimizer
            group = dist.distributed_c10d._get_default_group()
            fsdp_model: FSDP = self._get_wrapped_model(
                group, cuda_first=False, add_bn=False,
                cpu_offload=cpu_offload, backward_prefetch=backward_prefetch,
            )  # disable BN since the test uses varying batch sizes
            fsdp_model.eval()  # disable dropout
            device = torch.device("cuda")
            optim = torch.optim.SGD(fsdp_model.parameters(), lr=0.01, momentum=0.9)

            # Generate the sequence of batches, each containing the same data but
            # permuted
            def permute_tensor(x: torch.Tensor):
                return x.view(-1)[torch.randperm(x.numel())].view_as(x)

            batch: Tuple[torch.Tensor, ...] = fsdp_model.module.get_input(device)
            batches: List[Tuple[torch.Tensor, ...]] = [batch]
            for _ in range(num_iters_to_acc - 1):
                batches.append(tuple(permute_tensor(t) for t in batch))
            for (batch1, batch2) in itertools.combinations(batches, r=2):
                for t1, t2 in zip(batch1, batch2):
                    assert not torch.all(t1 == t2)

            # Concatenate the batches along the given batch dimension
            concat_batch: Tuple[torch.Tensor, ...] = tuple(
                torch.cat(ts, dim=batch_dim) for ts in zip(*batches)
            )

            # Establish reference gradients using the concatenated batch
            fsdp_model.zero_grad()
            output = fsdp_model(*concat_batch)
            ref_loss = fsdp_model.module.get_loss(concat_batch, output)
            ref_loss.backward()
            ref_grads = [p.grad.detach().clone() for p in fsdp_model.parameters()]

            # Compute the gradients by accumulating via `no_sync()`
            fsdp_model.zero_grad()
            losses = []
            with fsdp_model.no_sync():
                for batch in batches[:-1]:  # accumulate for all but the last batch
                    output = fsdp_model(*batch)
                    loss = fsdp_model.module.get_loss(batch, output)
                    loss.backward()
                    losses.append(loss)
            output = fsdp_model(*batches[-1])
            loss = fsdp_model.module.get_loss(batches[-1], output)
            loss.backward()
            losses.append(loss)
            acc_loss = sum(losses)
            acc_grads = [p.grad.detach().clone() for p in fsdp_model.parameters()]

            # Compare the losses and gradients
            torch.testing.assert_allclose(ref_loss, acc_loss)
            assert len(ref_grads) == len(acc_grads)
            for ref_grad, acc_grad in zip(ref_grads, acc_grads):
                assert ref_grad.device == acc_grad.device
                assert ref_grad.size() == acc_grad.size()
                assert ref_grad.dtype == acc_grad.dtype
                torch.testing.assert_allclose(ref_grad, acc_grad)

            # Check that the optimizer step does not error
            optim.step()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "num_iters_to_acc",
        [2, 4],
    )
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize(
        "backward_prefetch",
        [BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST, None]
    )
    def test_no_sync(
        self,
        num_iters_to_acc: int,
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
    ):
        """Tests the ``no_sync()`` context manager."""
        assert num_iters_to_acc >= 2, \
            "Accumulate for at least 2 iterations to be nontrivial"
        self._test_no_sync(
            batch_dim=1,
            num_iters_to_acc=num_iters_to_acc,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
        )


instantiate_parametrized_tests(TestNoSync)

if __name__ == "__main__":
    run_tests()
