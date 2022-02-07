# Owner(s): ["oncall: distributed"]

import itertools
import sys
from unittest.mock import patch
from typing import List, Tuple

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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


class TestGradAcc(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via its
    ``no_sync()`` context manager."""

    def _test_no_sync(self, batch_dim, num_iters_to_acc):
        """
        Tests ``no_sync()`` by comparing a run that trains sequentially through
        some batches while accumulating gradients with a run that trains on the
        concatenation of those batches in a single iteration. The number of
        batches, i.e. the number of iterations for which to accumulate
        gradients, is given by ``num_iters_to_acc``.
        """
        # Use double precision to avoid floating point drift
        torch.set_default_dtype(torch.float64)

        # Initialize the FSDP model
        group = dist.distributed_c10d._get_default_group()
        fsdp_model: FSDP = self._get_wrapped_model(
            group, cuda_first=False, add_bn=False,
        )  # disable BN since the test uses variable batch size
        fsdp_model.eval()  # disable dropout
        device = torch.device("cuda")

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

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "num_iters_to_acc",
        [2, 3, 4],
    )
    def test_no_sync(self, num_iters_to_acc):
        assert num_iters_to_acc >= 2, \
            "Accumulate for at least 2 iterations to be nontrivial"
        self._test_no_sync(batch_dim=1, num_iters_to_acc=num_iters_to_acc)

    def _test_communication(self):
        """Tests ``no_sync()`` by checking the communication cost in terms of
        calls to collective communication primitives."""
        group = dist.distributed_c10d._get_default_group()
        fsdp_model: FSDP = self._get_wrapped_model(
            group, cuda_first=False,
        )
        device = torch.device("cuda")
        batch = fsdp_model.module.get_input(device)

        with patch("torch.distributed.all_gather") as mock_all_gather, \
            patch("torch.distributed.reduce_scatter") as mock_reduce_scatter:
            # TODO: Add patches for `_reduce_scatter_base` and `_all_gather_base`

            # Check the communication cost when using `no_sync()``
            with fsdp_model.no_sync():
                output = fsdp_model(*batch)
                loss = fsdp_model.module.get_loss(batch, output)
                loss.backward()
            # TODO: ...

            # Check the normal communication cost (when not using `no_sync()`)
            output = fsdp_model(*batch)
            loss = fsdp_model.module.get_loss(batch, output)
            loss.backward()
            # TODO: ...


instantiate_parametrized_tests(TestGradAcc)

if __name__ == "__main__":
    run_tests()
