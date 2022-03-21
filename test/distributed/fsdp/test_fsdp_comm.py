# Owner(s): ["oncall: distributed"]

import sys
from unittest.mock import patch

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, NestedWrappedModule
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


class TestCommunication(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s collective communication usage."""
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "nested_model",
        [False, True],
    )
    @parametrize(
        "use_no_sync",
        [False, True],
    )
    def test_communication(
        self,
        nested_model: bool,
        use_no_sync: bool,
    ):
        """
        Tests FSDP's communication cost in terms of calls to collective
        communication primitives (i.e. all-gather and reduce-scatter).

        Arguments:
            nested_model (bool): If ``True``, uses ``NestedWrappedModule``,
                which has nested FSDP instances; if ``False``, uses the default
                model, which does not have nested FSDP instances.
            use_no_sync (bool): If ``True``, uses the ``no_sync()`` context
                manager to accumulate gradients for one iteration before
                synchronizing gradients in the second iteration; if ``False``,
                only checks the communication cost of normal execution.
        """
        # Initialize the model and inputs
        group = dist.distributed_c10d._get_default_group()
        device = torch.device("cuda")
        if nested_model:
            model = NestedWrappedModule(group, wrap_fsdp=True)
            fsdp_model: FSDP = FSDP(model, group).to(device)
        else:
            fsdp_model: FSDP = self._get_wrapped_model(group, cuda_first=False)
        batch = fsdp_model.module.get_input(device)

        # Count the number of FSDP instances
        num_fsdp = 0
        for m in fsdp_model.modules():  # includes `self`
            if isinstance(m, FSDP) and len(m.params) > 0:
                num_fsdp += 1

        # Count the number of all-gathers and reduce-scatters by mocking
        # `_all_gather_base()` and `_reducer_scatter_base()`
        # Both with and without `no_sync()`:
        #   Forward: `num_fsdp` all-gathers
        #   Backward: `num_fsdp` - 1 all-gathers (only excluding the root)
        expected_num_all_gather_no_sync = num_fsdp + (num_fsdp - 1)
        expected_num_all_gather_sync = num_fsdp + (num_fsdp - 1)
        expected_num_reduce_scatter_no_sync = 0
        expected_num_reduce_scatter_sync = num_fsdp

        num_no_sync_iters = 3
        num_sync_iters = 3
        with patch("torch.distributed._all_gather_base") as mock_all_gather, \
                patch("torch.distributed._reduce_scatter_base") as mock_reduce_scatter:
            def reset_mocks():
                mock_all_gather.reset_mock()
                mock_reduce_scatter.reset_mock()

            if use_no_sync:
                # Check the communication cost when using `no_sync()`
                for _ in range(num_no_sync_iters):
                    reset_mocks()
                    with fsdp_model.no_sync():
                        output = fsdp_model(*batch)
                        loss = fsdp_model.module.get_loss(batch, output)
                        loss.backward()
                    num_all_gather = mock_all_gather.call_count
                    num_reduce_scatter = mock_reduce_scatter.call_count
                    self.assertEqual(
                        num_all_gather,
                        expected_num_all_gather_no_sync,
                        f"Expected {expected_num_all_gather_no_sync} "
                        f"all-gathers but saw {num_all_gather} all-gathers "
                        f"when using `no_sync()`"
                    )
                    self.assertEqual(
                        num_reduce_scatter,
                        expected_num_reduce_scatter_no_sync,
                        f"Expected {expected_num_reduce_scatter_no_sync} "
                        f"reduce-scatters but saw {num_reduce_scatter} "
                        "reduce-scatters when using `no_sync()`"
                    )

            # Check the normal communication cost (when not using `no_sync()`)
            for _ in range(num_sync_iters):
                reset_mocks()
                output = fsdp_model(*batch)
                loss = fsdp_model.module.get_loss(batch, output)
                loss.backward()
                num_all_gather = mock_all_gather.call_count
                num_reduce_scatter = mock_reduce_scatter.call_count
                self.assertEqual(
                    num_all_gather,
                    expected_num_all_gather_sync,
                    f"Expected {expected_num_all_gather_sync} all-gathers "
                    f"but saw {num_all_gather} all-gathers when not using "
                    "`no_sync()`"
                )
                self.assertEqual(
                    num_reduce_scatter,
                    expected_num_reduce_scatter_sync,
                    f"Expected {expected_num_reduce_scatter_sync} reduce-"
                    f"scatters but saw {num_reduce_scatter} reduce-scatters "
                    "when not using `no_sync()`"
                )


instantiate_parametrized_tests(TestCommunication)

if __name__ == "__main__":
    run_tests()
