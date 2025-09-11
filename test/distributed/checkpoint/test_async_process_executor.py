# Owner(s): ["oncall: distributed checkpointing"]

import sys
from unittest.mock import patch

import torch
from torch import distributed as dist
from torch.distributed.checkpoint._async_process_executor import (
    _ProcessBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStorageWriter(StorageWriter):
    """Unified test storage writer with configurable behaviors."""

    def __init__(
        self,
        behavior="success",
    ):
        """
        Create a test storage writer with specified behavior.

        Args:
            behavior: "success", "fail_once"
        """
        self.behavior = behavior
        self.call_count = 0

    def _should_fail(self):
        """Determine if this call should fail based on the configured behavior."""
        self.call_count += 1

        if self.behavior == "success":
            return False
        elif self.behavior == "fail_once":
            return self.call_count == 1

        return False

    # Implement all required StorageWriter methods directly
    def reset(self, checkpoint_id=None):
        """Reset for new checkpoint."""

    def set_up_storage_writer(self, is_coordinator):
        """Set up storage writer."""

    def prepare_local_plan(self, plan):
        """Prepare local plan."""
        return plan

    def prepare_global_plan(self, plans):
        """Prepare global plan."""
        return plans

    def write_data(self, plan, planner):
        """Write data with policy-based failure behavior."""
        from torch.futures import Future

        # Check if we should fail based on policy
        if self._should_fail():
            raise RuntimeError(
                f"TestStorageWriter: {self.behavior} policy triggered failure on call {self.call_count}"
            )

        # Return a Future that completes to simple WriteResult-like objects
        future = Future()
        result = [{"success": True, "bytes_written": 100}]
        future.set_result(result)
        return future

    def finish(self, metadata, results):
        """Finish checkpoint."""
        return None

    def storage_meta(self):
        """Return storage metadata."""
        return None

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        """Validate checkpoint ID."""
        return True


class TestAsyncProcessExecutor(DTensorTestBase):
    """Test suite for async checkpoint process executor error handling using public APIs."""

    @with_comms
    def test_checkpoint_save_failure_continues_serving(self) -> None:
        """Test that checkpoint save failure doesn't exit process, continues serving."""

        test_state_dict = {
            "model": {"weight": torch.randn(4, 4), "bias": torch.randn(4)},
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
        }

        # 1. Simulate a failure in creating PG in background process.
        with patch(
            "torch.distributed.checkpoint._async_process_executor.get_free_port",
            return_value=-1,
        ):
            with self.assertRaises(ValueError) as _:
                proc_executor = _ProcessBasedAsyncCheckpointExecutor()
                fut = proc_executor.execute_save(
                    staging_future_or_state_dict=test_state_dict,
                )
                fut.result()

        # 2. Attempt save with failing storage writer
        with patch(
            "torch.distributed.checkpoint._async_process_executor.get_free_port",
            return_value=get_free_port(),
        ) as mock_get_free_port:
            proc_executor = _ProcessBasedAsyncCheckpointExecutor()
            fut = proc_executor.execute_save(
                staging_future_or_state_dict=test_state_dict,
                storage_writer=TestStorageWriter(behavior="fail_once"),
            )
            self.assertIn("fail_once policy triggered failure", str(fut.exception()))
            # Verify new process was created for this attempt
            if dist.get_rank() == 0:
                mock_get_free_port.assert_called_once()

        # 3. Second save attempt with successful storage writer - process should still be alive
        with patch(
            "torch.distributed.checkpoint._async_process_executor.get_free_port",
        ) as mock_get_free_port:
            proc_executor = _ProcessBasedAsyncCheckpointExecutor()
            fut = proc_executor.execute_save(
                staging_future_or_state_dict=test_state_dict,
                storage_writer=TestStorageWriter(behavior="success"),
            )
            result = fut.result()
            # Verify process is still alive
            mock_get_free_port.assert_not_called()
            # Verify successful save
            self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
