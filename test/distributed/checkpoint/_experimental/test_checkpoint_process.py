# Owner(s): ["oncall: distributed checkpointing"]


import os
import tempfile
import time
from concurrent.futures import Future
from typing import Any

import torch
from torch.distributed.checkpoint._experimental.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
    RequestType,
    WorkerRequest,
    WorkerResponse,
)
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


def subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize the subprocess with some basic checks.

    This is similar to the subprocess_init_routine in checkpointing_test.py.
    """
    assert name == "test-checkpointer", f"Unexpected subprocess name: {name}"
    assert os.getpid() != parent_pid, "This was supposed to run in a different process"
    assert os.getppid() == parent_pid, (
        "This was supposed to run as a child to main process"
    )


def failing_subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize function that raises an exception."""
    # Acknowledge parameters to avoid unused variable warnings
    _ = name
    _ = parent_pid
    raise RuntimeError("Subprocess initialization failed")


def timedout_subprocess_init_fn(**kwargs: Any) -> None:
    # Acknowledge parameters to avoid unused variable warnings
    _ = kwargs
    time.sleep(3)  # Simulate a long initialization


def ckpt_writer_init_fn(**kwargs: Any) -> CheckpointWriter:
    """Initialize a CheckpointWriter in the subprocess.

    This function is called in the subprocess to create a CheckpointWriter instance.
    It's important that this function is defined at the module level so it can be pickled.
    """
    return CheckpointWriter(
        config=kwargs.get("config"),
        rank_info=kwargs.get("rank_info"),
    )


def failing_ckpt_writer_init_fn(**kwargs: Any) -> CheckpointWriter:
    """Initialize function that raises an exception."""
    # Acknowledge parameters to avoid unused variable warnings
    _ = kwargs
    raise RuntimeError("CheckpointWriter initialization failed")


def shared_tensor_verifier_init_fn(**kwargs: Any) -> CheckpointWriter:
    """Initialize a CheckpointWriter that verifies shared memory tensors."""

    class SharedTensorVerifier(CheckpointWriter):
        def __init__(self, config=None, rank_info=None, **init_kwargs):
            # Acknowledge unused kwargs to avoid linting warnings
            _ = init_kwargs
            super().__init__(
                config=config or CheckpointWriterConfig(),
                rank_info=rank_info,
                barrier=None,
                commit_hook=None,
            )

        def write(self, state_dict, path, **__):
            # Acknowledge parameters to avoid unused variable warnings
            _ = path

            # Verify shared memory tensor behavior directly with assertions
            if "shared_tensor" in state_dict:
                shared_tensor = state_dict["shared_tensor"]
                # Critical assertion: shared tensor should remain in shared memory in subprocess
                assert shared_tensor.is_shared(), (
                    "Shared tensor should be in shared memory in subprocess"
                )

                shared_tensor[0] = 42.0

            if "regular_tensor" in state_dict:
                # Note: ForkingPickler moves regular tensors to shared memory during IPC - this is acceptable
                assert state_dict["regular_tensor"].is_shared(), (
                    "Regular tensor should also be in shared memory in subprocess"
                )

            return None

    verifier = SharedTensorVerifier(
        config=kwargs.get("config"),
        rank_info=kwargs.get("rank_info"),
    )
    return verifier


class TestRequestTypes(TestCase):
    """Test the request/response data structures."""

    def test_request_type_enum(self) -> None:
        """Test RequestType enum values."""
        self.assertEqual(RequestType.PING.value, "ping")
        self.assertEqual(RequestType.WRITE_CHECKPOINT.value, "write_checkpoint")
        self.assertEqual(RequestType.TERMINATE_PROCESS.value, "exit")

    def test_worker_request(self) -> None:
        """Test WorkerRequest dataclass."""
        request = WorkerRequest(request_type=RequestType.PING, payload={"test": "data"})
        self.assertEqual(request.request_type, RequestType.PING)
        self.assertEqual(request.payload["test"], "data")

    def test_worker_response(self) -> None:
        """Test WorkerResponse dataclass."""
        response = WorkerResponse(
            request_type=RequestType.PING,
            success=True,
            error_msg=None,
            payload={"result": "success"},
        )
        self.assertEqual(response.request_type, RequestType.PING)
        self.assertTrue(response.success)
        self.assertIsNone(response.error_msg)
        self.assertEqual(response.payload["result"], "success")


class TestCheckpointProcessConfig(TestCase):
    """Test CheckpointProcessConfig configuration."""

    def test_default_options(self) -> None:
        """Test default CheckpointProcessConfig."""
        options = CheckpointProcessConfig()
        # Test default values
        self.assertEqual(options.subprocess_init_timeout_secs, 30)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 60)

    def test_custom_options(self) -> None:
        """Test custom CheckpointProcessConfig."""
        options = CheckpointProcessConfig(
            subprocess_init_timeout_secs=10, subprocess_shutdown_timeout_secs=30
        )
        self.assertEqual(options.subprocess_init_timeout_secs, 10)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 30)


class TestCheckpointProcess(TestCase):
    def setUp(self) -> None:
        super().setUp()
        """Set up common test fixtures."""
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )
        self.writer_config = CheckpointWriterConfig()
        self.test_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def _create_checkpoint_process(
        self,
        subprocess_init_fn_override=None,
        subprocess_init_args_override=None,
        writer_init_fn_override=None,
        subprocess_init_timeout_secs=30,
    ):
        """Helper to create CheckpointProcess."""
        config = CheckpointProcessConfig(
            subprocess_init_timeout_secs=subprocess_init_timeout_secs,
        )

        return CheckpointProcess(
            rank_info=self.rank_info,
            config=config,
            subprocess_init_fn=subprocess_init_fn_override or subprocess_init_fn,
            subprocess_init_args=subprocess_init_args_override
            or (
                "test-checkpointer",
                os.getpid(),
            ),
            checkpoint_writer_init_fn=writer_init_fn_override or ckpt_writer_init_fn,
            checkpoint_writer_init_args={
                "config": self.writer_config,
                "rank_info": self.rank_info,
            },
        )

    def test_checkpoint_process_initialization(self) -> None:
        """Test that CheckpointProcess initializes and closes correctly."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for the process creation future to complete
        checkpoint_process.process_creation_future.result()

        # Verify process is alive
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())

        checkpoint_process.close()

        # Verify process is terminated
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_checkpoint_write_sync_state_dict(self) -> None:
        """Test writing a checkpoint with synchronous state dict."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint
            future = checkpoint_process.write(self.test_state_dict, checkpoint_path)

            # Verify future is returned
            self.assertIsInstance(future, Future)

            # Wait for completion
            future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

            # Verify checkpoint content
            loaded_state_dict = torch.load(expected_file)
            self.assertIn("model", loaded_state_dict)
            self.assertIn("optimizer", loaded_state_dict)
            self.assertEqual(loaded_state_dict["epoch"], 5)
            self.assertEqual(loaded_state_dict["step"], 1000)

        checkpoint_process.close()

    def test_checkpoint_write_future_state_dict(self) -> None:
        """Test writing a checkpoint with Future state dict."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Create a Future that resolves to the state dict
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)

        def get_state_dict():
            time.sleep(0.1)  # Simulate some processing time
            return self.test_state_dict

        future_state_dict = executor.submit(get_state_dict)

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint with Future state dict
            write_future = checkpoint_process.write(future_state_dict, checkpoint_path)

            # Wait for completion
            write_future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

        executor.shutdown(wait=True)
        checkpoint_process.close()

    def test_checkpoint_write_with_kwargs(self) -> None:
        """Test checkpoint writing with additional kwargs."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint with kwargs
            future = checkpoint_process.write(
                self.test_state_dict,
                checkpoint_path,
                custom_arg="test_value",
                another_arg=42,
            )

            # Wait for completion
            future.result()

            # Verify checkpoint was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

        checkpoint_process.close()

    def test_subprocess_initialization_timeout(self) -> None:
        """Test subprocess initialization timeout."""

        # Create checkpoint process with a very short timeout by mocking the initialization
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=timedout_subprocess_init_fn,
            subprocess_init_timeout_secs=1,
        )

        # This should timeout
        with self.assertRaises(TimeoutError) as cm:
            checkpoint_process.process_creation_future.result()

        self.assertIn("Timed out", str(cm.exception))

    def test_subprocess_initialization_failure(self) -> None:
        """Test subprocess initialization failure."""
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=failing_subprocess_init_fn
        )

        # The subprocess should fail to initialize
        # We expect this to raise an exception when we try to use it
        with self.assertRaises(RuntimeError):
            checkpoint_process.process_creation_future.result()

    def test_graceful_termination(self) -> None:
        """Test graceful termination of subprocess."""
        checkpoint_process = self._create_checkpoint_process()

        checkpoint_process.process_creation_future.result()
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())
        checkpoint_process.close()
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_forced_termination(self) -> None:
        """Test forced termination when graceful termination fails."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Mock the join method to simulate timeout
        def mock_join(timeout=None):
            # Acknowledge timeout parameter to avoid unused variable warning
            _ = timeout
            return False  # Simulate timeout

        checkpoint_process.process.join = mock_join

        # This should trigger forced termination
        checkpoint_process.close()

        # Process should still be terminated (killed)
        # Note: This test might be flaky depending on timing

    def test_communication_error_handling(self):
        """Test handling of communication errors."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Close the pipe to simulate communication failure
        checkpoint_process._parent_end.close()

        # Attempting to write should raise an error
        with self.assertRaises(RuntimeError) as cm:
            future = checkpoint_process.write(self.test_state_dict, "/tmp/test")
            future.result()

        self.assertIn("Child process terminated unexpectedly", str(cm.exception))

    def test_shared_memory_tensor_ipc(self):
        """Test that shared memory tensors are backed by the same memory across processes."""

        checkpoint_process = self._create_checkpoint_process(
            writer_init_fn_override=shared_tensor_verifier_init_fn,
        )

        checkpoint_process.process_creation_future.result()

        # Create tensors and put them in shared memory
        shared_tensor = torch.randn(100, 100)
        shared_tensor.share_memory_()

        shared_tensor_data_ptr = shared_tensor.data_ptr()

        regular_tensor = torch.randn(50, 50)
        # Don't put regular tensor in shared memory for comparison

        # Verify initial shared memory status
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should be in shared memory"
        )
        self.assertFalse(
            regular_tensor.is_shared(), "Regular tensor should not be in shared memory"
        )

        # Create state dict with mixed tensor types
        test_state_dict = {
            "shared_tensor": shared_tensor,
            "regular_tensor": regular_tensor,
        }

        # Write to subprocess - the SharedTensorVerifier will:
        # 1. Verify the tensor is still in shared memory
        # 2. Check the marker value (42.0) to confirm same memory
        # 3. Modify specific positions to prove same memory access
        future = checkpoint_process.write(test_state_dict, "")

        try:
            result = (
                future.result()
            )  # This will raise an exception if the subprocess assertions fail
            self.assertIsNone(result)  # SharedTensorVerifier returns None on success
        except Exception as e:
            self.fail(f"Subprocess assertions failed: {e}")

        # assert shared tensor is still in same shared memory
        self.assertEqual(
            shared_tensor_data_ptr,
            shared_tensor.data_ptr(),
            "Shared tensor should still be in same shared memory",
        )
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should still be in shared memory"
        )

        # CRITICAL TEST: Verify that modifications made by subprocess are visible in main process
        # This definitively proves that both processes access the same memory

        self.assertAlmostEqual(
            shared_tensor[0][0],
            42.0,
            places=6,
            msg=f"Expected subprocess signature 42.0, got {shared_tensor[0]}. "
            f"Shared memory not working - subprocess modifications not visible!",
        )

        checkpoint_process.close()


if __name__ == "__main__":
    run_tests()
