# Owner(s): ["oncall: distributed"]

"""
M11: Protocol Version Guardrail Tests

This module tests the distributed protocol version check that happens during
init_process_group. It verifies:
1. Matching versions allow initialization to succeed
2. Mismatched versions cause early, clear failure

The tests use the TORCH_DISTRIBUTED_PROTOCOL_VERSION_OVERRIDE environment
variable to simulate version mismatches without requiring multiple PyTorch
installations.
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed._protocol_version import (
    _PROTOCOL_VERSION_OVERRIDE_ENV,
    get_protocol_version,
    PROTOCOL_VERSION,
)
from torch.testing._internal.common_distributed import requires_gloo
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def _init_process_group_worker(
    rank: int,
    world_size: int,
    file_name: str,
    version_override: int | None,
    result_queue: mp.Queue,
) -> None:
    """
    Worker function for multiprocessing spawn.

    Args:
        rank: The rank of this process.
        world_size: Total number of processes.
        file_name: Path to the FileStore file.
        version_override: If not None, set as the protocol version override.
        result_queue: Queue to report success/failure back to the parent.
    """
    try:
        # Set the version override if specified
        if version_override is not None:
            os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = str(version_override)
        elif _PROTOCOL_VERSION_OVERRIDE_ENV in os.environ:
            del os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV]

        store = dist.FileStore(file_name, world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size,
        )
        # If we get here, init succeeded
        result_queue.put((rank, "success", None))
        dist.destroy_process_group()
    except Exception as e:
        result_queue.put((rank, "error", str(e)))


class TestProtocolVersion(TestCase):
    """Tests for the distributed protocol version guardrail."""

    def test_get_protocol_version_default(self) -> None:
        """Test that get_protocol_version returns PROTOCOL_VERSION by default."""
        # Ensure no override is set
        old_val = os.environ.pop(_PROTOCOL_VERSION_OVERRIDE_ENV, None)
        try:
            version = get_protocol_version()
            self.assertEqual(version, PROTOCOL_VERSION)
        finally:
            if old_val is not None:
                os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = old_val

    def test_get_protocol_version_override(self) -> None:
        """Test that get_protocol_version respects environment variable override."""
        old_val = os.environ.pop(_PROTOCOL_VERSION_OVERRIDE_ENV, None)
        try:
            os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = "42"
            version = get_protocol_version()
            self.assertEqual(version, 42)
        finally:
            if old_val is not None:
                os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = old_val
            else:
                os.environ.pop(_PROTOCOL_VERSION_OVERRIDE_ENV, None)

    def test_get_protocol_version_invalid_override(self) -> None:
        """Test that invalid override raises RuntimeError."""
        old_val = os.environ.pop(_PROTOCOL_VERSION_OVERRIDE_ENV, None)
        try:
            os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = "not_a_number"
            with self.assertRaises(RuntimeError) as cm:
                get_protocol_version()
            self.assertIn("Invalid value", str(cm.exception))
            self.assertIn("not_a_number", str(cm.exception))
        finally:
            if old_val is not None:
                os.environ[_PROTOCOL_VERSION_OVERRIDE_ENV] = old_val
            else:
                os.environ.pop(_PROTOCOL_VERSION_OVERRIDE_ENV, None)


class TestProtocolVersionMultiProcess(TestCase):
    """Multi-process tests for protocol version checking during init."""

    world_size = 2

    @requires_gloo()
    def test_matching_versions_succeed(self) -> None:
        """Test that matching protocol versions allow init to succeed."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name

        try:
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()

            processes = []
            for rank in range(self.world_size):
                # Both ranks use the same version (no override, use default)
                p = ctx.Process(
                    target=_init_process_group_worker,
                    args=(rank, self.world_size, file_name, None, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results
            results = []
            for _ in range(self.world_size):
                results.append(result_queue.get(timeout=60))

            # Wait for processes to finish
            for p in processes:
                p.join(timeout=10)

            # Verify all ranks succeeded
            for rank, status, error in results:
                self.assertEqual(
                    status,
                    "success",
                    f"Rank {rank} failed unexpectedly: {error}",
                )
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass

    @requires_gloo()
    def test_matching_overridden_versions_succeed(self) -> None:
        """Test that matching overridden versions also succeed."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name

        try:
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()

            processes = []
            for rank in range(self.world_size):
                # Both ranks use the same overridden version
                p = ctx.Process(
                    target=_init_process_group_worker,
                    args=(rank, self.world_size, file_name, 99, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results
            results = []
            for _ in range(self.world_size):
                results.append(result_queue.get(timeout=60))

            # Wait for processes to finish
            for p in processes:
                p.join(timeout=10)

            # Verify all ranks succeeded
            for rank, status, error in results:
                self.assertEqual(
                    status,
                    "success",
                    f"Rank {rank} failed unexpectedly: {error}",
                )
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass

    @requires_gloo()
    def test_mismatched_versions_fail(self) -> None:
        """Test that mismatched protocol versions cause init to fail early."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name

        try:
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()

            processes = []
            for rank in range(self.world_size):
                # Rank 0 uses version 1, Rank 1 uses version 2
                version_override = 1 if rank == 0 else 2
                p = ctx.Process(
                    target=_init_process_group_worker,
                    args=(rank, self.world_size, file_name, version_override, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results
            results = []
            for _ in range(self.world_size):
                results.append(result_queue.get(timeout=60))

            # Wait for processes to finish
            for p in processes:
                p.join(timeout=10)

            # At least one rank should have failed with a protocol version error
            errors = [error for _, status, error in results if status == "error"]
            self.assertGreater(
                len(errors),
                0,
                "Expected at least one rank to fail due to version mismatch",
            )

            # Verify the error message contains expected information
            error_messages = " ".join(errors)
            self.assertIn(
                "protocol version",
                error_messages.lower(),
                f"Error should mention 'protocol version': {error_messages}",
            )
        finally:
            try:
                os.remove(file_name)
            except OSError:
                pass


if __name__ == "__main__":
    run_tests()

