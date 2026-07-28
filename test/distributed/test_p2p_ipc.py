# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_p2p_ipc.py

import os
import unittest
from contextlib import contextmanager

import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    TEST_WITH_ROCM,
)


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@contextmanager
def _scoped_env(key: str, value: str):
    """Set an env var for the duration of the context, then restore it."""
    prior = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


@requires_cuda_p2p_access()
class P2PIpcTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "gloo"

    def _init_device(self, *, allocate: bool = True) -> None:
        # init and pin the process to the device. When `allocate` is False, we
        # only set the device without triggering a CUDA allocation. This is
        # required for the expandable_segments IPC regression test: if the
        # consumer allocates before fromShared(), that allocation would
        # "prime" the legacy process-global handle_type and mask the
        # #179220 (EBADF) bug we are guarding against.
        device_module.set_device(self.device)
        if allocate:
            torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _test_p2p_ipc_impl(
        self,
        tensor_size: int = 2333,
        *,
        consumer_prealloc: bool = True,
        free_producer_segment: bool = False,
    ) -> None:
        """
        Core P2P IPC test implementation.

        Test that cross-process P2P access works, by reducing a tensor,
        and then constructing a new tensor from the reduced tensor,
        while modifying the 6-th argument.

        This test is here to help stabilize the P2P share mechanism,
        preventing bc-breakage.

        Args:
            tensor_size: Size of the tensor to allocate. Larger sizes are needed
                        to trigger expandable segment allocation.
            consumer_prealloc: Whether ranks other than 0 allocate on the
                        device before receiving the IPC handle. Set to False
                        to exercise the expandable_segments IPC path where the
                        consumer has never touched the allocator (regression
                        guard for #179220).
            free_producer_segment: After the data check, have the consumer
                        release its import and the producer free the shared
                        segment, running ExpandableSegment::unmapHandles on the
                        exported handle (regression guard for #190860). Implies a
                        single consumer so the shared IPC ref-counter (init 1)
                        decrements to exactly 0 on release and the producer can
                        reclaim the block (with multiple consumers it never
                        reaches 0).
        """
        self._init_device(allocate=(self.rank == 0 or consumer_prealloc))

        # Freeing the shared segment needs exactly one consumer (see above).
        single_consumer = free_producer_segment
        is_consumer = self.rank != 0 and (not single_consumer or self.rank == 1)

        tensor = None
        tensor_meta = None
        if self.rank == 0:
            tensor = torch.randn(tensor_size, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            torch.distributed.broadcast_object_list([tensor_meta], src=0)
        else:
            recv_list = [None]
            torch.distributed.broadcast_object_list(recv_list, src=0)
            if is_consumer:
                func, args = recv_list[0]
                args = list(args)
                args[6] = self.rank
                tensor = func(*args)

        torch.distributed.barrier()

        if self.rank == 0:
            tensor.fill_(1)

        device_module.synchronize()
        torch.distributed.barrier()

        if self.rank == 0 or is_consumer:
            expected = torch.ones_like(tensor)
            self.assertEqual(tensor, expected, atol=0.0, rtol=0.0)

        if free_producer_segment:
            # Release the consumer's import (ref-counter 1 -> 0), then free the
            # producer's now-unreferenced segment -> unmapHandles.
            if is_consumer:
                del tensor
                device_module.synchronize()
            torch.distributed.barrier()  # consumer released before producer frees
            if self.rank == 0:
                del tensor_meta, tensor
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

        torch.distributed.barrier()

    def test_p2p_ipc(self) -> None:
        """Test P2P IPC with regular cudaMalloc allocations."""
        self._test_p2p_ipc_impl()

    @unittest.skipUnless(
        os.environ.get("TEST_CONFIG", "") == "h100-fabric",
        "fabric handle IPC path only exercised on the fabric-capable h100-fabric CI config; "
        "unconditionally skipped elsewhere (deadlocks on B200, see "
        "https://github.com/pytorch/pytorch/issues/189879)",
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "expandable_segments mode is not supported on ROCm"
    )
    def test_p2p_ipc_expandable_segments(self) -> None:
        """
        Test P2P IPC with expandable segments enabled. Exercises the
        SHAREABLE_CUDA_EXPANDABLE_SEGMENT path in ExpandableSegment::share /
        fromShared / unmapHandles and guards two bugs:

        - #179220 (EBADF): the consumer imports without pre-allocating, so its
          process-global handle_type stays UNSPECIFIED; if the wire header did
          not carry the producer's handle type it would misread a fabric handle
          as a POSIX fd. Checked by reading the producer's data through the
          shared segment.
        - #190860: after the consumer releases, the producer frees the shared
          segment, running unmapHandles on the exported fabric shareable_handle.
          Pre-fix, close(std::get<int>(*h.shareable_handle)) threw "std::get:
          wrong index for variant" for fabric handles (fabric-only; posix-fd
          stores an int, harmless no-op).

        A single consumer keeps the IPC ref-counter exact (1 -> 0 on release)
        so the producer can reclaim and free the segment.
        """
        # Enable IPC handles for expandable segments (disabled by default in
        # fbcode). Use a scoped env so state does not leak across tests.
        with _scoped_env("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", "1"):
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            torch.cuda.empty_cache()
            # 8MB > the 2MB default segment size, forcing an expandable segment.
            self._test_p2p_ipc_impl(
                tensor_size=2 * 1024 * 1024,
                consumer_prealloc=False,
                free_producer_segment=True,
            )

    @classmethod
    def tearDownClass(cls):
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        super().tearDownClass()


if __name__ == "__main__":
    run_tests()
