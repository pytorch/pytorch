# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_p2p_ipc.py

import os
import unittest

import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    skipIfRocm,
)


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_cuda_p2p_access()
class P2PIpcTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "gloo"

    def _init_device(self) -> None:
        # init and pin the process to the device
        device_module.set_device(self.device)
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _test_p2p_ipc_impl(self, tensor_size: int = 2333) -> None:
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
        """
        self._init_device()

        tensor: torch.Tensor

        if self.rank == 0:
            tensor = torch.randn(tensor_size, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            torch.distributed.broadcast_object_list([tensor_meta], src=0)
        else:
            recv_list = [None]
            torch.distributed.broadcast_object_list(recv_list, src=0)
            tensor_meta = recv_list[0]
            func, args = tensor_meta
            args = list(args)
            args[6] = self.rank
            tensor = func(*args)

        torch.distributed.barrier()

        if self.rank == 0:
            tensor.fill_(1)

        device_module.synchronize()
        torch.distributed.barrier()

        if not tensor.allclose(tensor, 1):
            raise AssertionError("Expected tensor to be close to 1")

        torch.distributed.barrier()

    def test_p2p_ipc(self) -> None:
        """Test P2P IPC with regular cudaMalloc allocations."""
        self._test_p2p_ipc_impl()

    @unittest.skip("Requires fix for expandable segments IPC handle type propagation")
    @skipIfRocm(msg="expandable_segments mode is not supported on ROCm")
    def test_p2p_ipc_expandable_segments(self) -> None:
        """
        Test P2P IPC with expandable segments enabled.

        This exercises the SHAREABLE_CUDA_EXPANDABLE_SEGMENT path in
        CUDACachingAllocator::shareIpcHandle() which uses
        cuMemExportToShareableHandle() instead of cudaIpcGetMemHandle().
        """
        # Enable IPC handles for expandable segments (disabled by default in fbcode)
        os.environ["TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC"] = "1"

        # Set expandable segments BEFORE any CUDA allocation
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        # Clear existing cached memory to force new allocations from expandable segments
        torch.cuda.empty_cache()

        # Use a larger tensor size (8MB) to ensure expandable segment allocation
        # Default segment size is 2MB, so this will trigger segment creation
        self._test_p2p_ipc_impl(tensor_size=2 * 1024 * 1024)

    @classmethod
    def tearDownClass(cls):
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        super().tearDownClass()


if __name__ == "__main__":
    run_tests()
