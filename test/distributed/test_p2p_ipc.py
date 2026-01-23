# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_p2p_ipc.py


import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


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

    def test_p2p_ipc(self) -> None:
        """
        Test that cross-process P2P access works, by reducing a tensor,
        and then constructing a new tensor from the reduced tensor,
        while modifying the 6-th argument.

        This test is here to help stabilize the P2P share mechanism,
        preventing bc-breakage.
        """
        self._init_device()

        tensor: torch.Tensor

        if self.rank == 0:
            tensor = torch.randn(2333, device=self.device)
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

        assert tensor.allclose(tensor, 1)

        torch.distributed.barrier()


if __name__ == "__main__":
    run_tests()
