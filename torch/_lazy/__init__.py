import torch._C._lazy
import torch.distributed as dist
import os

from datetime import timedelta


def mark_step(device: str = "lazy:0", wait=False):
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # TODO(whc) expand this to include backend hooks and align with XLA backend needs
    torch._C._lazy._mark_step(device, [], wait=wait)


def wait_device_ops(devices=None):
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)

def sync_multi(tensors, devices):
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(tensors, devices)

def get_tensor_id(tensor):
    """Return a unique id of the lazy tensor maintained by LTC"""
    return torch._C._lazy._get_tensor_id(tensor)

def create_lazy_process_group(store: dist.Store, rank: int, size: int, timeout: timedelta):
    """
    Create a ProcessGroup that is dedicated to handle lazy devices for torch.distributed.
    Args:
        store (dist.Store): key/value store accessible to all workers, used
                            to exchange connection/address information.
        rank (int): Rank of the current process (it should be a
                    number between 0 and ``size``-1).
        size (int): Number of processes participating in the job.
        timeout (timedelta): Timeout for operations executed against
                             the process group.
    Returns:
        A new LazyProcessGroup instance.
    """
    return LazyProcessGroup(store, rank, size, timeout)


def lazy_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    A hook that compose LazyTensor with DistributedDataParallel. Ideally, one just need LazyProcessGroup.
    Since dist._Work doesn't provide a way to set the value in its torch.futures.Future object, we have
    to rely on this hook to set the right value.

    Args:
        state (object): Not used at all.
        bucket: (dist.GradBucket): The bucket of gradients for all_reduce.

    Returns:
        A Future that contains a LazyTensor.
    """
    assert os.environ.get('LTC_TS_CUDA', False)

    buffer = bucket.buffer()

    cuda_buffer = buffer.to(torch.device('cuda', buffer.device.index))
    torch.distributed.all_reduce(cuda_buffer)
    buffer.copy_(cuda_buffer)

    fut = torch.futures.Future()
    fut.set_result(buffer)

    return fut


class LazyProcessGroup(dist.ProcessGroup):
    """
    A class to build a custom dist.ProcessGroup instance which overwrites
    the ``dist.ProcessGroup.getBackendName()``, ``dist.ProcessGroup.allgather()``,
    ``dist.ProcessGroup.allreduce()``, ``dist.ProcessGroup.broadcast()``
    methods to handle lazy devices. It currently only works for NCCL and
    CUDA backend devices.
    Args:
        store (dist.Store): key/value store accessible to all workers, used
                            to exchange connection/address information.
        rank (int): Rank of the current process (it should be a
                    number between 0 and ``size``-1).
        size (int): Number of processes participating in the job.
        timeout (timedelta): Timeout for operations executed against
                             the process group.
    """
    def __init__(self, store: dist.Store, rank: int, size: int, timeout: timedelta):
        assert os.environ.get('LTC_TS_CUDA', False)
        dist.ProcessGroup.__init__(self, rank, size)
        self.nccl_pg = dist.ProcessGroupNCCL(store, rank, size, timeout)

    def getBackendName(self):
        return "Lazy"

    def allgather(self, output_tensor_lists, input_tensor_list, opts=None):
        # The device index in output_tensor_lists could be wrong. Let's use the input_tensor_list's.
        cuda_output_tensor_lists = [
            [tensor.to(torch.device('cuda', input_tensor_list[0].device.index)) for tensor in tensor_list]
            for tensor_list in output_tensor_lists
        ]
        cuda_input_tensor_list = [tensor.to(torch.device('cuda', tensor.device.index)) for tensor in input_tensor_list]
        if opts == None:
            work = self.nccl_pg.allgather(cuda_output_tensor_lists, cuda_input_tensor_list)
        else:
            work = self.nccl_pg.allgather(cuda_output_tensor_lists, cuda_input_tensor_list, opts)
        work.wait()
        for tensor_list, cuda_tensor_list in zip(output_tensor_lists, cuda_output_tensor_lists):
            for tensor, cuda_tensor in zip(tensor_list, cuda_tensor_list):
                tensor.copy_(cuda_tensor)
        return work

    def allreduce(self, tensor_list, opts=None):
        cuda_tensor_list = [tensor.to(torch.device('cuda', tensor.device.index)) for tensor in tensor_list]
        work = self.nccl_pg.allreduce(cuda_tensor_list, opts)
        work.wait()
        for tensor, cuda_tensor in zip(tensor_list, cuda_tensor_list):
            tensor.copy_(cuda_tensor)
        return work

    def broadcast(self, tensor_list, opts=None):
        cuda_tensor_list = [tensor.to(torch.device('cuda', tensor.device.index)) for tensor in tensor_list]
        work = self.nccl_pg.broadcast(cuda_tensor_list, opts)
        work.wait()
        for tensor, cuda_tensor in zip(tensor_list, cuda_tensor_list):
            tensor.copy_(cuda_tensor)
        return work
