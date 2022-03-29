# flake8: noqa C101
import itertools
from typing import Iterator, List

import torch
import torch.distributed as dist


def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: dist.ProcessGroup
):
    """
    Averages all the given parameters.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params_it1, params_it2 = itertools.tee(params)
    # If the input parameters have different data types,
    # packing these parameters will trigger an implicit type up-casting.
    # The original parameter data types will be restored during the subsequent unpacking.
    flat_params = torch.cat([p.data.view(-1) for p in params_it1])
    flat_params /= dist.get_world_size(group_to_use)
    # Make sure the allreduce will not conflict with any other ongoing process group.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.all_reduce(flat_params, group=group_to_use)

    offset = 0
    for p in params_it2:
        p.data = flat_params[offset: offset + p.numel()].view_as(p).type_as(p)
        offset += p.numel()


class TensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    refer to https://github.com/epfml/LocalSGD-Code/blob/3d4811d01673af205a00176f5389ed008a1ddb37/distributed_code/pcode/utils/tensor_buffer.py#L5
    """

    def __init__(self, tensors: List[torch.Tensor], use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor.data[:] = entry


def flatten(tensors, shapes=None, use_cuda=True):
    """
    refer to https://github.com/epfml/LocalSGD-Code/blob/3d4811d01673af205a00176f5389ed008a1ddb37/distributed_code/pcode/utils/tensor_buffer.py#L5
    Args:
        tensors:
        shapes:
        use_cuda:

    Returns:

    """
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.reshape(-1)
    return vec


def get_param_list(param_groups: List[dict]):
    params_list = []
    for param_group in param_groups:
        for params in param_group["params"]:
            if params.grad is None:
                continue
            params_list.append(params)
    return params_list


def average_parameters_v1(
        params: torch.Tensor, process_group: dist.ProcessGroup
):
    """
    Averages all the given parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params /= dist.get_world_size(group_to_use)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.all_reduce(params, group=group_to_use)
    return params


def comm_average_param_fast(param_groups, process_group: dist.ProcessGroup = None):
    """
    use a faster communication but more memory average strategy
    Args:
        param_groups:

    Returns:

    """
    params_list = get_param_list(param_groups)
    params_tb = TensorBuffer(params_list)
    params_tb.buffer = average_parameters_v1(params_tb.buffer, process_group)
    # consistent the local models by assigning the consensus params.
    params_tb.unpack(params_list)


def comm_average_param_memory_efficient(param_groups, process_group: dist.ProcessGroup = None):
    """
    use a slower communication but less memory average strategy
    Args:
        param_groups:

    Returns:

    """
    for param_group in param_groups:
        for params in param_group["params"]:
            if params.grad is None:
                continue
            buffer = flatten([params])
            buffer = average_parameters_v1(buffer, process_group)
            params.data[:] = buffer.reshape(params.size())