import torch
from torch import Tensor


def my__foreach_mul_(tensors, others) -> ():
    """
    Updates tensors to be the result of pointwise multiplying with others.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: nothing, tensors is updated in place.
    """
    torch.ops.libtorch_agnostic_2_10.my__foreach_mul_.default(tensors, others)


def my__foreach_mul(tensors, others) -> list[Tensor]:
    """
    Returns a list of tensors that are the results of pointwise multiplying
    tensors and others.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: list of multiplied tensors
    """
    return torch.ops.libtorch_agnostic_2_10.my__foreach_mul.default(tensors, others)


def make_tensor_clones_and_call_foreach(t1, t2) -> list[Tensor]:
    """
    Returns a list of 2 tensors corresponding to the square of the inputs.

    Args:
        t1: Tensor
        t2: Tensor

    Returns: list of [t1^2, t2^2]
    """
    return torch.ops.libtorch_agnostic_2_10.make_tensor_clones_and_call_foreach.default(
        t1, t2
    )


def test_tensor_device(t):
    """
    Tests Tensor device() method.

    Args:
        t: Tensor - tensor to get device from

    Returns: Device - device of the tensor
    """
    return torch.ops.libtorch_agnostic_2_10.test_tensor_device.default(t)


def test_device_constructor(is_cuda, index, use_str):
    """
    Tests creating a Device from DeviceType and index, or from a string.

    Args:
        is_cuda: bool - if True, creates CUDA device; if False, creates CPU device
        index: int - device index
        use_str: bool - if True, constructs from string; if False, constructs from DeviceType

    Returns: Device - A device with the specified type and index
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_constructor.default(
        is_cuda, index, use_str
    )


def test_device_equality(d1, d2) -> bool:
    """
    Tests Device equality operator.

    Args:
        d1: Device - first device
        d2: Device - second device

    Returns: bool - True if devices are equal
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_equality.default(d1, d2)


def test_device_set_index(device, index):
    """
    Tests Device set_index() method.

    Args:
        device: Device - device to modify
        index: int - new device index

    Returns: Device - device with updated index
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_set_index.default(device, index)


def test_device_index(device) -> int:
    """
    Tests Device index() method.

    Args:
        device: Device - device to query

    Returns: int - device index
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_index.default(device)


def test_device_is_cuda(device) -> bool:
    """
    Tests Device is_cuda() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CUDA
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_is_cuda.default(device)


def test_device_is_cpu(device) -> bool:
    """
    Tests Device is_cpu() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CPU
    """
    return torch.ops.libtorch_agnostic_2_10.test_device_is_cpu.default(device)


def test_parallel_for(size, grain_size) -> Tensor:
    """
    Tests the parallel_for functionality by using it to fill a tensor with indices.
    Args:
        size: int - size of the tensor to create
        grain_size: int - grain size for parallel_for
    Returns: Tensor - a 1D int64 tensor where each element contains its index
        (if multiple threads are used the threadid will be encoded in the upper 32 bits)
    """
    return torch.ops.libtorch_agnostic_2_10.test_parallel_for.default(size, grain_size)


def test_get_num_threads() -> int:
    """
    Tests the get_num_threads functionality by returning the number of threads
    for the parallel backend.

    Returns: int - the number of threads for the parallel backend
    """
    return torch.ops.libtorch_agnostic_2_10.test_get_num_threads.default()


def my_empty(
    size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
) -> Tensor:
    """
    Creates an empty tensor with the specified size, dtype, layout, device, pin_memory, and memory_format.

    Args:
        size: list[int] - size of the tensor to create
        dtype: ScalarType or None - data type of the tensor
        layout: Layout or None - layout of the tensor
        device: Device or None - device on which to create the tensor
        pin_memory: bool or None - whether to use pinned memory
        memory_format: MemoryFormat or None - memory format of the tensor

    Returns: Tensor - an uninitialized tensor with the specified properties
    """
    return torch.ops.libtorch_agnostic_2_10.my_empty.default(
        size, dtype, layout, device, pin_memory, memory_format
    )


def my_reshape(t, shape) -> Tensor:
    """
    Returns a tensor with the same data but different shape.

    Args:
        t: Tensor - tensor to reshape
        shape: list[int] - new shape for the tensor

    Returns: Tensor - reshaped tensor
    """
    return torch.ops.libtorch_agnostic_2_10.my_reshape.default(t, shape)


def my_view(t, size) -> Tensor:
    """
    Returns a new tensor with the same data as the input tensor but of a different shape.

    Args:
        t: Tensor - tensor to view
        size: list[int] - new size for the tensor

    Returns: Tensor - tensor with new view
    """
    return torch.ops.libtorch_agnostic_2_10.my_view.default(t, size)


def my_shape(t) -> tuple[int]:
    """
    Returns a shape of the input tensor.

    Args:
        t: Tensor - input tensor

    Returns: tuple - shape of the imput tensor.
    """
    return torch.ops.libtorch_agnostic_2_10.my_shape.default(t)


def get_any_data_ptr(t, mutable) -> int:
    """
    Return data pointer value of the tensor.
    Args:
        t: Input tensor
        mutable: whether data pointer qualifier is mutable or const
    Returns: int - pointer value
    """
    return torch.ops.libtorch_agnostic_2_10.get_any_data_ptr.default(t, mutable)


def get_template_any_data_ptr(t, dtype, mutable) -> int:
    """
    Return data pointer value of the tensor iff it has dtype.
    Args:
        t: Input tensor
        dtype: Input dtype
        mutable: whether data pointer qualifier is mutable or const
    Returns: int - pointer value
    Raises RuntimeError when t.dtype() != dtype.
    """
    return torch.ops.libtorch_agnostic_2_10.get_template_any_data_ptr.default(
        t, dtype, mutable
    )
