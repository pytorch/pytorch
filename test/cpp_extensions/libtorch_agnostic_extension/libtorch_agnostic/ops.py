import torch
from torch import Tensor


def sgd_out_of_place(param, grad, weight_decay, lr, maximize) -> Tensor:
    """
    Computes a single step of SGD on a single parameter Tensor with grad.

    Assumes:
    - param and grad are the same shape and are 1D.
    - param and grad are float and on CPU

    Args:
        param: a 1D tensor of floats
        grad: a 1D tensor of floats
        weight_decay: a python double between 0 and 1
        lr: a python double

    Returns:
        a 1D float Tensor the same shape as param

    """
    return torch.ops.libtorch_agnostic.sgd_out_of_place.default(
        param, grad, weight_decay, lr, maximize
    )


def identity(t) -> Tensor:
    """
    Returns the input tensor

    Args:
        t: any Tensor

    Returns:
        a Tensor, the same as input.
    """
    return torch.ops.libtorch_agnostic.identity.default(t)


def my_abs(t) -> Tensor:
    """
    Returns abs on the input tensor, outputs a new Tensor

    Args:
        t: any Tensor

    Returns:
        a Tensor
    """
    return torch.ops.libtorch_agnostic.my_abs.default(t)


def my_is_cpu(t) -> bool:
    """
    Returns is_cpu on the input tensor.

    Args:
        t: any Tensor

    Returns:
        a bool
    """
    return torch.ops.libtorch_agnostic.my_is_cpu.default(t)


def my_ones_like(tensor, device) -> Tensor:
    """
    Returns a new Tensor like the input tensor, but with all ones

    Args:
        tensor: any Tensor
        device: a device string

    Returns:
        a ones Tensor with the same dtype and shape and other attributes
        like the input tensor
    """
    return torch.ops.libtorch_agnostic.my_ones_like.default(tensor, device)


def exp_neg_is_leaf(t1, t2, t3) -> tuple[Tensor, Tensor, bool]:
    """
    Returns a Tensor, Tensor, bool tuple corresponding to the respective inputs
    t1, t2, and t3.

    Args:
        t1: Tensor
        t2: Tensor
        t3: Tensor

    Returns:
        (exp(t1), neg(t2), is_leaf(t3))
    """
    return torch.ops.libtorch_agnostic.exp_neg_is_leaf.default(t1, t2, t3)


def neg_exp(t) -> Tensor:
    """
    Returns a Tensor composing neg of exp

    Args:
        t: Tensor

    Returns: neg(exp(t))
    """
    return torch.ops.libtorch_agnostic.neg_exp.default(t)


def divide_neg_exp(t) -> Tensor:
    """
    Returns a Tensor division of neg and exp

    Args:
        t: Tensor

    Returns: divide(neg(t), exp(t))
    """
    return torch.ops.libtorch_agnostic.divide_neg_exp.default(t)


def is_contiguous(t) -> bool:
    """
    Returns a bool indicating if the input tensor is contiguous

    Args:
        t: Tensor

    Returns: is_contiguous(t)
    """
    return torch.ops.libtorch_agnostic.is_contiguous.default(t)


def my_transpose(t, dim0, dim1) -> Tensor:
    """
    Returns t.transpose(dim0, dim1)

    Args:
        t: Tensor

    Returns: my_transpose(t, dim0, dim1)
    """
    return torch.ops.libtorch_agnostic.my_transpose.default(t, dim0, dim1)


def my_empty_like(t) -> Tensor:
    """
    Returns t.empty_like()

    Args:
        t: Tensor

    Returns: my_empty_like(t)
    """
    return torch.ops.libtorch_agnostic.my_empty_like.default(t)


def my_zero_(t) -> Tensor:
    """
    Returns t.zero_()

    Args:
        t: Tensor

    Returns: my_zero_(t)
    """
    return torch.ops.libtorch_agnostic.my_zero_.default(t)


def my_amax(t) -> Tensor:
    """
    Returns t.amax()

    Args:
        t: Tensor

    Returns: amax(t)
    """
    return torch.ops.libtorch_agnostic.my_amax.default(t)


def my_amax_vec(t) -> Tensor:
    """
    Returns t.amax()

    Args:
        t: Tensor

    Returns: amax(t)
    """
    return torch.ops.libtorch_agnostic.my_amax_vec.default(t)


def fill_infinity(t) -> Tensor:
    """
    Fills the tensor with inf.

    Args:
        t: Tensor to fill

    Returns: The modified tensor (same as input)
    """
    return torch.ops.libtorch_agnostic.fill_infinity.default(t)


def test_default_constructor(defined) -> bool:
    """
    Tests the default constructor for torch::stable::Tensor.

    Args:
        defined: bool - if True, tests defined tensor; if False, tests undefined tensor

    Returns: bool - result of calling .defined() on the tensor
    """
    return torch.ops.libtorch_agnostic.test_default_constructor.default(defined)


def test_tensor_device(t):
    """
    Tests Tensor device() method.

    Args:
        t: Tensor - tensor to get device from

    Returns: Device - device of the tensor
    """
    return torch.ops.libtorch_agnostic.test_tensor_device.default(t)


def my_pad(t) -> Tensor:
    """
    Pads the input tensor with hardcoded padding parameters.

    Args:
        t: Input tensor

    Returns: Padded tensor with padding [1, 2, 2, 1], mode "constant", value 0.0
    """
    return torch.ops.libtorch_agnostic.my_pad.default(t)


def my_narrow(t, dim, start, length) -> Tensor:
    """
    Returns a new tensor that is a narrowed version of the input tensor.

    Args:
        t: Input tensor
        dim: Dimension along which to narrow
        start: Starting position
        length: Length of the narrowed section

    Returns: Narrowed tensor
    """
    return torch.ops.libtorch_agnostic.my_narrow.default(t, dim, start, length)


def my_copy_(dst, src, non_blocking) -> Tensor:
    """
    Returns tensor dst that is updated with src elements.

    Args:
        dst: Destination tensor
        src: Source tensor
        non_blocking: bool

    Returns: Updated tensor
    """
    return torch.ops.libtorch_agnostic.my_copy_.default(dst, src, non_blocking)


def my_clone(t) -> Tensor:
    """
    Returns a clone of input tensor.

    Args:
        t: Input tensor

    Returns: Cloned tensor
    """
    return torch.ops.libtorch_agnostic.my_clone.default(t)


def test_device_guard(device_index) -> int:
    """
    Tests the DeviceGuard functionality by creating a device guard and returning an empty tensor.

    Args:
        device_index: Device index to set the guard to

    Returns: result of cudaGetDevice() as an integer after using the guard
    """
    return torch.ops.libtorch_agnostic.test_device_guard.default(device_index)


def test_device_guard_set_index() -> int:
    """
    Tests the DeviceGuard set_index functionality by creating a device guard with index 1,
    then setting it to index 0, and returning the current device.

    Returns: result of cudaGetDevice() as an integer after using set_index
    """
    return torch.ops.libtorch_agnostic.test_device_guard_set_index.default()


def test_stream(device_index) -> int:
    """
    Tests the Stream functionality by getting the current stream ID for the specified device.

    Args:
        device_index: Device index to get the stream for

    Returns: Stream ID as an integer
    """
    return torch.ops.libtorch_agnostic.test_stream.default(device_index)


def test_get_current_device_index() -> int:
    """
    Tests the getCurrentDeviceIndex functionality by getting the current device index.

    Returns: Current device index as an integer
    """
    return torch.ops.libtorch_agnostic.test_get_current_device_index.default()


def my_new_empty_dtype_variant(t) -> Tensor:
    """
    Returns a new empty tensor with shape [2, 5] and dtype bfloat16

    Args:
        t: Input tensor used as a reference for device and other properties

    Returns: New empty tensor with shape [2, 5] and dtype bfloat16
    """
    return torch.ops.libtorch_agnostic.my_new_empty_dtype_variant.default(t)


def my_new_zeros_dtype_variant(t) -> Tensor:
    """
    Returns a new tensor filled with 0s with shape [2, 5] and dtype Float

    Args:
        t: Input tensor used as a reference for device and other properties

    Returns: New zeros tensor
    """
    return torch.ops.libtorch_agnostic.my_new_zeros_dtype_variant.default(t)


def my__foreach_mul_(tensors, others) -> ():
    """
    Updates tensors to be the result of pointwise multiplying with others.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: nothing, tensors is updated in place.
    """
    torch.ops.libtorch_agnostic.my__foreach_mul_.default(tensors, others)


def my__foreach_mul(tensors, others) -> list[Tensor]:
    """
    Returns a list of tensors that are the results of pointwise multiplying
    tensors and others.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: list of multiplied tensors
    """
    return torch.ops.libtorch_agnostic.my__foreach_mul.default(tensors, others)


def make_tensor_clones_and_call_foreach(t1, t2) -> list[Tensor]:
    """
    Returns a list of 2 tensors corresponding to the square of the inputs.

    Args:
        t1: Tensor
        t2: Tensor

    Returns: list of [t1^2, t2^2]
    """
    return torch.ops.libtorch_agnostic.make_tensor_clones_and_call_foreach.default(
        t1, t2
    )


def test_device_constructor(is_cuda, index, use_str):
    """
    Tests creating a Device from DeviceType and index, or from a string.

    Args:
        is_cuda: bool - if True, creates CUDA device; if False, creates CPU device
        index: int - device index
        use_str: bool - if True, constructs from string; if False, constructs from DeviceType

    Returns: Device - A device with the specified type and index
    """
    return torch.ops.libtorch_agnostic.test_device_constructor.default(
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
    return torch.ops.libtorch_agnostic.test_device_equality.default(d1, d2)


def test_device_set_index(device, index):
    """
    Tests Device set_index() method.

    Args:
        device: Device - device to modify
        index: int - new device index

    Returns: Device - device with updated index
    """
    return torch.ops.libtorch_agnostic.test_device_set_index.default(device, index)


def test_device_index(device) -> int:
    """
    Tests Device index() method.

    Args:
        device: Device - device to query

    Returns: int - device index
    """
    return torch.ops.libtorch_agnostic.test_device_index.default(device)


def test_device_is_cuda(device) -> bool:
    """
    Tests Device is_cuda() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CUDA
    """
    return torch.ops.libtorch_agnostic.test_device_is_cuda.default(device)


def test_device_is_cpu(device) -> bool:
    """
    Tests Device is_cpu() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CPU
    """
    return torch.ops.libtorch_agnostic.test_device_is_cpu.default(device)


def test_parallel_for(size, grain_size) -> Tensor:
    """
    Tests the parallel_for functionality by using it to fill a tensor with indices.
    Args:
        size: int - size of the tensor to create
        grain_size: int - grain size for parallel_for
    Returns: Tensor - a 1D int64 tensor where each element contains its index
        (if multiple threads are used the threadid will be encoded in the upper 32 bits)
    """
    return torch.ops.libtorch_agnostic.test_parallel_for.default(size, grain_size)


def test_get_num_threads() -> int:
    """
    Tests the get_num_threads functionality by returning the number of threads
    for the parallel backend.

    Returns: int - the number of threads for the parallel backend
    """
    return torch.ops.libtorch_agnostic.test_get_num_threads.default()


def my_empty(size, dtype=None, device=None) -> Tensor:
    """
    Creates an empty tensor with the specified size, dtype, and device.

    Args:
        size: list[int] - size of the tensor to create
        dtype: ScalarType or None - data type of the tensor
        device: Device or None - device on which to create the tensor

    Returns: Tensor - an uninitialized tensor with the specified properties
    """
    return torch.ops.libtorch_agnostic.my_empty.default(size, dtype, device)
