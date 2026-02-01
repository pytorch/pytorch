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
    torch.ops.libtorch_agn_2_10.my__foreach_mul_.default(tensors, others)


def my__foreach_mul(tensors, others) -> list[Tensor]:
    """
    Returns a list of tensors that are the results of pointwise multiplying
    tensors and others.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: list of multiplied tensors
    """
    return torch.ops.libtorch_agn_2_10.my__foreach_mul.default(tensors, others)


def make_tensor_clones_and_call_foreach(t1, t2) -> list[Tensor]:
    """
    Returns a list of 2 tensors corresponding to the square of the inputs.

    Args:
        t1: Tensor
        t2: Tensor

    Returns: list of [t1^2, t2^2]
    """
    return torch.ops.libtorch_agn_2_10.make_tensor_clones_and_call_foreach.default(
        t1, t2
    )


def test_tensor_device(t):
    """
    Tests Tensor device() method.

    Args:
        t: Tensor - tensor to get device from

    Returns: Device - device of the tensor
    """
    return torch.ops.libtorch_agn_2_10.test_tensor_device.default(t)


def test_device_constructor(is_cuda, index, use_str):
    """
    Tests creating a Device from DeviceType and index, or from a string.

    Args:
        is_cuda: bool - if True, creates CUDA device; if False, creates CPU device
        index: int - device index
        use_str: bool - if True, constructs from string; if False, constructs from DeviceType

    Returns: Device - A device with the specified type and index
    """
    return torch.ops.libtorch_agn_2_10.test_device_constructor.default(
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
    return torch.ops.libtorch_agn_2_10.test_device_equality.default(d1, d2)


def test_device_set_index(device, index):
    """
    Tests Device set_index() method.

    Args:
        device: Device - device to modify
        index: int - new device index

    Returns: Device - device with updated index
    """
    return torch.ops.libtorch_agn_2_10.test_device_set_index.default(device, index)


def test_device_index(device) -> int:
    """
    Tests Device index() method.

    Args:
        device: Device - device to query

    Returns: int - device index
    """
    return torch.ops.libtorch_agn_2_10.test_device_index.default(device)


def test_device_is_cuda(device) -> bool:
    """
    Tests Device is_cuda() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CUDA
    """
    return torch.ops.libtorch_agn_2_10.test_device_is_cuda.default(device)


def test_device_is_cpu(device) -> bool:
    """
    Tests Device is_cpu() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is CPU
    """
    return torch.ops.libtorch_agn_2_10.test_device_is_cpu.default(device)


def test_parallel_for(size, grain_size) -> Tensor:
    """
    Tests the parallel_for functionality by using it to fill a tensor with indices.
    Args:
        size: int - size of the tensor to create
        grain_size: int - grain size for parallel_for
    Returns: Tensor - a 1D int64 tensor where each element contains its index
        (if multiple threads are used the threadid will be encoded in the upper 32 bits)
    """
    return torch.ops.libtorch_agn_2_10.test_parallel_for.default(size, grain_size)


def test_get_num_threads() -> int:
    """
    Tests the get_num_threads functionality by returning the number of threads
    for the parallel backend.

    Returns: int - the number of threads for the parallel backend
    """
    return torch.ops.libtorch_agn_2_10.test_get_num_threads.default()


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
    return torch.ops.libtorch_agn_2_10.my_empty.default(
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
    return torch.ops.libtorch_agn_2_10.my_reshape.default(t, shape)


def my_view(t, size) -> Tensor:
    """
    Returns a new tensor with the same data as the input tensor but of a different shape.

    Args:
        t: Tensor - tensor to view
        size: list[int] - new size for the tensor

    Returns: Tensor - tensor with new view
    """
    return torch.ops.libtorch_agn_2_10.my_view.default(t, size)


def my_shape(t) -> tuple[int]:
    """
    Returns a shape of the input tensor.

    Args:
        t: Tensor - input tensor

    Returns: tuple - shape of the input tensor.
    """
    return torch.ops.libtorch_agn_2_10.my_shape.default(t)


def get_any_data_ptr(t, mutable) -> int:
    """
    Return data pointer value of the tensor.
    Args:
        t: Input tensor
        mutable: whether data pointer qualifier is mutable or const
    Returns: int - pointer value
    """
    return torch.ops.libtorch_agn_2_10.get_any_data_ptr.default(t, mutable)


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
    return torch.ops.libtorch_agn_2_10.get_template_any_data_ptr.default(
        t, dtype, mutable
    )


def my_get_curr_cuda_blas_handle() -> int:
    """
    Return the current cuBlasHandle_t pointer value.
    """
    return torch.ops.libtorch_agn_2_10.my_get_curr_cuda_blas_handle.default()


def my_string_op(t, accessor, passthru) -> tuple[list[str], int]:
    """
    The purpose of this op is to test inputting and outputting strings in a
    stable custom op. This particular op takes in a Tensor, a string denoting
    which tensor metadata API to call, and a pass through string to return a
    string list and the value of the tensor metadata.

    If accessor is "size" or "stride", query along the 0th dim.

    Args:
        t: Tensor - input tensor to query
        accessor: str - which property to access ("dim", "size", or "stride")
        passthru: str - a string that gets returned as the last element of the list

    Returns: tuple - (list of [accessor, value, passthru] as strings, value)
    """
    return torch.ops.libtorch_agn_2_10.my_string_op.default(t, accessor, passthru)


def my_get_current_cuda_stream(device_index: int) -> int:
    """
    Return the current cudaStream_t pointer value.

    Args:
        device_index: int - device index
    """
    return torch.ops.libtorch_agn_2_10.my_get_current_cuda_stream.default(device_index)


def my_set_current_cuda_stream(stream: int, device_index: int):
    """
    Set the current stream to cudaStream_t pointer value.

    Args:
        stream: int - cudaStream_t pointer value
        device_index: int - device index
    """
    return torch.ops.libtorch_agn_2_10.my_set_current_cuda_stream.default(
        stream, device_index
    )


def my_get_cuda_stream_from_pool(high_priority: bool, device_index: int) -> int:
    """
    Return the cudaStream_t pointer value from pool.

    Args:
        high_priority: bool - if true, return a stream with high priority
        device_index: int - device index
    """
    return torch.ops.libtorch_agn_2_10.my_get_cuda_stream_from_pool.default(
        high_priority, device_index
    )


def my_cuda_stream_synchronize(stream: int, device_index: int):
    """
    Synchronize cuda stream.

    Args:
        stream: int - cudaStream_t pointer value
        device_index: int - device index
    """
    return torch.ops.libtorch_agn_2_10.my_cuda_stream_synchronize(stream, device_index)


def my_from_blob(data_ptr, sizes, strides, device, dtype) -> Tensor:
    """
    Creates a Tensor from existing memory using torch::stable::from_blob.

    Args:
        data_ptr: int - pointer to the data buffer
        sizes: tuple[int] - size of the tensor
        strides: tuple[int] - strides of the tensor
        device: Device - device on which the tensor resides
        dtype: ScalarType - data type of the tensor
        storage_offset: int - offset in the storage
        layout: Layout - layout of the tensor

    Returns: Tensor - tensor wrapping the existing memory
    """
    return torch.ops.libtorch_agn_2_10.my_from_blob.default(
        data_ptr, sizes, strides, device, dtype
    )


def test_std_cuda_check_success() -> int:
    """
    Test STD_CUDA_CHECK macro with a successful CUDA operation.
    Returns the current CUDA device index.
    """
    return torch.ops.libtorch_agn_2_10.test_std_cuda_check_success.default()


def test_std_cuda_check_error() -> None:
    """
    Test STD_CUDA_CHECK macro with a failing CUDA operation.
    This should raise a RuntimeError with the CUDA error message.
    """
    torch.ops.libtorch_agn_2_10.test_std_cuda_check_error.default()


def test_std_cuda_kernel_launch_check_success() -> None:
    """
    Test STD_CUDA_KERNEL_LAUNCH_CHECK macro with a successful kernel launch.
    Launches a simple kernel and checks for errors.
    """
    torch.ops.libtorch_agn_2_10.test_std_cuda_kernel_launch_check_success.default()


def test_std_cuda_kernel_launch_check_error() -> None:
    """
    Test STD_CUDA_KERNEL_LAUNCH_CHECK macro with an invalid kernel launch.
    This should raise a RuntimeError with the CUDA kernel launch error message.
    """
    torch.ops.libtorch_agn_2_10.test_std_cuda_kernel_launch_check_error.default()


def my__foreach_mul_vec(tensors, others) -> list[Tensor]:
    """
    Returns a list of tensors that are the results of pointwise multiplying
    tensors and others. This variant tests const std::vector<Tensor>& parameters.

    Args:
        tensors: list of tensors
        others: list of tensors (with the same corresponding shapes as tensors)

    Returns: list of multiplied tensors
    """
    return torch.ops.libtorch_agn_2_10.my__foreach_mul_vec.default(tensors, others)


def my_string_op_const_string_ref(t, accessor, passthru) -> tuple[list[str], int]:
    """
    Tests TORCH_BOX with const std::string& parameters.

    Args:
        t: Tensor - input tensor to query
        accessor: str - which property to access ("dim", "size", or "stride")
        passthru: str - a string that gets returned as the last element of the list

    Returns: tuple - (list of [accessor, value, passthru] as strings, value)
    """
    return torch.ops.libtorch_agn_2_10.my_string_op_const_string_ref.default(
        t, accessor, passthru
    )


def my_string_op_const_string_view_ref(t, accessor, passthru) -> tuple[list[str], int]:
    """
    Tests TORCH_BOX with const std::string_view& parameters.

    Args:
        t: Tensor - input tensor to query
        accessor: str - which property to access ("dim", "size", or "stride")
        passthru: str - a string that gets returned as the last element of the list

    Returns: tuple - (list of [accessor, value, passthru] as strings, value)
    """
    return torch.ops.libtorch_agn_2_10.my_string_op_const_string_view_ref.default(
        t, accessor, passthru
    )


def my_string_op_string_ref(t, accessor, passthru) -> tuple[list[str], int]:
    """
    Tests TORCH_BOX with std::string& (non-const) parameters.

    Args:
        t: Tensor - input tensor to query
        accessor: str - which property to access ("dim", "size", or "stride")
        passthru: str - a string that gets returned as the last element of the list

    Returns: tuple - (list of [accessor, value, passthru] as strings, value)
    """
    return torch.ops.libtorch_agn_2_10.my_string_op_string_ref.default(
        t, accessor, passthru
    )


def my_set_requires_grad(t, requires_grad) -> Tensor:
    """
    Sets the requires_grad attribute on a tensor.

    Args:
        t: Tensor - input tensor
        requires_grad: bool - whether the tensor requires gradient

    Returns: Tensor - the input tensor with requires_grad set
    """
    return torch.ops.libtorch_agn_2_10.my_set_requires_grad.default(t, requires_grad)


def my_to_device(t, device) -> Tensor:
    """
    Moves a tensor to the specified device.

    Args:
        t: Tensor - input tensor
        device: Device - target device

    Returns: Tensor - tensor on the new device
    """
    return torch.ops.libtorch_agn_2_10.my_to_device.default(t, device)


def my_to_dtype(t, dtype) -> Tensor:
    """
    Converts a tensor to the specified dtype.

    Args:
        t: Tensor - input tensor
        dtype: ScalarType - target dtype (e.g., torch.float64)

    Returns: Tensor - tensor with the new dtype
    """
    return torch.ops.libtorch_agn_2_10.my_to_dtype.default(t, dtype)


def my_contiguous(t) -> Tensor:
    """
    Returns a contiguous tensor with the default memory format.

    Args:
        t: Tensor - input tensor

    Returns: Tensor - contiguous tensor
    """
    return torch.ops.libtorch_agn_2_10.my_contiguous.default(t)


def my_contiguous_memory_format(t, memory_format) -> Tensor:
    """
    Returns a contiguous tensor with the specified memory format.

    Args:
        t: Tensor - input tensor
        memory_format: MemoryFormat - memory format (e.g., torch.channels_last)

    Returns: Tensor - contiguous tensor with specified memory format
    """
    return torch.ops.libtorch_agn_2_10.my_contiguous_memory_format.default(
        t, memory_format
    )


def my_to_dtype_layout(
    t,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    copy=False,
    memory_format=None,
) -> Tensor:
    """
    Converts a tensor to the specified dtype, layout, device, and/or memory format.

    Args:
        t: Tensor - input tensor
        dtype: ScalarType or None - target dtype (e.g., torch.float64)
        layout: Layout or None - target layout (e.g., torch.strided)
        device: Device or None - target device
        pin_memory: bool or None - whether to use pinned memory
        non_blocking: bool - if True, try to perform the operation asynchronously
        copy: bool - if True, always copy the tensor
        memory_format: MemoryFormat or None - target memory format (e.g., torch.channels_last)

    Returns: Tensor - converted tensor
    """
    return torch.ops.libtorch_agn_2_10.my_to_dtype_layout.default(
        t, dtype, layout, device, pin_memory, non_blocking, copy, memory_format
    )


def my_new_empty(
    self,
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> Tensor:
    """
    Creates a new uninitialized tensor with the specified size and properties.

    Args:
        self: Tensor - input tensor (used to infer default properties)
        size: list[int] - size of the new tensor
        dtype: ScalarType or None - data type of the new tensor
        layout: Layout or None - layout of the new tensor
        device: Device or None - device for the new tensor
        pin_memory: bool or None - whether to use pinned memory

    Returns: Tensor - new uninitialized tensor
    """
    return torch.ops.libtorch_agn_2_10.my_new_empty.default(
        self, size, dtype, layout, device, pin_memory
    )


def my_new_zeros(
    self,
    size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> Tensor:
    """
    Creates a new tensor filled with zeros with the specified size and properties.

    Args:
        self: Tensor - input tensor (used to infer default properties)
        size: list[int] - size of the new tensor
        dtype: ScalarType or None - data type of the new tensor
        layout: Layout or None - layout of the new tensor
        device: Device or None - device for the new tensor
        pin_memory: bool or None - whether to use pinned memory

    Returns: Tensor - new tensor filled with zeros
    """
    return torch.ops.libtorch_agn_2_10.my_new_zeros.default(
        self, size, dtype, layout, device, pin_memory
    )


def my_sum(self, dim=None, keepdim=False, dtype=None) -> Tensor:
    """
    Returns the sum of the tensor along the specified dimensions.

    Args:
        self: Tensor - input tensor
        dim: list[int] - dimensions to sum over
        keepdim: bool - whether to keep the reduced dimensions
        dtype: ScalarType or None - the desired data type of returned tensor

    Returns: Tensor - the sum of the tensor along the specified dimensions
    """
    return torch.ops.libtorch_agn_2_10.my_sum.default(self, dim, keepdim, dtype)


def my_sum_out(out, self, dim=None, keepdim=False, dtype=None) -> Tensor:
    """
    Computes the sum of the tensor along the specified dimensions and writes
    the result to the output tensor.

    Args:
        out: Tensor - output tensor (modified in-place)
        self: Tensor - input tensor
        dim: list[int] - dimensions to sum over
        keepdim: bool - whether to keep the reduced dimensions
        dtype: ScalarType or None - the desired data type of returned tensor

    Returns: Tensor - the output tensor (same as out parameter)
    """
    return torch.ops.libtorch_agn_2_10.my_sum_out.default(
        out, self, dim, keepdim, dtype
    )


def my_sum_all(self) -> Tensor:
    """
    Returns the sum of all elements in the tensor (reduces to a scalar).

    Args:
        self: Tensor - input tensor

    Returns: Tensor - scalar tensor containing the sum of all elements
    """
    return torch.ops.libtorch_agn_2_10.my_sum_all.default(self)


def my_sum_dim1(self) -> Tensor:
    """
    Returns the sum of the tensor along dimension 1.

    Args:
        self: Tensor - input tensor

    Returns: Tensor - the sum of the tensor along dimension 1
    """
    return torch.ops.libtorch_agn_2_10.my_sum_dim1.default(self)


def my_full(
    size, fill_value, dtype=None, layout=None, device=None, pin_memory=None
) -> Tensor:
    """
    Creates a tensor filled with a scalar value.

    Args:
        size: list[int] - size of the tensor to create
        fill_value: float - value to fill the tensor with
        dtype: ScalarType or None - data type of the tensor
        layout: Layout or None - layout of the tensor
        device: Device or None - device on which to create the tensor
        pin_memory: bool or None - whether to use pinned memory

    Returns: Tensor - a tensor filled with fill_value
    """
    return torch.ops.libtorch_agn_2_10.my_full.default(
        size, fill_value, dtype, layout, device, pin_memory
    )


def my_subtract(self, other, alpha=1.0) -> Tensor:
    """
    Subtracts other from self, scaled by alpha.

    Computes: self - alpha * other

    Args:
        self: Tensor - input tensor
        other: Tensor - tensor to subtract
        alpha: float - scaling factor for other (default: 1.0)

    Returns: Tensor - result of subtraction
    """
    return torch.ops.libtorch_agn_2_10.my_subtract.default(self, other, alpha)
