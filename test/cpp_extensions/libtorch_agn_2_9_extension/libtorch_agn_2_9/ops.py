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
    return torch.ops.libtorch_agn_2_9.sgd_out_of_place.default(
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
    return torch.ops.libtorch_agn_2_9.identity.default(t)


def my_abs(t) -> Tensor:
    """
    Returns abs on the input tensor, outputs a new Tensor

    Args:
        t: any Tensor

    Returns:
        a Tensor
    """
    return torch.ops.libtorch_agn_2_9.my_abs.default(t)


def my_is_cpu(t) -> bool:
    """
    Returns is_cpu on the input tensor.

    Args:
        t: any Tensor

    Returns:
        a bool
    """
    return torch.ops.libtorch_agn_2_9.my_is_cpu.default(t)


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
    return torch.ops.libtorch_agn_2_9.my_ones_like.default(tensor, device)


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
    return torch.ops.libtorch_agn_2_9.exp_neg_is_leaf.default(t1, t2, t3)


def neg_exp(t) -> Tensor:
    """
    Returns a Tensor composing neg of exp

    Args:
        t: Tensor

    Returns: neg(exp(t))
    """
    return torch.ops.libtorch_agn_2_9.neg_exp.default(t)


def divide_neg_exp(t) -> Tensor:
    """
    Returns a Tensor division of neg and exp

    Args:
        t: Tensor

    Returns: divide(neg(t), exp(t))
    """
    return torch.ops.libtorch_agn_2_9.divide_neg_exp.default(t)


def is_contiguous(t) -> bool:
    """
    Returns a bool indicating if the input tensor is contiguous

    Args:
        t: Tensor

    Returns: is_contiguous(t)
    """
    return torch.ops.libtorch_agn_2_9.is_contiguous.default(t)


def my_transpose(t, dim0, dim1) -> Tensor:
    """
    Returns t.transpose(dim0, dim1)

    Args:
        t: Tensor

    Returns: my_transpose(t, dim0, dim1)
    """
    return torch.ops.libtorch_agn_2_9.my_transpose.default(t, dim0, dim1)


def my_empty_like(t) -> Tensor:
    """
    Returns t.empty_like()

    Args:
        t: Tensor

    Returns: my_empty_like(t)
    """
    return torch.ops.libtorch_agn_2_9.my_empty_like.default(t)


def my_zero_(t) -> Tensor:
    """
    Returns t.zero_()

    Args:
        t: Tensor

    Returns: my_zero_(t)
    """
    return torch.ops.libtorch_agn_2_9.my_zero_.default(t)


def my_amax(t) -> Tensor:
    """
    Returns t.amax()

    Args:
        t: Tensor

    Returns: amax(t)
    """
    return torch.ops.libtorch_agn_2_9.my_amax.default(t)


def my_amax_vec(t) -> Tensor:
    """
    Returns t.amax()

    Args:
        t: Tensor

    Returns: amax(t)
    """
    return torch.ops.libtorch_agn_2_9.my_amax_vec.default(t)


def fill_infinity(t) -> Tensor:
    """
    Fills the tensor with inf.

    Args:
        t: Tensor to fill

    Returns: The modified tensor (same as input)
    """
    return torch.ops.libtorch_agn_2_9.fill_infinity.default(t)


def test_default_constructor(defined) -> bool:
    """
    Tests the default constructor for torch::stable::Tensor.

    Args:
        defined: bool - if True, tests defined tensor; if False, tests undefined tensor

    Returns: bool - result of calling .defined() on the tensor
    """
    return torch.ops.libtorch_agn_2_9.test_default_constructor.default(defined)


def mv_tensor_accessor(m, v) -> Tensor:
    """
    Returns matrix-vector product.

    Args:
        m: any 2-D Tensor with shape (N, M)
        v: any 1-D Tensor with shape (M,)

    Returns:
        a 1-D Tensor with shape (N,)
    """
    return torch.ops.libtorch_agn_2_9.mv_tensor_accessor.default(m, v)


def my_pad(t) -> Tensor:
    """
    Pads the input tensor with hardcoded padding parameters.

    Args:
        t: Input tensor

    Returns: Padded tensor with padding [1, 2, 2, 1], mode "constant", value 0.0
    """
    return torch.ops.libtorch_agn_2_9.my_pad.default(t)


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
    return torch.ops.libtorch_agn_2_9.my_narrow.default(t, dim, start, length)


def my_copy_(dst, src, non_blocking) -> Tensor:
    """
    Returns tensor dst that is updated with src elements.

    Args:
        dst: Destination tensor
        src: Source tensor
        non_blocking: bool

    Returns: Updated tensor
    """
    return torch.ops.libtorch_agn_2_9.my_copy_.default(dst, src, non_blocking)


def my_clone(t) -> Tensor:
    """
    Returns a clone of input tensor.

    Args:
        t: Input tensor

    Returns: Cloned tensor
    """
    return torch.ops.libtorch_agn_2_9.my_clone.default(t)


def test_device_guard(device_index) -> int:
    """
    Tests the DeviceGuard functionality by creating a device guard and returning an empty tensor.

    Args:
        device_index: Device index to set the guard to

    Returns: result of cudaGetDevice() as an integer after using the guard
    """
    return torch.ops.libtorch_agn_2_9.test_device_guard.default(device_index)


def test_device_guard_set_index() -> int:
    """
    Tests the DeviceGuard set_index functionality by creating a device guard with index 1,
    then setting it to index 0, and returning the current device.

    Returns: result of cudaGetDevice() as an integer after using set_index
    """
    return torch.ops.libtorch_agn_2_9.test_device_guard_set_index.default()


def test_stream(device_index) -> int:
    """
    Tests the Stream functionality by getting the current stream ID for the specified device.

    Args:
        device_index: Device index to get the stream for

    Returns: Stream ID as an integer
    """
    return torch.ops.libtorch_agn_2_9.test_stream.default(device_index)


def test_get_current_device_index() -> int:
    """
    Tests the getCurrentDeviceIndex functionality by getting the current device index.

    Returns: Current device index as an integer
    """
    return torch.ops.libtorch_agn_2_9.test_get_current_device_index.default()


def my_new_empty_dtype_variant(t) -> Tensor:
    """
    Returns a new empty tensor with shape [2, 5] and dtype bfloat16

    Args:
        t: Input tensor used as a reference for device and other properties

    Returns: New empty tensor with shape [2, 5] and dtype bfloat16
    """
    return torch.ops.libtorch_agn_2_9.my_new_empty_dtype_variant.default(t)


def my_new_zeros_dtype_variant(t) -> Tensor:
    """
    Returns a new tensor filled with 0s with shape [2, 5] and dtype Float

    Args:
        t: Input tensor used as a reference for device and other properties

    Returns: New zeros tensor
    """
    return torch.ops.libtorch_agn_2_9.my_new_zeros_dtype_variant.default(t)


def my_flatten(t, start_dim=0, end_dim=-1) -> Tensor:
    """
    Flattens the input tensor from start_dim to end_dim into a single dimension.

    Args:
        t: Tensor - tensor to flatten
        start_dim: int - first dimension to flatten (default: 0)
        end_dim: int - last dimension to flatten (default: -1)

    Returns: Tensor - flattened tensor
    """
    return torch.ops.libtorch_agn_2_9.my_flatten.default(t, start_dim, end_dim)


def my_optional_tensor_ref(maybe_tensor, default_size) -> Tensor:
    """
    Tests TORCH_BOX with const std::optional<Tensor>& parameter.
    Returns the tensor if present, otherwise returns a zeros tensor of specified size.

    Args:
        maybe_tensor: Optional[Tensor] - optional input tensor
        default_size: int - size of the default zeros tensor if maybe_tensor is None

    Returns: Tensor - the input tensor or a zeros tensor
    """
    return torch.ops.libtorch_agn_2_9.my_optional_tensor_ref.default(
        maybe_tensor, default_size
    )


def my_storage_offset(t) -> int:
    """
    Returns the storage offset of the input tensor.

    Args:
        t: Tensor - input tensor

    Returns: int - storage offset
    """
    return torch.ops.libtorch_agn_2_9.my_storage_offset.default(t)


def my_element_size(t) -> int:
    """
    Returns the element size in bytes of the input tensor.

    Args:
        t: Tensor - input tensor

    Returns: int - element size in bytes
    """
    return torch.ops.libtorch_agn_2_9.my_element_size.default(t)


def my_unsqueeze(t, dim) -> Tensor:
    """
    Returns a new tensor with a dimension of size one inserted at the specified position.

    Args:
        t: Tensor - input tensor
        dim: int - the index at which to insert the singleton dimension

    Returns: Tensor - unsqueezed tensor
    """
    return torch.ops.libtorch_agn_2_9.my_unsqueeze.default(t, dim)


def my_squeeze(t, dim) -> Tensor:
    """
    Returns a tensor with the specified dimension of size 1 removed.

    Args:
        t: Tensor - input tensor
        dim: int - the dimension to squeeze

    Returns: Tensor - squeezed tensor
    """
    return torch.ops.libtorch_agn_2_9.my_squeeze.default(t, dim)


def my_select(t, dim, index) -> Tensor:
    """
    Slices the tensor along the selected dimension at the given index.

    Args:
        t: Tensor - input tensor
        dim: int - the dimension to slice along
        index: int - the index to select

    Returns: Tensor - sliced tensor with one fewer dimension
    """
    return torch.ops.libtorch_agn_2_9.my_select.default(t, dim, index)


def my_matmul(self, other) -> Tensor:
    """
    Matrix product of two tensors.

    Args:
        self: Tensor - first tensor
        other: Tensor - second tensor

    Returns: Tensor - matrix product
    """
    return torch.ops.libtorch_agn_2_9.my_matmul.default(self, other)
