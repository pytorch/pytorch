import torch


def my_is_privateuseone(t) -> bool:
    """
    Returns is_privateuseone on the input tensor.

    Args:
        t: any Tensor

    Returns:
        a bool
    """
    return torch.ops.libtorch_agn_2_16.my_is_privateuseone.default(t)


def test_device_is_privateuseone(device) -> bool:
    """
    Tests Device is_privateuseone() method.

    Args:
        device: Device - device to check

    Returns: bool - True if device is privateuseone
    """
    return torch.ops.libtorch_agn_2_16.test_device_is_privateuseone.default(device)
