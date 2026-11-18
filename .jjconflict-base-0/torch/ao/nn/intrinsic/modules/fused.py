# mypy: allow-untyped-defs
import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
    ReLU,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, relu):
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(linear, relu)


class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class BNReLU2d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, bn):
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(linear, bn)


class LinearLeakyReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, leaky_relu):
        if not (type(linear) is Linear and type(leaky_relu) is torch.nn.LeakyReLU):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(leaky_relu).__name__}"
            )
        super().__init__(linear, leaky_relu)


class LinearTanh(_FusedModule):
    r"""This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, tanh):
        if not (type(linear) is Linear and type(tanh) is torch.nn.Tanh):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(tanh).__name__}"
            )
        super().__init__(linear, tanh)


class ConvAdd2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add):
        super().__init__(conv)
        self.add = add

    def forward(self, x1, x2):  # type: ignore[override]
        r"""Applies convolution to x1 and adds the result to x2."""
        return self.add(self[0](x1), x2)


class ConvAddReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add, relu):
        super().__init__(conv)
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):  # type: ignore[override]
        r"""Applies convolution to x1, adds the result to x2, and applies ReLU."""
        return self.relu(self.add(self[0](x1), x2))
