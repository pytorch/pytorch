import operator
from typing import Callable, Optional

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.quantization.fx._lower_to_native_backend as _lower_to_native_backend
import torch.ao.quantization.quantization_mappings as quantization_mappings
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.backend_config import get_native_backend_config

from .ns_types import NSNodeTargetType


toq = torch.ops.quantized


def get_base_name_to_sets_of_related_ops() -> dict[str, set[NSNodeTargetType]]:
    # note: this set is modified below by items from backend_config
    sets_of_related_ops: list[set[NSNodeTargetType]] = [
        # conv modules
        {
            nn.Conv1d,
        },
        {
            nn.Conv2d,
        },
        {
            nn.Conv3d,
        },
        # conv functionals
        {
            F.conv1d,
        },
        {
            F.conv2d,
        },
        {
            F.conv3d,
        },
        # linear modules
        {
            nn.Linear,
        },
        # linear functionals
        {
            F.linear,
        },
        # average pool
        {
            nn.AvgPool1d,
            torch.avg_pool1d,
        },
        {
            nn.AvgPool2d,
            torch._C._nn.avg_pool2d,
        },
        {
            nn.AvgPool3d,
            torch._C._nn.avg_pool3d,
        },
        # adaptive average pool
        {
            nn.AdaptiveAvgPool1d,
            F.adaptive_avg_pool1d,
        },
        {
            nn.AdaptiveAvgPool2d,
            F.adaptive_avg_pool2d,
        },
        {
            nn.AdaptiveAvgPool3d,
            F.adaptive_avg_pool3d,
        },
        # LSTM
        {
            nn.LSTM,
        },
        # add
        {
            torch.add,
            operator.add,  # x + y
        },
        # cat
        {
            torch.cat,
        },
        # mul
        {
            torch.mul,
            operator.mul,
        },
        # relu
        {
            F.relu,
            nn.ReLU,
            "relu",
            "relu_",
            torch.relu,
        },
        # maxpool
        {
            nn.MaxPool1d,
            F.max_pool1d,
        },
        {
            nn.MaxPool2d,
            F.max_pool2d,
        },
        {
            nn.MaxPool3d,
            F.max_pool3d,
        },
        # sigmoid
        {
            torch.sigmoid,
            "sigmoid",
            "sigmoid_",
            nn.Sigmoid,
            F.sigmoid,
        },
        # BatchNorm
        {
            nn.BatchNorm2d,
        },
        {
            nn.BatchNorm3d,
        },
        # ConvTranspose
        {
            nn.ConvTranspose1d,
        },
        {
            nn.ConvTranspose2d,
        },
        {
            nn.ConvTranspose3d,
        },
        # functional transposed conv
        {
            F.conv_transpose1d,
        },
        {
            F.conv_transpose2d,
        },
        {
            F.conv_transpose3d,
        },
        # ELU
        {
            nn.ELU,
        },
        # Embedding
        {
            nn.Embedding,
        },
        # EmbeddingBag
        {
            nn.EmbeddingBag,
        },
        # GroupNorm
        {
            nn.GroupNorm,
        },
        # Hardswish
        {
            nn.Hardswish,
        },
        # InstanceNorm
        {
            nn.InstanceNorm1d,
        },
        {
            nn.InstanceNorm2d,
        },
        {
            nn.InstanceNorm3d,
        },
        # LayerNorm
        {
            nn.LayerNorm,
        },
        # LeakyReLU
        {
            nn.LeakyReLU,
        },
        # ReLU6
        {
            nn.ReLU6,
            F.relu6,
        },
        # F.elu
        {
            F.elu,
        },
        # F.hardswish
        {
            F.hardswish,
        },
        # F.group_norm
        {
            F.group_norm,
        },
        # F.instance_norm
        {
            F.instance_norm,
        },
        # F.layer_norm
        {
            F.layer_norm,
        },
        # F.leaky_relu
        {
            F.leaky_relu,
        },
        # F.silu
        {
            nn.SiLU,
            F.silu,
        },
        # F.mish
        {
            nn.Mish,
            F.mish,
        },
        # F.tanh
        {
            nn.Tanh,
            F.tanh,
            torch.tanh,
            "tanh_",
            "tanh",
        },
        # F.hardsigmoid
        {
            "hardsigmoid_",
            "hardsigmoid",
            F.hardsigmoid,
            nn.Hardsigmoid,
        },
        # F.hardtanh
        {
            nn.Hardtanh,
            F.hardtanh,
            F.hardtanh_,
        },
        # floordiv
        {
            operator.floordiv,
        },
        # unsqueeze
        {
            torch.unsqueeze,
        },
        # stack
        {
            torch.stack,
        },
        # squeeze
        {
            torch.squeeze,
        },
        # sort
        {
            torch.sort,
        },
        # repeat_interleave
        {
            torch.repeat_interleave,
        },
        # min
        {
            torch.min,
        },
        # mean
        {
            torch.mean,
        },
        # max
        {
            torch.max,
        },
        # transpose
        {
            torch.transpose,
        },
        # flatten
        {
            torch.flatten,
        },
        # clamp
        {
            torch.clamp,
        },
        # chunk
        {
            torch.chunk,
        },
        # interpolate
        {
            torch.nn.functional.interpolate,
        },
        # dropout
        {
            nn.Dropout,
        },
        # F.dropout
        {
            F.dropout,
        },
        # matmul
        {
            torch.matmul,
        },
        # Softmax
        {
            nn.Softmax,
        },
        # PReLU
        {
            nn.PReLU,
            nnq.PReLU,
        },
        # F.prelu
        {
            F.prelu,
            toq.prelu,
        },
        # pixel shuffle
        {
            nn.PixelShuffle,
        },
        {
            F.pixel_shuffle,
        },
        # pixel unshuffle
        {
            nn.PixelUnshuffle,
        },
        {
            F.pixel_unshuffle,
        },
        # narrow
        {
            torch.narrow,
        },
    ]

    # for each floating point op, add versions of the op added by
    # backend_config
    backend_config = get_native_backend_config()

    new_connections: list[tuple[Callable, Callable]] = [
        # technical debt edge case
        (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear),
    ]

    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # pattern format: (c, (b, a))
        first_element = pattern
        # look from the end, because pattern is in reverse order
        while isinstance(first_element, (list, tuple)):
            first_element = first_element[-1]

        if config.fused_module is not None:
            # case 1: pattern fuses a pattern of ops into an op
            # example: nn.Conv1d, nn.ReLU fused into nni.ConvReLU1d
            new_connections.append((first_element, config.fused_module))

        if config.qat_module is not None:
            # case 2: pattern swaps a module into a QAT module
            # example: nni.ConvReLU1d swapped into nniqat.ConvReLU1d
            new_connections.append((first_element, config.qat_module))

        if config.reference_quantized_module is not None:
            # case 3: reference version of floating point module, such as
            # nn.Conv2d and nnqr.Conv2d
            new_connections.append((first_element, config.reference_quantized_module))

    #
    # Add reference module swaps from default lowering path
    #

    for source_to_target in (
        _lower_to_native_backend.STATIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.WEIGHT_ONLY_LOWER_MODULE_MAP,
        _lower_to_native_backend.SPECIAL_PATTERN_LOWER_MODULE_MAP,
    ):
        for source, target in source_to_target.items():  # type: ignore[attr-defined]
            new_connections.append((source, target))

    for source_to_double_target in (
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_MAP,
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_FUSED_MODULE_MAP,
    ):
        for source, (target1, target2) in source_to_double_target.items():  # type: ignore[attr-defined]
            new_connections.append((source, target1))
            new_connections.append((source, target2))

    #
    # Add function swaps from default lowering path
    #

    for source, (  # type:ignore[assignment]
        target1,
        target2,
    ) in _lower_to_native_backend.STATIC_LOWER_FUNCTIONAL_MAP.items():
        new_connections.append((source, target1))
        # pyrefly: ignore  # bad-argument-type
        new_connections.append((source, target2))

    for source_to_target in (
        _lower_to_native_backend.QBIN_OP_MAPPING,
        _lower_to_native_backend.QBIN_RELU_OP_MAPPING,
        quantization_mappings.DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
    ):
        for source, target in source_to_target.items():  # type:ignore[assignment]
            # pyrefly: ignore  # bad-argument-type
            new_connections.append((source, target))

    #
    # Add other swaps, ideally in the future this could be removed
    # after the lowering code stops using these.
    #
    for source_to_target in (
        quantization_mappings.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    ):
        for source, target in source_to_target.items():  # type:ignore[assignment]
            new_connections.append((source, target))

    # add the new connections from backend_config
    for item1, item2 in new_connections:
        for set_of_related_ops in sets_of_related_ops:
            if item1 in set_of_related_ops or item2 in set_of_related_ops:
                set_of_related_ops.add(item1)
                set_of_related_ops.add(item2)
                break

    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] = {}

    for counter, set_of_related_ops in enumerate(sets_of_related_ops):
        base_name = str(counter)
        base_name_to_sets_of_related_ops[base_name] = set_of_related_ops

    return base_name_to_sets_of_related_ops


def get_base_name_for_op(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    op: NSNodeTargetType,
) -> Optional[str]:
    for base_name, set_of_related_ops in base_name_to_sets_of_related_ops.items():
        if op in set_of_related_ops:
            return base_name
    return None


def add_op_to_sets_of_related_ops(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    op: NSNodeTargetType,
    related_op: Optional[NSNodeTargetType],
) -> None:
    if related_op is not None:
        for set_of_related_ops in base_name_to_sets_of_related_ops.values():
            if related_op in set_of_related_ops:
                set_of_related_ops.add(op)
                return
        # if we got here, related_op was not found
        raise AssertionError(f"{related_op} was not found")
    else:
        counter = 0
        while str(counter) in base_name_to_sets_of_related_ops:
            counter += 1
        base_name_to_sets_of_related_ops[str(counter)] = {op}


# TODO(future PR): clean this up
def get_node_type_to_io_type_map() -> dict[str, set[NSNodeTargetType]]:
    FUNS_IO_TYPE_FP32: set[NSNodeTargetType] = {
        F.linear,
        F.conv1d,
        F.conv2d,
        F.conv3d,
        torch.cat,
        F.elu,
        F.hardswish,
        F.instance_norm,
        F.layer_norm,
        F.leaky_relu,
        F.dropout,
        F.silu,
        F.mish,
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        torch.sum,
        F.prelu,
    }

    FUNS_IO_TYPE_FP16: set[NSNodeTargetType] = set()

    FUNS_IO_TYPE_INT8: set[NSNodeTargetType] = {
        toq.linear,
        toq.linear_relu,
        toq.conv1d,
        toq.conv1d_relu,
        toq.conv2d,
        toq.conv2d_relu,
        toq.conv3d,
        toq.conv3d_relu,
        toq.cat,
        toq.elu,
        toq.hardswish,
        toq.instance_norm,
        toq.layer_norm,
        toq.leaky_relu,
        toq.dropout,
        toq.prelu,
        # TODO(future PR): implement shadowing for binary ops and
        # uncomment below
        # toq.add,
        # toq.mul,
    }

    FUNS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        F.relu,
        F.tanh,
        torch.tanh,
        F.sigmoid,
        torch.sigmoid,
        F.hardsigmoid,
        operator.floordiv,
        torch.adaptive_avg_pool1d,
        F.adaptive_avg_pool2d,
        F.adaptive_avg_pool3d,
        F.dropout,
        F.hardtanh,
        F.hardtanh_,
        F.interpolate,
        F.max_pool1d,
        F.max_pool2d,
        F.max_pool3d,
        F.relu6,
        F.pixel_shuffle,
        F.pixel_unshuffle,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.cat,
        torch.chunk,
        torch.clamp,
        torch.flatten,
        torch.transpose,
        torch.max,
        torch.mean,
        torch.min,
        torch.narrow,
        torch.repeat_interleave,
        torch.sort,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        operator.add,
    }

    MODS_IO_TYPE_FP32: set[NSNodeTargetType] = {
        nn.Linear,
        nnqat.Linear,
        nnqatd.Linear,
        nnqd.Linear,
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nnqat.Conv1d,
        nnqat.Conv2d,
        nnqat.Conv3d,
        nnqat.Embedding,
        nnqat.EmbeddingBag,
        nn.LSTM,
        # note: nnqd.Linear is an instance of nnq.Linear, so this
        # check has to happen before the int8 module check
        nnqd.LSTM,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Dropout,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.ELU,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.ReLU6,
        nn.SiLU,
        nn.Mish,
        nn.Softmax,
        nn.PReLU,
        nni.BNReLU2d,
        nni.BNReLU3d,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
        nni.LinearReLU,
        nni.LinearBn1d,
        nni.ConvBn1d,
        nni.ConvBn2d,
        nni.ConvBn3d,
        nniqat.ConvBn1d,
        nniqat.ConvBn2d,
        nniqat.ConvBn3d,
        nniqat.ConvBnReLU1d,
        nniqat.ConvBnReLU2d,
        nniqat.ConvBnReLU3d,
        nniqat.ConvReLU1d,
        nniqat.ConvReLU2d,
        nniqat.ConvReLU3d,
        nniqat.LinearReLU,
        nniqat.LinearBn1d,
        nniqd.LinearReLU,
        nni.LinearLeakyReLU,
        nni.LinearTanh,
        nni.ConvAdd2d,
        nni.ConvAddReLU2d,
    }

    MODS_IO_TYPE_INT8: set[NSNodeTargetType] = {
        nnq.Linear,
        nnq.Conv1d,
        nnq.Conv2d,
        nnq.Conv3d,
        nnq.BatchNorm2d,
        nnq.BatchNorm3d,
        nnq.Dropout,
        nnq.ConvTranspose1d,
        nnq.ConvTranspose2d,
        nnq.ELU,
        nnq.InstanceNorm1d,
        nnq.InstanceNorm2d,
        nnq.InstanceNorm3d,
        nnq.LayerNorm,
        nnq.Hardswish,
        nnq.LeakyReLU,
        nnq.Embedding,
        nnq.EmbeddingBag,
        nnq.Dropout,
        nnq.Softmax,
        nnq.PReLU,
        nniq.BNReLU2d,
        nniq.BNReLU3d,
        nniq.ConvReLU1d,
        nniq.ConvReLU2d,
        nniq.ConvReLU3d,
        nniq.LinearReLU,
        nniq.LinearLeakyReLU,
        nniq.LinearTanh,
        nniq.ConvAdd2d,
        nniq.ConvAddReLU2d,
    }

    MODS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        nn.ReLU,
        nn.Tanh,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.Dropout,
        nn.Hardtanh,
        nn.Identity,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.PixelShuffle,
        nn.PixelUnshuffle,
        nn.ReLU6,
    }

    METHS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        "sigmoid_",
        "sigmoid",
        "tanh_",
        "tanh",
        "hardsigmoid_",
        "hardsigmoid",
        "relu_",
        "relu",
    }

    return {
        "funs_io_type_fp32": FUNS_IO_TYPE_FP32,
        "funs_io_type_fp16": FUNS_IO_TYPE_FP16,
        "funs_io_type_int8": FUNS_IO_TYPE_INT8,
        "funs_io_type_fp32_or_int8": FUNS_IO_TYPE_FP32_OR_INT8,
        "mods_io_type_fp32": MODS_IO_TYPE_FP32,
        "mods_io_type_int8": MODS_IO_TYPE_INT8,
        "mods_io_type_fp32_or_int8": MODS_IO_TYPE_FP32_OR_INT8,
        "meths_io_type_fp32_or_int8": METHS_IO_TYPE_FP32_OR_INT8,
    }


def get_unmatchable_types_map() -> dict[str, set[NSNodeTargetType]]:
    FUNS_UNMATCHABLE: set[NSNodeTargetType] = {
        torch.quantize_per_tensor,
        operator.getitem,
    }

    MODS_UNMATCHABLE: set[NSNodeTargetType] = {
        nn.Identity,
    }

    METHS_UNMATCHABLE: set[NSNodeTargetType] = {
        "to",
        "dequantize",
        "reshape",
        "view",
        "unsqueeze_",
        "unsqueeze",
        "transpose",
        "squeeze_",
        "squeeze",
        "size",
        "shape",
        "resize_",
        "repeat_interleave",
        "repeat",
        "permute",
        "numel",
        "mean",
        "detach_",
        "detach",
        "contiguous",
        "clamp",
        "chunk",
    }

    return {
        "funs_unmatchable": FUNS_UNMATCHABLE,
        "mods_unmatchable": MODS_UNMATCHABLE,
        "meths_unmatchable": METHS_UNMATCHABLE,
    }
