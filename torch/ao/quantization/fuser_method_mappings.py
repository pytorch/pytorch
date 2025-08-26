# mypy: allow-untyped-defs
import itertools
from typing import Any, Callable, Optional, Union

import torch.ao.nn.intrinsic as nni
import torch.nn as nn
from torch.ao.quantization.utils import get_combined_dict, MatchAllNode, Pattern


__all__ = [
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_linear_bn",
    "fuse_convtranspose_bn",
    "get_fuser_method",
    "get_fuser_method_new",
]


def fuse_conv_bn(is_qat, conv, bn):
    r"""Return the fused the conv and bn modules.
    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert conv.training == bn.training, (
        "Conv and BN both must be in the same mode (train or eval)."
    )

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if is_qat:
        assert bn.num_features == conv.out_channels, (
            "Output channel of Conv2d must match num_features of BatchNorm2d"
        )
        assert bn.affine, "Only support fusing BatchNorm2d with affine set to True"
        assert bn.track_running_stats, (
            "Only support fusing BatchNorm2d with tracking_running_stats set to True"
        )
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn)}")
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    r"""Return the fused conv and bv modules.

    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
    assert conv.training == bn.training == relu.training, (
        "Conv and BN both must be in the same mode (train or eval)."
    )
    fused_module: Optional[type[nn.Sequential]] = None
    if is_qat:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        assert bn.num_features == conv.out_channels, (
            "Output channel of Conv must match num_features of BatchNorm"
        )
        assert bn.affine, "Only support fusing BatchNorm with affine set to True"
        assert bn.track_running_stats, (
            "Only support fusing BatchNorm with tracking_running_stats set to True"
        )
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, relu)}")
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError(f"Cannot fuse eval modules: {(conv, bn, relu)}")


def fuse_linear_bn(is_qat, linear, bn):
    r"""Return the fused linear and bn modules.
    Given the linear and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer

    Examples::

        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert linear.training == bn.training, (
        "Linear and BN both must be in the same mode (train or eval)."
    )

    if is_qat:
        assert bn.num_features == linear.out_features, (
            "Output features of Linear must match num_features of BatchNorm1d"
        )
        assert bn.affine, "Only support fusing BatchNorm1d with affine set to True"
        assert bn.track_running_stats, (
            "Only support fusing BatchNorm1d with tracking_running_stats set to True"
        )
        return nni.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_convtranspose_bn(is_qat, convt, bn):
    r"""Return the fused ConvTranspose and bn modules.
    Given ConvTranspose and bn modules, fuses them and returns the fused module

    Args:
        convt: Module instance of type ConvTransposeNd
        bn: BatchNormNd instance that needs to be fused with the linear layer.
            batch norm N should match the ConvTranspose N

    Examples::

        >>> m1 = nn.ConvTranspose2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_convtranspose_bn(m1, b1)
    """
    assert convt.training == bn.training, (
        "ConvTranspose and BN both must be in the same mode (train or eval)."
    )

    if is_qat:
        raise Exception(  # noqa: TRY002
            "Fusing ConvTranspose+BatchNorm not yet supported in QAT."
        )
    else:
        return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)


def _sequential_wrapper2(sequential):
    """Return a sequential wrapped that for is_qat and two modules.
    Given a sequential class for two modules, return a function that takes
    is_qat, and then two modules as argument, that ignores the is_qat flag
    and always returns the sequential that combines the two input modules
    """

    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)

    return fuser_method


_DEFAULT_OP_LIST_TO_FUSER_METHOD: dict[tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU1d),
    (nn.Conv2d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU2d),
    (nn.Conv3d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU3d),
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): _sequential_wrapper2(nni.LinearReLU),
    (nn.BatchNorm2d, nn.ReLU): _sequential_wrapper2(nni.BNReLU2d),
    (nn.BatchNorm3d, nn.ReLU): _sequential_wrapper2(nni.BNReLU3d),
    (nn.ConvTranspose1d, nn.BatchNorm1d): fuse_convtranspose_bn,
    (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_convtranspose_bn,
    (nn.ConvTranspose3d, nn.BatchNorm3d): fuse_convtranspose_bn,
}


def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Get fuser method for the given list of module types.

    Get fuser method for the given list of module types,
    return None if fuser method does not exist
    """
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = {}
    all_mappings = get_combined_dict(
        _DEFAULT_OP_LIST_TO_FUSER_METHOD, additional_fuser_method_mapping
    )
    fuser_method = all_mappings.get(op_list, None)
    assert fuser_method is not None, f"did not find fuser method for: {op_list} "
    return fuser_method


def _reverse2(f):
    def reversed(is_qat, x, y):
        return f(is_qat, y, x)

    return reversed


def _reverse3(f):
    def reversed(is_qat, x, w):
        y, z = w
        return f(is_qat, z, y, x)

    return reversed


def _get_valid_patterns(op_pattern):
    """Return a list of valid patterns generated from the op_pattern.

    Returns a list of valid patterns generated from the op_pattern,
    since MatchAllNode can match all types of nodes,
    e.g. pattern (torch.nn.Conv2d, torch.add) should also be able to match keys like
    (MatchAllNode, torch.add) and (torch.nn.Conv2d, MatchAllNode)

    Example Input:
    (torch.add, (torch.nn.ReLU, torch.nn.Conv2d))

    Example Output:
    [(torch.add, (torch.nn.ReLU, torch.nn.Conv2d)),
     (torch.add, (torch.nn.ReLU, MatchAllNode)),
     (torch.add, (MatchAllNode, torch.nn.Conv2d)),
     (torch.add, (MatchAllNode, MatchAllNode)),
     (MatchAllNode, (torch.nn.ReLU, torch.nn.Conv2d)),
     (MatchAllNode, (torch.nn.ReLU, MatchAllNode)),
     (MatchAllNode, (MatchAllNode, torch.nn.Conv2d)),
     (MatchAllNode, (MatchAllNode, MatchAllNode)),
    ]
    """
    result: list[Any]
    if isinstance(op_pattern, (tuple, list)):
        sub_combs = [_get_valid_patterns(sub_pattern) for sub_pattern in op_pattern]
        result = list(itertools.product(*sub_combs))
    else:
        result = [op_pattern, MatchAllNode]
    return result


def get_fuser_method_new(
    op_pattern: Pattern,
    fuser_method_mapping: dict[Pattern, Union[nn.Sequential, Callable]],
):
    """Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    """
    op_patterns = _get_valid_patterns(op_pattern)
    fuser_method = None
    for op_pattern in op_patterns:
        fuser_method = fuser_method_mapping.get(op_pattern, None)
        if fuser_method is not None:
            break
    assert fuser_method is not None, f"did not find fuser method for: {op_pattern} "
    return fuser_method
