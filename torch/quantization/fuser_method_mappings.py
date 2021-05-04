import torch.nn as nn
import torch.nn.intrinsic as nni

from typing import Union, Callable, Tuple, Dict, Optional, Type

from .utils import get_combined_dict

def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn)))
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fused_module : Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))
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
            raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))

def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and returns the fused module

    Args:
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer

    Examples::

        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        raise Exception("Fusing Linear+BatchNorm not yet supported in training.")
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

DEFAULT_OP_LIST_TO_FUSER_METHOD : Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}

def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    ''' Get fuser method for the given list of module types,
    return None if fuser method does not exist
    '''
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = dict()
    all_mappings = get_combined_dict(DEFAULT_OP_LIST_TO_FUSER_METHOD,
                                     additional_fuser_method_mapping)
    fuser_method = all_mappings.get(op_list, None)
    assert fuser_method is not None, "did not find fuser method for: {} ".format(op_list)
    return fuser_method
