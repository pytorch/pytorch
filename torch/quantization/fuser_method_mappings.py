import torch.nn as nn
import torch.nn.intrinsic as nni

from typing import Union, Callable, Tuple, Dict

OP_LIST_TO_FUSER_METHOD : Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}

def register_fuser_method(op_list, fuser_method):
    r''' Register a fuser method for a tuple of ops, will be called
    during fusion step
    '''
    assert isinstance(op_list, tuple), 'op list must be a tuple'
    OP_LIST_TO_FUSER_METHOD[op_list] = fuser_method

def get_fuser_method(op_list):
    r''' Get fuser method for the given list of module types
    '''
    return OP_LIST_FUSER_METHOD.get(op_list)
