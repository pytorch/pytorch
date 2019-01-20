# DO NOT alter/distruct/free input object !
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six


def make_np(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, six.string_types):  # Caffe2 will pass name of blob(s) to fetch
        return prepare_caffe2(x)
    if np.isscalar(x):
        return np.array([x])
    if 'torch' in str(type(x)):
        return prepare_pytorch(x)
    if 'chainer' in str(type(x)):
        return prepare_chainer(x)
    if 'mxnet' in str(type(x)):
        return prepare_mxnet(x)
    raise NotImplementedError(
        'Got {}, but expected numpy array or torch tensor.'.format(type(x)))


def prepare_pytorch(x):
    import torch
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    x = x.cpu().numpy()
    return x


def prepare_theano(x):
    import theano
    pass


def prepare_caffe2(x):
    from caffe2.python import workspace
    x = workspace.FetchBlob(x)
    return x


def prepare_mxnet(x):
    x = x.asnumpy()
    return x


def prepare_chainer(x):
    import chainer
    x = chainer.cuda.to_cpu(x.data)
    return x
