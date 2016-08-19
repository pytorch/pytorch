import unittest
from itertools import product
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.autograd.leaf import Leaf

def get_cpu_type(t):
    assert t.__module__ == 'torch.cuda'
    return getattr(torch, t.__class__.__name__)


def get_gpu_type(t):
    assert t.__module__ == 'torch'
    return getattr(torch.cuda, t.__name__)


def to_gpu(obj, tensor_type=None):
    if torch.isTensor(obj):
        if tensor_type:
            return tensor_type(obj.size()).copy_(obj)
        return get_gpu_type(type(obj))(obj.size()).copy_(obj)
    elif torch.isStorage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, Variable):
        assert type(obj.creator) == Leaf
        return Variable(obj.data.clone().type(tensor_type))
    elif isinstance(obj, list):
        return [to_gpu(o, tensor_type) for o in obj]
    else:
        return deepcopy(obj)


def iter_indices(tensor):
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False


class TestCase(unittest.TestCase):
    precision = 1e-5

    def assertEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        if isinstance(x, Variable) and isinstance(y, Variable):
            x = x.data
            y = y.data

        if torch.isTensor(x) and torch.isTensor(y):
            max_err = 0
            super(TestCase, self).assertEqual(x.size().tolist(), y.size().tolist())
            for index in iter_indices(x):
                max_err = max(max_err, abs(x[index] - y[index]))
            self.assertLessEqual(max_err, prec)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        else:
            try:
                self.assertLessEqual(abs(x - y), prec)
                return
            except:
                pass
            super(TestCase, self).assertEqual(x, y)


def make_jacobian(input, num_out):
    if torch.isTensor(input) or isinstance(input, Variable):
        return torch.zeros(input.nElement(), num_out)
    else:
        return type(input)(make_jacobian(elem, num_out) for elem in input)


def iter_tensors(x):
    if torch.isTensor(x):
        yield x
    elif isinstance(x, Variable):
        yield x.data
    else:
        for elem in x:
            for result in iter_tensors(elem):
                yield result


def contiguous(input):
    if torch.isTensor(input):
        return input.contiguous()
    elif isinstance(input, Variable):
        return input.contiguous_()
    else:
        return type(input)(contiguous(e) for e in input)


def get_numerical_jacobian(fn, input, target):
    perturbation = 1e-6
    # To be able to use .view(-1) input must be contiguous
    input = contiguous(input)
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    outa = torch.Tensor(output_size)
    outb = torch.Tensor(output_size)

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1)
        for i in range(flat_tensor.nElement()):
            orig = flat_tensor[i]
            flat_tensor[i] = orig - perturbation
            outa.copy_(fn(input))
            flat_tensor[i] = orig + perturbation
            outb.copy_(fn(input))
            flat_tensor[i] = orig

            outb.add_(-1,outa).div_(2*perturbation)
            d_tensor[i] = outb

    return jacobian
