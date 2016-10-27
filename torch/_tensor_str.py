import math
import torch
from functools import reduce
from ._utils import _range

SCALE_FORMAT = '{:.5e} *\n'


def _number_format(tensor):
    min_sz = 2
    tensor = torch.DoubleTensor(tensor.nelement()).copy_(tensor).abs_()

    pos_inf_mask = tensor.eq(float('inf'))
    neg_inf_mask = tensor.eq(float('-inf'))
    nan_mask = tensor.ne(tensor)
    invalid_value_mask = pos_inf_mask + neg_inf_mask + nan_mask
    if invalid_value_mask.all():  # There are no regular numbers...
        tensor = torch.zeros(1)
    # Get any of non inf and nan values in the tensor
    example_value = tensor[invalid_value_mask.eq(0)][0]
    tensor[invalid_value_mask] = example_value
    if invalid_value_mask.any():
        min_sz = 3

    int_mode = True
    # TODO: use fmod?
    for value in tensor:
        if value != math.ceil(value):
            int_mode = False
            break

    exp_min = tensor.min()
    if exp_min != 0:
        exp_min = math.floor(math.log10(exp_min)) + 1
    else:
        exp_min = 1
    exp_max = tensor.max()
    if exp_max != 0:
        exp_max = math.floor(math.log10(exp_max)) + 1
    else:
        exp_max = 1

    scale = 1
    exp_max = int(exp_max)
    if int_mode:
        if exp_max > 9:
            format = '{:11.4e}'
            sz = max(min_sz, 11)
        else:
            sz = max(min_sz, exp_max + 1)
            format = '{:' + str(sz) + '.0f}'
    else:
        if exp_max - exp_min > 4:
            sz = 11
            if abs(exp_max) > 99 or abs(exp_min) > 99:
                sz = sz + 1
            sz = max(min_sz, sz)
            format = '{:' + str(sz) + '.4e}'
        else:
            if exp_max > 5 or exp_max < 0:
                sz = max(min_sz, 7)
                scale = math.pow(10, exp_max-1)
            else:
                if exp_max == 0:
                    sz = 7
                else:
                    sz = exp_max + 6
                sz = max(min_sz, sz)
            format = '{:' + str(sz) + '.4f}'
    return format, scale, sz


def _tensor_str(self):
    counter_dim = self.ndimension() - 2
    counter = torch.LongStorage(counter_dim).fill_(0)
    counter[0] = -1
    finished = False
    strt = ''
    while True:
        for i in _range(counter_dim):
            counter[i] += 1
            if counter[i] == self.size(i):
                if i == counter_dim-1:
                    finished = True
                counter[i] = 0
            else:
                break
        if finished:
            break
        if strt != '':
           strt += '\n'
        strt += '({},.,.) = \n'.format(','.join(str(i) for i in counter))
        submatrix = reduce(lambda t,i: t.select(0, i), counter, self)
        strt += _matrix_str(submatrix, ' ')
    return strt


def _matrix_str(self, indent=''):
    fmt, scale, sz = _number_format(self)
    nColumnPerLine = int(math.floor((80-len(indent))/(sz+1)))
    strt = ''
    firstColumn = 0
    while firstColumn < self.size(1):
        lastColumn = min(firstColumn + nColumnPerLine - 1, self.size(1)-1)
        if nColumnPerLine < self.size(1):
            strt += '\n' if firstColumn != 1 else ''
            strt += 'Columns {} to {} \n{}'.format(firstColumn, lastColumn, indent)
        if scale != 1:
            strt += SCALE_FORMAT.format(scale)
        for l in _range(self.size(0)):
            strt += indent + (' ' if scale != 1 else '')
            row_slice = self[l, firstColumn:lastColumn+1]
            strt += ' '.join(fmt.format(val/scale) for val in row_slice) + '\n'
        firstColumn = lastColumn + 1
    return strt


def _vector_str(self):
    fmt, scale, _ = _number_format(self)
    strt = ''
    ident = ''
    if scale != 1:
        strt += SCALE_FORMAT.format(scale)
        ident = ' '
    return strt + '\n'.join(ident + fmt.format(val/scale) for val in self) + '\n'


def _str(self):
    if self.ndimension() == 0:
        return '[{} with no dimension]\n'.format(torch.typename(self))
    elif self.ndimension() == 1:
        strt = _vector_str(self)
    elif self.ndimension() == 2:
        strt = _matrix_str(self)
    else:
        strt = _tensor_str(self)

    size_str = 'x'.join(str(size) for size in self.size())
    device_str = '' if not self.is_cuda else ' (GPU {})'.format(self.get_device())
    strt += '[{} of size {}{}]\n'.format(torch.typename(self),
            size_str, device_str)
    return '\n' + strt

