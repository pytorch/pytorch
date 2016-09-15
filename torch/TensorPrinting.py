import math
import torch
from functools import reduce

_pyrange = torch._pyrange
SCALE_FORMAT = '{:.5f} *\n'


def _printformat(storage):
    min_sz = 0
    tensor = torch.DoubleTensor(torch.DoubleStorage(storage.size()).copy_(storage)).abs()

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


def _printMatrix(self, indent=''):
    fmt, scale, sz = _printformat(self.storage())
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
        for l in _pyrange(self.size(0)):
            strt += indent + (' ' if scale != 1 else '')
            strt += ' '.join(fmt.format(val/scale) for val in self.select(0, l).narrow(0, firstColumn, lastColumn-firstColumn+1)) + '\n'
        firstColumn = lastColumn + 1
    return strt


def _printTensor(self):
    counter_dim = self.nDimension()-2
    counter = torch.LongStorage(counter_dim).fill_(0)
    counter[0] = -1
    finished = False
    strt = ''
    while True:
        for i in _pyrange(counter_dim):
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
        strt += _printMatrix(submatrix, ' ')
    return strt


def _printVector(tensor):
    fmt, scale, _ = _printformat(tensor.storage())
    strt = ''
    if scale != 1:
        strt += SCALE_FORMAT.format(scale)
    return '\n'.join(fmt.format(val/scale) for val in tensor) + '\n'


def printTensor(self):
    if self.nDimension() == 0:
        return '[{} with no dimension]\n'.format(torch.typename(self))
    elif self.nDimension() == 1:
        strt = _printVector(self)
    elif self.nDimension() == 2:
        strt = _printMatrix(self)
    else:
        strt = _printTensor(self)

    size_str = 'x'.join(str(size) for size in self.size())
    strt += '[{} of size {}]\n'.format(torch.typename(self), size_str)
    return '\n' + strt
