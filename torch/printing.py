import math
import torch
from functools import reduce

#TODO
def _printformat(storage):
    return '{:.2f}', 1, 5

SCALE_FORMAT = '{:.5f} *\n'

def _printMatrix(self, indent=''):
    fmt, scale, sz = _printformat(self.storage())
    nColumnPerLine = math.floor((80-len(indent))/(sz+1))
    strt = ''
    firstColumn = 0
    while firstColumn < self.size(1):
        lastColumn = min(firstColumn + nColumnPerLine - 1, self.size(1)-1)
        if nColumnPerLine < self.size(1):
            strt += '\n' if firstColumn != 1 else ''
            strt += 'Columns {} to {} \n{}'.format(firstColumn, lastColumn, indent)
        if scale != 1:
            strt += SCALE_FORMAT.format(scale)
        for l in range(self.size(0)):
            strt += indent + (' ' if scale != 1 else '')
            strt += ' '.join(fmt.format(val/scale) for val in self.select(0, l)) + '\n'
        firstColumn = lastColumn + 1
    return strt

def _printTensor(self):
    counter_dim = self.nDimension()-2
    counter = torch.LongStorage(counter_dim).fill(0)
    counter[0] = -1
    finished = False
    strt = ''
    while counter[-1] < self.size(0)-1:
        for i in range(counter_dim):
            counter[i] += 1
            if counter[i] == self.size(i):
                counter[i] = 0
            else:
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
