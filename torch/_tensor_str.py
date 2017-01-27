import math
import torch
from functools import reduce
from ._utils import _range


class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80


PRINT_OPTS = __PrinterOptions()
SCALE_FORMAT = '{:.5e} *\n'


# We could use **kwargs, but this will give better docs
def set_printoptions(
        precision=None,
        threshold=None,
        edgeitems=None,
        linewidth=None,
        profile=None,
):
    """Set options for printing. Items shamelessly taken from Numpy

    Args:
        precision: Number of digits of precision for floating point output
            (default 8).
        threshold: Total number of array elements which trigger summarization
            rather than full repr (default 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default 80). Thresholded matricies will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (default, short, full)
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = float('inf')
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth


def _number_format(tensor, min_sz=-1):
    min_sz = max(min_sz, 2)
    tensor = torch.DoubleTensor(tensor.nelement()).copy_(tensor).abs_()

    pos_inf_mask = tensor.eq(float('inf'))
    neg_inf_mask = tensor.eq(float('-inf'))
    nan_mask = tensor.ne(tensor)
    invalid_value_mask = pos_inf_mask + neg_inf_mask + nan_mask
    if invalid_value_mask.all():
        example_value = 0
    else:
        example_value = tensor[invalid_value_mask.eq(0)][0]
    tensor[invalid_value_mask] = example_value
    if invalid_value_mask.any():
        min_sz = max(min_sz, 3)

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
    prec = PRINT_OPTS.precision
    if int_mode:
        if exp_max > prec + 1:
            format = '{{:11.{}e}}'.format(prec)
            sz = max(min_sz, 7 + prec)
        else:
            sz = max(min_sz, exp_max + 1)
            format = '{:' + str(sz) + '.0f}'
    else:
        if exp_max - exp_min > prec:
            sz = 7 + prec
            if abs(exp_max) > 99 or abs(exp_min) > 99:
                sz = sz + 1
            sz = max(min_sz, sz)
            format = '{{:{}.{}e}}'.format(sz, prec)
        else:
            if exp_max > prec + 1 or exp_max < 0:
                sz = max(min_sz, 7)
                scale = math.pow(10, exp_max - 1)
            else:
                if exp_max == 0:
                    sz = 7
                else:
                    sz = exp_max + 6
                sz = max(min_sz, sz)
            format = '{{:{}.{}f}}'.format(sz, prec)
    return format, scale, sz


def _tensor_str(self):
    n = PRINT_OPTS.edgeitems
    has_hdots = self.size()[-1] > 2 * n
    has_vdots = self.size()[-2] > 2 * n
    print_full_mat = not has_hdots and not has_vdots
    formatter = _number_format(self, min_sz=3 if not print_full_mat else 0)
    print_dots = self.numel() >= PRINT_OPTS.threshold

    dim_sz = max(2, max(len(str(x)) for x in self.size()))
    dim_fmt = "{:^" + str(dim_sz) + "}"
    dot_fmt = u"{:^" + str(dim_sz + 1) + "}"

    counter_dim = self.ndimension() - 2
    counter = torch.LongStorage(counter_dim).fill_(0)
    counter[counter.size() - 1] = -1
    finished = False
    strt = ''
    while True:
        nrestarted = [False for i in counter]
        nskipped = [False for i in counter]
        for i in _range(counter_dim - 1, -1, -1):
            counter[i] += 1
            if print_dots and counter[i] == n and self.size(i) > 2 * n:
                counter[i] = self.size(i) - n
                nskipped[i] = True
            if counter[i] == self.size(i):
                if i == 0:
                    finished = True
                counter[i] = 0
                nrestarted[i] = True
            else:
                break
        if finished:
            break
        elif print_dots:
            if any(nskipped):
                for hdot in nskipped:
                    strt += dot_fmt.format('...') if hdot \
                        else dot_fmt.format('')
                strt += '\n'
            if any(nrestarted):
                strt += ' '
                for vdot in nrestarted:
                    strt += dot_fmt.format(u'\u22EE' if vdot else '')
                strt += '\n'
        if strt != '':
            strt += '\n'
        strt += '({},.,.) = \n'.format(
            ','.join(dim_fmt.format(i) for i in counter))
        submatrix = reduce(lambda t, i: t.select(0, i), counter, self)
        strt += _matrix_str(submatrix, ' ', formatter, print_dots)
    return strt


def __repr_row(row, indent, fmt, scale, sz, truncate=None):
    if truncate is not None:
        dotfmt = " {:^5} "
        return (indent +
                ' '.join(fmt.format(val / scale) for val in row[:truncate]) +
                dotfmt.format('...') +
                ' '.join(fmt.format(val / scale) for val in row[-truncate:]) +
                '\n')
    else:
        return indent + ' '.join(fmt.format(val / scale) for val in row) + '\n'


def _matrix_str(self, indent='', formatter=None, force_truncate=False):
    n = PRINT_OPTS.edgeitems
    has_hdots = self.size(1) > 2 * n
    has_vdots = self.size(0) > 2 * n
    print_full_mat = not has_hdots and not has_vdots

    if formatter is None:
        fmt, scale, sz = _number_format(self,
                                        min_sz=5 if not print_full_mat else 0)
    else:
        fmt, scale, sz = formatter
    nColumnPerLine = int(math.floor((PRINT_OPTS.linewidth - len(indent)) / (sz + 1)))
    strt = ''
    firstColumn = 0

    if not force_truncate and \
       (self.numel() < PRINT_OPTS.threshold or print_full_mat):
        while firstColumn < self.size(1):
            lastColumn = min(firstColumn + nColumnPerLine - 1, self.size(1) - 1)
            if nColumnPerLine < self.size(1):
                strt += '\n' if firstColumn != 1 else ''
                strt += 'Columns {} to {} \n{}'.format(
                    firstColumn, lastColumn, indent)
            if scale != 1:
                strt += SCALE_FORMAT.format(scale)
            for l in _range(self.size(0)):
                strt += indent + (' ' if scale != 1 else '')
                row_slice = self[l, firstColumn:lastColumn + 1]
                strt += ' '.join(fmt.format(val / scale) for val in row_slice)
                strt += '\n'
            firstColumn = lastColumn + 1
    else:
        if scale != 1:
            strt += SCALE_FORMAT.format(scale)
        if has_vdots and has_hdots:
            vdotfmt = "{:^" + str((sz + 1) * n - 1) + "}"
            ddotfmt = u"{:^5}"
            for row in self[:n]:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
            strt += indent + ' '.join([vdotfmt.format('...'),
                                       ddotfmt.format(u'\u22F1'),
                                       vdotfmt.format('...')]) + "\n"
            for row in self[-n:]:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
        elif not has_vdots and has_hdots:
            for row in self:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
        elif has_vdots and not has_hdots:
            vdotfmt = u"{:^" + \
                str(len(__repr_row(self[0], '', fmt, scale, sz))) + \
                "}\n"
            for row in self[:n]:
                strt += __repr_row(row, indent, fmt, scale, sz)
            strt += vdotfmt.format(u'\u22EE')
            for row in self[-n:]:
                strt += __repr_row(row, indent, fmt, scale, sz)
        else:
            for row in self:
                strt += __repr_row(row, indent, fmt, scale, sz)
    return strt


def _vector_str(self):
    fmt, scale, sz = _number_format(self)
    strt = ''
    ident = ''
    n = PRINT_OPTS.edgeitems
    dotfmt = u"{:^" + str(sz) + "}\n"
    if scale != 1:
        strt += SCALE_FORMAT.format(scale)
        ident = ' '
    if self.numel() < PRINT_OPTS.threshold:
        return (strt +
                '\n'.join(ident + fmt.format(val / scale) for val in self) +
                '\n')
    else:
        return (strt +
                '\n'.join(ident + fmt.format(val / scale) for val in self[:n]) +
                '\n' + (ident + dotfmt.format(u"\u22EE")) +
                '\n'.join(ident + fmt.format(val / scale) for val in self[-n:]) +
                '\n')


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
    device_str = '' if not self.is_cuda else \
        ' (GPU {})'.format(self.get_device())
    strt += '[{} of size {}{}]\n'.format(torch.typename(self),
                                         size_str, device_str)
    return '\n' + strt
