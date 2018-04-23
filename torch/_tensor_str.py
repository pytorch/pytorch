import math
import torch
from functools import reduce
from sys import float_info


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
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 8).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
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


def _get_min_log_scale():
    min_positive = float_info.min * float_info.epsilon  # get smallest denormal
    if min_positive == 0:  # use smallest normal if DAZ/FTZ is set
        min_positive = float_info.min
    return math.ceil(math.log(min_positive, 10))


def _number_format(tensor, min_sz=-1):
    floating_dtype = tensor.dtype.is_floating_point  # save this because we cast later
    _min_log_scale = _get_min_log_scale()
    min_sz = max(min_sz, 2)
    tensor = torch.DoubleTensor(tensor.size()).copy_(tensor).abs_().view(tensor.nelement())

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
        if value != math.ceil(value.item()):
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
    include_decimal_int_mode = floating_dtype and int_mode

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
            if include_decimal_int_mode:
                format += '.'
                sz += 1
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
                scale = math.pow(10, max(exp_max - 1, _min_log_scale))
            else:
                if exp_max == 0:
                    sz = 7
                else:
                    sz = exp_max + 6
                sz = max(min_sz, sz)
            format = '{{:{}.{}f}}'.format(sz, prec)
    return format, scale, sz


def _scalar_str(self, fmt, scale):
    scalar_str = fmt.format(self.item() / scale)
    # The leading space for positives is ugly on scalars, so we strip it
    return scalar_str.lstrip()


def _vector_str(self, indent, fmt, scale, sz, summarize):
    element_length = sz + 3
    elements_per_line = int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length)))
    char_per_line = element_length * elements_per_line

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = ([fmt.format(val.item() / scale) for val in self[:PRINT_OPTS.edgeitems]] +
                [' ...'] +
                [fmt.format(val.item() / scale) for val in self[-PRINT_OPTS.edgeitems:]])
    else:
        data = [fmt.format(val.item() / scale) for val in self]

    data_lines = [data[i:i + elements_per_line] for i in range(0, len(data), elements_per_line)]
    lines = [', '.join(line) for line in data_lines]
    return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'


def _tensor_str(self, indent, fmt, scale, sz, summarize):
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, fmt, scale)
    if dim == 1:
        return _vector_str(self, indent, fmt, scale, sz, summarize)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = ([_tensor_str(self[i], indent + 1, fmt, scale, sz, summarize)
                   for i in range(0, PRINT_OPTS.edgeitems)] +
                  ['...'] +
                  [_tensor_str(self[i], indent + 1, fmt, scale, sz, summarize)
                   for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
    else:
        slices = [_tensor_str(self[i], indent + 1, fmt, scale, sz, summarize) for i in range(0, self.size(0))]

    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'


def _str(self):
    if self.is_sparse:
        size_str = str(tuple(self.shape)).replace(' ', '')
        return '{} of size {} with indices:\n{}\nand values:\n{}'.format(
            self.type(), size_str, self._indices(), self._values())

    prefix = 'tensor('
    indent = len(prefix)
    summarize = self.numel() > PRINT_OPTS.threshold

    suffix = ')'
    if not torch._C._is_default_type_cuda():
        if self.device.type == 'cuda':
            suffix = ', device=\'' + str(self.device) + '\'' + suffix
    else:
        if self.device.type == 'cpu' or torch.cuda.current_device() != self.device.index:
            suffix = ', device=\'' + str(self.device) + '\'' + suffix

    if self.numel() == 0:
        # In an empty tensor, there are no elements to infer if the dtype should be int64,
        # so it must be shown explicitly.
        if self.dtype != torch.get_default_dtype():
            suffix = ', dtype=' + str(self.dtype) + suffix
        tensor_str = '[]'
    else:
        if self.dtype != torch.get_default_dtype() and self.dtype != torch.int64:
            suffix = ', dtype=' + str(self.dtype) + suffix

        fmt, scale, sz = _number_format(self)
        if scale != 1:
            prefix = prefix + SCALE_FORMAT.format(scale) + ' ' * indent
        tensor_str = _tensor_str(self, indent, fmt, scale, sz, summarize)

    return prefix + tensor_str + suffix
