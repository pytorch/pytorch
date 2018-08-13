import math
import torch
from functools import reduce
from sys import float_info
from torch._six import inf, nan


class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80


PRINT_OPTS = __PrinterOptions()


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
            PRINT_OPTS.threshold = inf
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


class _Formatter(object):
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        if not self.floating_dtype:
            copy = torch.empty(tensor.size(), dtype=torch.long).copy_(tensor).view(tensor.nelement())
            for value in copy.tolist():
                value_str = '{}'.format(value)
                self.max_width = max(self.max_width, len(value_str))

        else:
            copy = torch.empty(tensor.size(), dtype=torch.float64).copy_(tensor).view(tensor.nelement())
            copy_list = copy.tolist()
            try:
                for value in copy_list:
                    if value != math.ceil(value):
                        self.int_mode = False
                        break
            # nonfinites will throw errors
            except (ValueError, OverflowError):
                self.int_mode = False

            if self.int_mode:
                for value in copy_list:
                    value_str = '{:.0f}'.format(value)
                    if math.isnan(value) or math.isinf(value):
                        self.max_width = max(self.max_width, len(value_str))
                    else:
                        # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
                        # to indicate that the tensor is of floating type. add 1 to the len to account for this.
                        self.max_width = max(self.max_width, len(value_str) + 1)

            else:
                copy_abs = copy.abs()
                pos_inf_mask = copy_abs.eq(inf)
                neg_inf_mask = copy_abs.eq(-inf)
                nan_mask = copy_abs.ne(copy)
                invalid_value_mask = pos_inf_mask + neg_inf_mask + nan_mask
                if invalid_value_mask.all():
                    example_value = 0
                else:
                    example_value = copy_abs[invalid_value_mask.eq(0)][0]
                copy_abs[invalid_value_mask] = example_value

                exp_min = copy_abs.min()
                if exp_min != 0:
                    exp_min = math.floor(math.log10(exp_min)) + 1
                else:
                    exp_min = 1
                exp_max = copy_abs.max()
                if exp_max != 0:
                    exp_max = math.floor(math.log10(exp_max)) + 1
                else:
                    exp_max = 1

                # these conditions for using scientific notation are based on numpy
                if exp_max - exp_min > PRINT_OPTS.precision or exp_max > 8 or exp_min < -4:
                    self.sci_mode = True
                    for value in copy_list:
                        value_str = ('{{:.{}e}}').format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in copy_list:
                        value_str = ('{{:.{}f}}').format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))

    def width(self):
        return self.max_width

    def format(self, value):
        if self.floating_dtype:
            if self.int_mode:
                ret = '{:.0f}'.format(value)
                if not (math.isinf(value) or math.isnan(value)):
                    ret += '.'
            elif self.sci_mode:
                ret = ('{{:{}.{}e}}').format(self.max_width, PRINT_OPTS.precision).format(value)
            else:
                ret = ('{{:.{}f}}').format(PRINT_OPTS.precision).format(value)
        else:
            ret = '{}'.format(value)
        return (self.max_width - len(ret)) * ' ' + ret


def _scalar_str(self, formatter):
    return formatter.format(self.item())


def _vector_str(self, indent, formatter, summarize):
    # length includes spaces and comma between elements
    element_length = formatter.width() + 2
    elements_per_line = max(1, int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length))))
    char_per_line = element_length * elements_per_line

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = ([formatter.format(val) for val in self[:PRINT_OPTS.edgeitems].tolist()] +
                [' ...'] +
                [formatter.format(val) for val in self[-PRINT_OPTS.edgeitems:].tolist()])
    else:
        data = [formatter.format(val) for val in self.tolist()]

    data_lines = [data[i:i + elements_per_line] for i in range(0, len(data), elements_per_line)]
    lines = [', '.join(line) for line in data_lines]
    return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'


def _tensor_str(self, indent, formatter, summarize):
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, formatter)
    if dim == 1:
        return _vector_str(self, indent, formatter, summarize)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = ([_tensor_str(self[i], indent + 1, formatter, summarize)
                   for i in range(0, PRINT_OPTS.edgeitems)] +
                  ['...'] +
                  [_tensor_str(self[i], indent + 1, formatter, summarize)
                   for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
    else:
        slices = [_tensor_str(self[i], indent + 1, formatter, summarize) for i in range(0, self.size(0))]

    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'


def _maybe_wrap_suffix(suffix, indent, tensor_str):
    suffix_len = len(suffix)
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1
    if suffix_len > 2 and last_line_len + suffix_len > PRINT_OPTS.linewidth:
        return ',\n' + ' ' * indent + suffix[2:]
    return suffix


def get_summarized_data(self):
    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return torch.cat((self[:PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems:]))
        else:
            return self
    if self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [get_summarized_data(self[i]).reshape(-1) for i in range(0, PRINT_OPTS.edgeitems)]
        end = ([get_summarized_data(self[i]).reshape(-1)
               for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
        return torch.cat((start + end))
    else:
        return self


def _str(self):
    if self.is_sparse:
        size_str = str(tuple(self.shape)).replace(' ', '')
        return '{} of size {} with indices:\n{}\nand values:\n{}'.format(
            self.type(), size_str, self._indices(), self._values())

    prefix = 'tensor('
    indent = len(prefix)
    summarize = self.numel() > PRINT_OPTS.threshold

    suffix = ''
    if not torch._C._is_default_type_cuda():
        if self.device.type == 'cuda':
            suffix += ', device=\'' + str(self.device) + '\''
    else:
        if self.device.type == 'cpu' or torch.cuda.current_device() != self.device.index:
            suffix += ', device=\'' + str(self.device) + '\''

    if self.numel() == 0:
        # Explicitly print the shape if it is not (0,), to match NumPy behavior
        if self.dim() != 1:
            suffix += ', size=' + str(tuple(self.shape))

        # In an empty tensor, there are no elements to infer if the dtype should be int64,
        # so it must be shown explicitly.
        if self.dtype != torch.get_default_dtype():
            suffix += ', dtype=' + str(self.dtype)
        tensor_str = '[]'
    else:
        if self.dtype != torch.get_default_dtype() and self.dtype != torch.int64:
            suffix += ', dtype=' + str(self.dtype)

        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        tensor_str = _tensor_str(self, indent, formatter, summarize)

    if self.grad_fn is not None:
        suffix += ', grad_fn=<{}>'.format(type(self.grad_fn).__name__)
    elif self.requires_grad:
        suffix += ', requires_grad=True'

    suffix += ')'

    suffix = _maybe_wrap_suffix(suffix, indent, tensor_str)

    return prefix + tensor_str + suffix
