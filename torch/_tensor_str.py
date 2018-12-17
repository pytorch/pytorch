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
            (default = 4).
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

        with torch.no_grad():
            tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = '{}'.format(value)
                self.max_width = max(self.max_width, len(value_str))

        else:
            nonzero_finite_vals = torch.masked_select(tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0))

            if nonzero_finite_vals.numel() == 0:
                # no valid number, do nothing
                return

            # Convert to double for easy calculation. HalfTensor overflows with 1e8, and there's no div() on CPU.
            nonzero_finite_abs = nonzero_finite_vals.abs().double()
            nonzero_finite_min = nonzero_finite_abs.min().double()
            nonzero_finite_max = nonzero_finite_abs.max().double()

            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break

            if self.int_mode:
                # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
                # to indicate that the tensor is of floating type. add 1 to the len to account for this.
                if nonzero_finite_max / nonzero_finite_min > 1000. or nonzero_finite_max > 1.e8:
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = ('{{:.{}e}}').format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = ('{:.0f}').format(value)
                        self.max_width = max(self.max_width, len(value_str) + 1)
            else:
                # Check if scientific representation should be used.
                if nonzero_finite_max / nonzero_finite_min > 1000.\
                        or nonzero_finite_max > 1.e8\
                        or nonzero_finite_min < 1.e-4:
                    self.sci_mode = True
                    for value in nonzero_finite_vals:
                        value_str = ('{{:.{}e}}').format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = ('{{:.{}f}}').format(PRINT_OPTS.precision).format(value)
                        self.max_width = max(self.max_width, len(value_str))

    def width(self):
        return self.max_width

    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = ('{{:{}.{}e}}').format(self.max_width, PRINT_OPTS.precision).format(value)
            elif self.int_mode:
                ret = '{:.0f}'.format(value)
                if not (math.isinf(value) or math.isnan(value)):
                    ret += '.'
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


def _tensor_str_with_formatter(self, indent, formatter, summarize):
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, formatter)
    if dim == 1:
        return _vector_str(self, indent, formatter, summarize)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = ([_tensor_str_with_formatter(self[i], indent + 1, formatter, summarize)
                   for i in range(0, PRINT_OPTS.edgeitems)] +
                  ['...'] +
                  [_tensor_str_with_formatter(self[i], indent + 1, formatter, summarize)
                   for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
    else:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, formatter, summarize)
                  for i in range(0, self.size(0))]

    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'


def _tensor_str(self, indent):
    if self.numel() == 0:
        return '[]'

    summarize = self.numel() > PRINT_OPTS.threshold
    if self.dtype is torch.float16:
        self = self.float()
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
    return _tensor_str_with_formatter(self, indent, formatter, summarize)


def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(', ' + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(')')
    return ''.join(tensor_strs)


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
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = ([self[i]
               for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
        return torch.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return torch.stack([get_summarized_data(x) for x in self])


def _str(self):
    prefix = 'tensor('
    indent = len(prefix)

    suffixes = []
    if not torch._C._is_default_type_cuda():
        if self.device.type == 'cuda':
            suffixes.append('device=\'' + str(self.device) + '\'')
    else:
        if self.device.type == 'cpu' or torch.cuda.current_device() != self.device.index:
            suffixes.append('device=\'' + str(self.device) + '\'')

    has_default_dtype = self.dtype == torch.get_default_dtype() or self.dtype == torch.int64

    if self.is_sparse:
        suffixes.append('size=' + str(tuple(self.shape)))
        suffixes.append('nnz=' + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append('dtype=' + str(self.dtype))
        indices_prefix = 'indices=tensor('
        indices = self._indices().detach()
        indices_str = _tensor_str(indices, indent + len(indices_prefix))
        if indices.numel() == 0:
            indices_str += ', size=' + str(tuple(indices.shape))
        values_prefix = 'values=tensor('
        values = self._values().detach()
        values_str = _tensor_str(values, indent + len(values_prefix))
        if values.numel() == 0:
            values_str += ', size=' + str(tuple(values.shape))
        tensor_str = indices_prefix + indices_str + '),\n' + ' ' * indent + values_prefix + values_str + ')'
    else:
        if self.numel() == 0 and not self.is_sparse:
            # Explicitly print the shape if it is not (0,), to match NumPy behavior
            if self.dim() != 1:
                suffixes.append('size=' + str(tuple(self.shape)))

            # In an empty tensor, there are no elements to infer if the dtype
            # should be int64, so it must be shown explicitly.
            if self.dtype != torch.get_default_dtype():
                suffixes.append('dtype=' + str(self.dtype))
            tensor_str = '[]'
        else:
            if not has_default_dtype:
                suffixes.append('dtype=' + str(self.dtype))
            tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        suffixes.append('layout=' + str(self.layout))

    if self.grad_fn is not None:
        name = type(self.grad_fn).__name__
        if name == 'CppFunction':
            name = self.grad_fn.name().rsplit('::', maxsplit=1)[-1]
        suffixes.append('grad_fn=<{}>'.format(name))
    elif self.requires_grad:
        suffixes.append('requires_grad=True')

    return _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse)
