import torch
from torch._six import container_abcs
from itertools import repeat
from torch.nn.modules.module import _addindent

def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float), wt_zp.to(torch.float), observer.ch_axis, torch.quint8)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""
    def parse(x):
        while isinstance(x, container_abcs.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(repeat(x, n))
    return parse

def hide_packed_params_repr(self, params):
    # We don't want to show `PackedParams` children, hence custom
    # `__repr__`. This is the same as nn.Module.__repr__, except the check
    # for the `params module`.
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in self._modules.items():
        if isinstance(module, params):
            continue
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = self._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str

_pair_from_first = _ntuple_from_first(2)
