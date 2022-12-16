import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent

class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """Wrapper for quantized modules than can be lowered from reference modules."""
    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        raise NotImplementedError

def _get_weight_observer(observer):
    # FakeQuantize observer
    if hasattr(observer, "activation_post_process"):
        observer = observer.activation_post_process
    # UniformQuantizationObserverBase observer
    return observer

def _needs_weight_clamping(observer, dtype):
    observer = _get_weight_observer(observer)
    if dtype in [torch.qint8, torch.quint8, torch.qint32]:
        info = torch.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False

def _clamp_weights(qweight, observer, scale, zp):
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight

    observer = _get_weight_observer(observer)
    min_, max_ = observer.quant_min, observer.quant_max

    # Doing this because can't use torch.ops.quantized.clamp() with per_channel qscheme yet.
    qw_int_max = torch.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = torch.clone(qweight.int_repr()).fill_(min_)
    qw_int = torch.minimum(torch.maximum(qweight.int_repr(), qw_int_min), qw_int_max)

    if observer.qscheme in [torch.per_tensor_symmetric,
                            torch.per_tensor_affine]:
        qweight = torch._make_per_tensor_quantized_tensor(qw_int, scale.item(), zp.item())
    elif observer.qscheme in [torch.per_channel_symmetric,
                              torch.per_channel_affine,
                              torch.per_channel_affine_float_qparams]:
        qweight = torch._make_per_channel_quantized_tensor(qw_int, scale, zp, axis=observer.ch_axis)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float), wt_zp.to(torch.float), observer.ch_axis, observer.dtype)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""
    def parse(x):
        while isinstance(x, collections.abc.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(itertools.repeat(x, n))
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
