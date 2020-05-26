import torch
from torch._six import container_abcs
from itertools import repeat

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
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""
    def parse(x):
        while isinstance(x, container_abcs.Iterable):
            if len(x) == n:
                break
            x = x[0]
        return tuple(repeat(x, n))
    return parse

_pair_from_first = _ntuple_from_first(2)
