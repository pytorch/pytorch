
import torch

def quantize(x, qtype, scale=None, zero_point=None):
    """Quantizes the tensor x.
    The scale and zero_point are derived from min/max."""
    qinfo = torch.iinfo(qtype)
    qmin = qinfo.min
    qmax = qinfo.max

    fmin = x.cpu().detach().min().item()
    fmax = x.cpu().detach().max().item()

    fmin = min(fmin, 0)
    fmax = max(fmax, 0)

    if scale is None:
        scale = (fmax - fmin) / (qmax - qmin)
    if zero_point is None:
        zero_point = qmin - int(round(fmin / scale))
    return torch.quantize_per_tensor(x, scale, zero_point, qtype)
