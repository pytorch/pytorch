import torch

def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
    else:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), 0, torch.qint8)
    return qweight

"""Creates an alias for an existing class with a different docstring."""
def __add_alias(aliased_class, docstring):
    class _(aliased_class):
        pass
    _.__doc__ = docstring
    _.__name__ = aliased_class.__name__
    return _
