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

"""Creates an alias for an existing class with a different docstring.

Args:
    aliased_class: Class to alias
    module: New module name in case different package location.
            Ideally should be set to '__name__'
    docstring: New docstring
"""
def __alias(aliased_class, module=None, docstring=''):
    attr_dict = {'__doc__': docstring}
    if module is not None:
        attr_dict['__module__'] = module
    alias = type(aliased_class.__name__, (aliased_class,), attr_dict)
    return alias
