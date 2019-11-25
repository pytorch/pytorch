import torch

def _quantize_weight(float_wt, observer):
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_scale, wt_zp, wt_axis = observer.calculate_qparams()
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight
