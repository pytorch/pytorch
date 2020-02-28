import torch

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


properties_to_save = (
    '_parameters',
    '_buffers',
    '_backward_hooks',
    '_forward_hooks',
    '_forward_pre_hooks',
    '_state_dict_hooks',
    '_load_state_dict_pre_hooks',
    '_modules',
)


def _get_module_state(mod):
    return tuple(getattr(mod, p) for p in properties_to_save)


def _set_module_state(mod, state, start):
    for index, p in enumerate(properties_to_save):
        setattr(mod, p, state[start + index])
