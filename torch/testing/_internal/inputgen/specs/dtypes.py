import torch

_int = [torch.uint8, torch.int8, torch.short, torch.int, torch.long]
_int_and_bool = [torch.bool] + _int
_floating = [torch.float, torch.double]
_real = _int + _floating
_real_and_bool = [torch.bool] + _int + _floating
_floating_and_half = [torch.half] + _floating
_complex = [torch.chalf, torch.cfloat, torch.cdouble]
_quant = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]
_all = [torch.bool] + _int + _floating + _complex + _quant
