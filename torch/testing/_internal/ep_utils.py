from torch._export.converter import TS2EPConverter
from torch.export._trace import _convert_ts_to_export_experimental


def test_ep_conversion(mod, inp):
    ep = TS2EPConverter(mod, inp).convert()
    out = ep.module()(*inp)
    print(out)


def test_ep_retracing(mod, inp):
    em = _convert_ts_to_export_experimental(
        mod, inp
    )
    out = em(*inp)
    print(out)
