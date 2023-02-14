import sys
import torch
import types
from typing import List

# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_id(qengine: str) -> int:
    if qengine == 'none' or qengine == '' or qengine is None:
        ret = 0
    elif qengine == 'fbgemm':
        ret = 1
    elif qengine == 'qnnpack':
        ret = 2
    elif qengine == 'onednn':
        ret = 3
    elif qengine == 'x86':
        ret = 4
    else:
        ret = -1
        raise RuntimeError("{} is not a valid value for quantized engine".format(qengine))
    return ret

# This function should correspond to the enums present in c10/core/QEngine.h
def _get_qengine_str(qengine: int) -> str:
    all_engines = {0 : 'none', 1 : 'fbgemm', 2 : 'qnnpack', 3 : 'onednn', 4 : 'x86'}
    return all_engines.get(qengine, '*undefined')

class _QEngineProp:
    def __get__(self, obj, objtype) -> str:
        return _get_qengine_str(torch._C._get_qengine())

    def __set__(self, obj, val: str) -> None:
        torch._C._set_qengine(_get_qengine_id(val))

class _SupportedQEnginesProp:
    def __get__(self, obj, objtype) -> List[str]:
        qengines = torch._C._supported_qengines()
        return [_get_qengine_str(qe) for qe in qengines]

    def __set__(self, obj, val) -> None:
        raise RuntimeError("Assignment not supported")

class QuantizedEngine(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
engine: str
supported_engines: List[str]
