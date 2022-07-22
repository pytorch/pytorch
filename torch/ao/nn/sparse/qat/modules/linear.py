import torch
import torch.ao.nn.sparse.quantized as ao_qsparsenn
from typing import Type, Any

__all__ = ['SparseQATLinear']

class SparseQATLinear(ao_qsparsenn.Linear):
    _FLOAT_MODULE: Type[Any] = torch.nn.qat.modules.linear.Linear

    @classmethod
    def _get_name(cls) -> str:
        return "SparseQATLinear"

    @classmethod
    def from_observed(cls, *args, **kwargs):
        r"""Because we are using the custom modules for FX, we need to define
        this class method for conversion. In practice this is the same as the
        from_float"""
        return cls.from_float(*args, **kwargs)
