import torch
from typing import Type, Any

__all__ = ['SparseQuantizedLinearReLU']

class SparseQuantizedLinearReLU(torch.nn.Module):
    r"""Sparse Quantized linear + relu fusion.

    This is a fustion of the linear and the relu layers.
    This is just a combination of two layers, without any physical fusion.

    Note: Because TorchScript doesn't support inheritance, we must use
          composition for fused modules. To enable that we take the
          `sparse_qlinear` as an argument, which is set in the `from_float`
          method. That way, we have the proper linear layer and don't
          break the torchscript.
          See: https://github.com/pytorch/pytorch/issues/42885
    """
    _FLOAT_MODULE: Type[Any] = torch.nn.intrinsic.modules.fused.LinearReLU

    def __init__(self, sparse_qlinear: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.sparse_qlinear: torch.nn.Module = sparse_qlinear

    def forward(self, x):
        y = self.sparse_qlinear(x)
        return torch.relu(y)

    @classmethod
    # pyre-fixme[14]: `_get_name` overrides method defined in `Module` inconsistently.
    def _get_name(cls) -> str:
        return "SparseQuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod) -> "SparseQuantizedLinearReLU":
        assert type(mod) == cls._FLOAT_MODULE, (
            cls._get_name()
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
            + " (got "
            + str(type(mod))
            + ")"
        )
        # We insert the parameters into the `nn.Linear` only,
        # There should be none in the LinearReLU wrapper here, but
        # only in the underlying linear layer.
        assert hasattr(mod[0], "sparse_params"), (
            f"Expecting {cls._FLOAT_MODULE.__name__} to have `sparse_params` "
            "in its linear portion in order "
            f"to be able to convert to {cls._get_name()}. "
            "Did you forget to add arguments to the "
            "`sparsifier.squash_mask(...)`?"
        )
        # LinearReLU is just a Sequential[Linear, ReLU]
        # However, the parameters are stored in the LinearReLU, not Linear
        mod[0].activation_post_process = mod.activation_post_process
        qmod = torch.ao.nn.sparse.quantized.Linear.from_float(mod[0])
        fused_mod = cls(qmod)
        # qmod.__class__ = cls
        return fused_mod

    @classmethod
    def from_observed(cls, *args, **kwargs) -> "SparseQuantizedLinearReLU":
        r"""Because we are using the custom modules for FX, we need to define
        this class method for conversion. In practice this is the same as the
        from_float"""
        return cls.from_float(*args, **kwargs)
