import torch
import torch.ao.nn.sparse.intrinsic.quantized as ao_iqsparsenn

__all__ = ['SparseQATLinearReLU']

class SparseQATLinearReLU(ao_iqsparsenn.SparseQuantizedLinearReLU):
    r"""Sparse QAT module (fused).

    Note: There is no fused linear + relu for sparse quantized operation.
    This module replaces it with sparse_quantized followed by torch.relu.

    Ideally, we would want to inherit from the spars
    Script doesn't like
    inheritance in the forward. That's why we use composition to override the
    behavior.
    """
    _FLOAT_MODULE = torch.nn.intrinsic.qat.modules.linear_relu.LinearReLU

    @classmethod
    def _get_name(cls) -> str:
        return "SparseQATLinearReLU"

    @classmethod
    def from_float(cls, mod) -> "SparseQATLinearReLU":
        assert type(mod) == cls._FLOAT_MODULE, (
            cls._get_name()
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
            + " (got "
            + str(type(mod))
            + ")"
        )
        # Because the qat.LinearReLU is the same as the Linear itself,
        # but with the ReLU activation attached to it, the sparse_params
        # are inserted in it directly (not in the underlying submodules).
        assert hasattr(mod, "sparse_params"), (
            f"Expecting {cls._FLOAT_MODULE.__name__} to have `sparse_params` "
            f"to be able to convert to {cls._get_name()}. "
            "Did you forget to add arguments to the "
            "`sparsifier.squash_mask(...)`?"
        )
        # There are a lot of different types of quant linear layers
        # By using to_float we are making sure we are converting any
        # QAT type.
        float_mod = mod.to_float()[0]
        float_mod.sparse_params = mod.sparse_params
        float_mod.qconfig = mod.qconfig
        float_mod.activation_post_process = mod.activation_post_process
        qmod = torch.ao.nn.sparse.quantized.Linear.from_float(float_mod)
        fused_mod = cls(qmod)
        return fused_mod
