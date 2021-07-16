from typing import Optional

from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern

import torch
import torch.nn.intrinsic as nni
from torch.nn.quantized.modules.utils import _quantize_weight, hide_packed_params_repr


class Linear(torch.nn.Module):
    r"""
    A dynamically quantized sparse linear module with float tensor as inputs and outputs.
    """
    _version = 1
    _op_type = "sparse_dynamic"
    _FLOAT_MODULE = torch.nn.Linear

    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias=True, dtype=torch.qint8):
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError("Only QINT8 is supported for Sparse Quantized Linear Dynamic")

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        qweight = torch._empty_affine_quantized([out_features, in_features],
                                                scale=1, zero_point=0, dtype=torch.qint8)
        self._packed_params = linear.LinearPackedParams(dtype)
        self._packed_params.set_weight_bias(qweight, bias, row_block_size, col_block_size)

    def _get_name(self):
        return 'SparseQuantizedDynamicLinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, qscheme={}'.format(
            self.in_features, self.out_features, self.weight().qscheme()
        )

    def __repr__(self):
        return hide_packed_params_repr(self, linear.LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear_dynamic(x, self._packed_params._packed_params)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'op_type'] = self._op_type

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        op_type = int(state_dict[prefix + 'op_type'])
        assert op_type == 'sparse', \
            "Cannot load from op_type [{}], expecting [{}]".format(op_type, self._op_type)
        state_dict.pop(prefix + 'op_type')

        version = local_metadata.get('version', None)
        assert version <= self._version

        # Is this code valid? In old quantization it seemed to be used to load
        # older model
        weight = state_dict.pop(prefix + 'weight')
        bias = state_dict.pop(prefix + 'bias')
        state_dict.update({prefix + '_packed_params.weight': weight,
                           prefix + '_packed_params.bias': bias})

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor],
                        row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        assert row_block_size is not None and col_block_size is not None
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized sparse dynamic module from a float module.

        We only care about the convert at this stage, no need for observers just yet.
        """
        assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        if type(mod) == nni.LinearReLU:
            mod = mod[0]
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.qconfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()

        # It is important to multiply by the mask BEFORE calling the `weight_observer`
        # TODO (zaf): Mask might not be part of the qconfig (T83295194)
        weight = mod.weight
        if getattr(mod.qconfig, 'mask', False):
            weight = mod.qconfig.mask * mod.weight

        weight_observer(weight)
        dtype = weight_observer.dtype
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        w_sc, w_zp = weight_observer.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            assert not torch.any(w_zp.bool()), "All weight zero points must map to 0"
        else:
            assert w_zp == 0, 'Weight zero point must map to 0'
        qweight = _quantize_weight(weight.float(), weight_observer)

        row_block_size, col_block_size = LinearBlockSparsePattern.block_size()
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      row_block_size,
                      col_block_size,
                      dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        return qlinear
