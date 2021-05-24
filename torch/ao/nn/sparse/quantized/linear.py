from typing import Optional

from .. import Linear as SparseLinear

import torch
import torch.nn.intrinsic as nni
from torch.nn.quantized.modules.utils import _quantize_weight, hide_packed_params_repr

# TODO (zaf): Inherit from `quantized.LinearPackedParams` (T83294430)
class LinearPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, row_block_size=1, col_block_size=4, dtype=torch.qint8):
        super().__init__()
        self.prepack_op = torch.ops.sparse.qlinear_prepack
        self.unpack_op = torch.ops.sparse.qlinear_unpack

        if dtype != torch.qint8:
            raise NotImplementedError("Linear prepacking only supports QINT8")
        self.dtype = dtype
        wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        self.set_weight_bias(wq, None, row_block_size, col_block_size)
        # Hack to make torch.jit.script/torch.jit.load work
        # Once we have self.unpack_op working we wont need this.
        self.__annotations__['bias'] = Optional[torch.Tensor]

    def _get_name(self):
        return "SparseQuantizedLinearPackedParams"

    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor],
                        row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        assert row_block_size is not None and col_block_size is not None
        self._packed_params = self.prepack_op(weight, bias, row_block_size, col_block_size)
        # TODO: We will save the original weight and bias, because the unpacking is not yet there.
        self.weight = weight
        self.bias = bias
        self.row_block_size = row_block_size
        self.col_block_size = col_block_size

    @torch.jit.export
    def _weight_bias(self):
        # TODO: The unpacking is not yet implemented
        # return self.unpack_op(self._packed_params)
        return self.weight, self.bias, self.row_block_size, self.col_block_size

    def forward(self, x):
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_params'] = self._weight_bias()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        assert version <= self._version

        self.dtype = state_dict.pop(prefix + 'dtype')
        weight, bias, row_block_size, col_block_size = state_dict.pop(prefix + '_packed_params')
        self.set_weight_bias(weight, bias, row_block_size, col_block_size)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def __getstate__(self):
        qweight, bias, row_block_size, col_block_size = self._weight_bias()
        return qweight, bias, row_block_size, col_block_size, self.training, self.dtype

    @torch.jit.export
    def __setstate__(self, state):
        self.set_weight_bias(state[0], state[1], state[2], state[3])
        self.training = state[4]
        self.dtype = state[5]

    def __repr__(self):
        return self._weight_bias().__repr__()

# TODO (zaf): Inherit from `quantized.Linear` (T83294430)
class Linear(torch.nn.Module):
    r"""
    A quantized sparse linear module with quantized tensor as inputs and outputs.
    """
    _version = 1
    _FLOAT_MODULE = (torch.nn.Linear, SparseLinear)

    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias=True, dtype=torch.qint8):
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError("Only QINT8 is supported for Sparse Quantized Linear")

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        qweight = torch._empty_affine_quantized([out_features, in_features],
                                                scale=1, zero_point=0, dtype=torch.qint8)
        self._packed_params = LinearPackedParams(dtype)
        self._packed_params.set_weight_bias(qweight, bias, row_block_size, col_block_size)
        self.scale = 1.0
        self.zero_point = 0

    @classmethod
    def _get_name(cls):
        return 'SparseQuantizedLinear'

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, zero_point={}, qscheme={}'.format(
            self.in_features, self.out_features, self.scale, self.zero_point, self.weight().qscheme()
        )

    def __repr__(self):
        return hide_packed_params_repr(self, LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear(x, self._packed_params._packed_params, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        op_type = int(state_dict[prefix + 'op_type'])
        state_dict.pop(prefix + 'op_type')

        version = local_metadata.get('version', None)
        assert version <= self._version

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
    def from_float(cls, mod, row_block_size=1, col_block_size=4):
        r"""Create a quantized sparse module from a float module.

        We only care about the convert at this stage, no need for observers just yet.

        TODO: Need to figure out how to store the block shapes in the mod
        """
        assert type(mod) in cls._FLOAT_MODULE, cls._get_name + \
            '.from_float only works for ' + \
            str([n.__name__ for n in cls._FLOAT_MODULE])
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()

        # It is important to multiply by the mask BEFORE calling the `weight_post_process`
        weight = mod.weight * mod.mask

        weight_post_process(weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        w_sc, w_zp = weight_post_process.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            assert not torch.any(w_zp.bool()), "All weight zero points must map to 0"
        else:
            assert w_zp == 0, 'Weight zero point must map to 0'
        qweight = _quantize_weight(weight.float(), weight_post_process)

        # Use these default values until we figure out how to augment
        # `mod` to contain sparse config
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      row_block_size,
                      col_block_size,
                      dtype=dtype)
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear
