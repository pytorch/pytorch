# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight


__all__ = [
    "Linear",
]


class Linear(nnq.Linear):
    r"""
    A dynamic quantized linear module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable floating point bias of the module of shape
                       :math:`(\text{out\_features})`. If :attr:`bias` is ``True``,
                       the values are initialized to zero.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # version used in this class is different from the parent class nnq.Linear
    _version = 4

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias_, dtype=dtype)
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        self.version = 4

    def forward(self, x):
        # Note that we can handle self.bias == None case.
        if self._packed_params.dtype == torch.qint8:
            if self.version is None or self.version < 4:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params
                )
            else:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params, reduce_range=True
                )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")
        return Y.to(x.dtype)

    def _get_name(self):
        return "DynamicQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = f"in_features={self.in_features}, out_features={self.out_features}, dtype={self._packed_params.dtype}"
        if self._packed_params.dtype == torch.qint8:
            extra_repr_str += f", qscheme={self.weight().qscheme()}"
        return extra_repr_str

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        self.version = version
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a dynamic quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
        float_modules = [
            torch.nn.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.ao.nn.intrinsic.modules.fused.LinearReLU,
            torch.ao.nn.qat.dynamic.Linear,
        ]

        if type(mod) not in float_modules:
            raise AssertionError(
                "nn.quantized.dynamic.Linear.from_float only works for one of"
                + str([float_mod.__name__ for float_mod in float_modules])
                + f", got {type(mod)}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if type(mod) is nni.LinearReLU:
            mod = mod[0]

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer = default_dynamic_qconfig.weight()
        dtype = weight_observer.dtype
        if dtype not in [torch.qint8, torch.float16]:
            raise AssertionError(
                f"The only supported dtypes for dynamic quantized linear are qint8 and float16, got: {dtype}"
            )
        weight_observer(mod.weight)
        if dtype == torch.qint8:
            qweight = _quantize_weight(mod.weight.float(), weight_observer)
        elif dtype == torch.float16:
            qweight = mod.weight.float()
        else:
            raise RuntimeError(
                "Unsupported dtype specified for dynamic quantized Linear!"
            )
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)

        qlinear.set_weight_bias(qweight, mod.bias)
        return qlinear

    @classmethod
    def from_reference(cls, ref_qlinear):  # type: ignore[override]
        """Create a (fbgemm/qnnpack) dynamic quantized module from a reference quantized
        module
        Args:
            ref_qlinear (Module): a reference quantized  module, either produced by
            torch.ao.quantization functions or provided by the user
        """
        qlinear = cls(
            ref_qlinear.in_features,
            ref_qlinear.out_features,
            dtype=ref_qlinear.weight_dtype,
        )
        qweight = ref_qlinear.get_quantized_weight()
        bias = ref_qlinear.bias
        qlinear.set_weight_bias(qweight, bias)
        return qlinear
