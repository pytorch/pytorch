import torch
from torch.nn.parameter import Parameter
from typing import List

__all__: List[str] = []

class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`channel_len` defines the length of the channel when initializing scale and zero point
      for the per channel case.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static estimation for
      scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """
    def __init__(self, observer, quant_min=0, quant_max=255, scale=1., zero_point=0., channel_len=-1,
                 use_grad_scaling=False, **observer_kwargs):
        super(_LearnableFakeQuantize, self).__init__()
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.use_grad_scaling = use_grad_scaling
        if channel_len == -1:
            self.scale = Parameter(torch.tensor([scale]))
            self.zero_point = Parameter(torch.tensor([zero_point]))
        else:
            assert isinstance(channel_len, int) and channel_len > 0, "Channel size must be a positive integer."
            self.scale = Parameter(torch.tensor([scale] * channel_len))
            self.zero_point = Parameter(torch.tensor([zero_point] * channel_len))

        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
            'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
            'quant_max out of bound'
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([0], dtype=torch.uint8))

        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        r"""Enables static observer estimates and disbales learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def enable_static_observation(self):
        r"""Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)  # type: ignore[operator]
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        print('_LearnableFakeQuantize Scale: {}'.format(self.scale.detach()))
        print('_LearnableFakeQuantize Zero Point: {}'.format(self.zero_point.detach()))

    @torch.jit.export
    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    def forward(self, X):
        if self.static_enabled[0] == 1:  # type: ignore[index]
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.zero_()

            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (
                    torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)

        return X
