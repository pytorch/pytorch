from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class _FakeQuantizeOp(torch.autograd.Function):
    r"""
        The backpropagation routine is developed based on the following literature:
        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS

        A helper class to perform the necessary fake quantization on the activated
        outputs/weights for any given layer.

        Note that in the backward function, the gradient with respect to the
        scale during quantization is computed as:

        :math:
            \frac{\partial x_f}{\partial s} =
            \left\lfloor{ \frac{x_f}{s} \right\rceil} - \frac{x_f}{s}
    """
    @staticmethod
    def forward(ctx, X, scale, zero_point, g, ch_axis, qscheme, q_min, q_max):
        ctx.save_for_backward(X, scale)
        scale_v = float(scale.item())
        zp_v = int(zero_point.item())
        if qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
            X_fq = torch.fake_quantize_per_channel_affine(
                X, scale_v, zp_v, ch_axis, q_min, q_max)
        else:
            X_fq = torch.fake_quantize_per_tensor_affine(
                X, scale_v, zp_v, q_min, q_max)
        ctx.other = g, q_min, q_max, zero_point, X_fq
        return X_fq


    @staticmethod
    def backward(ctx, grad_X):
        X, scale = ctx.saved_tensors
        g, q_min, q_max, zero_point, X_fq = ctx.other
        zp_v = int(zero_point.item())
        X_q = ((X / scale).round() + zero_point).clamp(q_min, q_max)

        indicate_small = (X_q == q_min).float()
        indicate_big = (X_q == q_max).float()
        indicate_middle = torch.ones(indicate_small.shape) - \
            indicate_small - indicate_big

        grad_small = q_min - zp_v
        grad_big = q_max - zp_v
        grad_middle = (X_fq - X) / scale


        grad_scale = indicate_small * grad_small + indicate_big * grad_big + \
            indicate_middle * grad_middle

        grad_scale = (grad_scale * grad_X / g).sum().unsqueeze(dim=0)
        return grad_X, grad_scale, None, None, None, None, None, None


class _FakeQuantize(nn.Module):
    r""" This is inspired by the FakeQuantize Module in fake_quantize.py, which
    supports more generalized lower-bit quantization. For all other attribute definitions,
    please see the FakeQuantize class.

    Below are the new attributes introduced in this class:

    * :attr: `scale` defines the scale parameter that will be learned for QAT.

    * :attr: `bitwidth` defines the number of bits used during quantization.

    * :attr: `init_param` defines whether or not the parameters are to be initialized.

    * :attr: `static_enabled` defines whether or not the module is using static initialization of parameters.

    * :attr: `learning_enabled` defines whether or not the parameters are learned during fake quantization.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(_FakeQuantize, self).__init__()
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # Initializes the scale as a learnable parameter and registers
        # zero points and other flags as buffers.
        self.scale = Parameter(torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('init_param', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([0], dtype=torch.uint8))
        # Declares the observer and other necessary parameters for fake quantization.
        self.activation_post_process = observer(**observer_kwargs)
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        # Instantiates the bitwidth of the FakeQuantize operator.
        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())

    @torch.jit.export
    def toggle_gradient_update(self, enabled=True):
        # Enables learning via backpropagation on the quantization
        # parameters and prevents observer from updating the values.
        self.toggle_static_update(not enabled) \
            .toggle_scale_learning(enabled) \
            .toggle_fake_quant(enabled)
        return self

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        # Enables static estimation from observer to be used
        # to update parameters; results are still fake quantized.
        self.toggle_static_update(enabled) \
            .toggle_scale_learning(not enabled) \
            .toggle_fake_quant(enabled)
        return self

    @torch.jit.export
    def toggle_observer_only(self, enabled=True):
        # Enables pure observation and update of data
        # by the observer; the input is directly returned as output.
        self.toggle_static_update(enabled) \
            .toggle_scale_learning(not enabled) \
            .toggle_fake_quant(not enabled)
        return self

    @torch.jit.export
    def toggle_scale_learning(self, enabled=True):
        # Toggles learning of the scale via QAT.
        self.learning_enabled[0] = int(enabled)
        self.scale.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        # Toggles whether the output is fake quantized.
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def toggle_static_update(self, enabled=True):
        # Toggles whether static estimates from the observers
        # are used to update scale and zero point.
        self.static_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def signal_param_initialized(self, initialized=True):
        # Signals the fake quantization parameters have been initialized.
        self.init_param[0] = int(initialized)
        return self

    @torch.jit.export
    def get_signed_repr(self):
        # Retrieves the signed representation for quantization range.
        q_min = -2 ** (self.bitwidth - 1)
        q_max = 2 ** (self.bitwidth - 1) - 1
        return q_min, q_max

    @torch.jit.export
    def get_unsigned_repr(self):
        # Retrieves the unsigned representation for quantization range.
        q_umin = 0
        q_umax = 2 ** (self.bitwidth) - 1
        return q_umin, q_umax

    @torch.jit.export
    def get_proper_qrepr(self):
        # Retrieves the proper representation for qmin and qmax based on
        # the observer's datatype.
        observer = self.activation_post_process
        if observer.dtype == torch.quint8:
            q_min, q_max = self.get_unsigned_repr()
        else:
            q_min, q_max = self.get_signed_repr()
        # Reduces range if necessary.
        if observer.reduce_range:
            q_min, q_max = q_min // 2, q_max // 2
        return q_min, q_max

    @torch.jit.export
    def get_gradient_constant(self, X):
        # Calculates the gradient constant for better convergence.
        return (X.numel() * self.quant_max) ** 0.5

    @torch.jit.export
    def print_quantization_params(self):
        # Prints the quantizaton parmaeters for tracking purposes.
        print('Scale: {:.6f}, Zero Point: {}'
              .format(float(self.scale), int(self.zero_point)))

    @torch.jit.export
    def calculate_qparams(self):
        q_min, q_max = self.get_proper_qrepr()
        return self.activation_post_process.calculate_qparams(q_min, q_max)

    @torch.jit.export
    def update_zero_point(self):
        q_min, q_max = self.get_proper_qrepr()
        # Updates the zero point based on the scale parameter.
        min_val = self.activation_post_process.min_val
        min_val = torch.min(min_val, torch.zeros_like(min_val))
        _zero_point = (self.quant_min - torch.round(min_val / self.scale)).item()
        _zero_point = torch.tensor([_zero_point]).to(self.zero_point.device).clamp(q_min, q_max)
        self.zero_point.copy_(_zero_point)

    @torch.jit.export
    def update_qparams(self, scale, zero_point):
        _scale, _zero_point = scale.to(self.scale.device), zero_point.to(self.zero_point.device)
        self.scale.data.copy_(_scale)
        self.zero_point.resize_(_zero_point.shape)
        self.zero_point.copy_(_zero_point)

    @torch.jit.export
    def observe_qparams(self):
        print('_FakeQuantize Scale: {}'.format(self.scale.item()))
        print('_FakeQuantize Zero Point: {}'.format(self.zero_point.item()))

    def forward(self, X):
        # Observes the input values.
        self.activation_post_process(X.detach())

        if self.static_enabled[0] == 1 and self.init_param[0] == 0:
            # This represents the typical static QAT process; can also be
            # used for initializing parameters in preparation for learning scale.
            _scale, _zero_point = self.calculate_qparams()
            self.update_qparams(_scale, _zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.learning_enabled[0] == 1:
                # Updates zero point using cumulative moving average min max observations
                # instead of instantaneous minimum.
                self.update_zero_point()
                # Scale parmaeter has been initializaed, in the following iteration,
                # we disable initialization once fake quantization actually begins.
                self.signal_param_initialized(True)
                g = self.get_gradient_constant(X)
                self.scale.data.copy_(torch.abs(self.scale.data))
                X = _FakeQuantizeOp.apply(
                    X, self.scale, self.zero_point, g, self.ch_axis, self.qscheme,
                    self.quant_min, self.quant_max)
            else:
                # In this case, we proceed with the regular fake quantization process.
                # No learning is enabled.
                if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, float(self.scale.item()), int(self.zero_point.item()), self.quant_min, self.quant_max)

        return X

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We will be saving the static state of scale (instead of as a dynamic parameter).
        super(_FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale.data
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    self.scale = Parameter(torch.tensor([val]))
                else:
                    setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(_FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                         missing_keys, unexpected_keys, error_msgs)
