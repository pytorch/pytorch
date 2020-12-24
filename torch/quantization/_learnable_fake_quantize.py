# type: ignore
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .observer import _with_args


def _quantize(x, scale, zp):
    r"""Reference function for quantizing x -- non-clamped.
    """
    return ((x / scale) + zp).round()

def _quantize_vectorized(x, ch_axis, scale, zp):
    r"""Reference function for quantizing a vectorized vesion of x;
    applies to per channel fake quantization.
    """
    axis_mask = [1] * x.ndim
    axis_mask[ch_axis] = x.shape[ch_axis]
    scale_remasked = scale.reshape(axis_mask)
    zp_remasked = zp.reshape(axis_mask)
    return (x / scale_remasked + zp_remasked).round()

def _calculate_X_grad(dY, Xq, q_min, q_max):
    r"""Reference function for calculating the gradient per tensor for the input.
    The gradient for input is calculated as below.

    Let Xq be the quantized version of X (clamped at qmin and qmax).

    :math:
        \frac{dy}{dx} =
            \begin{cases}
                dy& \text{ if } q_{\min} \le X_q \le q_{\max} \\
                0& \text{ else }
            \end{cases}
    """
    mask = (Xq >= q_min) * (Xq <= q_max)
    dX = torch.zeros_like(dY)
    dX[mask] = dY[mask]
    return dX

def _calculate_scale_grad(dY, X, X_fq, X_q, scale, zero_point, q_min, q_max, device):
    r"""Reference function for calculating the gradient for scale.
    The gradient for scale is calculated as below:

    Let Xfq be the fake quantized version of X.
    Let Xq be the quantized version of X (clamped at qmin and qmax).
    Let Delta and z be the scale and the zero point.

    :math:
        \frac{dy}{d\Delta} =
            \begin{cases}
                q_{\min} - z& \text{ if } X_q= q_{\min} \\
                q_{\max} - z& \text{ if } X_q= q_{\max} \\
                (X_{fq} - X) / \Delta & \text{ else }
            \end{cases}
    """
    indicate_small_scale = (X_q == q_min).float().to(device)
    indicate_big_scale = (X_q == q_max).float().to(device)
    indicate_middle_scale = torch.ones(indicate_small_scale.shape, device=device) - \
        indicate_small_scale - indicate_big_scale

    dScale_small = q_min - zero_point
    dScale_big = q_max - zero_point
    dScale_middle = (X_fq - X) / scale

    dScale = indicate_small_scale * dScale_small + \
        indicate_big_scale * dScale_big + \
        indicate_middle_scale * dScale_middle

    return dScale * dY

def _calculate_zero_point_grad(dY, X, X_fq, X_q, scale, zero_point, q_min, q_max, device):
    r"""Reference function for calculating the gradient for zero point.
    The gradient for zero point is calculated as below:

    Let Xfq be the fake quantized version of X.
    Let Xq be the quantized version of X (clamped at qmin and qmax).
    Let Delta and z be the scale and the zero point.

    :math:
        \frac{dy}{dz} =
            \begin{cases}
                -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
                0 & \text{ else }
            \end{cases}
    """
    indicate_saturate_zp = ((X_q == q_min).float() + (X_q == q_max).float()).to(device)
    indicate_unsaturate_zp = torch.ones(indicate_saturate_zp.shape, device=device) - \
        indicate_saturate_zp

    dZeroPoint_saturate = -scale
    dZeroPoint_unsaturate = 0

    dZeroPoint = indicate_saturate_zp * dZeroPoint_saturate + \
        indicate_unsaturate_zp * dZeroPoint_unsaturate

    return dZeroPoint * dY

class _LearnableFakeQuantizePerTensorOp(torch.autograd.Function):
    r"""A helper class to perform the necessary per tensor fake quantization on
    the activated outputs/weights for any given layer.

    The backpropagation routines for scale and zero point are developed
    based on the following literature:
    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
    @staticmethod
    def forward(ctx, X, scale, zero_point, q_min, q_max, grad_factor):
        ctx.save_for_backward(X, scale, zero_point)
        scale_val = float(scale.item())
        zp_val = int((zero_point + 0.5).clamp(q_min, q_max).item())
        X_fq = torch.fake_quantize_per_tensor_affine(
            X, scale_val, zp_val, q_min, q_max)
        ctx.other = q_min, q_max, X_fq, grad_factor
        return X_fq

    @staticmethod
    def backward(ctx, dY):
        X, scale, zero_point = ctx.saved_tensors
        device = X.device
        dY = dY.to(device)
        q_min, q_max, X_fq, grad_factor = ctx.other

        zero_point = int((zero_point + 0.5).clamp(q_min, q_max).item())
        X_q = _quantize(X, scale, zero_point).to(device)
        dX = _calculate_X_grad(dY, X_q, q_min, q_max).to(device)

        X_q = X_q.clamp(q_min, q_max)
        dScale = _calculate_scale_grad(
            dY, X, X_fq, X_q, scale, zero_point, q_min, q_max, device).sum().unsqueeze(0)
        dZeroPoint = _calculate_zero_point_grad(
            dY, X, X_fq, X_q, scale, zero_point, q_min, q_max, device).sum().unsqueeze(0)

        dScale *= grad_factor
        dZeroPoint *= grad_factor

        return dX, dScale, dZeroPoint, None, None, None


class _LearnableFakeQuantizePerChannelOp(torch.autograd.Function):
    r"""A helper class to perform the necessary per channel fake quantization on
    the activated outputs/weights for any given layer. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    """
    @staticmethod
    def forward(ctx, X, scale, zero_point, ch_axis, q_min, q_max, grad_factor):
        ctx.save_for_backward(X, scale, zero_point)
        scale_vec = scale.detach().type(torch.float32)
        zp_vec = ((zero_point.detach() + 0.5).clamp(q_min, q_max)).type(torch.int64)
        X_fq = torch.fake_quantize_per_channel_affine(
            X, scale_vec, zp_vec, ch_axis, q_min, q_max)
        ctx.other = q_min, q_max, X_fq, ch_axis, grad_factor
        return X_fq

    @staticmethod
    def backward(ctx, dY):
        X, scale, zero_point = ctx.saved_tensors
        device = X.device
        dY = dY.to(device)
        q_min, q_max, X_fq, ch_axis, grad_factor = ctx.other

        axis_mask = [1] * X.ndim
        axis_mask[ch_axis] = X.shape[ch_axis]

        scale_vec = scale.detach().type(torch.float32)
        zp_vec = ((zero_point.detach() + 0.5).clamp(q_min, q_max)).type(torch.int64)

        scale_vec = scale_vec.reshape(axis_mask).to(device)
        zp_vec = zp_vec.reshape(axis_mask).to(device)

        X_q = _quantize_vectorized(X, ch_axis, scale_vec, zp_vec).to(device)
        dX = _calculate_X_grad(dY, X_q, q_min, q_max).to(device)

        axis_for_reduction = set(range(X_fq.ndim))
        axis_for_reduction.remove(ch_axis)
        axis_for_reduction = tuple(axis_for_reduction)

        X_q = X_q.clamp(q_min, q_max)
        dScale = _calculate_scale_grad(
            dY, X, X_fq, X_q, scale_vec, zp_vec, q_min, q_max, device).sum(axis_for_reduction)
        dZeroPoint = _calculate_zero_point_grad(
            dY, X, X_fq, X_q, scale_vec, zp_vec, q_min, q_max, device).sum(axis_for_reduction)

        dScale *= grad_factor
        dZeroPoint *= grad_factor

        return dX, dScale, dZeroPoint, None, None, None, None


class _LearnableFakeQuantize(nn.Module):
    r""" This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr: `channel_len` defines the length of the channel when initializing scale and zero point
             for the per channel case.

    * :attr: `use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
              normalized by the constant, which is proportional to the square root of the number of
              elements in the tensor. The related literature justifying the use of this particular constant
              can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr: `fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr: `static_enabled` defines the flag for using observer's static estimation for
             scale and zero point.

    * attr: `learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """
    def __init__(self, observer, quant_min=0, quant_max=255, scale=1., zero_point=0., channel_len=-1,
                 use_grad_scaling=False, **observer_kwargs):
        super(_LearnableFakeQuantize, self).__init__()
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
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
        self.static_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
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
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        self.activation_post_process(X.detach())
        _scale, _zero_point = self.calculate_qparams()
        _scale = _scale.to(self.scale.device)
        _zero_point = _zero_point.to(self.zero_point.device)

        if self.static_enabled[0] == 1:
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.learning_enabled[0] == 1:
                if self.use_grad_scaling:
                    grad_factor = 1.0 / (self.weight.numel() * self.quant_max) ** 0.5
                else:
                    grad_factor = 1.0
                if self.qscheme in (
                        torch.per_channel_symmetric, torch.per_channel_affine):
                    X = _LearnableFakeQuantizePerChannelOp.apply(
                        X, self.scale, self.zero_point, self.ch_axis,
                        self.quant_min, self.quant_max, grad_factor)
                else:
                    X = _LearnableFakeQuantizePerTensorOp.apply(
                        X, self.scale, self.zero_point,
                        self.quant_min, self.quant_max, grad_factor)
            else:
                if self.qscheme == torch.per_channel_symmetric or \
                        self.qscheme == torch.per_channel_affine:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale, self.zero_point, self.ch_axis,
                        self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, float(self.scale.item()), int(self.zero_point.item()),
                        self.quant_min, self.quant_max)

        return X

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We will be saving the static state of scale (instead of as a dynamic param).
        super(_LearnableFakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale.data
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    self.scale.data.copy_(val)
                else:
                    setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(_LearnableFakeQuantize, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    with_args = classmethod(_with_args)
