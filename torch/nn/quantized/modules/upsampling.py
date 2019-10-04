
from torch import nn
from .utils import __alias

_module_name = __name__

UpsamplingBilinear2d = __alias(nn.UpsamplingBilinear2d, module=_module_name, docstring=r"""
Applies a 2D bilinear upsampling to a quantized input signal composed of several
quantized input channels.

To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
as it's constructor argument.

When :attr:`size` is given, it is the output size of the image `(h, w)`.

.. note:: The output quantization parameters are the same as input.

Args:
    size (int or Tuple[int, int], optional): output spatial sizes
    scale_factor (float or Tuple[float, float], optional): multiplier for
        spatial size.

.. warning::
    This class is deprecated in favor of :func:`~nn.quantized.functional.interpolate`. It is
    equivalent to ``nn.quantized.functional.interpolate(..., mode='bilinear', align_corners=True)``.

Shape:
    - Input: :math:`(N, C, H_{in}, W_{in})`
    - Output: :math:`(N, C, H_{out}, W_{out})` where

.. math::
    H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

.. math::
    W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

Examples::

    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> input
    tensor([[[[1., 2.],
              [3., 4.]]]], size=(1, 1, 2, 2), dtype=torch.qint32,
           quantization_scheme=torch.per_tensor_affine, scale=1e-06, zero_point=0)
    >>>
    >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
    >>> m(input)
    tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
              [1.6667, 2.0000, 2.3333, 2.6667],
              [2.3333, 2.6667, 3.0000, 3.3333],
              [3.0000, 3.3333, 3.6667, 4.0000]]]], size=(1, 1, 4, 4),
           dtype=torch.qint32, quantization_scheme=torch.per_tensor_affine,
           scale=1e-06, zero_point=0)

See :class:`~torch.nn.UpsamplingBilinear2d`
""")

UpsamplingNearest2d = __alias(nn.UpsamplingNearest2d, module=_module_name, docstring=r"""
Applies a 2D nearest neighbor upsampling to a quantized input signal composed of
several quantized input channels.

To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
as it's constructor argument.

When :attr:`size` is given, it is the output size of the image `(h, w)`.

.. note:: The output quantization parameters are the same as input.

Args:
    size (int or Tuple[int, int], optional): output spatial sizes
    scale_factor (float or Tuple[float, float], optional): multiplier for
        spatial size.

.. warning::
    This class is deprecated in favor of :func:`~nn.quantized.functional.interpolate`.

Shape:
    - Input: :math:`(N, C, H_{in}, W_{in})`
    - Output: :math:`(N, C, H_{out}, W_{out})` where

.. math::
      H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

.. math::
      W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

Examples::

    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input = torch.quantize_per_tensor(input, 1e-6, 0, torch.qint32)
    >>> input
    tensor([[[[1., 2.],
              [3., 4.]]]], size=(1, 1, 2, 2), dtype=torch.qint32,
           quantization_scheme=torch.per_tensor_affine, scale=1e-06, zero_point=0)
    >>> m = nn.UpsamplingNearest2d(scale_factor=2)
    >>> m(input)
    tensor([[[[1., 1., 2., 2.],
              [1., 1., 2., 2.],
              [3., 3., 4., 4.],
              [3., 3., 4., 4.]]]], size=(1, 1, 4, 4), dtype=torch.qint32,
           quantization_scheme=torch.per_tensor_affine, scale=1e-06, zero_point=0)

See :class:`~torch.nn.UpsamplingNearest2d`
""")
