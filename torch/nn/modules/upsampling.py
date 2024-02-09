from .module import Module
from .. import functional as F

from torch import Tensor
from typing import Optional
from ..common_types import _size_2_t, _ratio_2_t, _size_any_t, _ratio_any_t

__all__ = ['Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d']


class Upsample(Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.

    Shape:
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, `bicubic`, and `trilinear`) don't proportionally
        align the output and input pixels, and thus the output values can depend
        on the input size. This was the default behavior for these modes up to
        version 0.3.1. Since then, the default behavior is
        ``align_corners = False``. See below for concrete examples on how this
        affects the outputs.

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='nearest')
        >>> m(input)
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]])

        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> m(input)
        tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
                  [1.5000, 1.7500, 2.2500, 2.5000],
                  [2.5000, 2.7500, 3.2500, 3.5000],
                  [3.0000, 3.2500, 3.7500, 4.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> m(input)
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  [1.6667, 2.0000, 2.3333, 2.6667],
                  [2.3333, 2.6667, 3.0000, 3.3333],
                  [3.0000, 3.3333, 3.6667, 4.0000]]]])

        >>> # Try scaling the same data in a larger tensor
        >>> input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
        >>> input_3x3[:, :, :2, :2].copy_(input)
        tensor([[[[1., 2.],
                  [3., 4.]]]])
        >>> input_3x3
        tensor([[[[1., 2., 0.],
                  [3., 4., 0.],
                  [0., 0., 0.]]]])

        >>> # xdoctest: +IGNORE_WANT("seems to fail when other tests are run in the same session")
        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> # Notice that values in top left corner are the same with the small input (except at boundary)
        >>> m(input_3x3)
        tensor([[[[1.0000, 1.2500, 1.7500, 1.5000, 0.5000, 0.0000],
                  [1.5000, 1.7500, 2.2500, 1.8750, 0.6250, 0.0000],
                  [2.5000, 2.7500, 3.2500, 2.6250, 0.8750, 0.0000],
                  [2.2500, 2.4375, 2.8125, 2.2500, 0.7500, 0.0000],
                  [0.7500, 0.8125, 0.9375, 0.7500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> # Notice that values in top left corner are now changed
        >>> m(input_3x3)
        tensor([[[[1.0000, 1.4000, 1.8000, 1.6000, 0.8000, 0.0000],
                  [1.8000, 2.2000, 2.6000, 2.2400, 1.1200, 0.0000],
                  [2.6000, 3.0000, 3.4000, 2.8800, 1.4400, 0.0000],
                  [2.4000, 2.7200, 3.0400, 2.5600, 1.2800, 0.0000],
                  [1.2000, 1.3600, 1.5200, 1.2800, 0.6400, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
    """

    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name', 'recompute_scale_factor']
    name: str
    size: Optional[_size_any_t]
    scale_factor: Optional[_ratio_any_t]
    mode: str
    align_corners: Optional[bool]
    recompute_scale_factor: Optional[bool]

    def __init__(self, size: Optional[_size_any_t] = None, scale_factor: Optional[_ratio_any_t] = None,
                 mode: str = 'nearest', align_corners: Optional[bool] = None,
                 recompute_scale_factor: Optional[bool] = None) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                             recompute_scale_factor=self.recompute_scale_factor)

    def __setstate__(self, state):
        if 'recompute_scale_factor' not in state:
            state['recompute_scale_factor'] = True

        super().__setstate__(state)

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = 'scale_factor=' + repr(self.scale_factor)
        else:
            info = 'size=' + repr(self.size)
        info += ', mode=' + repr(self.mode)
        return info


class UpsamplingNearest2d(Upsample):
    r"""Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
          H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
          W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(input)
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]])
    """

    def __init__(self, size: Optional[_size_2_t] = None, scale_factor: Optional[_ratio_2_t] = None) -> None:
        super().__init__(size, scale_factor, mode='nearest')


class UpsamplingBilinear2d(Upsample):
    r"""Applies a 2D bilinear upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`. It is
        equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> # xdoctest: +IGNORE_WANT("do other tests modify the global state?")
        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(input)
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  [1.6667, 2.0000, 2.3333, 2.6667],
                  [2.3333, 2.6667, 3.0000, 3.3333],
                  [3.0000, 3.3333, 3.6667, 4.0000]]]])
    """

    def __init__(self, size: Optional[_size_2_t] = None, scale_factor: Optional[_ratio_2_t] = None) -> None:
        super().__init__(size, scale_factor, mode='bilinear', align_corners=True)
