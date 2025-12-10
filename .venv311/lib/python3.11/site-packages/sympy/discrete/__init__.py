"""This module contains functions which operate on discrete sequences.

Transforms - ``fft``, ``ifft``, ``ntt``, ``intt``, ``fwht``, ``ifwht``,
            ``mobius_transform``, ``inverse_mobius_transform``

Convolutions - ``convolution``, ``convolution_fft``, ``convolution_ntt``,
            ``convolution_fwht``, ``convolution_subset``,
            ``covering_product``, ``intersecting_product``
"""

from .transforms import (fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from .convolutions import convolution, covering_product, intersecting_product

__all__ = [
    'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
    'inverse_mobius_transform',

    'convolution', 'covering_product', 'intersecting_product',
]
