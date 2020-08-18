import sys

import torch
from torch._C import _add_docstr, _fft  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds the doc strings for the spectral ops, but
# connects the torch.fft Python namespace to the torch._C._fft builtins.

fft = _add_docstr(_fft.fft_fft, r"""
fft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Note:

    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms, even though for real inputs the
    negative frequencies are redundant. :func:`torch.fft.rfft` returns the more
    efficient "half-complex" representation where only the positive frequencies
    are stored explicitly.

Args:
    input (Tensor): the input tensor
    n (int, optional): FFT length. If given, the input will either be zero-padded
        or trimmed to this length before taking the FFT.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.fft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.ifft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    >>> t = tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
    >>> torch.fft.fft(t)
    tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
""")

ifft = _add_docstr(_fft.fft_ifft, r"""
ifft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional inverse discrete Fourier transform of :attr:`input`.

Args:
    input (Tensor): the input tensor
    n (int, optional): FFT length. If given, the input will either be zero-padded
        or trimmed to this length before taking the FFT.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.fft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.ifft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    >>> import torch.fft
    >>> t = torch.tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> torch.fft.ifft(t)
    tensor([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
""")

rfft = _add_docstr(_fft.fft_rfft, r"""
rfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional discrete Fourier transform of real :attr:`input`.

The FFT of a real signal is hermitian-symmetric, ``X[i] = conj(X[-i])`` so
the output contains only the positive frequencies below the nyquist frequency.
To compute the full output, use :func:`torch.fft.fft`

Args:
    input (Tensor): the real input tensor
    n (int, optional): FFT length. If given, the input will either be zero-padded
        or trimmed to this length before taking the FFT.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.rfft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.irfft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.rfft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j])

    Compare against the full output from :func:`torch.fft.fft`:

    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    Notice that the symmetric element ``T[-1] == T[1].conj()`` is ommitted.
    At the Nyquist freqency ``T[-2] == T[2].conj() == T[2]`` is given but must
    always be purely real.
""")

irfft = _add_docstr(_fft.fft_irfft, r"""
irfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the inverse of :func:`torch.fft.rfft`.

:attr:`input` is a one-sided hermitian signal in the fourrier domain, as
produced by :func:`torch.fft.rfft`. By the hermitian property, the time-domain
signal will be purely real.

Note that some input frequencies must be purely real to satisfy the hermitian
property. In these cases the imaginary component will be ignored.

Args:
    input (Tensor): the input tensor representing a half-hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        output signal. If given, the input will either be zero-padded or trimmed to this
        length before taking the IFFT. Defaults to odd output: ``n=input.size(dim) * 2 - 1``.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.rfft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.irfft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> T = torch.fft.rfft(t)
    >>> T
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j])

    Without specifying the output length to :func:`torch.fft.irfft`, the output
    will not round-trip properly because the input is even-length:

    >>> torch.fft.irfft(T)
    tensor([-0.4000,  0.8392,  1.1298,  2.0702,  2.3608])

    So, it is recommended to always pass the signal length :attr:`n`:

    >>> torch.fft.irfft(T, t.numel())
    tensor([0., 1., 2., 3.])
""")

hfft = _add_docstr(_fft.fft_hfft, r"""
hfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional discrete Fourier transform of a hermitian
symmetric :attr:`input` signal.

Because the signal is hermitian in the time-domain, the result will be real in
the frequency domain. Note that some input frequencies must be purely real to
satisfy the hermitian property. In these cases the imaginary component will be
ignored.

Args:
    input (Tensor): the input tensor representing a half-hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        real output. If given, the input will either be zero-padded or trimmed to this
        length before taking the FFT. Defaults to odd output: ``n=input.size(dim) * 2 - 1``.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.hfft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.ihfft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    Taking a frequency signal that's purely real and bringing into the time
    domain gives hermitian symmetric output:

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> T = torch.fft.ifft(t)
    >>> T
    tensor([ 1.5000+-0.0000j, -0.5000-0.5000j, -0.5000+-0.0000j, -0.5000+0.5000j])

    Note that ``T[1] == T[-1].conj()`` is redundant. We can thus compute the
    forward transform without considering negative frequencies:

    >>> torch.fft.hfft(T[:3], n=4)
    tensor([0., 1., 2., 3.])

    Like with :func:`torch.fft.irfft`, the output length must be given in order
    to recover an even length output:

    >>> torch.fft.hfft(T[:3])
    tensor([-1.5000,  1.6521,  0.2432,  2.5410,  1.4590,  3.7568,  2.3479])
""")

ihfft = _add_docstr(_fft.fft_ihfft, r"""
ihfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the inverse one dimensional Fourier transform of a real signal.

The IFFT of a real signal is hermitian-symmetric, ``X[i] = conj(X[-i])`` so
the output contains only the positive frequencies below the nyquist frequency.
To compute the full output, use :func:`torch.fft.ifft`.

Args:
    input (Tensor): the real input tensor
    n (int, optional): FFT length. If given, the input will either be zero-padded
        or trimmed to this length before taking the FFT.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode:
        ``"forward"`` - the forward transform (:func:`~torch.fft.hfft`) is normalized by ``1/n``,
        ``"backward"`` - the backward transform (:func:`~torch.fft.ihfft`) is normalized by ``1/n``,
        ``"ortho"`` - both transform directions are normalized by ``1/sqrt(n)`` (making it orthonormal).
        Default is ``"backward"``.

Example::

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.ihfft(t)
    tensor([ 1.5000+-0.0000j, -0.5000-0.5000j, -0.5000+-0.0000j])

    Compare against the full output from :func:`torch.fft.ifft`:

    >>> torch.fft.ifft(t)
    tensor([ 1.5000+-0.0000j, -0.5000-0.5000j, -0.5000+-0.0000j, -0.5000+0.5000j])
""")
