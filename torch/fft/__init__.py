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
    negative frequencies are redundant. :func:`~torch.fft.rfft` returns the more
    efficient "half-complex" representation where only the positive frequencies
    are stored explicitly.

Args:
    input (Tensor): the input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.fft` and :func:`~torch.fft.ifft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (no normalization).

Example:

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
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the IFFT.
    dim (int, optional): The dimension along which to take the one dimensional IFFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.fft` and :func:`~torch.fft.ifft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> import torch.fft
    >>> t = torch.tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> torch.fft.ifft(t)
    tensor([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
""")

fftn = _add_docstr(_fft.fft_fftn, r"""
fftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the N dimensional discrete Fourier transform of :attr:`input`.

Note:

    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])`. This
    function always returns all positive and negative frequency terms, even
    though for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfftn` returns the more efficient "half-complex"
    representation where only the positive frequencies of the last dimension
    are stored explicitly.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions. If
        given, the input will either be zero-padded or trimmed to this length
        before computing the FFT.
    dim (Tuple[int], optional): Dimensions to be Fourier transformed.
        Default: all dimensions.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.fftn` and :func:`~torch.fft.ifftn`) a total
        normalization factor of ``1/n`` is applied, where ``n = prod(s)``.
        The mode determines where this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (no normalization).

Example:

    >>> import torch.fft
    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fftn = torch.fft.fftn(t)

    The Fourier transform is separable, so :func:`torch.fft.fftn` here is
    equivalent to two one-dimensional :func:`torch.fft.fft`s:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.allclose(fftn, two_ffts)

""")

ifftn = _add_docstr(_fft.fft_ifftn, r"""
ifftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the N dimensional inverse discrete Fourier transform of :attr:`input`.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions. If
        given, the input will either be zero-padded or trimmed to this length
        before computing the IFFT.
    dim (Tuple[int], optional): Dimensions to be inverse Fourier transformed.
        Default: all dimensions.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.fftn` and :func:`~torch.fft.ifftn`) a total
        normalization factor of ``1/n`` is applied, where ``n = prod(s)``.
        The mode determines where this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> import torch.fft
    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifftn = torch.fft.ifftn(t)

    The Fourier transform is separable, so :func:`torch.fft.ifftn` here is
    equivalent to two one-dimensional :func:`torch.fft.ifft`s:

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.allclose(ifftn, two_iffts)

""")

rfft = _add_docstr(_fft.fft_rfft, r"""
rfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional Fourier transform of real-valued :attr:`input`.

The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
the output contains only the positive frequencies below the Nyquist frequency.
To compute the full output, use :func:`~torch.fft.fft`

Args:
    input (Tensor): the real input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the real FFT.
    dim (int, optional): The dimension along which to take the one dimensional real FFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.rfft` and :func:`~torch.fft.irfft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (no normalization).

Example:

    >>> import torch.fft
    >>> t = torch.arange(4)
    >>> t
    tensor([0, 1, 2, 3])
    >>> torch.fft.rfft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j])

    Compare against the full output from :func:`~torch.fft.fft`:

    >>> torch.fft.fft(t)
    tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

    Notice that the symmetric element ``T[-1] == T[1].conj()`` is omitted.
    At the Nyquist frequency ``T[-2] == T[2]`` is it's own symmetric pair,
    and therefore must always be real-valued.
""")

irfft = _add_docstr(_fft.fft_irfft, r"""
irfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfft`. By the Hermitian property, the
time-domain signal will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`:

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        output signal. If given, the input will either be zero-padded or trimmed to this
        length before computing the real IFFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional real IFFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.rfft` and :func:`~torch.fft.irfft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> import torch.fft
    >>> t = torch.arange(5)
    >>> t
    tensor([0, 1, 2, 3, 4])
    >>> T = torch.fft.rfft(t)
    >>> T
    tensor([10.0000+0.0000j, -2.5000+3.4410j, -2.5000+0.8123j])

    Without specifying the output length to :func:`~torch.fft.irfft`, the output
    will not round-trip properly because the input is odd-length:

    >>> torch.fft.irfft(T)
    tensor([0.6250, 1.4045, 3.1250, 4.8455])

    So, it is recommended to always pass the signal length :attr:`n`:

    >>> torch.fft.irfft(T, t.numel())
    tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])
""")

rfftn = _add_docstr(_fft.fft_rfftn, r"""
rfftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the N-dimensional discrete Fourier transform of real :attr:`input`.

The FFT of a real signal is Hermitian-symmetric,
``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])`` so the full
:func:`~torch.fft.fftn` output contains redundant information.
:func:`~torch.fft.rfftn` instead omits the negative frequencies in the
last dimension.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions. If
        given, the input will either be zero-padded or trimmed to this length
        before computing the real FFT.
    dim (Tuple[int], optional): Dimensions to be Fourier transformed.
        Default: all dimensions.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.rfftn` and :func:`~torch.fft.irfftn`) a
        total normalization factor of ``1/n`` is applied, where ``n = prod(s)``.
        The mode determines where this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (no normalization).

Example:

    >>> import torch.fft
    >>> t = torch.rand(10, 10)
    >>> rfftn = torch.fft.rfftn(t)
    >>> rfftn.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.fftn`, we have all
    elements up to the Nyquist frequency.

    >>> fftn = torch.fft.fftn(t)
    >>> torch.allclose(fftn[..., :6], rfftn)
    True

    The Fourier transform is separable, so :func:`torch.fft.rfftn` here is
    equivalent to a combination of :func:`~torch.fft.fft` and
    :func:`~torch.fft.rfft`:

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(x, dim=1), dim=0)
    >>> torch.allclose(rfftn, two_ffts)

""")

irfftn = _add_docstr(_fft.fft_irfftn, r"""
irfftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfftn`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfftn`. By the Hermitian property, the
time-domain signal will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`s`:

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Output size in the transformed dimensions. If
        given, the input will either be zero-padded or trimmed to this length
        before computing the real IFFT.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be inverse Fourier transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: all dimensions.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.rfftn` and :func:`~torch.fft.irfftn`) a
        total normalization factor of ``1/n`` is applied, where ``n = prod(s)``.
        The mode determines where this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> import torch.fft
    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfftn(t)

    Without specifying the output length to :func:`~torch.fft.irfft`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfft(T).size()
    torch.Size([10, 10])

    So, it is recommended to always pass the signal length :attr:`s`:

    >>> roundtrip = torch.fft.irfft(T, t.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.allclose(roundtrip, t)
    True

""")

hfft = _add_docstr(_fft.fft_hfft, r"""
hfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional discrete Fourier transform of a Hermitian
symmetric :attr:`input` signal.

Note:

    :func:`~torch.fft.hfft`/:func:`~torch.fft.ihfft` are analogous to
    :func:`~torch.fft.rfft`/:func:`~torch.fft.irfft`. The real FFT expects
    a real signal in the time-domain and gives a Hermitian symmetry in the
    frequency-domain. The Hermitian FFT is the opposite; Hermitian symmetric in
    the time-domain and real-valued in the frequency-domain. For this reason,
    special care needs to be taken with the length argument :attr:`n`, in the
    same way as with :func:`~torch.fft.irfft`.

Note:
    Because the signal is Hermitian in the time-domain, the result will be real
    in the frequency domain. Note that some input frequencies must be
    real-valued to satisfy the Hermitian property. In these cases the imaginary
    component will be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`:

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        real output. If given, the input will either be zero-padded or trimmed to this
        length before computing the Hermitian FFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian FFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.hfft` and :func:`~torch.fft.ihfft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (no normalization).

Example:

    Taking a real-valued frequency signal and bringing it into the time domain
    gives Hermitian symmetric output:

    >>> import torch.fft
    >>> t = torch.arange(5)
    >>> t
    tensor([0, 1, 2, 3, 4])
    >>> T = torch.fft.ifft(t)
    >>> T
    tensor([ 2.0000+-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j, -0.5000+0.1625j,
            -0.5000+0.6882j])

    Note that ``T[1] == T[-1].conj()`` and ``T[2] == T[-2].conj()`` is
    redundant. We can thus compute the forward transform without considering
    negative frequencies:

    >>> torch.fft.hfft(T[:3], n=5)
    tensor([0., 1., 2., 3., 4.])

    Like with :func:`~torch.fft.irfft`, the output length must be given in order
    to recover an even length output:

    >>> torch.fft.hfft(T[:3])
    tensor([0.5000, 1.1236, 2.5000, 3.8764])
""")

ihfft = _add_docstr(_fft.fft_ihfft, r"""
ihfft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the inverse of :func:`~torch.fft.hfft`.

:attr:`input` must be a real-valued signal, interpreted in the Fourier domain.
The IFFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])``.
:func:`~torch.fft.ihfft` represents this in the onesided form where only the
positive frequencies below the Nyquist frequency are included. To compute the
full output, use :func:`~torch.fft.ifft`.

Args:
    input (Tensor): the real input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the Hermitian IFFT.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian IFFT.
    norm (str, optional): Normalization mode. Between the forward and backward
        transforms (:func:`~torch.fft.hfft` and :func:`~torch.fft.ihfft`) a total
        normalization factor of ``1/n`` is applied. The mode determines where
        this normalization is applied:

        * ``"forward"`` - the forward transform is normalized by ``1/n``, backward is unnormalized.
        * ``"backward"`` - the backward transform is normalized by ``1/n``, forward is unnormalized.
        * ``"ortho"`` - both transforms are normalized by ``1/sqrt(n)`` (making it orthonormal).

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> import torch.fft
    >>> t = torch.arange(5)
    >>> t
    tensor([0, 1, 2, 3, 4])
    >>> torch.fft.ihfft(t)
    tensor([ 2.0000+-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j])

    Compare against the full output from :func:`~torch.fft.ifft`:

    >>> torch.fft.ifft(t)
    tensor([ 2.0000+-0.0000j, -0.5000-0.6882j, -0.5000-0.1625j, -0.5000+0.1625j,
        -0.5000+0.6882j])
""")
