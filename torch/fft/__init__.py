import sys

import torch
from torch._C import _add_docstr, _fft  # type: ignore
from torch._torch_docs import factory_common_args

__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
           'Tensor']

Tensor = torch.Tensor

# Note: This not only adds the doc strings for the spectral ops, but
# connects the torch.fft Python namespace to the torch._C._fft builtins.

fft = _add_docstr(_fft.fft_fft, r"""
fft(input, n=None, dim=-1, norm=None) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Note:

    The Fourier domain representation of any real signal satisfies the
    Hermitian property: `X[i] = conj(X[-i])`. This function always returns both
    the positive and negative frequency terms even though, for real inputs, the
    negative frequencies are redundant. :func:`~torch.fft.rfft` returns the
    more compact one-sided representation where only the positive frequencies
    are returned.

Args:
    input (Tensor): the input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the FFT.
    dim (int, optional): The dimension along which to take the one dimensional FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.ifft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

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
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.fft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> t = torch.tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> torch.fft.ifft(t)
    tensor([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
""")

fft2 = _add_docstr(_fft.fft_fft2, r"""
fft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor

Computes the 2 dimensional discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.fftn` but FFTs only the last two dimensions by default.

Note:
    The Fourier domain representation of any real signal satisfies the
    Hermitian property: ``X[i, j] = conj(X[-i, -j])``. This
    function always returns all positive and negative frequency terms even
    though, for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfft2` returns the more compact one-sided representation
    where only the positive frequencies of the last dimension are returned.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifft2`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifft2` the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fft2 = torch.fft.fft2(t)

    The discrete Fourier transform is separable, so :func:`~torch.fft.fft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.fft` calls:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.allclose(fft2, two_ffts)

""")

ifft2 = _add_docstr(_fft.fft_ifft2, r"""
ifft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor

Computes the 2 dimensional inverse discrete Fourier transform of :attr:`input`.
Equivalent to :func:`~torch.fft.ifftn` but IFFTs only the last two dimensions by default.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.fft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifft2 = torch.fft.ifft2(t)

    The discrete Fourier transform is separable, so :func:`~torch.fft.ifft2`
    here is equivalent to two one-dimensional :func:`~torch.fft.ifft` calls:

    >>> two_iffts = torch.fft.ifft(torch.fft.ifft(x, dim=0), dim=1)
    >>> torch.allclose(ifft2, two_iffts)

""")

fftn = _add_docstr(_fft.fft_fftn, r"""
fftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the N dimensional discrete Fourier transform of :attr:`input`.

Note:

    The Fourier domain representation of any real signal satisfies the
    Hermitian property: ``X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n])``. This
    function always returns all positive and negative frequency terms even
    though, for real inputs, half of these values are redundant.
    :func:`~torch.fft.rfftn` returns the more compact one-sided representation
    where only the positive frequencies of the last dimension are returned.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.fftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.ifftn`) with the same
        normalization mode will apply an overall normalization of ``1/n``
        between the two transforms. This is required to make
        :func:`~torch.fft.ifftn` the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> fftn = torch.fft.fftn(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.fftn`
    here is equivalent to two one-dimensional :func:`~torch.fft.fft` calls:

    >>> two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
    >>> torch.allclose(fftn, two_ffts)

""")

ifftn = _add_docstr(_fft.fft_ifftn, r"""
ifftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the N dimensional inverse discrete Fourier transform of :attr:`input`.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the IFFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ifftn`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.fftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ifftn`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> x = torch.rand(10, 10, dtype=torch.complex64)
    >>> ifftn = torch.fft.ifftn(x)

    The discrete Fourier transform is separable, so :func:`~torch.fft.ifftn`
    here is equivalent to two one-dimensional :func:`~torch.fft.ifft` calls:

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
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.irfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

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
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`.

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        output signal. If given, the input will either be zero-padded or trimmed to this
        length before computing the real IFFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional real IFFT.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.rfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

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

rfft2 = _add_docstr(_fft.fft_rfft2, r"""
rfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor

Computes the 2-dimensional discrete Fourier transform of real :attr:`input`.
Equivalent to :func:`~torch.fft.rfftn` but FFTs only the last two dimensions by default.

The FFT of a real signal is Hermitian-symmetric, ``X[i, j] = conj(X[-i, -j])``,
so the full :func:`~torch.fft.fft2` output contains redundant information.
:func:`~torch.fft.rfft2` instead omits the negative frequencies in the last
dimension.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfft2`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.irfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

    >>> t = torch.rand(10, 10)
    >>> rfft2 = torch.fft.rfft2(t)
    >>> rfft2.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.fft2`, we have all
    elements up to the Nyquist frequency.

    >>> fft2 = torch.fft.fft2(t)
    >>> torch.allclose(fft2[..., :6], rfft2)
    True

    The discrete Fourier transform is separable, so :func:`~torch.fft.rfft2`
    here is equivalent to a combination of :func:`~torch.fft.fft` and
    :func:`~torch.fft.rfft`:

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(x, dim=1), dim=0)
    >>> torch.allclose(rfft2, two_ffts)

""")

irfft2 = _add_docstr(_fft.fft_irfft2, r"""
irfft2(input, s=None, dim=(-2, -1), norm=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfft2`.
Equivalent to :func:`~torch.fft.irfftn` but IFFTs only the last two dimensions by default.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfft2`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal shape :attr:`s`.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: last two dimensions.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfft2`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.rfft2`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfft2`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfft2(t)

    Without specifying the output length to :func:`~torch.fft.irfft2`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfft2(T).size()
    torch.Size([10, 10])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.irfft2(T, t.size())
    >>> roundtrip.size()
    torch.Size([10, 9])
    >>> torch.allclose(roundtrip, t)
    True

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
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Default: ``s = [input.size(d) for d in dim]``
    dim (Tuple[int], optional): Dimensions to be transformed.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.rfftn`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real FFT orthonormal)

        Where ``n = prod(s)`` is the logical FFT size.
        Calling the backward transform (:func:`~torch.fft.irfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfftn`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

    >>> t = torch.rand(10, 10)
    >>> rfftn = torch.fft.rfftn(t)
    >>> rfftn.size()
    torch.Size([10, 6])

    Compared against the full output from :func:`~torch.fft.fftn`, we have all
    elements up to the Nyquist frequency.

    >>> fftn = torch.fft.fftn(t)
    >>> torch.allclose(fftn[..., :6], rfftn)
    True

    The discrete Fourier transform is separable, so :func:`~torch.fft.rfftn`
    here is equivalent to a combination of :func:`~torch.fft.fft` and
    :func:`~torch.fft.rfft`:

    >>> two_ffts = torch.fft.fft(torch.fft.rfft(x, dim=1), dim=0)
    >>> torch.allclose(rfftn, two_ffts)

""")

irfftn = _add_docstr(_fft.fft_irfftn, r"""
irfftn(input, s=None, dim=None, norm=None) -> Tensor

Computes the inverse of :func:`~torch.fft.rfftn`.

:attr:`input` is interpreted as a one-sided Hermitian signal in the Fourier
domain, as produced by :func:`~torch.fft.rfftn`. By the Hermitian property, the
output will be real-valued.

Note:
    Some input frequencies must be real-valued to satisfy the Hermitian
    property. In these cases the imaginary component will be ignored.
    For example, any imaginary component in the zero-frequency term cannot
    be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`s`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal shape :attr:`s`.

Args:
    input (Tensor): the input tensor
    s (Tuple[int], optional): Signal size in the transformed dimensions.
        If given, each dimension ``dim[i]`` will either be zero-padded or
        trimmed to the length ``s[i]`` before computing the real FFT.
        If a length ``-1`` is specified, no padding is done in that dimension.
        Defaults to even output in the last dimension:
        ``s[-1] = 2*(input.size(dim[-1]) - 1)``.
    dim (Tuple[int], optional): Dimensions to be transformed.
        The last dimension must be the half-Hermitian compressed dimension.
        Default: all dimensions, or the last ``len(s)`` dimensions if :attr:`s` is given.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.irfftn`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)

        Where ``n = prod(s)`` is the logical IFFT size.
        Calling the forward transform (:func:`~torch.fft.rfftn`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.irfftn`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

    >>> t = torch.rand(10, 9)
    >>> T = torch.fft.rfftn(t)

    Without specifying the output length to :func:`~torch.fft.irfft`, the output
    will not round-trip properly because the input is odd-length in the last
    dimension:

    >>> torch.fft.irfftn(T).size()
    torch.Size([10, 10])

    So, it is recommended to always pass the signal shape :attr:`s`.

    >>> roundtrip = torch.fft.irfftn(T, t.size())
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
    Because the signal is Hermitian in the time-domain, the result will be
    real in the frequency domain. Note that some input frequencies must be
    real-valued to satisfy the Hermitian property. In these cases the imaginary
    component will be ignored. For example, any imaginary component in
    ``input[0]`` would result in one or more complex frequency terms which
    cannot be represented in a real output and so will always be ignored.

Note:
    The correct interpretation of the Hermitian input depends on the length of
    the original data, as given by :attr:`n`. This is because each input shape
    could correspond to either an odd or even length signal. By default, the
    signal is assumed to be even length and odd signals will not round-trip
    properly. So, it is recommended to always pass the signal length :attr:`n`.

Args:
    input (Tensor): the input tensor representing a half-Hermitian signal
    n (int, optional): Output signal length. This determines the length of the
        real output. If given, the input will either be zero-padded or trimmed to this
        length before computing the Hermitian FFT.
        Defaults to even output: ``n=2*(input.size(dim) - 1)``.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian FFT.
    norm (str, optional): Normalization mode. For the forward transform
        (:func:`~torch.fft.hfft`), these correspond to:

        * ``"forward"`` - normalize by ``1/n``
        * ``"backward"`` - no normalization
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the Hermitian FFT orthonormal)

        Calling the backward transform (:func:`~torch.fft.ihfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft`
        the exact inverse.

        Default is ``"backward"`` (no normalization).

Example:

    Taking a real-valued frequency signal and bringing it into the time domain
    gives Hermitian symmetric output:

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
:func:`~torch.fft.ihfft` represents this in the one-sided form where only the
positive frequencies below the Nyquist frequency are included. To compute the
full output, use :func:`~torch.fft.ifft`.

Args:
    input (Tensor): the real input tensor
    n (int, optional): Signal length. If given, the input will either be zero-padded
        or trimmed to this length before computing the Hermitian IFFT.
    dim (int, optional): The dimension along which to take the one dimensional Hermitian IFFT.
    norm (str, optional): Normalization mode. For the backward transform
        (:func:`~torch.fft.ihfft`), these correspond to:

        * ``"forward"`` - no normalization
        * ``"backward"`` - normalize by ``1/n``
        * ``"ortho"`` - normalize by ``1/sqrt(n)`` (making the IFFT orthonormal)

        Calling the forward transform (:func:`~torch.fft.hfft`) with the same
        normalization mode will apply an overall normalization of ``1/n`` between
        the two transforms. This is required to make :func:`~torch.fft.ihfft`
        the exact inverse.

        Default is ``"backward"`` (normalize by ``1/n``).

Example:

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

fftfreq = _add_docstr(_fft.fft_fftfreq, r"""
fftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the discrete Fourier Transform sample frequencies for a signal of size :attr:`n`.

Note:
    By convention, :func:`~torch.fft.fft` returns positive frequency terms
    first, followed by the negative frequencies in reverse order, so that
    ``f[-i]`` for all :math:`0 < i \leq n/2`` in Python gives the negative
    frequency terms. For an FFT of length :attr:`n` and with inputs spaced in
    length unit :attr:`d`, the frequencies are::

        f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. :func:`~torch.fft.fftfreq` follows NumPy's
    convention of taking it to be negative.

Args:
    n (int): the FFT length
    d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units.

Keyword Args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example:

    >>> torch.fft.fftfreq(5)
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

    For even input, we can see the Nyquist frequency at ``f[2]`` is given as
    negative:

    >>> torch.fft.fftfreq(4)
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

""".format(**factory_common_args))

rfftfreq = _add_docstr(_fft.fft_rfftfreq, r"""
rfftfreq(n, d=1.0, *, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Computes the sample frequencies for :func:`~torch.fft.rfft` with a signal of size :attr:`n`.

Note:
    :func:`~torch.fft.rfft` returns Hermitian one-sided output, so only the
    positive frequency terms are returned. For a real FFT of length :attr:`n`
    and with inputs spaced in length unit :attr:`d`, the frequencies are::

        f = torch.arange((n + 1) // 2) / (d * n)

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. Unlike :func:`~torch.fft.fftfreq`,
    :func:`~torch.fft.rfftfreq` always returns it as positive.

Args:
    n (int): the real FFT length
    d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units.

Keyword Args:
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example:

    >>> torch.fft.rfftfreq(5)
    tensor([ 0.0000,  0.2000,  0.4000])

    >>> torch.fft.rfftfreq(4)
    tensor([ 0.0000,  0.2500, 0.5000])

    Compared to the output from :func:`~torch.fft.fftfreq`, we see that the
    Nyquist frequency at ``f[2]`` has changed sign:
    >>> torch.fft.fftfreq(4)
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

""".format(**factory_common_args))

fftshift = _add_docstr(_fft.fft_fftshift, r"""
fftshift(input, dim=None) -> Tensor

Reorders n-dimensional FFT data, as provided by :func:`~torch.fft.fftn`, to have
negative frequency terms first.

This performs a periodic shift of n-dimensional data such that the origin
``(0, ..., 0)`` is moved to the center of the tensor. Specifically, to
``input.shape[dim] // 2`` in each selected dimension.

Note:
    By convention, the FFT returns positive frequency terms first, followed by
    the negative frequencies in reverse order, so that ``f[-i]`` for all
    :math:`0 < i \leq n/2` in Python gives the negative frequency terms.
    :func:`~torch.fft.fftshift` rearranges all frequencies into ascending order
    from negative to positive with the zero-frequency term in the center.

Note:
    For even lengths, the Nyquist frequency at ``f[n/2]`` can be thought of as
    either negative or positive. :func:`~torch.fft.fftshift` always puts the
    Nyquist term at the 0-index. This is the same convention used by
    :func:`~torch.fft.fftfreq`.

Args:
    input (Tensor): the tensor in FFT order
    dim (int, Tuple[int], optional): The dimensions to rearrange.
        Only dimensions specified here will be rearranged, any other dimensions
        will be left in their original order.
        Default: All dimensions of :attr:`input`.

Example:

    >>> f = torch.fft.fftfreq(4)
    >>> f
    tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

    >>> torch.fft.fftshift(f)
    tensor([-0.5000, -0.2500,  0.0000,  0.2500])

    Also notice that the Nyquist frequency term at ``f[2]`` was moved to the
    beginning of the tensor.

    This also works for multi-dimensional transforms:

    >>> x = torch.fft.fftfreq(5, d=1/5) + 0.1 * torch.fft.fftfreq(5, d=1/5).unsqueeze(1)
    >>> x
    tensor([[ 0.0000,  1.0000,  2.0000, -2.0000, -1.0000],
            [ 0.1000,  1.1000,  2.1000, -1.9000, -0.9000],
            [ 0.2000,  1.2000,  2.2000, -1.8000, -0.8000],
            [-0.2000,  0.8000,  1.8000, -2.2000, -1.2000],
            [-0.1000,  0.9000,  1.9000, -2.1000, -1.1000]])

    >>> torch.fft.fftshift(x)
    tensor([[-2.2000, -1.2000, -0.2000,  0.8000,  1.8000],
            [-2.1000, -1.1000, -0.1000,  0.9000,  1.9000],
            [-2.0000, -1.0000,  0.0000,  1.0000,  2.0000],
            [-1.9000, -0.9000,  0.1000,  1.1000,  2.1000],
            [-1.8000, -0.8000,  0.2000,  1.2000,  2.2000]])

    :func:`~torch.fft.fftshift` can also be useful for spatial data. If our
    data is defined on a centered grid (``[-(N//2), (N-1)//2]``) then we can
    use the standard FFT defined on an uncentered grid (``[0, N)``) by first
    applying an :func:`~torch.fft.ifftshift`.

    >>> x_centered = torch.arange(-5, 5)
    >>> x_uncentered = torch.fft.ifftshift(x_centered)
    >>> fft_uncentered = torch.fft.fft(x_uncentered)

    Similarly, we can convert the frequency domain components to centered
    convention by applying :func:`~torch.fft.fftshift`.

    >>> fft_centered = torch.fft.fftshift(fft_uncentered)

    The inverse transform, from centered Fourier space back to centered spatial
    data, can be performed by applying the inverse shifts in reverse order:

    >>> x_centered_2 = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fft_centered)))
    >>> torch.allclose(x_centered.to(torch.complex64), x_centered_2)
    True


""")

ifftshift = _add_docstr(_fft.fft_ifftshift, r"""
ifftshift(input, dim=None) -> Tensor

Inverse of :func:`~torch.fft.fftshift`.

Args:
    input (Tensor): the tensor in FFT order
    dim (int, Tuple[int], optional): The dimensions to rearrange.
        Only dimensions specified here will be rearranged, any other dimensions
        will be left in their original order.
        Default: All dimensions of :attr:`input`.

Example:

    >>> f = torch.fft.fftfreq(5)
    >>> f
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

    A round-trip through :func:`~torch.fft.fftshift` and
    :func:`~torch.fft.ifftshift` gives the same result:

    >>> shifted = torch.fftshift(f)
    >>> torch.ifftshift(shifted)
    tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

""")
