import warnings

from .windows import cosine, exponential, gaussian

from torch._torch_docs import factory_common_args

__all__ = [
    'cosine',
    'exponential',
    'gaussian',
]


def _add_docstr(function, docstr):
    function.__doc__ = docstr


_add_docstr(cosine, r"""
Computes a window with a simple cosine waveform.
Also known as the sine window.

The cosine window is defined as follows:

.. math::
    w(n) = \cos{\left(\frac{\pi n}{M} - \frac{\pi}{2}\right)} = \sin{\left(\frac{\pi n}{M}\right)}

Where `M` is the length of the window.
    """ +
            r"""
        
        Args:
            window_length (int): the length of the output window.
                In other words, the number of points of the cosine window.
            periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
                If `False`, returns a symmetric window suitable for use in filter design. Default: `True`.
        
        Keyword args:
            {dtype}
            {layout}
            {device}
            {requires_grad}
        
        Examples:
            >>> # Generate a cosine window without keyword args.
            >>> torch.signal.windows.cosine(10)
            tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1.0000, 0.9595, 0.8413, 0.6549,
            0.4154])
        
            >>> # Generate a symmetric cosine window.
            >>> torch.signal.windows.cosine(10,periodic=False)
            tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540,
            0.1564])
        
        .. note::
            The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
            and `periodic` is `False`.
        """.format(
                **factory_common_args
            ))

_add_docstr(exponential, r"""
Computes a window with an exponential waveform.
Also known as Poisson window.

The exponential window is defined as follows:

.. math::
    w(n) = \exp{\left(-\frac{|n - center|}{\tau}\right)}
    """ +
            r"""
        
        Args:
            window_length (int): the length of the output window.
                In other words, the number of points of the ee window.
            periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
                If `False`, returns a symmetric window suitable for use in filter design. Default: `True`.
            center (float, optional): his value defines where the center of the window will be located.
                In other words, at which sample the peak of the window can be found.
                Default: `window_length / 2` if `periodic` is `True` (default), else `(window_length - 1) / 2`.
            tau (float, optional): the decay value.
                For `center = 0`, it's suggested to use :math:`\tau = -\frac{(M - 1)}{\ln(x)}`,
                if `x` is the fraction of the window remaining at the end. Default: 1.0.
            """ +
            r"""
        
        Keyword args:
            {dtype}
            {layout}
            {device}
            {requires_grad}
        
        Examples:
            >>> # Generate an exponential window without keyword args.
            >>> torch.signal.windows.exponential(10)
            tensor([0.0067, 0.0183, 0.0498, 0.1353, 0.3679, 1.0000, 0.3679, 0.1353, 0.0498,
            0.0183])
        
            >>> # Generate a symmetric exponential window and decay factor equal to .5
            >>> torch.signal.windows.exponential(10,periodic=False,tau=.5)
            tensor([1.2341e-04, 9.1188e-04, 6.7379e-03, 4.9787e-02, 3.6788e-01, 3.6788e-01,
            4.9787e-02, 6.7379e-03, 9.1188e-04, 1.2341e-04])
        
        .. note::
            The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
            and `periodic` is `False`.
        """.format(
                **factory_common_args
            ))

_add_docstr(gaussian, r"""
Computes a window with a gaussian waveform.

The gaussian window is defined as follows:

.. math::
    w(n) = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
    """ +
            r"""
        
        Args:
            window_length (int): the length of the output window.
                In other words, the number of points of the cosine window.
            periodic (bool, optional): If `True`, returns a periodic window suitable for use in spectral analysis.
                If `False`, returns a symmetric window suitable for use in filter design. Default: `True`
            std (float, optional): the standard deviation of the gaussian. It controls how narrow or wide the window is.
                Default: 0.5.
        
        Keyword args:
            {dtype}
            {layout}
            {device}
            {requires_grad}
        
        Examples:
            >>> # Generate a gaussian window without keyword args.
            >>> torch.signal.windows.gaussian(10)
            tensor([1.9287e-22, 1.2664e-14, 1.5230e-08, 3.3546e-04, 1.3534e-01, 1.0000e+00,
            1.3534e-01, 3.3546e-04, 1.5230e-08, 1.2664e-14])
        
            >>> # Generate a symmetric gaussian window and standard deviation equal to 0.9.
            >>> torch.signal.windows.gaussian(10,periodic=False,std=0.9)
            tensor([3.7267e-06, 5.1998e-04, 2.1110e-02, 2.4935e-01, 8.5700e-01, 8.5700e-01,
            2.4935e-01, 2.1110e-02, 5.1998e-04, 3.7267e-06])
        
        .. note::
            The window is normalized to 1 (maximum value is 1), however, the 1 doesn't appear if `M` is even
            and `periodic` is `False`.
        """.format(
                **factory_common_args
            ))
