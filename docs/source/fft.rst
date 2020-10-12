.. role:: hidden
    :class: hidden-section

.. _torch-fft-module:

torch.fft
=========

Discrete Fourier transforms and related functions.

To use these functions the torch.fft module must be imported since its name
conflicts with the :func:`torch.fft` function.

.. automodule:: torch.fft
    :noindex:

.. currentmodule:: torch.fft

Fast Fourier Transforms
-----------------------

.. autofunction:: fft
.. autofunction:: ifft
.. autofunction:: fft2
.. autofunction:: ifft2
.. autofunction:: fftn
.. autofunction:: ifftn
.. autofunction:: rfft
.. autofunction:: irfft
.. autofunction:: rfft2
.. autofunction:: irfft2
.. autofunction:: rfftn
.. autofunction:: irfftn
.. autofunction:: hfft
.. autofunction:: ihfft

Helper Functions
----------------

.. autofunction:: fftfreq
.. autofunction:: rfftfreq
.. autofunction:: fftshift
.. autofunction:: ifftshift
