.. _complex_numbers-doc:

Complex Numbers
===============

Complex numbers are numbers that can be expressed in the form :math:`a + bj`, where a and b are real numbers,
and *j* is a solution of the equation :math:`x^2 = −1`. Complex numbers frequently occur in mathematics and
engineering, especially in signal processing. Traditionally many users and libraries (e.g., TorchAudio) have
handled complex numbers by representing the data in float tensors with shape :math:`(..., 2)` where the last
dimension contains the real and imaginary values.

Tensors of complex dtypes provide a more natural user experience for working with complex numbers. Operations on
complex tensors (e.g., :func:`torch.mv`, :func:`torch.matmul`) are likely to be faster and more memory efficient
than operations on float tensors mimicking them. Operations involving complex numbers in PyTorch are optimized
to use vectorized assembly instructions and specialized kernels (e.g. LAPACK, cuBlas).

.. note::
     Spectral operations (e.g., :func:`torch.fft`, :func:`torch.stft` etc.) currently don't use complex tensors but
     the API will be soon updated to use complex tensors.

.. warning ::
     Complex tensors is a beta feature and subject to change.

Creating Complex Tensors
------------------------

We support two complex dtypes: `torch.cfloat` and `torch.cdouble`

::

     >>> x = torch.randn(2,2, dtype=torch.cfloat)
     >>> x
     tensor([[-0.4621-0.0303j, -0.2438-0.5874j],
          [ 0.7706+0.1421j,  1.2110+0.1918j]])

.. note::

     The default dtype for complex tensors is determined by the default floating point dtype.
     If the default floating point dtype is `torch.float64` then complex numbers are inferred to
     have a dtype of `torch.complex128`, otherwise they are assumed to have a dtype of `torch.complex64`.

All factory functions apart from :func:`torch.linspace`, :func:`torch.logspace`, and :func:`torch.arange` are
supported for complex tensors.

Transition from the old representation
--------------------------------------

Users who currently worked around the lack of complex tensors with real tensors of shape :math:`(..., 2)`
can easily to switch using the complex tensors in their code using :func:`torch.view_as_complex`
and :func:`torch.view_as_real`. Note that these functions don’t perform any copy and return a
view of the input tensor.

::

     >>> x = torch.randn(3, 2)
     >>> x
     tensor([[ 0.6125, -0.1681],
          [-0.3773,  1.3487],
          [-0.0861, -0.7981]])
     >>> y = torch.view_as_complex(x)
     >>> y
     tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])
     >>> torch.view_as_real(y)
     tensor([[ 0.6125, -0.1681],
          [-0.3773,  1.3487],
          [-0.0861, -0.7981]])

Accessing real and imag
-----------------------

The real and imaginary values of a complex tensor can be accessed using the :attr:`real` and
:attr:`imag`.

.. note::
     Accessing `real` and `imag` attributes doesn't allocate any memory, and in-place updates on the
     `real` and `imag` tensors will update the original complex tensor. Also, the
     returned `real` and `imag` tensors are not contiguous.

::

     >>> y.real
     tensor([ 0.6125, -0.3773, -0.0861])
     >>> y.imag
     tensor([-0.1681,  1.3487, -0.7981])

     >>> y.real.mul_(2)
     tensor([ 1.2250, -0.7546, -0.1722])
     >>> y
     tensor([ 1.2250-0.1681j, -0.7546+1.3487j, -0.1722-0.7981j])
     >>> y.real.stride()
     (2,)

Angle and abs
-------------

The angle and absolute values of a complex tensor can be computed using :func:`torch.angle` and
`torch.abs`.

::

     >>> x1=torch.tensor([3j, 4+4j])
     >>> x1.abs()
     tensor([3.0000, 5.6569])
     >>> x1.angle()
     tensor([1.5708, 0.7854])

Linear Algebra
--------------

Currently, there is very minimal linear algebra operation support for complex tensors.
We currently support :func:`torch.mv`, :func:`torch.svd`, :func:`torch.qr`, and :func:`torch.inverse`
(the latter three are only supported on CPU). However we are working to add support for more
functions soon: :func:`torch.matmul`, :func:`torch.solve`, :func:`torch.eig`,
:func:`torch.symeig`. If any of these would help your use case, please
`search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
if an issue has already been filed and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.


Serialization
-------------

Complex tensors can be serialized, allowing data to be saved as complex values.

::

     >>> torch.save(y, 'complex_tensor.pt')
     >>> torch.load('complex_tensor.pt')
     tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])


Autograd
--------

PyTorch supports autograd for complex tensors. The autograd APIs can be
used for both holomorphic and non-holomorphic functions. For holomorphic functions,
you get the regular complex gradient. For :math:`C → R` real-valued loss functions,
`grad.conj()` gives a descent direction. For more details, check out the note :ref:`complex_autograd-doc`.

We do not support the following subsystems:

* Quantization

* JIT

* Sparse Tensors

* Distributed

If any of these would help your use case, please `search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
if an issue has already been filed and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.
