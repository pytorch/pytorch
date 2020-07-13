.. _complex_numbers-doc:

Complex Numbers
===============

Complex Numbers are numbers that can be expressed in the form :math:`a + bi`, where a and b are real numbers,
and *i* is a solution of the equation :math:`x^2 = −1`. Complex numbers frequently occur in mathematics and
engineering, especially in signal processing. The aim of introducing tensors of complex dtypes is to provide
a more natural user experience for users and libraries (eg. TorchAudio) that currently work around the
lack of complex tensors by using Float Tensors with shape :math:`(..., 2)` where the last dimension contains
the real and imaginary values.

We support many functions for Complex Tensors eg. :func:`torch.svd`, :func:`torch.qr` etc. Operations on complex tensors are likely to be
faster than operations on float tensors mimicking them. Operations involving complex numbers in PyTorch are
optimized to use vectorized assembly instructions and specialized kernels (e.g. LAPACK, CuBlas). Thus using
functions for complex tensors will provide performance benefits as opposed to users defining their own functions.

.. warning ::
     Complex Tensors is a beta feature and subject to change.

Creating Complex Tensors
----------------------

We support two complex dtypes: `torch.cfloat` and `torch.cdouble`

::

     >>> x = torch.randn(2,2, dtype=torch.cfloat)
     >>> x
     tensor([[-0.4621-0.0303j, -0.2438-0.5874j],
          [ 0.7706+0.1421j,  1.2110+0.1918j]])

.. note::

     The default dtype for complex tensors is determined by the default floating point dtype.
     If the default floating point dtype is torch.float64 then complex numbers are inferred to
     have a dtype of torch.complex128, otherwise they are assumed to have a dtype of torch.complex64.

The following factory functions can be used to create complex tensors:

- :func:`torch.tensor`
- :func:`torch.as_tensor`
- :func:`torch.empty`
- :func:`torch.rand`
- :func:`torch.randn`
- :func:`torch.as_strided`
- :func:`torch.from_numpy`
- :func:`torch.zeros`
- :func:`torch.ones`
- :func:`torch.full`
- :func:`torch.eye`

Transition from the old representation
--------------------------------------

Users who currently worked around the lack of complex tensors with real tensors of shape `(..., 2)`
can easily to switch using the complex tensors in their code using :func:`torch.view_as_complex` and
- :func:`torch.view_as_real` view functions:

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
:attr:`imag` views.

::

     >>> y.real
     tensor([ 0.6125, -0.3773, -0.0861])
     >>> y.imag
     tensor([-0.1681,  1.3487, -0.7981])

Angle and abs
-------------

The angle and absolute values of a complex tensor can be accesses using :func:`torch.angle` and
`torch.abs`.

::

     >>> x1=torch.tensor([3j, 4+4j])
     >>> x1.abs()
     tensor([3.0000, 5.6569])
     >>> x1.angle()
     tensor([1.5708, 0.7854])

Linear Algebra
--------------

Currently, there is very minimal linear algebraic operation support for complex tensors.
We currently support :func:`torch.mv`, :func:`torch.svd`, :func:`torch.qr`, and :func:`torch.inverse`
(the latter three are only supported on CPU). However we are working to add support for more
functions soon: :func:`torch.matmul`, :func:`torch.solve`, :func:`torch.eig`, :func:`torch.eig`,
:func:`torch.symeig`. If any of the other ops would help your use case, please search if an issue has
already been filed (https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22)
and if not, file one (https://github.com/pytorch/pytorch/issues/new/choose).

Serialization
-------------

Complex Tensors can be serialized, allowing data to be saved as complex values.

::

     >>> torch.save(y, 'complex_tensor.pt')
     >>> torch.load('complex_tensor.pt')
     tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])


Autograd
--------

PyTorch supports Autograd for Complex Tensors.

1. :func:`torch.functional.backward` can be used for holomorphic :math:`C -> C` functions.
   For non-holomorphic functions, the gradient is evaluated as if it were holomorphic.
2. :func:`torch.functional.backward` can be used to optimize :math:`C -> R` functions, like
   real-values loss functions of complex parameters :math:`x` by taking steps in the direction
   of conjugate of :math:`x.grad`.
3. mention the current behavior of backward for spectral ops?

For more details, check out the Autograd note :ref:`complex_autograd-doc`.

::

     x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)>>> x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
     >>> y = x.detach().requires_grad_(True)
     >>> x0 = x.clone()
     >>> x1 = torch.view_as_complex(x0)
     >>> x2 = torch.view_as_real(x1)
     >>> x2.mul_(2)
     tensor([[[ 4.2425, -0.1076],
          [ 3.2731,  2.3156]],

          [[ 4.1179,  0.7358],
          [-1.7711, -0.4389]]], dtype=torch.float64,
          grad_fn=<ViewAsRealBackward>)
     >>> x2.sum().backward()
     >>> y0 = y.clone()
     >>> y0.mul_(2)
     tensor([[[ 4.2425, -0.1076],
          [ 3.2731,  2.3156]],

          [[ 4.1179,  0.7358],
          [-1.7711, -0.4389]]], dtype=torch.float64, grad_fn=<MulBackward0>)
     >>> y0.sum().backward()
     >>> x.grad
     tensor([[[2., 2.],
          [2., 2.]],

          [[2., 2.],
          [2., 2.]]], dtype=torch.float64)
     >>> y.grad
     tensor([[[2., 2.],
          [2., 2.]],

          [[2., 2.],
          [2., 2.]]], dtype=torch.float64)

We do not support the following subsystems:

Quantization

JIT

Sparse Tensors

Distributed

Multiprocessing

If any of these would help your use case, please search if an issue has already been filed (https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22)
and if not, file one (https://github.com/pytorch/pytorch/issues/new/choose).
