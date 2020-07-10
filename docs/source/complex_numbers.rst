.. _complex_numbers-doc:

Complex Numbers
===============

Complex Numbers are numbers that can be expressed in the form :math:`a + bi`, where a and b are real numbers,
and *i* is a solution of the equation :math:`x^2 = −1`. Complex numbers frequently occur in mathematics and
engineering, especially in signal processing. The aim of introducing Complex Tensors is to provide a more natural
user experience for users and libraries (eg. TorchAudio) that currently workaround the lack of Complex Tensors
by using Float Tensors with shape :math:`(..., 2)` where the last dimension contains the real and imaginary values.

We support many functions for Complex Tensors eg. SVD, QR etc. Operations on complex tensors are likely to be
faster than operations on float tensors mimicking them. We also have a `Vec256 class` for `ComplexFloat` and
`ComplexDouble` to benefit from Intel assembly instructions for performing vectorized operations on CPU. In
addition, operations on Complex Tensors utilize specialized kernels for complex provided by LAPACK (for CPU)
and BLAS (for CUDA) or kernels in `torch` specially written for complex tensors. Thus using functions Complex
Tensors will provide performance benefits as opposed to users defining their own functions.

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

The default dtype for complex tensors is updated based on the current default tensor type.
cdouble if double else float.

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
- :func:`torch.view_as_real`:

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
::

     >>> y.real
     tensor([ 0.6125, -0.3773, -0.0861])
     >>> y.imag
     tensor([-0.1681,  1.3487, -0.7981])

Angle and abs
-------------
::

     >>> x1=torch.tensor([3j, 4+4j])
     >>> x1.abs()
     tensor([3.0000, 5.6569])
     >>> x1.angle()
     tensor([1.5708, 0.7854])

Serialization
-------------

::

     >>> torch.save(y, 'complex_tensor.pt')
     >>> torch.load('complex_tensor.pt')
     tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])

Autograd Support (with and without view_as_real, view_as_complex)
-----------------------------------------------------------------


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

We also do not support the following subsystems:

Quantization

JIT
distributions

multiprocessing

distributed

ONNX

If any of these would help your use case, please search if an issue has already been filed (https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+named+tensor%22)
and if not, file one(https://github.com/pytorch/pytorch/issues/new/choose).
