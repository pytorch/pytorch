.. _complex_numbers-doc:

Complex Numbers
===============

Complex numbers are numbers that can be expressed in the form :math:`a + bj`, where a and b are real numbers,
and *j* is a solution of the equation :math:`x^2 = −1`. Complex numbers frequently occur in mathematics and
engineering, especially in signal processing. Tensors of complex dtypes provide a more natural user experience
for users and libraries (eg. TorchAudio) that previously worked around the lack of complex tensors by using
float tensors with shape :math:`(..., 2)` where the last dimension contained the real and imaginary values.

Operations on complex tensors (eg :func:`torch.mv`, :func:`torch.matmul`) are likely to be faster and more
memory efficient than operations on float tensors mimicking them. Operations involving complex numbers in
PyTorch are optimized to use vectorized assembly instructions and specialized kernels (e.g. LAPACK, CuBlas).
Thus using functions for complex tensors will provide performance benefits as opposed to users defining
their own functions.

.. note::
     Spectral Ops currently don't use complex tensors but the API would be soon updated to use complex tensors.

.. warning ::
     Complex Tensors is a beta feature and subject to change.

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
     If the default floating point dtype is torch.float64 then complex numbers are inferred to
     have a dtype of torch.complex128, otherwise they are assumed to have a dtype of torch.complex64.

All factory functions apart from :func:`torch.linspace`, :func:`torch.logspace`, and :func:`torch.arange` are
supported for complex tensors.

Transition from the old representation
--------------------------------------

Users who currently worked around the lack of complex tensors with real tensors of shape `(..., 2)`
can easily to switch using the complex tensors in their code using :func:`torch.view_as_complex` and
- :func:`torch.view_as_real`. Note that these functions don't perform any copy and
return a view of the input Tensor.

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

The angle and absolute values of a complex tensor can be accessed using :func:`torch.angle` and
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
functions soon: :func:`torch.matmul`, :func:`torch.solve`, :func:`torch.eig`, :func:`torch.eig`,
:func:`torch.symeig`. If any of these would help your use case, please
`search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
if an issue has already been filed and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.


Serialization
-------------

Complex Tensors can be serialized, allowing data to be saved as complex values.

::

     >>> torch.save(y, 'complex_tensor.pt')
     >>> torch.load('complex_tensor.pt')
     tensor([ 0.6125-0.1681j, -0.3773+1.3487j, -0.0861-0.7981j])


Autograd
--------

PyTorch supports Autograd for Complex Tensors. The autograd APIs can be
used for both holomorphic and non-holomorphic functions. For non-holomorphic
functions, the gradient is evaluated as if it were holomorphic. For more details,
check out the note :ref:`complex_autograd-doc`.

Gradient calculation can also be easily done for functions not supported for complex tensors
yet by enclosing the unsupported operations between :func:`torch.view_as_real` and
:func:`torch.view_as_complex` functions. The example shown below computes the pointwise multiplication
of two complex tensors, in one case by performing operations on complex tensors, and in the other
by by performing operations on complex tensors viewed as real tensors. As shown below, the gradients
computed have same values in both cases.

::

     >>> x = torch.randn(2, dtype=torch.cfloat, requires_grad=True)
     >>> y = torch.randn(2, dtype=torch.cfloat, requires_grad=True)
     >>> x_ = x.detach().requires_grsad_(True)
     >>> y_ = y.detach().requires_grad_(True)
     >>> z = x[0]*y[0] + x[1]*y[1]
     >>> z
     tensor(0.2114-1.1952j, grad_fn=<AddBackward0>)

     >>> x0 = x_.clone()
     >>> y0 = y_.clone()
     >>> x1 = torch.view_as_real(x0)
     >>> y1 = torch.view_as_real(y0)
     >>> z_ = torch.empty_like(x1)
     >>> z_[:, 0] = x1[:, 0] * y1[:, 0] - x1[:, 1] * y1[:, 1]
     >>> z_[:, 1] = x1[:, 0] * y1[:, 1] + x1[:, 1] * y1[:, 0]
     >>> z1 = torch.view_as_complex(z_)
     >>> z2 = z1.sum()
     >>> z2
     tensor(0.2114-1.1952j, grad_fn=<SumBackward0>)

     >>> z.backward()
     >>> z2.backward()
     >>> x.grad, y.grad
     tensor([-0.6815+0.5931j,  0.5333-1.0872j]) tensor([-0.4869+0.9011j,  0.3673+0.2007j])
     >>> x_.grad, y_.grad
     tensortensor([-0.6815+0.5931j,  0.5333-1.0872j]) tensor([-0.4869+0.9011j,  0.3673+0.2007j])

We do not support the following subsystems:

* Quantization

* JIT

* Sparse Tensors

* Distributed

If any of these would help your use case, please `search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
if an issue has already been filed and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.
