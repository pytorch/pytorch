.. _complex_numbers-doc:

Complex Numbers
===============

Complex numbers are numbers that can be expressed in the form :math:`a + bj`, where a and b are real numbers,
and *j* is called the imaginary unit, which satisfies the equation :math:`j^2 = -1`. Complex numbers frequently occur in mathematics and
engineering, especially in topics like signal processing. Traditionally many users and libraries (e.g., TorchAudio) have
handled complex numbers by representing the data in float tensors with shape :math:`(..., 2)` where the last
dimension contains the real and imaginary values.

Tensors of complex dtypes provide a more natural user experience while working with complex numbers. Operations on
complex tensors (e.g., :func:`torch.mv`, :func:`torch.matmul`) are likely to be faster and more memory efficient
than operations on float tensors mimicking them. Operations involving complex numbers in PyTorch are optimized
to use vectorized assembly instructions and specialized kernels (e.g. LAPACK, cuBlas).

.. note::
     Spectral operations in the `torch.fft module <https://pytorch.org/docs/stable/fft.html#torch-fft>`_ support
     native complex tensors.

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
:func:`torch.abs`.

::

     >>> x1=torch.tensor([3j, 4+4j])
     >>> x1.abs()
     tensor([3.0000, 5.6569])
     >>> x1.angle()
     tensor([1.5708, 0.7854])

Linear Algebra
--------------

Many linear algebra operations, like :func:`torch.matmul`, :func:`torch.linalg.svd`, :func:`torch.linalg.solve` etc., support complex numbers.
If you'd like to request an operation we don't currently support, please `search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
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

PyTorch supports autograd for complex tensors. The gradient computed is the Conjugate Wirtinger derivative,
the negative of which is precisely the direction of steepest descent used in Gradient Descent algorithm. Thus,
all the existing optimizers can be implemented to work out of the box with complex parameters. For more details,
check out the note :ref:`complex_autograd-doc`.


Optimizers
----------

Semantically, we define stepping through a PyTorch optimizer with complex parameters as being equivalent to stepping
through the same optimizer on the :func:`torch.view_as_real` equivalent of the complex params. More concretely:

::

     >>> params = [torch.rand(2, 3, dtype=torch.complex64) for _ in range(5)]
     >>> real_params = [torch.view_as_real(p) for p in params]

     >>> complex_optim = torch.optim.AdamW(params)
     >>> real_optim = torch.optim.AdamW(real_params)


`real_optim` and `complex_optim` will compute the same updates on the parameters, though there may be slight numerical
discrepancies between the two optimizers, similar to numerical discrepancies between foreach vs forloop optimizers
and capturable vs default optimizers. For more details, see https://pytorch.org/docs/stable/notes/numerical_accuracy.html.

Specifically, while you can think of our optimizer's handling of complex tensors as the same as optimizing over their
`p.real` and `p.imag` pieces separately, the implementation details are not precisely that. Note that the
:func:`torch.view_as_real` equivalent will convert a complex tensor to a real tensor with shape :math:`(..., 2)`,
whereas splitting a complex tensor into two tensors is 2 tensors of size :math:`(...)`. This distinction has no impact on
pointwise optimizers (like AdamW) but will cause slight discrepancy in optimizers that do global reductions (like LBFGS).
We currently do not have optimizers that do per-Tensor reductions and thus do not yet define this behavior. Open an issue
if you have a use case that requires precisely defining this behavior.


We do not fully support the following subsystems:

* Quantization

* JIT

* Sparse Tensors

* Distributed

If any of these would help your use case, please `search <https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+complex>`_
if an issue has already been filed and if not, `file one <https://github.com/pytorch/pytorch/issues/new/choose>`_.
