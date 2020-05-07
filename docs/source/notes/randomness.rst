Reproducibility
===============

Completely reproducible results are not guaranteed across PyTorch releases,
individual commits or different platforms. Furthermore, results need not be
reproducible between CPU and GPU executions, even when using identical seeds.

However, in order to make computations deterministic on your specific problem on
one specific platform and PyTorch release, there are a couple of steps to take.

There are two pseudorandom number generators involved in PyTorch, which you will
need to seed manually to make runs reproducible. Furthermore, you should ensure
that all other libraries your code relies on and which use random numbers also
use a fixed seed.

PyTorch
.......
You can use :meth:`torch.manual_seed()` to seed the RNG for all devices (both
CPU and CUDA)::

    import torch
    torch.manual_seed(0)


There are some PyTorch functions that use CUDA functions that can be a source
of nondeterminism. One class of such CUDA functions are atomic operations,
in particular :attr:`atomicAdd`, which can lead to the order of additions being
nondetermnistic. Because floating-point addition is not perfectly associative
for floating-point operands, :attr:`atomicAdd` with floating-point operands can
introduce different floating-point rounding errors on each evaluation, which
introduces a source of nondeterministic variance (aka noise) in the result.

PyTorch functions that use :attr:`atomicAdd` in the forward kernels include
:meth:`torch.Tensor.index_add_`, :meth:`torch.Tensor.scatter_add_`,
:meth:`torch.bincount`.

A number of operations have backwards kernels that use :attr:`atomicAdd`,
including :meth:`torch.nn.functional.embedding_bag`,
:meth:`torch.nn.functional.ctc_loss`, :meth:`torch.nn.functional.interpolate`,
and many forms of pooling, padding, and sampling.

There is currently no simple way of avoiding nondeterminism in these functions.

Additionally, the backward path for :meth:`repeat_interleave` operates
nondeterministically on the CUDA backend because :meth:`repeat_interleave`
is implemented using :meth:`index_select`, the backward path for
which is implemented using :meth:`index_add_`, which is known to operate
nondeterministically (in the forward direction) on the CUDA backend (see above).

CuDNN
.....
When running on the CuDNN backend, two further options must be set::

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

.. warning::

    Deterministic operation may have a negative single-run performance impact,
    depending on the composition of your model. Due to different underlying
    operations, which may be slower, the processing speed (e.g. the number of
    batches trained per second) may be lower than when the model functions
    nondeterministically. However, even though single-run speed may be
    slower, depending on your application determinism may save time by
    facilitating experimentation, debugging, and regression testing.

Numpy
.....
If you or any of the libraries you are using rely on Numpy, you should seed the
Numpy RNG as well. This can be done with::

    import numpy as np
    np.random.seed(0)
