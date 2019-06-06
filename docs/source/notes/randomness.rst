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
of non-determinism. One class of such CUDA functions are atomic operations,
in particular :attr:`atomicAdd`, where the order of parallel additions to the
same value is undetermined and, for floating-point variables, a source of
variance in the result. PyTorch functions that use :attr:`atomicAdd` in the forward
include :meth:`torch.Tensor.index_add_`, :meth:`torch.Tensor.scatter_add_`,
:meth:`torch.bincount`.

A number of operations have backwards that use :attr:`atomicAdd`, in particular
:meth:`torch.nn.functional.embedding_bag`,
:meth:`torch.nn.functional.ctc_loss` and many forms of pooling, padding, and sampling.
There currently is no simple way of avoiding non-determinism in these functions.


CuDNN
.....
When running on the CuDNN backend, two further options must be set::

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

.. warning::

    Deterministic mode can have a performance impact, depending on your model. This means that due to the deterministic nature of the model, the processing speed (i.e. processed batch items per second) can be lower than when the model is non-deterministic.

Numpy
.....
If you or any of the libraries you are using rely on Numpy, you should seed the
Numpy RNG as well. This can be done with::

    import numpy as np
    np.random.seed(0)
