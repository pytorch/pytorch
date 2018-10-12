
Reproducibility
===============

Completely reproducible results are not guaranteed across PyTorch releases,
individual commits or different platforms. Furthermore, results need not be
reproducible between CPU and GPU executions, even when using identical seeds.

However, in order to make computations deterministic on your specific problem on
one specific platform and PyTorch release, there are a couple of steps to take.

There are two pseudorandom number generators involved in PyTorch, which you will
need to seed manually to make runs reproducible. Furthermore, you should ensure
that all other libraries your code relies on an which use random numbers also
use a fixed seed.

PyTorch
.......
You can use :meth:`torch.manual_seed()` to seed the RNG for all devices (both
CPU and CUDA)

    import torch
    torch.manual_seed(0)


CuDNN
.....
When running on the CuDNN backend, one further option must be set::

    torch.backends.cudnn.deterministic = True

.. warning::

    Deterministic mode can have a performance impact, depending on your model.

Numpy
.....
If you or any of the libraries you are using rely on Numpy, you should seed the
Numpy RNG as well. This can be done with::

    import numpy as np
    np.random.seed(0)
