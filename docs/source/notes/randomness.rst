
Reproducibility
===============

Completely reproducible results are not guaranteed across PyTorch releases,
individual commits or different platforms. Furthermore, results need to be
reproducible between CPU and GPU executions, even when using identical seeds.

However, in order to make computations deterministic on your specific problem on
one specific platform and PyTorch release, there are a couple of steps to take.

There are several pseudorandom number generators involved in various layers of
the library, which you will need to seed manually to make runs reproducible.

Numpy
.....
The numpy rng can be seeded with::

    import numpy as np
    np.random.seed(0)

PyTorch
.......
When running on the CPU, you should seed the RNG with::

    import torch
    torch.manual_seed(0)

While this will seed all CUDA RNGs as well, you can do so explicitly with
:meth:`torch.cuda.manual_seed` for the current device, or
:meth:`torch.cuda.manual_seed_all` to seed the RNG for all GPUs

CuDNN
.....
When running on the CuDNN backend, one further option must be set::

    torch.backends.cudnn.deterministic = True

.. warning::

    Deterministic mode can have a performance impact, depending on your model.
