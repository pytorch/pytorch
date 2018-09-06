
Reproducibility
===============

There are several pseudorandom number generators involved in various layers of
the library, which you will need to seed manually to make experiments
reproducible.

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
