.. currentmodule:: torch.distributed.ops

Autograd-Aware Collective Functions
===================================

.. automodule:: torch.distributed.ops

.. warning::
    The ``torch.distributed.nn.functional`` module which offers a similar set of
    functions is deprecated and should not be used for future development.

Example
^^^^^^^
::

    # Worker 1 ###########################################
    import torch
    import torch.distributed as dist
    import torch.distributed.ops as distops

    dist.init_process_group(
        backend="gloo", init_method="tcp://localhost:12345", world_size=2, rank=0
    )

    x = torch.tensor([[2.0, 2.0, 1.0], [1.0, 3.0, 1.0]], requires_grad=True)

    y = x ** 3

    # y:
    # tensor([[8,  8, 1],
    #         [1, 27, 1]])

    # Take the product across all ranks.
    s = distops.prod(y)

    # s:
    # tensor([[8, 512, 27],
    #         [8,  27,  1]])

    s.backward(torch.ones([2, 3]))

    # x_grad:
    # tensor([[24, 1536, 162]
    #         [48,   54,  6]])



    # Worker 2 ###########################################
    import torch
    import torch.distributed as dist
    import torch.distributed.ops as distops

    dist.init_process_group(
        backend="gloo", init_method="tcp://localhost:12345", world_size=2, rank=1
    )

    x = torch.tensor([[1.0, 4.0, 3.0], [2.0, 1.0, 1.0]], requires_grad=True)

    y = x ** 3

    # y:
    # tensor([[1, 64, 27],
    #         [8,  1,  1]])

    # Take the product across all ranks.
    s = distops.prod(y)

    # s:
    # tensor([[8, 512, 27],
    #         [8,  27,  1]])

    s.backward(torch.ones([2, 3]))

    # x_grad:
    # tensor([[48, 768, 54]
    #         [24, 162,  6]])

Functions
^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:

    sum
    sum_on_rank
    prod
    minimum
    minimum
    copy
