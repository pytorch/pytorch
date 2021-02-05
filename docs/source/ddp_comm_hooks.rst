DDP Communication Hooks
=======================

DDP communication hook is a generic interface to control how to communicate
gradients across workers by overriding the vanilla allreduce in
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.>`_.
A few built-in communication hooks are provided,
and users can easily apply any of these hooks to optimize communication.
Besides, the hook interface can also support user-defined communication
strategies for more advanced use cases.

.. warning ::
    DDP communication hook is experimental and subject to change.

.. warning ::
    DDP communication hooks can only support single process single device mode
    on NCCL backend.

How to Use a Communication Hook?
--------------------------------

To use a communication hook, the user just needs to let the DDP model register
the hook before the training loop by calling
:func:`torch.nn.parallel.DistributedDataParallel.register_comm_hook`.

Default Communication Hooks
---------------------------

Default communication hooks are simple **stateless** hooks, so the input state
in ``register_comm_hook`` is either a process group or ``None``.

.. automodule:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
    :members:

PowerSGD Communication Hook
---------------------------

PowerSGD communication hook is a **stateful** hook used for gradient
compression, and the user needs to provide a state defined as below.
The performance is `on par with <https://observablehq.com/@tvogels/powersgd-benchmark>`_
the implementation in the original `paper <https://arxiv.org/abs/1905.13727>`_.

PowerSGD State
^^^^^^^^^^^^^^^^

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
.. autoclass:: PowerSGDState

PowerSGD Hooks
^^^^^^^^^^^^^^^^

.. warning ::
    As a biased compressor, PowerSGD requires an extra copy of gradients to
    enable error feedback for a higher accuracy.
    This may be infeasible for the use cases that have a memory constraint.

.. warning ::
    The current implementation may cause gradient overflow for FP16 input.

.. autofunction:: powerSGD_hook
.. autofunction:: batched_powerSGD_hook

Acknowledgements
----------------

Many thanks to PowerSGD paper author **Thijs Vogels** for the code review on
PowerSGD communication hook and the
`comparison experiments <https://observablehq.com/@tvogels/powersgd-benchmark>`_.
