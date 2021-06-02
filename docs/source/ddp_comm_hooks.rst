DDP Communication Hooks
=======================

DDP communication hook is a generic interface to control how to communicate
gradients across workers by overriding the vanilla allreduce in
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.>`_.
A few built-in communication hooks are provided,
and users can easily apply any of these hooks to optimize communication.
Besides, the hook interface can also support user-defined communication
strategies for more advanced use cases.

How to Use a Communication Hook?
--------------------------------

To use a communication hook, the user just needs to let the DDP model register
the hook before the training loop as below.

:func:`torch.nn.parallel.DistributedDataParallel.register_comm_hook`

What Does a Communication Hook Operate On?
------------------------------------------

Communication hook provides a flexible way to allreduce gradients.
Therefore, it mainly operates on the gradients on each replica before allreduce,
which are bucketized to increase the overlap between communication and computation.
Particularly, :class:`torch.distributed.GradBucket` represents a bucket of gradient tensors to be allreduced.

.. autoclass:: torch.distributed.GradBucket

.. autofunction:: torch.distributed.GradBucket.get_index
.. autofunction:: torch.distributed.GradBucket.get_tensor
.. autofunction:: torch.distributed.GradBucket.get_per_parameter_tensors
.. autofunction:: torch.distributed.GradBucket.is_the_last_bucket_to_allreduce
.. autofunction:: torch.distributed.GradBucket.set_tensor

Default Communication Hooks
---------------------------

Default communication hooks are simple **stateless** hooks, so the input state
in ``register_comm_hook`` is either a process group or ``None``.
The input ``bucket`` is a :class:`torch.distributed.GradBucket` object.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
.. autofunction:: allreduce_hook
.. autofunction:: fp16_compress_hook

Additionally, a communication hook wraper is provided to support :meth:`~fp16_compress_hook` as a wrapper,
which can be combined with other communication hooks.

.. autofunction:: fp16_compress_wrapper

PowerSGD Communication Hook
---------------------------

PowerSGD (`Vogels et al., NeurIPS 2019 <https://arxiv.org/abs/1905.13727>`_)
is a gradient compression algorithm, which can provide very high compression
rates and accelerate bandwidth-bound distributed training.
This algorithm needs to maintain both some hyperparameters and the internal
state. Therefore, PowerSGD communication hook is a **stateful** hook,
and the user needs to provide a state object defined as below.

PowerSGD State
^^^^^^^^^^^^^^^^

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
.. autoclass:: PowerSGDState

PowerSGD Hooks
^^^^^^^^^^^^^^^^

.. warning ::
    PowerSGD typically requires extra memory of the same size as the model's
    gradients to enable error feedback, which can compensate for biased
    compressed communication and improve accuracy.

.. warning ::
    PowerSGD hooks may conflict with `Apex automatic mixed precision package <https://github.com/NVIDIA/apex>`_.
    Please use PyTorch `native automatic mixed precision package <https://pytorch.org/docs/stable/amp.html>`_
    instead.

.. autofunction:: powerSGD_hook
.. autofunction:: batched_powerSGD_hook

Acknowledgements
----------------

Many thanks to PowerSGD paper author **Thijs Vogels** for the code review on
PowerSGD communication hook, as well as the
`comparison experiments <https://observablehq.com/@tvogels/powersgd-benchmark>`_,
which show that the performance of PowerSGD communication hook is on par with
the implementation in the original `paper <https://arxiv.org/abs/1905.13727>`_.
