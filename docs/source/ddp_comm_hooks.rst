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

A communication hook provides a flexible way to allreduce gradients.
Therefore, it mainly operates on the gradients on each replica before allreduce,
which are bucketized to increase the overlap between communication and computation.
Particularly, :class:`torch.distributed.GradBucket` represents a bucket of gradient tensors to be allreduced.

.. autoclass:: torch.distributed.GradBucket

.. autofunction:: torch.distributed.GradBucket.index
.. autofunction:: torch.distributed.GradBucket.buffer
.. autofunction:: torch.distributed.GradBucket.gradients
.. autofunction:: torch.distributed.GradBucket.is_last
.. autofunction:: torch.distributed.GradBucket.set_buffer
.. autofunction:: torch.distributed.GradBucket.parameters

Default Communication Hooks
---------------------------

Default communication hooks are simple **stateless** hooks, so the input state
in ``register_comm_hook`` is either a process group or ``None``.
The input ``bucket`` is a :class:`torch.distributed.GradBucket` object.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.default_hooks
.. autofunction:: allreduce_hook
.. autofunction:: fp16_compress_hook
.. autofunction:: bf16_compress_hook

Additionally, a communication hook wrapper is provided to support :meth:`~fp16_compress_hook` or :meth:`~bf16_compress_hook` as a wrapper,
which can be combined with other communication hooks.

.. autofunction:: fp16_compress_wrapper
.. autofunction:: bf16_compress_wrapper

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

Debugging Communication Hooks
-----------------------------

As the name implies, debugging communication hooks are **only** used for debugging and performance optimization purpose.

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks

.. warning ::
    Debugging communication hooks do not necessarily output the correct results.

.. autofunction:: noop_hook

Checkpointing of Communication Hooks
------------------------------------

.. currentmodule:: torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook

A stateful communication hook can be saved as a part of model checkpointing to enable trainer restarts.
To make a hook serializable, ``__setstate__`` and ``__getstate__`` should be defined.

.. warning ::
    ``__getstate__`` should exclude non-serializable attributes from a returned dictionary.

.. warning ::
    ``__setstate__`` should properly initialize non-serializable attributes, excluded from a provided ``state``.

:class:`PowerSGDState` has ``__setstate__`` and ``__getstate__`` implemented and can be used as a reference.

.. class:: PowerSGDState
    :noindex:

    .. automethod:: PowerSGDState.__getstate__
    .. automethod:: PowerSGDState.__setstate__

Here is a simple, end-to-end example of saving and reloading PowerSGD state and hook.

::

    import os
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    import torch.multiprocessing as mp

    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(24,24)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(24,12)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    def run_demo(demo_fn, world_size):
        mp.spawn(
            demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True)

    def demo_serialization(rank, world_size):
        setup(rank, world_size)

        CHECKPOINT = tempfile.gettempdir() + "/checkpoint.pt"

        model = SimpleModel().to(rank)
        ddp_model = DistributedDataParallel(model, device_ids=[rank])

        powersgd_hook = powerSGD.powerSGD_hook
        powersgd_state = powerSGD.PowerSGDState(process_group=None)

        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        state = {
            'state_dict': ddp_model.state_dict(),
            'comm_hook': powersgd_hook,
            'comm_hook_state': powersgd_state}

        if rank == 0:
            torch.save(state, CHECKPOINT)

        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(CHECKPOINT, map_location=map_location)

        new_ddp_model = DistributedDataParallel(SimpleModel().to(rank), device_ids=[rank])
        new_ddp_model.load_state_dict(checkpoint['state_dict'])
        powersgd_hook = checkpoint['comm_hook']
        powersgd_state = checkpoint['comm_hook_state']

        new_ddp_model.register_comm_hook(powersgd_state, powersgd_hook)

        if rank == 0:
            os.remove(CHECKPOINT)

        cleanup()

    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(demo_serialization, world_size)

Acknowledgements
----------------

Many thanks to PowerSGD paper author **Thijs Vogels** for the code review on
PowerSGD communication hook, as well as the
`comparison experiments <https://observablehq.com/@tvogels/powersgd-benchmark>`_,
which show that the performance of PowerSGD communication hook is on par with
the implementation in the original `paper <https://arxiv.org/abs/1905.13727>`_.
