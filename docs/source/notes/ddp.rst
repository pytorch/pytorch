.. _ddp:

Distributed Data Parallel
=========================

.. warning::
  The implementation of :class:`torch.nn.parallel.DistributedDataParallel`
  evolves over time. This design note is written based on the states as of v1.4.


:class:`torch.nn.parallel.DistributedDataParallel` replicates model, splits
input data, and transparently performs distributed training. This page
describes how it works and reveals implementation details.

Example
^^^^^^^

Below is a toy example of :class:`torch.nn.parallel.DistributedDataParallel`.

.. code::

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP


    def example(rank, world_size):
        # setup
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        model = nn.Linear(10, 10).to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        # run one iteration
        # forward pass
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()

    def main():
        world_size = 2
        mp.spawn(example,
            args=(world_size,),
            nprocs=world_size,
            join=True)

    if __name__=="__main__":
        main()



System Design
^^^^^^^^^^^^^

This section explains actions performed by
:class:`torch.nn.parallel.DistributedDataParallel` in different steps of one
training iteration.

- **Prerequisite**: DDP relies on c10d ``ProcessGroup`` for communications.
  Hence, applications must create ``ProcessGroup`` instances before constructing
  DDP.
- **Construction**: The DDP constructor takes a reference to the local module,
  and broadcasts ``state_dict()`` from rank0 process to all other processes in
  the group to make sure that all model replicas start from the exactly same
  state. Then, each DDP process creates a local ``Reducer``, which later will
  take care of the gradients synchronization during the backward pass. To
  improve communication efficiency, the ``Reducer`` organizes parameter
  gradients into buckets, and reduces one bucket at a time. The mapping from
  parameter gradients to buckets is determined at the construction time, based
  on the bucket size limit and the reverse order of ``Model.parameters()`` from
  the given model. The reason for using the reverse order is because DDP expects
  gradients to become ready during the backward pass in approximately that
  order. Of course, this assumption might not always be true, and when that
  happens it could hurt DDP backward speed as the ``Reducer`` cannot keep of the
  communication at the earliest possible time. Besides bucketing, the
  ``Reducer`` also registers autograd hooks during construction, one hook per
  parameter. These hooks will be triggered during backward pass when the
  gradient becomes ready.
- **Forward Pass**: The DDP takes the input and passes it to the local model,
  and then analyze the output from the local model if ``find_unused_parameters``
  is set to ``True``. This mode allows running backward on a subgraph of the
  model, and DDP finds out which parameters are involved in the backward pass by
  traversing the autograd graph from the model output and mark all unused
  parameters as ready for reduction. During the backward pass, the ``Reducer``
  would only wait for unready parameters, but it would still reduce all bucket
  as for now. Marking a parameter gradient as ready does not help DDP skip
  buckets as for now, but it will prevent DDP from waiting for absent gradients
  forever during the backward pass. Note that traversing graph introduces extra
  overheads, so applications should only set ``find_unused_parameters`` to
  ``True`` when necessary.
- **Backward Pass**: The ``backward()`` function is directly invoked on the loss
  ``Tensor``, which is out of DDP's control, and DDP uses autograd hooks to
  trigger its actions. When one gradient becomes ready, its corresponding DDP
  hook will fire, and DDP will then mark that parameter gradient as ready. When
  gradients in one bucket are all ready, the ``Reducer`` kicks of an
  asynchronous ``allreduce`` on that bucket to calculate mean of gradient across
  all processes. When all buckets are ready, the ``Reducer`` will block waiting
  for all ``allreduce`` operations to finish. When this is done, averaged
  gradients are written to the ``param.grad`` field of all parameters.
- **Optimizer Step**: From the optimizer's perspective, it is optimizing a local
  model. Model replicas on all DDP processes can keep in sync become they all
  start from the same state and they receive the same averaged gradients in
  every iteration.


.. image:: https://user-images.githubusercontent.com/16999635/72313121-4e7c1c80-3658-11ea-9e3a-ca75f697b9de.png
    :alt: ddp_grad_sync.png
    :width: 500 px

Implementation
^^^^^^^^^^^^^^

Below are pointers to the implementation.

ProcessGroup
------------

- `ProcessGroup.hpp <https://github.com/pytorch/pytorch/blob/v1.4.0/torch/lib/c10d/ProcessGroup.hpp>`__:
  contains the abstract API of all process group implementations. The ``c10d``
  library provides 4 implementations out of box, namely,
  `ProcessGroupGloo`, `ProcessGroupNCCL`, `ProcessGroupMPI`, and
  `ProcessGroupRoundRobin`, where `ProcessGroupRoundRobin` is a composition of
  multiple process group instances and launches collective communications in a
  round robin manner. ``DistributedDataParallel`` uses
  ``ProcessGroup::broadcast()`` to send model states from rank0 to others during
  initialization and ``ProcessGroup::allreduce()`` to sum gradients.


- `Store.hpp <https://github.com/pytorch/pytorch/blob/v1.4.0/torch/lib/c10d/Store.hpp>`__:
  assists the rendezvous service for process group instances to find each other.

DistributedDataParallel
-----------------------

- `distributed.py <https://github.com/pytorch/pytorch/blob/v1.4.0/torch/nn/parallel/distributed.py>`__:
  is the Python entry point for DDP. It implements the initialization and the
  ``forward`` function for the ``nn.parallel.DistributedDataParallel`` module
  which call into C++ libraries.

- `comm.h <https://github.com/pytorch/pytorch/blob/v1.4.0/torch/csrc/distributed/c10d/comm.h>`__:
  implements the coalesced broadcast helper function which is invoked to
  broadcast model states during initialization and synchronize model buffers
  before the forward pass.

- `reducer.h <https://github.com/pytorch/pytorch/blob/v1.4.0/torch/csrc/distributed/c10d/comm.h>`__:
  provides the core implementation for gradient synchronization in the backward
  pass.

The following stack graph shows the structure of code.

.. image:: https://user-images.githubusercontent.com/16999635/72313120-4e7c1c80-3658-11ea-9c6d-44336b2daeac.png
    :alt: ddp_code.png
    :width: 500 px
