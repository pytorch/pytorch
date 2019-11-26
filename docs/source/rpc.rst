.. _distributed-rpc-framework:

Distributed RPC Framework
=========================

The distributed RPC framework provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a
higher-level API to automatically differentiate models split across several
machines.

.. warning::
  The RPC API is experimental and subject to change.

RPC and RRef Framework
----------------------

Before using RPC and distributed autograd primitives, initialization must take
place. To initialize the RPC framework we need to use
:meth:`~torch.distributed.rpc.init_rpc` which would initialize the RPC
framework, RRef framework and distributed autograd. By default, this will also
initialize the `ProcessGroup` (:meth:`~torch.distributed.init_process_group`)
backend for RPC communication. The `ProcessGroup` backend internally uses gloo
for communication.


.. automodule:: torch.distributed.rpc
.. autofunction:: init_rpc

.. _rref:

RRef
----

An `RRef` (Remote REFerence) is a reference to a value of some type `T`
(e.g. `Tensor`) on a remote worker. This handle keeps the referenced remote
value alive on the owner, but there is no implication that the value will be
transferred to the local worker in the future. RRefs can be used in
multi-machine training by holding references to `nn.Modules
<https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ that exist on
other workers, and calling the appropriate functions to retrieve or modify their
parameters during training. See :ref:`remote-reference-protocol` for more
details.

.. autoclass:: RRef
    :members:

RPC and RRef primitives
-----------------------

This library provides primitives allowing users to create and modify references
(RRefs) to remote data as well as remotely execute functions.

.. automodule:: torch.distributed.rpc
.. autofunction:: rpc_sync
.. autofunction:: rpc_async
.. autofunction:: remote
.. autofunction:: get_worker_info
.. autofunction:: wait_all_workers

Distributed Autograd Framework
------------------------------

This module provides an RPC-based distributed autograd framework that can be
used for applications such as model parallel training. In short, applications
may send and receive gradient recording tensors over RPC. In the forward pass,
we record when gradient recording tensors are sent over RPC and during the
backward pass we use this information to perform a distributed backward pass
using RPC. For more details see :ref:`distributed-autograd-design`.

.. automodule:: torch.distributed.autograd
    :members: context, backward, get_gradients

Distributed Optimizer
---------------------

.. automodule:: torch.distributed.optim
    :members: DistributedOptimizer
