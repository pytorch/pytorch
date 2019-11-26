.. _distributed-rpc-framework:

Distributed RPC Framework
=========================

The distributed RPC framework provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a
higher-level API to automatically differentiate models split across several
machines.

.. warning::
  The RPC API is experimental and subject to change.


Basics
------

The distributed RPC framework makes it easy to run functions remotely, access
remote objects, and have autograd and optimizer marching through those
functions. This will be especially helpful if the forward pass needs to span
multiple machines when conducting, e.g., distributed model parallel training,
parameter-server training, etc. This framework contains four main sets of APIs.

1) **Remote Procedure Call (RPC)** allows run a function on the specified
   destination worker with given arguments and get (RRef of) the return value
   back.
2) **Remote Reference (RRef)** serves as a distributed shared pointer to a local
   or remote object. It can be shared with other workers and reference counting
   will be handled transparently.
3) **Distributed Autograd** stitches together local autograd engines on all the
   workers involved in the forward pass, and automatically reach out to them
   during the backward pass to compute gradients.
4) **Distributed Optimizer** takes a list of parameter RRefs in the constructor
   and updates them all accordingly when running `step()`.


.. _rpc:

RPC
---

Before using RPC and distributed autograd primitives, initialization must take
place. To initialize the RPC framework we need to use
:meth:`~torch.distributed.rpc.init_rpc` which would initialize the RPC
framework, RRef framework and distributed autograd. By default, this will also
initialize the `ProcessGroup` (:meth:`~torch.distributed.init_process_group`)
backend for RPC communication. The `ProcessGroup` backend internally uses gloo
for communication.


.. automodule:: torch.distributed.rpc
.. autofunction:: init_rpc

The following APIs provide primitives allowing users to remotely execute
functions as well as create (RRefs) to remote data objects.

.. automodule:: torch.distributed.rpc
.. autofunction:: rpc_sync
.. autofunction:: rpc_async
.. autofunction:: remote
.. autofunction:: get_worker_info
.. autofunction:: wait_all_workers


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
