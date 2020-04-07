.. _distributed-rpc-framework:

Distributed RPC Framework
=========================

The distributed RPC framework provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a
higher-level API to automatically differentiate models split across several
machines.

.. warning ::
     APIs in the RPC package are stable. There are multiple ongoing work items to improve performance and error handling, which will ship in future releases.


Basics
------

The distributed RPC framework makes it easy to run functions remotely, supports
referencing remote objects without copying the real data around, and provides
autograd and optimizer APIs to transparently run backward and update parameters
across RPC boundaries. These features can be categorized into four sets of APIs.

1) **Remote Procedure Call (RPC)** supports running a function on the specified
   destination worker with the given arguments and getting the return value back
   or creating a reference to the return value. There are three main RPC APIs:
   :meth:`~torch.distributed.rpc.rpc_sync` (synchronous),
   :meth:`~torch.distributed.rpc.rpc_async` (asynchronous), and
   :meth:`~torch.distributed.rpc.remote` (asynchronous and returns a reference
   to the remote return value). Use the synchronous API if the user code cannot
   proceed without the return value. Otherwise, use the asynchronous API to get
   a future, and wait on the future when the return value is needed on the
   caller. The :meth:`~torch.distributed.rpc.remote` API is useful when the
   requirement is to create something remotely but never need to fetch it to
   the caller. Imagine the case that a driver process is setting up a parameter
   server and a trainer. The driver can create an embedding table on the
   parameter server and then share the reference to the embedding table with the
   trainer, but itself will never use the embedding table locally. In this case,
   :meth:`~torch.distributed.rpc.rpc_sync` and
   :meth:`~torch.distributed.rpc.rpc_async` are no longer appropriate, as they
   always imply that the return value will be returned to the caller
   immediately or in the future.
2) **Remote Reference (RRef)** serves as a distributed shared pointer to a local
   or remote object. It can be shared with other workers and reference counting
   will be handled transparently. Each RRef only has one owner and the object
   only lives on that owner. Non-owner workers holding RRefs can get copies of
   the object from the owner by explicitly requesting it. This is useful when
   a worker needs to access some data object, but itself is neither the creator
   (the caller of :meth:`~torch.distributed.rpc.remote`) or the owner of the
   object. The distributed optimizer, as we will discuss below, is one example
   of such use cases.
3) **Distributed Autograd** stitches together local autograd engines on all the
   workers involved in the forward pass, and automatically reach out to them
   during the backward pass to compute gradients. This is especially helpful if
   the forward pass needs to span multiple machines when conducting, e.g.,
   distributed model parallel training, parameter-server training, etc. With
   this feature, user code no longer needs to worry about how to send gradients
   across RPC boundaries and in which order should the local autograd engines
   be launched, which can become quite complicated where there are nested and
   inter-dependent RPC calls in the forward pass.
4) **Distributed Optimizer**'s constructor takes a
   :meth:`~torch.optim.Optimizer` (e.g., :meth:`~torch.optim.SGD`,
   :meth:`~torch.optim.Adagrad`, etc.) and a list of parameter RRefs, creates an
   :meth:`~torch.optim.Optimizer` instance on each distinct RRef owner, and
   updates parameters accordingly when running ``step()``. When you have
   distributed forward and backward passes, parameters and gradients will be
   scattered across multiple workers, and hence it requires an optimizer on each
   of the involved workers. Distributed Optimizer wraps all those local
   optimizers into one, and provides a concise constructor and ``step()`` API.


.. _rpc:

RPC
---

Before using RPC and distributed autograd primitives, initialization must take
place. To initialize the RPC framework we need to use
:meth:`~torch.distributed.rpc.init_rpc` which would initialize the RPC
framework, RRef framework and distributed autograd. By default, this will also
initialize the ``ProcessGroup`` (:meth:`~torch.distributed.init_process_group`)
backend for RPC communication. The ``ProcessGroup`` backend internally uses gloo
for communication.

.. automodule:: torch.distributed.rpc
.. autofunction:: init_rpc

The following APIs allow users to remotely execute functions as well as create
references (RRefs) to remote data objects. In these APIs, when passing a
``Tensor`` as an argument or a return value, the destination worker will try to
create a ``Tensor`` with the same meta (i.e., shape, stride, etc.). We
intentionally disallow transmitting CUDA tensors because it might crash if the
device lists on source and destination workers do not match. In such cases,
applications can always explicitly move the input tensors to CPU on the caller
and move it to the desired devices on the callee if necessary.

.. warning::
  TorchScript support in RPC is experimental and subject to change.

.. autofunction:: rpc_sync
.. autofunction:: rpc_async
.. autofunction:: remote
.. autofunction:: get_worker_info
.. autofunction:: shutdown
.. autoclass:: WorkerInfo
    :members:
.. autoclass:: ProcessGroupRpcBackendOptions
    :members:
    :inherited-members:

.. _rref:


RRef
----

An ``RRef`` (Remote REFerence) is a reference to a value of some type ``T``
(e.g. ``Tensor``) on a remote worker. This handle keeps the referenced remote
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
