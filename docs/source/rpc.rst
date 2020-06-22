:orphan:

.. contents:: :local:
    :depth: 2

.. _distributed-rpc-framework:

Distributed RPC Framework
=========================

The distributed RPC framework provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a
higher-level API to automatically differentiate models split across several
machines.

.. warning ::
     APIs in the RPC package are stable. There are multiple ongoing work items
     to improve performance and error handling, which will ship in future releases.


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
framework, RRef framework and distributed autograd.

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
  TorchScript support in RPC is experimental and subject to change. Since
  v1.5.0, ``torch.distributed.rpc`` supports calling TorchScript functions as
  RPC target functions, and this will help improve parallelism on the callee
  side as executing TorchScript functions does not require GIL.

.. autofunction:: rpc_sync
.. autofunction:: rpc_async
.. autofunction:: remote
.. autofunction:: get_worker_info
.. autofunction:: shutdown
.. autoclass:: WorkerInfo
    :members:


The RPC package also provides decorators which allow applications to specify
how a given function should be treated on the callee side.

.. warning::
  The ``rpc.functions`` package is experimental and subject to change.

.. autofunction:: torch.distributed.rpc.functions.async_execution


.. _rpc-backends:

Backends
^^^^^^^^

The RPC module can leverage different backends to perform the communication
between the nodes. The backend to be used can be specified in the
:func:`~torch.distributed.rpc.init_rpc` function, by passing a certain value of
the :class:`~torch.distributed.rpc.BackendType` enum. Regardless of what backend
is used, the rest of the RPC API won't change. Each backend also defines its own
subclass of the :class:`~torch.distributed.rpc.RpcBackendOptions` class, an
instance of which can also be passed to :func:`~torch.distributed.rpc.init_rpc`
to configure the backend's behavior.

.. autoclass:: BackendType

.. autoclass:: RpcBackendOptions
    :members:


Process Group Backend
"""""""""""""""""""""

The Process Group agent, which is the default, instantiates a process group from
the :mod:`~torch.distributed` module and utilizes its point-to-point
communication capabilities to send RPC messages across. Internally, the process
group uses `the Gloo library <https://github.com/facebookincubator/gloo/>`_.

Gloo has been hardened by years of extensive use in PyTorch and is thus very
reliable. However, as it was designed to perform collective communication, it
may not always be the best fit for RPC. For example, each networking operation
is synchronous and blocking, which means that it cannot be run in parallel with
others. Moreover, it opens a connection between all pairs of nodes, and brings
down all of them when one fails, thus reducing the resiliency and the elasticity
of the system.

Example::

    >>> import os
    >>> from torch.distributed import rpc
    >>> os.environ['MASTER_ADDR'] = 'localhost'
    >>> os.environ['MASTER_PORT'] = '29500'
    >>>
    >>> rpc.init_rpc(
    >>>     "worker1",
    >>>     rank=0,
    >>>     world_size=2,
    >>>     rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
    >>>         num_send_recv_threads=16,
    >>>         rpc_timeout=20 # 20 second timeout
    >>>     )
    >>> )
    >>>
    >>> # omitting init_rpc invocation on worker2


.. autoclass:: ProcessGroupRpcBackendOptions
    :members:
    :inherited-members:


TensorPipe Backend
""""""""""""""""""

.. warning::
    The TensorPipe backend is a **beta feature**.

The TensorPipe agent leverages `the TensorPipe library
<https://github.com/pytorch/tensorpipe>`_, which provides a natively
point-to-point communication primitive specifically suited for machine learning
that fundamentally addresses some of the limitations of Gloo. Compared to Gloo,
it has the advantage of being asynchronous, which allows a large number of
transfers to occur simultaneously, each at their own speed, without blocking
each other. It will only open pipes between pairs of nodes when needed, on
demand, and when one node fails only its incident pipes will be closed, while
all other ones will keep working as normal. In addition, it is able to support
multiple different transports (TCP, of course, but also shared memory, NVLink,
InfiniBand, ...) and can automatically detect their availability and negotiate
the best transport to use for each pipe.

The TensorPipe backend has been introduced in PyTorch v1.6 and is being actively
developed. At the moment, it only supports CPU tensors, with GPU support coming
soon. It comes with a TCP-based transport, just like Gloo. It is also able to
automatically chunk and multiplex large tensors over multiple sockets and
threads in order to achieve very high bandwidths. In addition to that, it packs
two Linux-specific transports for communication between processes on a same
machine (one based on ringbuffers stored in shared memory, the other on the
cross-memory attach syscalls) which can achieve lower latencies than TCP.
The agent will be able to pick the best transport on its own, with no
intervention required.

Example::

    >>> import os
    >>> from torch.distributed import rpc
    >>> os.environ['MASTER_ADDR'] = 'localhost'
    >>> os.environ['MASTER_PORT'] = '29500'
    >>>
    >>> rpc.init_rpc(
    >>>     "worker1",
    >>>     rank=0,
    >>>     world_size=2,
    >>>     backend=rpc.BackendType.TENSORPIPE,
    >>>     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    >>>         num_worker_threads=8,
    >>>         rpc_timeout=20 # 20 second timeout
    >>>     )
    >>> )
    >>>
    >>> # omitting init_rpc invocation on worker2

.. autoclass:: TensorPipeRpcBackendOptions
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
    :inherited-members:


.. toctree::
    :caption: More Information about RRef

    rpc/rref


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

.. toctree::
    :caption: More Information about RPC Autograd

    rpc/distributed_autograd


Distributed Optimizer
---------------------

.. automodule:: torch.distributed.optim
    :members: DistributedOptimizer

Design Notes
------------
The distributed autograd design note covers the design of the RPC-based distributed autograd framework that is useful for applications such as model parallel training.

-  :ref:`distributed-autograd-design`

The RRef design note covers the design of the :ref:`rref` (Remote REFerence) protocol used to refer to values on remote workers by the framework.

-  :ref:`remote-reference-protocol`

Tutorials
---------
The RPC tutorial introduces users to the RPC framework and provides two example applications using :ref:`torch.distributed.rpc<distributed-rpc-framework>` APIs.

-  `Getting started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__
-  `Implementing a Parameter Server using Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html>`__