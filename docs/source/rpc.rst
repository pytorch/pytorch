.. role:: hidden
    :class: hidden-section

Distributed RPC Framework
=========================

The distributed RPC framework provides mechanisms for multi-machine model
training through a set of primitives to allow for remote communication, and a
higher-level API to automatically differentiate models split across several
machines.

RPC and RRef Framework
----------------------

Before using RPC and distributed autograd primitives, initialization must take
place. First, a backend over which RPCs can be sent over must be initialized.
The default (and currently, only available) implementation is the `ProcessGroup`
backend, and must be initialized with `torch.distributed.init_process_group
<https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`_
before using other functions. See the `documentation for
torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ for
additional details. Next, to initialize the RPC framework we need to use
`init_model_parallel` which would initialize the RPC framework, RRef framework
and distributed autograd.

.. automodule:: torch.distributed.rpc
.. autofunction:: init_model_parallel

RRef
----

An `RRef` (Remote REFerence) is a reference to a value of some type `T`
(e.g. `Tensor`) on a remote worker. This handle keeps the referenced remote
value alive on the owner, but there is no implication that the value will be
transferred to the local worker in the future. RRefs can be used in
multi-machine training by holding references to `nn.Modules
<https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ that exist on
other workers, and calling the appropriate functions to retrieve or modify their
parameters during training.

.. autoclass:: RRef
    :members:

RPC and RRef primitives
-----------------------

This library provides primitives allowing users to create and modify references
(RRefs) to remote data as well as remotely execute functions.

.. automodule:: torch.distributed.rpc.api
    :members:

Distributed Autograd Framework
------------------------------

This module provides an RPC-based distributed autograd framework that can be
used for applications such as model parallel training. In short, applications
may send and receive gradient recording tensors over RPC. In the forward pass,
we record when gradient recording tensors are sent over RPC and during the
backward pass we use this information to perform a distributed backward pass
using RPC. For more details see the design doc
`here <https://github.com/pytorch/pytorch/pull/29175>`_.

.. automodule:: torch.distributed.autograd
    :members:
