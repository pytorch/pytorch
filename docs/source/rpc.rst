.. role:: hidden
    :class: hidden-section

Distributed RPC Framework
=====================================================

The distributed RPC framework provides mechanisms for multi-machine model training through a set of primitives to allow for remote communication, and a higher-level API to automatically differentiate models split across several machines.

RPC and RRef Framework
====================================================

Basics
------

Before using RPC and distributed autograd primitives, initialization must take place. First, a backend over which RPCs can be sent over must be initialized. The default (and currently, only available) implementation is the `ProcessGroup` backend, and must be initialized with `torch.distributed.init_process_group` before using other function. See the `documentation for torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ for additional details. Next, the local RPC agent can be initialized, after which the process will be able to send and receive RPCs from all other connected processes.

.. automodule:: torch.distributed.rpc
    :members:

RRef
------

An RRef (Remote REFerence) is a reference to a value of some type T (e.g. Tensor) on a remote worker. This handle keeps the referenced remote value alive on the owner, but there is no implication that the value will be transferred to the local worker in the future. RRefs can be used in multi-machine training through holding references to `nn.Modules` that exist on other workers, and calling the appropriate functions to retrieve or modify their parameters during training.

.. autoclass:: RRef
    :members:

RPC and RRef primitives
------

This library provides primitives allowing users to create and modify references to remote data and remotely execute functions and retrieve the result.

.. automodule:: torch.distributed.rpc.api
    :members:

Distributed Autograd Framework
====================================================
This module provides an RPC-based distributed autograd framework that can be used for applications such as model parallel training. In short, applications may run RPC to execute code remotely in the forward pass, and autograd will automatically travel across RPC boundaries in the backward pass.


.. automodule:: torch.distributed.autograd
    :members:
