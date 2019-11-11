.. role:: hidden
    :class: hidden-section

Distributed RPC Framework
=====================================================

The distributed RPC framework provides mechanisms for multi-machine model training through a set of primitives to allow for remote communication, and a higher-level API to automatically differentiate models split across several machines.

RPC and RRef Framework
====================================================

Bbasics
------

An RRef is a reference to a value of some type T (e.g. Tensor) on a remote worker. This handle keeps the referenced remote tensor value alive on the owner, but there is no implication that the value will be transferred to the local worker in the future.

.. automodule:: torch.distributed.rpc
    :members:
.. autoclass:: RRef
    :members:
.. automodule:: torch.distributed.rpc.api
    :members:

Distributed Autograd Framework
====================================================
This module provides an RPC-based distributed autograd framework that can be used for applications such as model parallel training. In short, applications may run RPC to execute code remotely in the forward pass, and autograd will automatically travel across RPC boundaries in the backward pass.


.. automodule:: torch.distributed.autograd
    :members:
