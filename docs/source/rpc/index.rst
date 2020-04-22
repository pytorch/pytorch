.. _rpc-index:

Distributed RPC Framework
==============================

The distributed RPC framework provides mechanisms for multi-machine model training through a set of primitives to allow for remote communication, and a higher-level API to automatically differentiate models split across several machines.

-  :ref:`distributed-rpc-framework`

Design Notes
-----------
The distributed autograd design note covers the design of the RPC-based distributed autograd framework that is useful for applications such as model parallel training.

-  :ref:`distributed-autograd-design`

The RRef design note covers the design of the :ref:`rref` (Remote REFerence) protocol used to refer to values on remote workers by the framework.

-  :ref:`remote-reference-protocol`

Tutorials
---------
The RPC tutorial introduces users to the RPC framework and provides two example applications using :ref:`torch.distributed.rpc<distributed-rpc-framework>` APIs.

-  `Getting started with Distributed RPC Framework <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__
