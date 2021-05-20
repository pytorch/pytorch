.. _rendezvous-api:

Rendezvous
==========

.. automodule:: torch.distributed.elastic.rendezvous

Below is a state diagram describing how rendezvous works.

.. image:: etcd_rdzv_diagram.png

Registry
--------------------

.. autoclass:: RendezvousParameters

.. automodule:: torch.distributed.elastic.rendezvous.registry

Handler
--------------------

.. currentmodule:: torch.distributed.elastic.rendezvous

.. autoclass:: RendezvousHandler
   :members:

Exceptions
-------------
.. autoclass:: RendezvousError
.. autoclass:: RendezvousClosedError
.. autoclass:: RendezvousTimeoutError
.. autoclass:: RendezvousConnectionError
.. autoclass:: RendezvousStateError

Implmentations
----------------

Etcd Rendezvous
****************

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_rendezvous

.. autoclass:: EtcdRendezvousHandler

.. autoclass:: EtcdRendezvous
   :members:

.. autoclass:: EtcdStore
   :members:

Etcd Server
*************

The ``EtcdServer`` is a convenience class that makes it easy for you to
start and stop an etcd server on a subprocess. This is useful for testing
or single-node (multi-worker) deployments where manually setting up an
etcd server on the side is cumbersome.

.. warning:: For production and multi-node deployments please consider
             properly deploying a highly available etcd server as this is
             the single point of failure for your distributed jobs.

.. currentmodule:: torch.distributed.elastic.rendezvous.etcd_server

.. autoclass:: EtcdServer
