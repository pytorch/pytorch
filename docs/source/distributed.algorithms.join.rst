.. role:: hidden
    :class: hidden-section

Generic Join Context Manager
============================
The generic join context manager facilitates distributed training on uneven
inputs. This page outlines the API of the relevant classes: :class:`Join`,
:class:`Joinable`, and :class:`JoinHook`. For a tutorial, see
`Distributed Training with Uneven Inputs Using the Join Context Manager`_.

.. autoclass:: torch.distributed.algorithms.Join
    :members:

.. autoclass:: torch.distributed.algorithms.Joinable
    :members:

.. autoclass:: torch.distributed.algorithms.JoinHook
    :members:

.. _Distributed Training with Uneven Inputs Using the Join Context Manager: https://pytorch.org/tutorials/advanced/generic_join.html
