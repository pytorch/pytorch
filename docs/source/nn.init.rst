.. role:: hidden
    :class: hidden-section

.. _nn-init-doc:

torch.nn.init
=============

.. warning::
    All the functions in this module are intended to be used to initialize neural network
    parameters, so they all run in :func:`torch.no_grad` mode and will not be taken into
    account by autograd.

.. currentmodule:: torch.nn.init
.. autofunction:: calculate_gain
.. autofunction:: uniform_
.. autofunction:: normal_
.. autofunction:: constant_
.. autofunction:: ones_
.. autofunction:: zeros_
.. autofunction:: eye_
.. autofunction:: dirac_
.. autofunction:: xavier_uniform_
.. autofunction:: xavier_normal_
.. autofunction:: kaiming_uniform_
.. autofunction:: kaiming_normal_
.. autofunction:: trunc_normal_
.. autofunction:: orthogonal_
.. autofunction:: sparse_
