.. role:: hidden
    :class: hidden-section

torch.linalg
============

Common linear algebra operations.

This module is in BETA. New functions are still being added, and some
functions may change in future PyTorch releases. See the documentation of each
function for details.

.. automodule:: torch.linalg
.. currentmodule:: torch.linalg

Matrix Properties
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    norm
    vector_norm
    det
    slogdet
    cond
    matrix_rank

Decompositions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky
    qr
    eig
    eigvals
    eigh
    eigvalsh
    svd
    svdvals

Solvers
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    solve
    lstsq

Inverses
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inv
    pinv

Matrix Products
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    matrix_power
    multi_dot
    householder_product

Tensor Operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorinv
    tensorsolve
