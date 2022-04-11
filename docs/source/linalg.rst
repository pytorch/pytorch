.. role:: hidden
    :class: hidden-section

torch.linalg
============

Common linear algebra operations.

.. automodule:: torch.linalg
.. currentmodule:: torch.linalg

Matrix Properties
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    norm
    vector_norm
    matrix_norm
    diagonal
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
    lu_factor
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
    solve_triangular
    lstsq

Inverses
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inv
    pinv

Matrix Functions
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    matrix_exp
    matrix_power

Matrix Products
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cross
    matmul
    multi_dot
    householder_product

Tensor Operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorinv
    tensorsolve

Experimental Functions
----------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky_ex
    inv_ex
    lu_factor_ex
