.. role:: hidden
    :class: hidden-section

torch.linalg
============

Common linear algebra operations.

See :ref:`Linear Algebra Stability` for some common numerical edge-cases.

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
    lu
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
    lu_solve
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
    vecdot
    multi_dot
    householder_product

Tensor Operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorinv
    tensorsolve

Misc
----

.. autosummary::
    :toctree: generated
    :nosignatures:

    vander

Experimental Functions
----------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    cholesky_ex
    inv_ex
    solve_ex
    lu_factor_ex
    ldl_factor
    ldl_factor_ex
    ldl_solve
