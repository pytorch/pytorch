```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.linalg

Common linear algebra operations.

See {ref}`Linear Algebra Stability` for some common numerical edge-cases.

```{eval-rst}
.. automodule:: torch.linalg
.. currentmodule:: torch.linalg
```

## Matrix Properties

```{eval-rst}
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
```

## Decompositions

```{eval-rst}
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
```

(linalg solvers)=

## Solvers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    solve
    solve_triangular
    lu_solve
    lstsq
```

(linalg inverses)=

## Inverses

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    inv
    pinv
```

## Matrix Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    matrix_exp
    matrix_power
```

## Matrix Products

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    cross
    matmul
    vecdot
    multi_dot
    householder_product
```

## Tensor Operations

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorinv
    tensorsolve
```

## Misc

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    vander
```

## Experimental Functions

```{eval-rst}
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
```
