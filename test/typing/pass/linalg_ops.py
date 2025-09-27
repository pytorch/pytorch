# flake8: noqa
import torch


# Test tensors for linalg operations
a = torch.randn(3, 3)
b = torch.randn(3, 3)
vec = torch.randn(3)
vectors = torch.randn(4, 3)
batch_matrices = torch.randn(2, 3, 3)
rhs = torch.randn(3, 2)

# ========== Basic Linear Algebra Operations ==========

# Cross product
torch.linalg.cross(vectors, vectors)
torch.linalg.cross(vectors, vectors, dim=1)
torch.linalg.cross(vec, vec, dim=-1)

# Determinant
torch.linalg.det(a)
torch.linalg.det(batch_matrices)

# Matrix inverse
torch.linalg.inv(a)
pos_def = a @ a.T + torch.eye(3)
torch.linalg.inv(pos_def)

# Solve linear systems
torch.linalg.solve(a, rhs)
torch.linalg.solve(a, vec)
torch.linalg.solve(batch_matrices, torch.randn(2, 3, 2))

# Cholesky decomposition
torch.linalg.cholesky(pos_def)
torch.linalg.cholesky(pos_def, upper=False)
torch.linalg.cholesky(pos_def, upper=True)

# ========== Eigendecomposition ==========

# General eigendecomposition
torch.linalg.eig(a)
torch.linalg.eigvals(a)

# Hermitian eigendecomposition
hermitian = a @ a.T
torch.linalg.eigh(hermitian)
torch.linalg.eigh(hermitian, UPLO="L")
torch.linalg.eigh(hermitian, UPLO="U")
torch.linalg.eigvalsh(hermitian)
torch.linalg.eigvalsh(hermitian, UPLO="L")

# ========== SVD ==========

# Singular Value Decomposition
torch.linalg.svd(a)
torch.linalg.svd(a, full_matrices=True)
torch.linalg.svd(a, full_matrices=False)
torch.linalg.svdvals(a)

# ========== QR Decomposition ==========

torch.linalg.qr(a)
torch.linalg.qr(a, mode="reduced")
torch.linalg.qr(a, mode="complete")

# ========== Norms ==========

# Vector norms
torch.linalg.vector_norm(vec)
torch.linalg.vector_norm(vec, ord=1)
torch.linalg.vector_norm(vec, ord=2)
torch.linalg.vector_norm(vec, ord=float("inf"))
torch.linalg.vector_norm(vectors, dim=1)

# Matrix norms
torch.linalg.matrix_norm(a)
torch.linalg.matrix_norm(a, ord="fro")
torch.linalg.matrix_norm(a, ord="nuc")
torch.linalg.matrix_norm(a, ord=1)
torch.linalg.matrix_norm(a, ord=-1)
torch.linalg.matrix_norm(a, ord=2)
torch.linalg.matrix_norm(a, ord=-2)

# General norm
torch.linalg.norm(a)
torch.linalg.norm(vec)
torch.linalg.norm(a, ord="fro")
torch.linalg.norm(vec, ord=2)

# ========== Matrix Functions ==========

# Condition number
torch.linalg.cond(a)
torch.linalg.cond(a, p=1)
torch.linalg.cond(a, p=2)
torch.linalg.cond(a, p="fro")

# Matrix rank
torch.linalg.matrix_rank(a)
torch.linalg.matrix_rank(a, hermitian=False)
torch.linalg.matrix_rank(a, tol=1e-6)

# Pseudoinverse
torch.linalg.pinv(a)
torch.linalg.pinv(a, rcond=1e-6)
torch.linalg.pinv(a, hermitian=False)

# Matrix exponential
torch.linalg.matrix_exp(a)

# Matrix power
torch.linalg.matrix_power(a, 2)
torch.linalg.matrix_power(a, -1)
torch.linalg.matrix_power(a, 0)

# ========== LU Decomposition ==========

# LU factorization
torch.linalg.lu_factor(a)
torch.linalg.lu_factor(a, pivot=True)
torch.linalg.lu_factor(a, pivot=False)

# Full LU decomposition
torch.linalg.lu(a)
torch.linalg.lu(a, pivot=True)

# LU solve
LU, pivots = torch.linalg.lu_factor(a)
torch.linalg.lu_solve(LU, pivots, rhs)
torch.linalg.lu_solve(LU, pivots, rhs, left=True)
torch.linalg.lu_solve(LU, pivots, rhs, left=False)

# ========== Other Decompositions ==========

# Sign and log determinant
torch.linalg.slogdet(a)

# LDL factorization
torch.linalg.ldl_factor(hermitian)
torch.linalg.ldl_factor(hermitian, hermitian=True)

# LDL solve
LD, pivots = torch.linalg.ldl_factor(hermitian)
torch.linalg.ldl_solve(LD, pivots, rhs)
torch.linalg.ldl_solve(LD, pivots, rhs, hermitian=True)

# Householder product
reflectors = torch.randn(3, 3)
tau = torch.randn(3)
torch.linalg.householder_product(reflectors, tau)

# ========== Triangular Operations ==========

# Triangular solve
upper_tri = torch.triu(a)
torch.linalg.solve_triangular(upper_tri, rhs, upper=True)
torch.linalg.solve_triangular(upper_tri, rhs, upper=True, left=True)
torch.linalg.solve_triangular(upper_tri, rhs, upper=True, unitriangular=False)

# ========== Least Squares ==========

torch.linalg.lstsq(a, vec)
torch.linalg.lstsq(a, rhs)
torch.linalg.lstsq(a, vec, rcond=None)
torch.linalg.lstsq(a, vec, driver="gelsy")

# ========== Tensor Operations ==========

# Multi-dot product
matrices = (a, b, a)
torch.linalg.multi_dot(matrices)

# Tensor inverse and solve
t4d = torch.randn(6, 8, 3, 4)
torch.linalg.tensorinv(t4d, ind=2)

A = torch.randn(2, 3, 4, 5)
B = torch.randn(2, 3)
torch.linalg.tensorsolve(A, B)
torch.linalg.tensorsolve(A, B, dims=(2, 3))

# ========== Utility Functions ==========

# Vandermonde matrix
torch.linalg.vander(vec)
torch.linalg.vander(vec, N=5)

# Vector dot product
torch.linalg.vecdot(vec, vec)
torch.linalg.vecdot(vectors, vectors, dim=1)

# Matrix multiplication
torch.linalg.matmul(a, b)
torch.linalg.matmul(vectors, a.T)

# Diagonal extraction
torch.linalg.diagonal(a)
torch.linalg.diagonal(a, offset=0)
torch.linalg.diagonal(a, offset=1)
torch.linalg.diagonal(batch_matrices, dim1=-2, dim2=-1)

# ========== Extended Functions (with error checking) ==========

# These functions return additional info about computation status
torch.linalg.inv_ex(a)
torch.linalg.inv_ex(a, check_errors=False)

torch.linalg.cholesky_ex(pos_def)
torch.linalg.cholesky_ex(pos_def, upper=False, check_errors=True)

torch.linalg.solve_ex(a, rhs)
torch.linalg.solve_ex(a, rhs, left=True, check_errors=False)

torch.linalg.lu_factor_ex(a)
torch.linalg.lu_factor_ex(a, pivot=True, check_errors=True)

torch.linalg.ldl_factor_ex(hermitian)
torch.linalg.ldl_factor_ex(hermitian, hermitian=True, check_errors=False)

# ========== With Output Parameters ==========

# Many functions support output parameters
out_tensor = torch.empty_like(a)
torch.linalg.inv(a, out=out_tensor)

det_out = torch.empty(a.shape[:-2])
torch.linalg.det(a, out=det_out)

norm_out = torch.empty(())
torch.linalg.norm(vec, out=norm_out)
