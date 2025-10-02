import torch


# Create test tensors for linalg operations
t2d = torch.randn(3, 3)  # 2D tensor for matrix operations
t_batch = torch.randn(2, 3, 3)  # Batched 3x3 matrices
t_vec = torch.randn(3)  # 1D vector
t_vectors = torch.randn(4, 3)  # Batch of vectors

# ========== Linear Algebra Functions ==========

# Cross product
cross_result = torch.linalg.cross(t_vectors, t_vectors)
reveal_type(cross_result)  # E: torch._tensor.Tensor

# Determinant
det_result = torch.linalg.det(t2d)
reveal_type(det_result)  # E: torch._tensor.Tensor

# Matrix inverse
inv_result = torch.linalg.inv(t2d)
reveal_type(inv_result)  # E: torch._tensor.Tensor

# Solve linear system
b = torch.randn(3, 2)
solve_result = torch.linalg.solve(t2d, b)
reveal_type(solve_result)  # E: torch._tensor.Tensor

# Cholesky decomposition
chol_result = torch.linalg.cholesky(
    t2d @ t2d.T + torch.eye(3)
)  # Make positive definite
reveal_type(chol_result)  # E: torch._tensor.Tensor

# Eigendecomposition - returns tuple
eig_result = torch.linalg.eig(t2d)
reveal_type(eig_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

# Extract eigenvalues and eigenvectors
eig_vals, eig_vecs = eig_result
reveal_type(eig_vals)  # E: torch._tensor.Tensor
reveal_type(eig_vecs)  # E: torch._tensor.Tensor

# Hermitian eigendecomposition - returns tuple
eigh_result = torch.linalg.eigh(t2d @ t2d.T)  # Make Hermitian
reveal_type(
    eigh_result
)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor, fallback=torch.return_types._linalg_eigh]

# SVD - returns tuple
svd_result = torch.linalg.svd(t2d)
reveal_type(
    svd_result
)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor, torch._tensor.Tensor, fallback=torch.return_types._linalg_svd]

# Extract SVD components
U, S, Vh = svd_result
reveal_type(U)  # E: torch._tensor.Tensor
reveal_type(S)  # E: torch._tensor.Tensor
reveal_type(Vh)  # E: torch._tensor.Tensor

# QR decomposition - returns tuple
qr_result = torch.linalg.qr(t2d)
reveal_type(
    qr_result
)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor, fallback=torch.return_types.qr]

# Matrix norms
matrix_norm_result = torch.linalg.matrix_norm(t2d)
reveal_type(matrix_norm_result)  # E: torch._tensor.Tensor

# Vector norms
vector_norm_result = torch.linalg.vector_norm(t_vec)
reveal_type(vector_norm_result)  # E: torch._tensor.Tensor

# General norm function
norm_result = torch.linalg.norm(t2d)
reveal_type(norm_result)  # E: torch._tensor.Tensor

# Condition number
cond_result = torch.linalg.cond(t2d)
reveal_type(cond_result)  # E: torch._tensor.Tensor

# Matrix rank
rank_result = torch.linalg.matrix_rank(t2d)
reveal_type(rank_result)  # E: torch._tensor.Tensor

# Pseudoinverse
pinv_result = torch.linalg.pinv(t2d)
reveal_type(pinv_result)  # E: torch._tensor.Tensor

# Matrix exponential
matrix_exp_result = torch.linalg.matrix_exp(t2d)
reveal_type(matrix_exp_result)  # E: torch._tensor.Tensor

# Matrix power
matrix_power_result = torch.linalg.matrix_power(t2d, 3)
reveal_type(matrix_power_result)  # E: torch._tensor.Tensor

# LU factorization - returns tuple
lu_result = torch.linalg.lu_factor(t2d)
reveal_type(lu_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

# LU solve using factorization
LU, pivots = lu_result
lu_solve_result = torch.linalg.lu_solve(LU, pivots, b)
reveal_type(lu_solve_result)  # E: torch._tensor.Tensor

# Sign and log determinant - returns tuple
slogdet_result = torch.linalg.slogdet(t2d)
reveal_type(
    slogdet_result
)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor, torch._tensor.Tensor, torch._tensor.Tensor, fallback=torch.return_types._linalg_slogdet] # noqa: B950

# Householder product
tau = torch.randn(3)
householder_result = torch.linalg.householder_product(t2d, tau)
reveal_type(householder_result)  # E: torch._tensor.Tensor

# Multi-dot product
matrices = (t2d, t2d, t2d)
multi_dot_result = torch.linalg.multi_dot(matrices)
reveal_type(multi_dot_result)  # E: torch._tensor.Tensor

# Tensor operations
t4d = torch.randn(2, 3, 4, 5)
tensorinv_result = torch.linalg.tensorinv(t4d, ind=2)
reveal_type(tensorinv_result)  # E: torch._tensor.Tensor

# Tensor solve
A = torch.randn(2, 3, 3, 4)
B = torch.randn(2, 3)
tensorsolve_result = torch.linalg.tensorsolve(A, B)
reveal_type(tensorsolve_result)  # E: torch._tensor.Tensor

# Vandermonde matrix
vander_result = torch.linalg.vander(t_vec)
reveal_type(vander_result)  # E: torch._tensor.Tensor

# Vector dot product
vecdot_result = torch.linalg.vecdot(t_vec, t_vec)
reveal_type(vecdot_result)  # E: torch._tensor.Tensor

# Least squares - returns tuple
lstsq_result = torch.linalg.lstsq(t2d, t_vec)
reveal_type(
    lstsq_result
)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor, torch._tensor.Tensor, torch._tensor.Tensor]

# Diagonal extraction
diagonal_result = torch.linalg.diagonal(t2d)
reveal_type(diagonal_result)  # E: torch._tensor.Tensor

# SVD values only
svdvals_result = torch.linalg.svdvals(t2d)
reveal_type(svdvals_result)  # E: torch._tensor.Tensor

# Eigenvalues only
eigvals_result = torch.linalg.eigvals(t2d)
reveal_type(eigvals_result)  # E: torch._tensor.Tensor

# Hermitian eigenvalues only
eigvalsh_result = torch.linalg.eigvalsh(t2d @ t2d.T)
reveal_type(eigvalsh_result)  # E: torch._tensor.Tensor

# Matrix multiplication
matmul_result = torch.linalg.matmul(t2d, t2d)
reveal_type(matmul_result)  # E: torch._tensor.Tensor

# ========== Extended/Exception Functions ==========

# Functions that return additional info (with _ex suffix)
inv_ex_result = torch.linalg.inv_ex(t2d)
reveal_type(inv_ex_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

chol_ex_result = torch.linalg.cholesky_ex(t2d @ t2d.T + torch.eye(3))
reveal_type(chol_ex_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

solve_ex_result = torch.linalg.solve_ex(t2d, b)
reveal_type(solve_ex_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

# LDL factorization
ldl_factor_result = torch.linalg.ldl_factor(t2d @ t2d.T)
reveal_type(ldl_factor_result)  # E: tuple[torch._tensor.Tensor, torch._tensor.Tensor]

# LDL solve
LD, pivots = ldl_factor_result
ldl_solve_result = torch.linalg.ldl_solve(LD, pivots, b)
reveal_type(ldl_solve_result)  # E: torch._tensor.Tensor

# Triangular solve
solve_triangular_result = torch.linalg.solve_triangular(t2d, b, upper=True)
reveal_type(solve_triangular_result)  # E: torch._tensor.Tensor

# Exception: LinAlgError should be available
try:
    torch.linalg.cholesky(torch.randn(3, 3))  # Might fail if not positive definite
except torch.linalg.LinAlgError:
    pass
reveal_type(
    torch.linalg.LinAlgError
)  # E: def (*args: builtins.object) -> torch._C._LinAlgError
