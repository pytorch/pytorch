# flake8: noqa
import torch


# Test tensors
a = torch.randn(3, 3)
vec = torch.randn(3)

# ========== Type errors that should fail ==========

# Cross product requires same shape
torch.linalg.cross(torch.randn(3), torch.randn(4))  # E:

# Det requires square matrices
torch.linalg.det(torch.randn(3, 4))  # E:

# Invalid ord parameter for vector_norm
torch.linalg.vector_norm(vec, ord="invalid")  # E:

# Invalid UPLO parameter
torch.linalg.eigh(a, UPLO="X")  # E:

# Invalid mode for QR
torch.linalg.qr(a, mode="invalid")  # E:

# Matrix power with non-integer
torch.linalg.matrix_power(a, 2.5)  # E:

# Solve with incompatible shapes
torch.linalg.solve(torch.randn(3, 3), torch.randn(4))  # E:

# LU solve with wrong pivot tensor
LU = torch.randn(3, 3)
wrong_pivots = torch.randn(4)  # Should be integer type and right size
torch.linalg.lu_solve(LU, wrong_pivots, torch.randn(3))  # E:

# Invalid driver for lstsq
torch.linalg.lstsq(a, vec, driver="invalid_driver")  # E:

# Tensorinv with invalid ind
torch.linalg.tensorinv(torch.randn(2, 3, 4, 5), ind=0)  # E:
