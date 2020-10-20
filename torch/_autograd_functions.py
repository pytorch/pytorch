import torch

class _LU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, pivot=True, get_infos=False):
        LU, pivots, infos = torch._lu_with_info(self, pivot=pivot, check_errors=(not get_infos))
        ctx.save_for_backward(LU, pivots)
        ctx.mark_non_differentiable(pivots, infos)
        return LU, pivots, infos

    @staticmethod
    def backward(ctx, LU_grad, pivots_grad, infors_grad):
        LU, pivots = ctx.saved_tensors
        P, L, U = torch.lu_unpack(LU, pivots)

        Lt_inv = L.inverse().transpose(-1, -2)
        Ut_inv = U.inverse().transpose(-1, -2)

        phi_L = (L.transpose(-1, -2) @ LU_grad).tril_()
        phi_L.diagonal(dim1=-2, dim2=-1).mul_(0.0)
        phi_U = (LU_grad @ U.transpose(-1, -2)).triu_()

        self_grad_perturbed = Lt_inv @ (phi_L + phi_U) @ Ut_inv
        return P @ self_grad_perturbed, None, None

