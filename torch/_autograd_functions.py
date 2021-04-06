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
        """
        Here we derive the gradients for the LU decomposition.
        LIMITATIONS: square inputs of full rank.
        If not stated otherwise, for tensors A and B,
        `A B` means the matrix product of A and B.

        Forward AD:
        Note that PyTorch returns packed LU, it is a mapping
        A -> (B:= L + U - I, P), such that A = P L U, and
        P is a permutation matrix, and is non-differentiable.

        Using B = L + U - I, A = P L U, we get

        dB = dL + dU and     (*)
        P^T dA = dL U + L dU (**)

        By left/right multiplication of (**) with L^{-1}/U^{-1} we get:
        L^{-1} P^T dA U^{-1} = L^{-1} dL + dU U^{-1}.

        Note that L^{-1} dL is lower-triangular with zero diagonal,
        and dU U^{-1} is upper-triangular.
        Define 1_U := triu(ones(n, n)), and 1_L := ones(n, n) - 1_U, so

        L^{-1} dL = 1_L * (L^{-1} P^T dA U^{-1}),
        dU U^{-1} = 1_U * (L^{-1} P^T dA U^{-1}), where * denotes the Hadamard product.

        Hence we finally get:
        dL = L 1_L * (L^{-1} P^T dA U^{-1}),
        dU = 1_U * (L^{-1} P^T dA U^{-1}) U

        Backward AD:
        The backward sensitivity is then:
        Tr(B_grad^T dB) = Tr(B_grad^T dL) + Tr(B_grad^T dU) = [1] + [2].

        [1] = Tr(B_grad^T dL) = Tr(B_grad^T L 1_L * (L^{-1} P^T dA U^{-1}))
            = [using Tr(A (B * C)) = Tr((A * B^T) C)]
            = Tr((B_grad^T L * 1_L^T) L^{-1} P^T dA U^{-1})
            = [cyclic property of trace]
            = Tr(U^{-1} (B_grad^T L * 1_L^T) L^{-1} P^T dA)
            = Tr((P L^{-T} (L^T B_grad * 1_L) U^{-T})^T dA).
        Similar, [2] can be rewritten as:
        [2] = Tr(P L^{-T} (B_grad U^T * 1_U) U^{-T})^T dA, hence
        Tr(A_grad^T dA) = [1] + [2]
                        = Tr((P L^{-T} (L^T B_grad * 1_L + B_grad U^T * 1_U) U^{-T})^T dA), so
        A_grad = P L^{-T} (L^T B_grad * 1_L + B_grad U^T * 1_U) U^{-T}.

        In the code below we use the name `LU` instead of `B`, so that there is no confusion
        in the derivation above between the matrix product and a two-letter variable name.
        """
        LU, pivots = ctx.saved_tensors
        P, L, U = torch.lu_unpack(LU, pivots)

        # To make sure MyPy infers types right
        assert (L is not None) and (U is not None)

        I = LU_grad.new_zeros(LU_grad.shape)
        I.diagonal(dim1=-2, dim2=-1).fill_(1)

        Lt_inv = torch.triangular_solve(I, L, upper=False).solution.transpose(-1, -2)
        Ut_inv = torch.triangular_solve(I, U, upper=True).solution.transpose(-1, -2)

        phi_L = (L.transpose(-1, -2) @ LU_grad).tril_()
        phi_L.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        phi_U = (LU_grad @ U.transpose(-1, -2)).triu_()

        self_grad_perturbed = Lt_inv @ (phi_L + phi_U) @ Ut_inv
        return P @ self_grad_perturbed, None, None
