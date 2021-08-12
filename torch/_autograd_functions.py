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

        Let A^H = (A^T).conj()

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
        Tr(B_grad^H dB) = Tr(B_grad^H dL) + Tr(B_grad^H dU) = [1] + [2].

        [1] = Tr(B_grad^H dL) = Tr(B_grad^H L 1_L * (L^{-1} P^T dA U^{-1}))
            = [using Tr(A (B * C)) = Tr((A * B^T) C)]
            = Tr((B_grad^H L * 1_L^T) L^{-1} P^T dA U^{-1})
            = [cyclic property of trace]
            = Tr(U^{-1} (B_grad^H L * 1_L^T) L^{-1} P^T dA)
            = Tr((P L^{-H} (L^H B_grad * 1_L) U^{-H})^H dA).
        Similar, [2] can be rewritten as:
        [2] = Tr(P L^{-H} (B_grad U^H * 1_U) U^{-H})^H dA, hence
        Tr(A_grad^H dA) = [1] + [2]
                        = Tr((P L^{-H} (L^H B_grad * 1_L + B_grad U^H * 1_U) U^{-H})^H dA), so
        A_grad = P L^{-H} (L^H B_grad * 1_L + B_grad U^H * 1_U) U^{-H}.

        In the code below we use the name `LU` instead of `B`, so that there is no confusion
        in the derivation above between the matrix product and a two-letter variable name.
        """
        LU, pivots = ctx.saved_tensors
        P, L, U = torch.lu_unpack(LU, pivots)

        # To make sure MyPy infers types right
        assert (L is not None) and (U is not None) and (P is not None)

        # phi_L = L^H B_grad * 1_L
        phi_L = (L.transpose(-1, -2).conj() @ LU_grad).tril_()
        phi_L.diagonal(dim1=-2, dim2=-1).fill_(0.0)
        # phi_U = B_grad U^H * 1_U
        phi_U = (LU_grad @ U.transpose(-1, -2).conj()).triu_()
        phi = phi_L + phi_U

        # using the notation from above plus the variable names, note
        # A_grad = P L^{-H} phi U^{-H}.
        # Instead of inverting L and U, we solve two systems of equations, i.e.,
        # the above expression could be rewritten as
        # L^H P^T A_grad U^H = phi.
        # Let X = P^T A_grad U_H, then
        # X = L^{-H} phi, where L^{-H} is upper triangular, or
        # X = torch.triangular_solve(phi, L^H)
        # using the definition of X we see:
        # X = P^T A_grad U_H => P X = A_grad U_H => U A_grad^H = X^H P^T, so
        # A_grad = (U^{-1} X^H P^T)^H, or
        # A_grad = torch.triangular_solve(X^H P^T, U)^H
        X = torch.triangular_solve(phi, L.transpose(-1, -2).conj(), upper=True).solution
        A_grad = torch.triangular_solve(X.transpose(-1, -2).conj() @ P.transpose(-1, -2), U, upper=True) \
            .solution.transpose(-1, -2).conj()

        return A_grad, None, None
