import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        return

    torch.manual_seed(0)

    # Build a small SPD matrix so the system is solvable.
    spd = torch.rand(8, 5, device="cuda", dtype=torch.float32)
    A = (spd.T @ spd).to_sparse_csr()

    # 2D RHS (n, k) should be supported by cuDSS-backed sparse CSR solve.
    B = torch.rand(A.size(0), 3, device="cuda", dtype=torch.float32)

    X_sparse = torch.linalg.solve(A, B)
    X_dense = torch.linalg.solve(A.to_dense(), B)

    max_err = (X_sparse - X_dense).abs().max().item()
    print("max_abs_err:", max_err)


if __name__ == "__main__":
    main()

