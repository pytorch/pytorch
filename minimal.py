import torch

if __name__ == '__main__':
    spd = torch.rand(4, 3)
    A = spd.T @ spd
    b = torch.rand(3).to(torch.float64).cuda()
    A = A.to_sparse_csr().to(torch.float64).cuda()

    x = torch.linalg.solve(A, b)
    print((A @ x - b).norm())
