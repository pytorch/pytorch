import torch
torch.manual_seed(42)

def diff(a, b):
    return (a - b).abs().max()

for n in range(4001, 3999, -1):
    for nexp in range(20):
        x = torch.randn(n, n, dtype=torch.cfloat)
        q, _ = torch.linalg.qr(x)
        s = 2 * torch.rand(n, dtype=torch.float) + 1
        s[::2].mul_(-1)
        x = (q * s.unsqueeze(-2)) @ q.mH
        x = x + x.mH

        ld, piv, _ = torch.linalg.ldl_factor_ex(x, hermitian=True)
        l, p, _ = torch.linalg.ldl_factor_ex(x.cuda(), hermitian=True)

        if diff(piv.cuda(), p) != 0:
            raise ValueError(f"{n=}, {nexp=}, torch.{x}")
            breakpoint()

        if (err := diff(ld.cuda(), l.tril())) > 1e-4:
            raise ValueError(f"{err=}, {n=}, {nexp=}, torch.{x}")
            breakpoint()
