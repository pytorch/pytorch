import torch
torch.manual_seed(42)

def diff(a, b):
    return (a - b).abs().max()

for n in range(32, 1, -1):
    for nexp in range(500):
        x = torch.randn(n, n, dtype=torch.cdouble)
        q, _ = torch.linalg.qr(x)
        s = 2 * torch.rand(n, dtype=torch.double) + 1
        x = (q * s.unsqueeze(-2)) @ q
        x = x + x.mH

        ld, piv, _ = torch.linalg.ldl_factor_ex(x, hermitian=True)
        l, p, _ = torch.linalg.ldl_factor_ex(x.cuda(), hermitian=True)

        if (err := diff(ld.cuda(), l.tril())) > 1e-12:
            raise ValueError(f"{err=}, {n=}, {nexp=}, torch.{x}")
            breakpoint()
