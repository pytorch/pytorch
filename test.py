import torch
torch.manual_seed(42)

def diff(a, b):
    return (a - b).abs().max()

for n in [37, 111, 247, 1023]:
    for nexp in range(20):
        x = torch.randn(n, n, dtype=torch.cdouble)
        q, _ = torch.linalg.qr(x)
        s = 2 * torch.rand(n, dtype=torch.double) + 1
        s[::2].mul_(-1)
        x = (q * s.unsqueeze(-2)) @ q.mH
        x = x + x.mH
        x_cuda = x.cuda()

        l, p, _ = torch.linalg.ldl_factor_ex(x.cuda(), hermitian=True)
        sol = torch.linalg.ldl_solve(l.cpu(), p.cpu(), (x_cuda @ x_cuda).cpu(), hermitian=True)
        if (err := diff(sol.cuda(), x_cuda)) > 1e-11:
            raise ValueError(f"{err=}, {n=}, {nexp=}, torch.{x}")
            breakpoint()

        #if diff(piv.cuda(), p) != 0:
        #    raise ValueError(f"{n=}, {nexp=}, torch.{x}")
        #    breakpoint()

        #if (err := diff(ld.cuda(), l.tril())) > 1e-4:
        #    raise ValueError(f"{err=}, {n=}, {nexp=}, torch.{x}")
        #    breakpoint()
