import torch
from tqdm import tqdm

torch.manual_seed(0)
device = 'mps'

size = 700

a = torch.rand((size, size)).to(device)
a = a @ a.t().conj()

b = torch.linalg.cholesky_ex(a, check_errors=True)[1]

for i in tqdm(range(5)):
    b += torch.linalg.cholesky_ex(a)[1]

A = torch.randn(size, size).triu_().to(device)
B = torch.randn(size, size).to(device)
X = torch.linalg.solve_triangular(A, B, upper=True)
torch.mps.profiler.start(wait_until_completed=True)
for i in tqdm(range(100000)):
    X += torch.linalg.solve_triangular(A, B, upper=True)
torch.mps.profiler.stop()
