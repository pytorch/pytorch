import torch
import scipy.special
import numpy as np
torch.set_printoptions(precision=23)

print(*torch.__config__.show().split("\n"), sep="\n")

# V = 7.9765129089355468750
# V = 15
V = 7.97082901000976562500000

t = torch.tensor(V, dtype=torch.float32)
t_vec = torch.tensor([V] * 20, dtype=torch.float32)

print("SCALAR T:", torch.special.i1(t))
# print("VEC T:", torch.special.i1(t_vec))
print("SCIPY SCALAR:", scipy.special.i1(t.numpy()))
print("SCIPY SCALAR DOUBLE:", scipy.special.i1(t.numpy().astype(np.double)))
print("DIFF SCALAR:", scipy.special.i1(t.numpy()) - torch.special.i1(t).numpy())
# print("DIFF VEC:", max(scipy.special.i1(t_vec.numpy()) - torch.special.i1(t_vec).numpy()))
