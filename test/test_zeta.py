# import scipy.special
# import torch

# torch.set_printoptions(precision=10)

# x = torch.tensor(2, dtype=torch.float)
# q = torch.tensor(0.75, dtype=torch.float)

# print(scipy.special.zeta(x.numpy(), q.numpy()))
# print(torch.special.zeta(x, q))


# import jax

# x_np = x.numpy()

# def _zeta(x_np):
#     return jax.scipy.special.zeta(2, x_np)

# print(jax.grad(jax.grad(_zeta))(1.2))
# # print(-2 * torch.special.zeta(torch.tensor(3., dtype=torch.float), torch.tensor(1.2, dtype=torch.float)))

import torch

x = torch.ones(2, 2)
a = x
b = torch.ones(5)

print("A SHAPE:", a.shape)
print("X SHAPE:", x.shape)


a.resize_as_(b)
a = b

print("A SHAPE:", a.shape)
print("X SHAPE:", x.shape)