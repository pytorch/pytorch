# import torch
# import time
# from tqdm import tqdm

# cpu_tensor = torch.rand((12, 60, 640, 64), dtype=torch.float)
# mps_tensor = cpu_tensor.to('mps')

# num = 20

# torch.mvlgamma(cpu_tensor, 1)

# t1 = time.time()
# for i in tqdm(range(num)):
#     cpu_result = torch.mvlgamma(cpu_tensor, 1)
#     cpu_result + 1
# print(cpu_result[0, 0, 0, 0])
# t2 = time.time()

# for i in range(5):
#     mps_result = torch.mvlgamma(mps_tensor, 1)
# t3 = time.time()
# results = []
# for i in tqdm(range(num)):
#     mps_result = torch.mvlgamma(mps_tensor, 1)
#     mps_result.to('cpu') + 1
# t4 = time.time()

# print(f'cpu: {t2 - t1}')
# print(f'mps: {t4- t3}')

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, Beta
from tqdm import tqdm


# def log_mixture_nb(x, mu_1, mu_2, theta_1, theta_2, pi, eps=1e-8):
#     """Note: All inputs should be torch Tensors
#     log likelihood (scalar) of a minibatch according to a mixture nb model.
#     pi is the probability to be in the first component.

#     For totalVI, the first component should be background.

#     Parameters
#     ----------
#     mu1: mean of the first negative binomial component (has to be positive support) (shape: minibatch x genes)
#     theta1: first inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
#     mu2: mean of the second negative binomial (has to be positive support) (shape: minibatch x genes)
#     theta2: second inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
#         If None, assume one shared inverse dispersion parameter.
#     eps: numerical stability constant


#     Returns
#     -------

#     """
#     theta = theta_1
#     if theta.ndimension() == 1:
#         theta = theta.view(
#             1, theta.size(0)
#         )  # In this case, we reshape theta for broadcasting

#     log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
#     log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
#     lgamma_x_theta = torch.mvlgamma(x + theta, 5)
#     lgamma_theta = torch.mvlgamma(theta, 50)
#     lgamma_x_plus_1 = torch.mvlgamma(x + 1, 2)
#     # lgamma_x_theta = x + theta
#     # lgamma_theta = theta
#     # lgamma_x_plus_1 = x + 1

#     log_nb_1 = (
#         theta * (torch.log(theta + eps) - log_theta_mu_1_eps)
#         + x * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
#         + lgamma_x_theta
#         - lgamma_theta
#         - lgamma_x_plus_1
#     )
#     log_nb_2 = (
#         theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
#         + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
#         + lgamma_x_theta
#         - lgamma_theta
#         - lgamma_x_plus_1
#     )

#     logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi)), dim=0)
#     softplus_pi = F.softplus(-pi)

#     log_mixture_nb = logsumexp - softplus_pi

#     return log_mixture_nb

# dim1 = 6400
# dim2 = 2000
# device = 'mps'
# x = torch.rand((dim1, dim2)).to(device)
# mu1 = torch.rand((dim1, dim2)).to(device)
# mu2 = torch.rand((dim1, dim2)).to(device)
# theta1 = torch.rand((dim1, dim2)).to(device)

# for i in tqdm(range(200)):
#     log_mixture_nb(x, mu1, mu2, theta1, None, torch.Tensor([3.13]).to(device))

order = 1

a = torch.arange(-10, 1, .1, dtype=torch.float).to('mps')
print(a.dtype)
b = a.to('cpu')
mps = torch.polygamma(order, a)
mps = [x.item() for x in mps]
cpu = torch.polygamma(order, b)
cpu = [round(x.item(), 4) for x in cpu]
diff = torch.polygamma(order, a).to('cpu') - torch.polygamma(order, b)
nums = [round(x.item(), 2) for x in b]
diff_list = [round(x.item(), 5) for x in diff]

for x, y, z, l in zip(nums, diff_list, mps, cpu):
    print(x, ':', y, ';', z, ' ', l)

from scipy import special
x = [-2.1, -0.9]
print(special.polygamma(4, x))

t = torch.tensor([float('inf')])
print(torch.polygamma(2, t))
print(special.polygamma(2, t.numpy()))
print(torch.polygamma(3, t))
print(special.polygamma(3, t.numpy()))
print(torch.polygamma(4, t))
print(special.polygamma(4, t.numpy()))
