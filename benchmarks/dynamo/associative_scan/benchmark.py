import torch
from associative_scan_memeff import associative_scan as associative_scan_memeff
from associative_scan_matrix import associative_scan as associative_scan_matrix

from time import perf_counter

import numpy as np

import jax
import jax.numpy as jnp

###############################################################################
# Auxiliary function for timing
###############################################################################
def time_fn(fn, xs, warm_up=1):
    for _ in range(warm_up):
        result = fn(xs)

    t_start = perf_counter() 
    result = fn(xs)
    t_stop = perf_counter()
    t_ = t_stop - t_start
    return result, t_

###############################################################################
# Define the SSM kernels for PyTorch and for JAX
###############################################################################
# Define the SSM kernel module for PyTorch
class S5Kernel(torch.nn.Module):
    def __init__(self, op):
        super(S5Kernel, self).__init__()
        self.op = op
    
    def forward(self, xs: torch.Tensor):
        def s5_operator(x: torch.Tensor, y: torch.Tensor):
            A_i, Bu_i = x
            A_j, Bu_j = y
            return A_j * A_i, A_j * Bu_i + Bu_j

        result = self.op(s5_operator, xs, dim=0,)
        return result
    
# Define models
model_matrix = S5Kernel(associative_scan_matrix)
model_memeff = S5Kernel(associative_scan_memeff)

def binary_operator_efficient_jax(x, y):
    A_i, Bu_i = x
    A_j, Bu_j = y
    return A_j * A_i, A_j * Bu_i + Bu_j

def s5_operator_jax(xs):
    result_jax = jax.lax.associative_scan(binary_operator_efficient_jax, xs, axis=0)
    return result_jax

@jax.jit
def s5_operator_jax_jit(xs):
    result_jax = jax.lax.associative_scan(binary_operator_efficient_jax, xs, axis=0)
    return result_jax

###############################################################################
# Define the inputs
###############################################################################
warm_up_cycles = 3
timesteps = 200
state_dim = 40
A = torch.randn(state_dim, device='cuda')
B = torch.randn(timesteps, state_dim, device='cuda', requires_grad=True)
xs = (torch.tensor(A.repeat((timesteps, 1)).cpu().numpy(), device='cuda', requires_grad=True), B)

# Prepare JAX inputs
jax_device = jax.devices('cuda')[0]
xs_jax = tuple([jnp.array(el.detach().cpu(), device=jax_device) for el in xs])

###############################################################################
# Run models forward (inference-only)
###############################################################################

# Matrix implementation for associative_scan
result_matrix, t_matrix = time_fn(model_matrix, xs, warm_up=warm_up_cycles)

# Memory-efficient implementation for associative_scan
result_memeff, t_memeff = time_fn(model_memeff, xs, warm_up=warm_up_cycles)
torch.testing.assert_close(result_matrix, result_memeff)

# Non-jit compiled JAX model
result_jax, t_jax = time_fn(s5_operator_jax, xs_jax, warm_up=warm_up_cycles)
for jax_array, torch_array in zip([np.asarray(el) for el in result_jax], [el.detach().cpu().numpy() for el in result_memeff]):
    if not np.array_equal(jax_array, torch_array):
        raise Exception('results not equal')

# jit compiled JAX model
result_jax_jit, t_jax_jit = time_fn(s5_operator_jax_jit, xs_jax, warm_up=warm_up_cycles)
for jax_array, torch_array in zip([np.asarray(el) for el in result_jax], [el.detach().cpu().numpy() for el in result_memeff]):
    if not np.array_equal(jax_array, torch_array):
        raise Exception('results not equal')

print(f"Matrix {t_matrix} vs. Mem eff. {t_memeff} vs. JAX {t_jax} vs. JAX (JIT) {t_jax_jit}")

###############################################################################
# Run models with backward
###############################################################################
def wrapper_torch_bwd(fn, xs):
    result = fn(xs)
    result = result[0][:10, :30].sum()
    # grad_init = [torch.ones_like(el) for el in result]
    # grad = torch.autograd.grad(result, xs, grad_init)
    grad = torch.autograd.grad(result, xs)
    return grad

def wrapper_jax_bwd(fn, xs):
    def inner_fn(xs):
        result = fn(xs)
        result = result[0][:10, :30].sum()
        return result
    result = inner_fn(xs)
    grad = jax.grad(inner_fn)(xs)
    return grad

def wrapper_jax_jit_bwd(fn, xs):
    @jax.jit
    def inner_fn(xs):
        result = fn(xs)
        result = result[0][:10, :30].sum()
        return result
    result = inner_fn(xs)
    grad = jax.grad(inner_fn)(xs)
    return grad
    
# Matrix implementation for associative_scan
result_grad_matrix, t_grad_matrix = time_fn(lambda xs: wrapper_torch_bwd(model_matrix, xs), xs, warm_up=warm_up_cycles)

# Memory-efficient implementation for associative_scan
result_grad_memeff, t_grad_memeff = time_fn(lambda xs: wrapper_torch_bwd(model_memeff, xs), xs, warm_up=warm_up_cycles)

# Non-jit compiled JAX model
result_grad_jax, t_grad_jax = time_fn(lambda xs: wrapper_jax_bwd(s5_operator_jax, xs), xs_jax, warm_up=warm_up_cycles)

# jit compiled JAX model
result_grad_jax_jit, t_grad_jax_jit = time_fn(lambda xs: wrapper_jax_jit_bwd(s5_operator_jax, xs), xs_jax, warm_up=warm_up_cycles)

print(f"Matrix {t_grad_matrix} vs. Mem eff. {t_grad_memeff} vs. JAX {t_grad_jax} vs. JAX (JIT) {t_grad_jax_jit}")