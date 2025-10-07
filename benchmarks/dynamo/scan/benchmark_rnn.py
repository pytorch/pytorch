import torch
from torch._higher_order_ops.scan import scan

from time import perf_counter

###############################################################################
# Auxiliary function for timing
###############################################################################
def time_fn(fn, xs, warm_up=1):
    for _ in range(warm_up):
        result = fn(*xs)

    t_start = perf_counter() 
    result = fn(*xs)
    t_stop = perf_counter()
    t_ = t_stop - t_start
    return result, t_

###############################################################################
# Define the RNN for PyTorch
###############################################################################
class RNN(torch.nn.Module):
    def __init__(self, Wih, bih, Whh, bhh):
        super(RNN, self).__init__()
        self.Wih = Wih
        self.bih = bih
        self.Whh = Whh
        self.bhh = bhh
        
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def rnn_combine(carry, x):
            h = torch.tanh(x @ self.Wih + self.bih + carry @ self.Whh + self.bhh)
            # Needed to not trigger aliasing errors in inductor
            return h + 0., h.clone()
        
        carry, outs = scan(rnn_combine, init, xs, dim=0)
        
        return carry, outs

# Define the RNN with pure PyTorch
rnn_pytorch = torch.nn.RNN(3, 5)

rnn_scan = RNN(rnn_pytorch.weight_ih_l0.T, rnn_pytorch.bias_ih_l0, 
               rnn_pytorch.weight_hh_l0.T, rnn_pytorch.bias_hh_l0)

###############################################################################
# Define the inputs
###############################################################################
warm_up_cycles = 3
xs = torch.randn(4, 3, requires_grad=True)
init = torch.zeros(5, requires_grad=True)

###############################################################################
# Run models forward (inference-only)
###############################################################################
# torch.testing.assert_close(result_pytorch, result)

# grad_init = torch.ones_like(result_pytorch)
# grad_pytorch = torch.autograd.grad(result_pytorch, xs, grad_init)[0]
# grad = torch.autograd.grad(result, xs, grad_init)[0]

# # The map function computes x ** y for each element, where y = 2
# # Therefore, we expect the correct gradients to be x * 2
# print("Gradient of PyTorch:\n", grad_pytorch)
# print("Gradient of cond:\n", grad)
# torch.testing.assert_close(grad_pytorch, grad)


# Native PyTorch implementation
result_pytorch, t_pytorch = time_fn(rnn_pytorch, (xs,), warm_up=warm_up_cycles)

# Implementation with scan
result_scan, t_scan = time_fn(rnn_scan, (init, xs), warm_up=warm_up_cycles)
torch.testing.assert_close(result_pytorch[0], result_scan[1])

# Implementation with scan - compiled
result_scan, t_scan_comp = time_fn(torch.compile(rnn_scan, fullgraph=True), (init, xs), warm_up=warm_up_cycles)
torch.testing.assert_close(result_pytorch[0], result_scan[1])

print(f"PyTorch {t_pytorch} vs. Scan {t_scan} vs. Scan (Comp.) {t_scan_comp}")

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