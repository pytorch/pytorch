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
class LSTM(torch.nn.Module):
    def __init__(self, Wii, bii, Whi, bhi, Wif, bif, Whf, bhf, Wig, big, Whg, bhg, Wio, bio, Who, bho):
        super(LSTM, self).__init__()
        self.Wii = Wii.clone()
        self.bii = bii.clone()
        self.Whi = Whi.clone()
        self.bhi = bhi.clone()
        self.Wif = Wif.clone()
        self.bif = bif.clone()
        self.Whf = Whf.clone()
        self.bhf = bhf.clone()
        self.Wig = Wig.clone()
        self.big = big.clone()
        self.Whg = Whg.clone()
        self.bhg = bhg.clone()
        self.Wio = Wio.clone()
        self.bio = bio.clone()
        self.Who = Who.clone()
        self.bho = bho.clone()
        
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def lstm_combine(carry, x):
            h, c = carry
            
            i = torch.sigmoid(x @ self.Wii + self.bii + h @ self.Whi + self.bhi)
            f = torch.sigmoid(x @ self.Wif + self.bif + h @ self.Whf + self.bhf)
            g = torch.tanh(x @ self.Wig + self.big + h @ self.Whg + self.bhg)
            o = torch.sigmoid(x @ self.Wio + self.bio + h @ self.Who + self.bho)
            
            c_new = f * c + i * g
            h_new = o * torch.tanh(c_new)
            
            return (h_new, c_new), o + 0.
        
        carry, outs = scan(lstm_combine, init, xs, dim=0)
        
        return carry, outs

# Define the RNN with pure PyTorch
lstm_pytorch = torch.nn.LSTM(3, 5)

Wii, Wif, Wig, Wio = torch.chunk(lstm_pytorch.weight_ih_l0, 4)
Whi, Whf, Whg, Who = torch.chunk(lstm_pytorch.weight_hh_l0, 4)
bii, bif, big, bio = torch.chunk(lstm_pytorch.bias_ih_l0, 4)
bhi, bhf, bhg, bho = torch.chunk(lstm_pytorch.bias_hh_l0, 4)

lstm_scan = LSTM(
                Wii.T, bii,
                Whi.T, bhi,
                
                Wif.T, bif,
                Whf.T, bhf,
                
                Wig.T, big,
                Whg.T, bhg,
                
                Wio.T, bio,
                Who.T, bho,
                )

###############################################################################
# Define the inputs
###############################################################################
warm_up_cycles = 3
xs = torch.randn(1, 3, requires_grad=True)
init = (torch.zeros(5, requires_grad=True), torch.zeros(5, requires_grad=True))

###############################################################################
# Run models forward (inference-only)
###############################################################################
# Native PyTorch implementation
result_pytorch, t_pytorch = time_fn(lstm_pytorch, (xs,), warm_up=warm_up_cycles)
# result_pytorch = result_pytorch[0]

# Implementation with scan
result_scan, t_scan = time_fn(lstm_scan, (init, xs), warm_up=warm_up_cycles)
# torch.testing.assert_close(result_pytorch[0], result_scan[1])

# Implementation with scan - compiled
result_scan, t_scan_comp = time_fn(torch.compile(lstm_scan, fullgraph=True), (init, xs), warm_up=warm_up_cycles)
# torch.testing.assert_close(result_pytorch[0], result_scan[1])

print(f"PyTorch {t_pytorch} vs. Scan {t_scan} vs. Scan (Comp.) {t_scan_comp}")

###############################################################################
# Run models with backward
###############################################################################
# TODO: This is to be done!
def wrapper_torch_bwd(fn, xs):
    result = fn(xs)
    result = result[0][:10, :30].sum()
    # grad_init = [torch.ones_like(el) for el in result]
    # grad = torch.autograd.grad(result, xs, grad_init)
    grad = torch.autograd.grad(result, xs)
    return grad


# torch.testing.assert_close(result_pytorch, result)

# grad_init = torch.ones_like(result_pytorch)
# grad_pytorch = torch.autograd.grad(result_pytorch, xs, grad_init)[0]
# grad = torch.autograd.grad(result, xs, grad_init)[0]

# # The map function computes x ** y for each element, where y = 2
# # Therefore, we expect the correct gradients to be x * 2
# print("Gradient of PyTorch:\n", grad_pytorch)
# print("Gradient of cond:\n", grad)
# torch.testing.assert_close(grad_pytorch, grad)

    
# # Matrix implementation for associative_scan
# result_grad_matrix, t_grad_matrix = time_fn(lambda xs: wrapper_torch_bwd(model_matrix, xs), xs, warm_up=warm_up_cycles)

# # Memory-efficient implementation for associative_scan
# result_grad_memeff, t_grad_memeff = time_fn(lambda xs: wrapper_torch_bwd(model_memeff, xs), xs, warm_up=warm_up_cycles)

# # Non-jit compiled JAX model
# result_grad_jax, t_grad_jax = time_fn(lambda xs: wrapper_jax_bwd(s5_operator_jax, xs), xs_jax, warm_up=warm_up_cycles)

# # jit compiled JAX model
# result_grad_jax_jit, t_grad_jax_jit = time_fn(lambda xs: wrapper_jax_jit_bwd(s5_operator_jax, xs), xs_jax, warm_up=warm_up_cycles)

# print(f"Matrix {t_grad_matrix} vs. Mem eff. {t_grad_memeff} vs. JAX {t_grad_jax} vs. JAX (JIT) {t_grad_jax_jit}")