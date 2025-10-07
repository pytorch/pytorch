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
        self.Wih = Wih.clone()
        self.bih = bih.clone()
        self.Whh = Whh.clone()
        self.bhh = bhh.clone()
        
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def rnn_combine(carry, x):
            h = torch.tanh(x @ self.Wih + self.bih + carry @ self.Whh + self.bhh)
            # Needed to not trigger aliasing errors in inductor
            return h, h.clone()
        
        outs = scan(rnn_combine, init, xs, dim=0)[1]
        return outs

# Define the RNN with pure PyTorch
rnn_pytorch = torch.nn.RNN(
            input_size=5,
            hidden_size=7,
        )

W_ih = rnn_pytorch.weight_ih_l0.T.clone()
b_ih = rnn_pytorch.bias_ih_l0.clone()
W_hh = rnn_pytorch.weight_hh_l0.T.clone()
b_hh = rnn_pytorch.bias_hh_l0.clone()

rnn_scan = RNN(W_ih,
               b_ih, 
               W_hh,
               b_hh)

###############################################################################
# Define the inputs
###############################################################################
warm_up_cycles = 3
xs = torch.randn(10, 5, requires_grad=True)
init = torch.zeros(7, requires_grad=True)

###############################################################################
# Run models forward (inference-only)
###############################################################################

# Native PyTorch implementation
result_pytorch, t_pytorch = time_fn(rnn_pytorch, (xs,), warm_up=warm_up_cycles)

# Implementation with scan
result_scan, t_scan = time_fn(rnn_scan, (init, xs), warm_up=warm_up_cycles)
torch.testing.assert_close(result_pytorch[0], result_scan)

# Implementation with scan - compiled
result_scan, t_scan_comp = time_fn(torch.compile(rnn_scan, fullgraph=True), (init, xs), warm_up=warm_up_cycles)
torch.testing.assert_close(result_pytorch[0], result_scan)

print(f"PyTorch {t_pytorch} vs. Scan {t_scan} vs. Scan (Comp.) {t_scan_comp}")

###############################################################################
# Run models with backward
###############################################################################
# TODO To be done