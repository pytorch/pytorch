import torch
from torch._higher_order_ops.scan import scan

from time import perf_counter

###############################################################################
# Auxiliary function for timing
###############################################################################
def time_fn(fn, args, warm_up=1):
    t_initial = -1.
    for ind in range(warm_up):
        t_start = perf_counter() 
        result = fn(*args)
        t_stop = perf_counter()
        if ind == 0:
            t_initial = t_stop - t_start

    t_start = perf_counter() 
    result = fn(*args)
    t_stop = perf_counter()
    t_run = t_stop - t_start
    return result, t_initial, t_run

###############################################################################
# Define the RNN models
###############################################################################
warm_up_cycles = 3
# input_size = 15
input_size = 50
# hidden_size = 20
hidden_size = 200
time_steps = 20

class LSTM_forloop(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_forloop, self).__init__()
        
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        
    # Implementation adopted from 
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        # The input `xs` has the time as the first dimsion
        output = []
        for i in range(xs.size()[0]):
            hx, cx = self.lstm_cell(xs[i], init)
            init = (hx, cx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        return output

class LSTM_scan(torch.nn.Module):
    def __init__(self, Wii, bii, Whi, bhi, Wif, bif, Whf, bhf, Wig, big, Whg, bhg, Wio, bio, Who, bho):
        super(LSTM_scan, self).__init__()
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
            
            return (h_new, c_new.clone()), h_new.clone()
        
        carry, outs = scan(lstm_combine, init, xs, dim=0)
        
        return carry, outs

# Define the for-loop LSTM model
lstm_forloop = LSTM_forloop(input_size, hidden_size)
lstm_forloop_comp = torch.compile(lstm_forloop, fullgraph=True)


# Define the LSTM using CUDA kernels
lstm_forloop_state_dict = lstm_forloop.state_dict()
lstm_cuda_state_dict = {}
for key, value in lstm_forloop_state_dict.items():
    new_key = key.replace('lstm_cell.', '') + '_l0'
    lstm_cuda_state_dict[new_key] = value.clone()
lstm_cuda = torch.nn.LSTM(input_size, hidden_size)
lstm_cuda.load_state_dict(lstm_cuda_state_dict)

# Define the LSTM model using scan
Wii, Wif, Wig, Wio = torch.chunk(lstm_cuda.weight_ih_l0, 4)
Whi, Whf, Whg, Who = torch.chunk(lstm_cuda.weight_hh_l0, 4)
bii, bif, big, bio = torch.chunk(lstm_cuda.bias_ih_l0, 4)
bhi, bhf, bhg, bho = torch.chunk(lstm_cuda.bias_hh_l0, 4)
lstm_scan = LSTM_scan(
                Wii.T, bii,
                Whi.T, bhi,
                
                Wif.T, bif,
                Whf.T, bhf,
                
                Wig.T, big,
                Whg.T, bhg,
                
                Wio.T, bio,
                Who.T, bho,
                )
lstm_scan_comp = torch.compile(lstm_scan, fullgraph=True)

###############################################################################
# Define the inputs
###############################################################################
xs = torch.randn(time_steps, input_size, requires_grad=True)
init = (torch.zeros(hidden_size, requires_grad=True), torch.zeros(hidden_size, requires_grad=True))

###############################################################################
# Run models forward (inference-only)
###############################################################################
# For-loop model
result_forloop, time_initial_forloop, time_run_forloop = time_fn(lstm_forloop, (init, xs), warm_up=warm_up_cycles)

# For-loop model compiled
result_forloop_comp, time_initial_forloop_comp, time_run_forloop_comp = time_fn(lstm_forloop_comp, (init, xs), warm_up=warm_up_cycles)

# CUDA model
result_cuda, time_initial_cuda, time_run_cuda = time_fn(lstm_cuda, (xs.clone(), (init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0))), warm_up=warm_up_cycles)

# Scan model
result_scan, time_initial_scan, time_run_scan = time_fn(lstm_scan, ((init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0)), xs.clone()), warm_up=warm_up_cycles)

# Scan model compiled
result_scan_comp, time_initial_scan_comp, time_run_scan_comp = time_fn(lstm_scan_comp, ((init[0].clone().unsqueeze(0), init[1].clone().unsqueeze(0)), xs.clone()), warm_up=warm_up_cycles)

torch.testing.assert_close(result_forloop, result_forloop_comp)
torch.testing.assert_close(result_forloop, result_cuda[0])
torch.testing.assert_close(result_forloop, result_scan[1][:, 0, :])
torch.testing.assert_close(result_forloop, result_scan_comp[1][:, 0, :])

print(f'T={time_steps}:')
print(f'Compile times:\n\
For-Loop        : {time_initial_forloop_comp:.5f}\n\
Scan            : {time_initial_scan_comp:.5f}\n')
print(f'Run times       :\n\
For-Loop        : {time_run_forloop:.5f} \n\
For-Loop compile: {time_run_forloop_comp:.5f} \n\
CUDA            : {time_run_cuda:.5f} \n\
Scan            : {time_run_scan:.5f} \n\
Scan compile    : {time_run_scan_comp:.5f}')

###############################################################################
# Run models with backward
###############################################################################
# TODO To be done