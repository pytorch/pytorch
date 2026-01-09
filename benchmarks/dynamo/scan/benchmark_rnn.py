from time import perf_counter

import torch
from torch._higher_order_ops.scan import scan


###############################################################################
# Auxiliary function for timing
###############################################################################
def time_fn(fn, args, warm_up=1):
    t_initial = -1.0
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
input_size = 50
hidden_size = 200


class RNN_forloop(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.rnn_cell = torch.nn.RNNCell(input_size, hidden_size)

    # Implementation adopted from
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        # The input `xs` has the time as the first dimsion
        output = []
        for i in range(xs.size()[0]):
            init = self.rnn_cell(xs[i], init)
            output.append(init)
        output = torch.stack(output, dim=0)
        return output


class RNN_scan(torch.nn.Module):
    def __init__(self, Wih, bih, Whh, bhh):
        super().__init__()
        self.Wih = Wih.clone()
        self.bih = bih.clone()
        self.Whh = Whh.clone()
        self.bhh = bhh.clone()

    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def rnn_combine(carry, x):
            h = torch.tanh(x @ self.Wih + self.bih + carry @ self.Whh + self.bhh)
            # Needed to not trigger aliasing errors in inductor
            return h + 0.0, h.clone()

        carry, outs = scan(rnn_combine, init, xs, dim=0)
        return carry, outs


class RNN_scan_cell(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.rnn_cell = torch.nn.RNNCell(input_size, hidden_size)
        self.rnn_cell.load_state_dict(state_dict)

    def forward(self, init: torch.Tensor, xs: torch.Tensor):
        def rnn_combine(carry, x):
            hx = self.rnn_cell(x, carry)
            return hx, hx.clone()

        carry, outs = scan(rnn_combine, init, xs, dim=0)
        return carry, outs


# Define the for-loop LSTM model
rnn_forloop = RNN_forloop(input_size, hidden_size)
rnn_forloop_comp = torch.compile(rnn_forloop, fullgraph=True)

# Define the LSTM using CUDA kernels
rnn_forloop_state_dict = rnn_forloop.state_dict()
rnn_cuda_state_dict = {}
for key, value in rnn_forloop_state_dict.items():
    new_key = key.replace("rnn_cell.", "") + "_l0"
    rnn_cuda_state_dict[new_key] = value.clone()
rnn_cuda = torch.nn.RNN(input_size, hidden_size)
rnn_cuda.load_state_dict(rnn_cuda_state_dict)

# Define the LSTM model using scan
W_ih = rnn_cuda.weight_ih_l0.T.clone()
b_ih = rnn_cuda.bias_ih_l0.clone()
W_hh = rnn_cuda.weight_hh_l0.T.clone()
b_hh = rnn_cuda.bias_hh_l0.clone()
rnn_scan = RNN_scan(W_ih, b_ih, W_hh, b_hh)
rnn_scan_comp = torch.compile(rnn_scan, fullgraph=True)

# Define the LSTM model using scan and LSTMCell
rnn_scan_cell_state_dict = {}
for key, value in rnn_forloop_state_dict.items():
    new_key = key.replace("rnn_cell.", "")
    rnn_scan_cell_state_dict[new_key] = value.clone()
rnn_scan_cell = RNN_scan_cell(rnn_scan_cell_state_dict)
rnn_scan_cell_comp = torch.compile(rnn_scan_cell, fullgraph=True)

for time_steps in [3, 20, 70, 100]:
    ###############################################################################
    # Define the inputs
    ###############################################################################
    xs = torch.randn(time_steps, input_size, requires_grad=True)
    init = torch.zeros(hidden_size, requires_grad=True)

    ###############################################################################
    # Run models forward (inference-only)
    ###############################################################################
    # For-loop model
    result_forloop, time_initial_forloop, time_run_forloop = time_fn(
        rnn_forloop, (init, xs), warm_up=warm_up_cycles
    )

    # For-loop model compiled
    result_forloop_comp, time_initial_forloop_comp, time_run_forloop_comp = time_fn(
        rnn_forloop_comp, (init, xs), warm_up=warm_up_cycles
    )

    # CUDA model
    result_cuda, time_initial_cuda, time_run_cuda = time_fn(
        rnn_cuda, (xs.clone(), init.clone().unsqueeze(0)), warm_up=warm_up_cycles
    )

    # Scan model
    result_scan, time_initial_scan, time_run_scan = time_fn(
        rnn_scan, (init.clone().unsqueeze(0), xs.clone()), warm_up=warm_up_cycles
    )

    # Scan model compiled
    result_scan_comp, time_initial_scan_comp, time_run_scan_comp = time_fn(
        rnn_scan_comp, (init.clone().unsqueeze(0), xs.clone()), warm_up=warm_up_cycles
    )

    # Scan cell model
    result_scan_cell, time_initial_scan_cell, time_run_scan_cell = time_fn(
        rnn_scan_cell, (init.clone(), xs.clone()), warm_up=warm_up_cycles
    )

    # Scan cell model compiled
    result_scan_cell_comp, time_initial_scan_cell_comp, time_run_scan_cell_comp = (
        time_fn(rnn_scan_cell_comp, (init.clone(), xs.clone()), warm_up=warm_up_cycles)
    )

    torch.testing.assert_close(result_forloop, result_forloop_comp)
    torch.testing.assert_close(result_forloop, result_cuda[0])
    torch.testing.assert_close(result_forloop, result_scan[1][:, 0, :])
    torch.testing.assert_close(result_forloop, result_scan_comp[1][:, 0, :])
    torch.testing.assert_close(result_forloop, result_scan_cell[1])
    torch.testing.assert_close(result_forloop, result_scan_cell_comp[1])

    print(f"T={time_steps}:")
    print(
        f"Compile times       :\n\
For-Loop            : {time_initial_forloop_comp:.5f}\n\
Scan                : {time_initial_scan_comp:.5f}\n\
Scan Cell           : {time_initial_scan_cell_comp:.5f}\n"
    )
    print(
        f"Run times           :\n\
For-Loop            : {time_run_forloop:.5f}\n\
For-Loop compile.   : {time_run_forloop_comp:.5f}\n\
CUDA                : {time_run_cuda:.5f}\n\
Scan                : {time_run_scan:.5f}\n\
Scan compile        : {time_run_scan_comp:.5f}\n\
Scan RNNCell        : {time_run_scan_cell:.5f}\n\
Scan RNNCell compile: {time_run_scan_cell_comp:.5f}"
    )

###############################################################################
# Run models with backward
###############################################################################
# TODO To be done
