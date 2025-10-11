from typing import List, Tuple, Optional, overload, Union, cast
import torch
import numpy as np
import time
import torch.optim as optim
from torch.nn.parameter import Parameter

def RNNScript(
    input,
    param1,
    param2,
    ):

    state1 = torch.zeros(32, 340, dtype=input.dtype, device=input.device)
    
    outs = []

    Wx = input @ param1
    Wx_inp, Wx_rec = torch.tensor_split(Wx, 2, 2)
    for wt_inp, wt_rec in zip(Wx_inp, Wx_rec):
        rec_mul_inp, rec_mul_rec = torch.tensor_split(state1 @ param2, 2, 1)
        input_prev = (wt_inp + rec_mul_inp)
        output_gate = (wt_rec + rec_mul_rec)

        state1 = input_prev * torch.sigmoid(output_gate)
        outs.append(state1)
    
    outs = torch.stack(outs)

    return outs, (outs)

if __name__ == "__main__":

    input_size = 140
    hidden_size = 340
    num_layers = 1
    num_timesteps = 111
    batch_size = 32

    bi_dir = True 
    rnnt_input = False
    num_threads = -1
    use_gpu = True
    load_weights = False

    forward_times = []
    backward_times = []

    if use_gpu:
        device = torch.device('cuda:0')
    else:
        device = None

    parameters = []

    w_ih = torch.empty((input_size, hidden_size), device=device)
    w_io = torch.empty((input_size, hidden_size), device=device)
    w_i_comb = Parameter(torch.cat([w_ih,w_io],1))
    parameters.append(w_i_comb)

    w_hh = torch.empty((hidden_size, hidden_size), device=device)
    w_ho = torch.empty((hidden_size, hidden_size), device=device)
    w_h_comb = Parameter(torch.cat([w_hh,w_ho],1))
    parameters.append(w_h_comb)
    
    def count_kernels(guard):
        print("[pt2_compile] guard failed: ", guard)

    rnnscript = torch.compile(RNNScript, mode='reduce-overhead', dynamic=True, fullgraph=True)
    #backend = torch._TorchCompileInductorWrapper('reduce-overhead', None, True)
    #rnnscript = torch._dynamo.optimize(backend=backend, nopython=True, dynamic=True, guard_fail_fn=count_kernels)(RNNScript)
    #rnnscript = RNNScript
    snu = lambda x: rnnscript(x, w_i_comb, w_h_comb)

    optimizer = optim.SGD(parameters, 0.1)

    inp = torch.rand((num_timesteps, batch_size, input_size))

    if use_gpu:
        inp = inp.cuda()

    optimizer.zero_grad()
    for execution in range(5):
        start_forward = time.time_ns()
        t_rnd = np.random.randint(0, 200)
        inp = torch.rand((t_rnd, batch_size, input_size))
        if use_gpu:
            inp = inp.cuda()
        out, state = snu(inp)

        if use_gpu:
            torch.cuda.synchronize()
        stop_forward = time.time_ns()
        forward_times.append((stop_forward - start_forward) / (10 ** 9))

        loss = 1. - torch.sum(out)

        start_time_backward = time.time_ns()
        #loss.backward()
        if use_gpu:
            torch.cuda.synchronize()
        stop_time_backward = time.time_ns()
        backward_times.append((stop_time_backward - start_time_backward) / (10 ** 9))

    print('================================================================')
    print('Model with sSNU-os:')
    print('# Layers: ' + str(num_layers))
    print('# Units per layer: ' + str(hidden_size))
    print('Bidirectional: ' + str(bi_dir))
    print('Load weights: '  + str(load_weights))
    print('RNN-T input: ' + str(rnnt_input))
    print('# CPU threads: ' + str(num_threads))
    print('GPU support: ' + str(use_gpu))
    print('----------------------------------------------------------------')
    print('Timing summary')
    print('Time of forward computation: {:.4f} +- {:.4f} s'.format(np.mean(np.array(forward_times)), np.std(np.array(forward_times))))
    print('Time of backward computation: {:.4f} +- {:.4f} s'.format(np.mean(np.array(backward_times)), np.std(np.array(backward_times))))

