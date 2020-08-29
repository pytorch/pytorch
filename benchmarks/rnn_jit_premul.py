import os
import argparse
import sys
import torch
from timeit import default_timer as timer

def cell(igates, hidden, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy

def dynamic_rnn(input, hidden, wih, whh, bih, bhh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    hx, cx = hidden
    outputs = []
    inputs = torch.matmul(input, wih.t()).unbind(0)
    hy, cy = hx[0], cx[0]
    for seq_idx in range(len(inputs)):
        hy, cy = cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
        outputs += [hy]
    return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                miniBatch=64, dropout=0.0, return_module=False, device='cuda', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, dropout=dropout)
    if 'cuda' in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None

def setUp(fuser='te', executor=None):
#     torch._C._jit_logging_enable_timers(False)
    assert fuser in ['te', 'old', 'none']
    if fuser == 'te':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_set_num_profiled_runs(2)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser == 'old':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser == 'none':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

    # --executor overrides settings of --fuser
    if executor == 'profiling':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_set_num_profiled_runs(2)
    elif executor == 'simple':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(False)
    elif executor == 'legacy':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

def main():
    parser = argparse.ArgumentParser(description='Profile RNNs')

    parser.add_argument('--fuser', default='te', type=str,
                        help='The fuser backend to use. One of: te, old, or none')
    parser.add_argument('--executor', default=None, type=str,
                        help='The executor to use. One of: legacy, simple, profiling')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--showgraph', action='store_true')

    args = parser.parse_args()
    setUp(fuser=args.fuser, executor=args.executor)
    input, hidden, params, _ = lstm_inputs()
    inputs = [input, hidden] + params[0]

    s = torch.jit.script(dynamic_rnn)

    # Warmup runs
    print('Warming up..', file=sys.stderr)
    sys.stderr.flush()
    for _ in range(10):
        s(*inputs)
    sys.stderr.flush()

    # Benchmarking run
    print('Benchmarking..', file=sys.stderr)
    sys.stderr.flush()
    N = 20
    start = timer()
    if args.bench:
        if args.backward:
            for _ in range(N):
                fwd_output = s(*inputs)
                fwd_output, _ = fwd_output
                grad_output = torch.randn_like(fwd_output)
                fwd_output.backward(grad_output)
        else:
            for _ in range(N):
                fwd_output = s(*inputs)
    else:
#         torch._C._jit_logging_enable_timers(True)
        with torch.autograd.profiler.profile() as prof:
            for _ in range(N):
                fwd_output = s(*inputs)
                if args.backward:
                    fwd_output, _ = fwd_output
                    grad_output = torch.randn_like(fwd_output)
                    fwd_output.backward(grad_output)
#         torch._C._jit_logging_enable_timers(False)
    end = timer()
    sys.stderr.flush()
    if args.showgraph:
        print(s.graph_for(*inputs))
    if not args.bench:
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print('Time: %.3f s' % ((end-start)))

if __name__ == '__main__':
    main()
