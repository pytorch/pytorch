import argparse
import torch
import torch.nn as nn

from .cells import lstm_cell
from .factory import pytorch_lstm_creator, varlen_pytorch_lstm_creator
from .runner import get_nn_runners


def barf():
    import pdb
    pdb.set_trace()


def assertEqual(tensor, expected, threshold=0.001):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for t, e in zip(tensor, expected):
            assertEqual(t, e)
    else:
        if (tensor - expected).abs().max() > threshold:
            barf()


def filter_requires_grad(tensors):
    return [t for t in tensors if t.requires_grad]


def test_rnns(experim_creator, control_creator, check_grad=True, verbose=False,
              seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
              miniBatch=64, device='cuda', seed=17):
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)

    print("Setting up...")
    control = control_creator(**creator_args)
    experim = experim_creator(**creator_args)

    # Precondition
    assertEqual(experim.inputs, control.inputs)
    assertEqual(experim.params, control.params)

    print("Checking outputs...")
    control_outputs = control.forward(*control.inputs)
    experim_outputs = experim.forward(*experim.inputs)
    assertEqual(experim_outputs, control_outputs)

    print("Checking grads...")
    assert control.backward_setup is not None
    assert experim.backward_setup is not None
    assert control.backward is not None
    assert experim.backward is not None
    control_backward_inputs = control.backward_setup(control_outputs, seed)
    experim_backward_inputs = experim.backward_setup(experim_outputs, seed)

    control.backward(*control_backward_inputs)
    experim.backward(*experim_backward_inputs)

    control_grads = [p.grad for p in control.params]
    experim_grads = [p.grad for p in experim.params]
    assertEqual(experim_grads, control_grads)

    if verbose:
        print(experim.forward.graph_for(*experim.inputs))
    print('')


def test_vl_py(**test_args):
    # XXX: This compares vl_py with vl_lstm.
    # It's done this way because those two don't give the same outputs so
    # the result isn't an apples-to-apples comparison right now.
    control_creator = varlen_pytorch_lstm_creator
    name, experim_creator, context = get_nn_runners('vl_py')[0]
    with context():
        print('testing {}...'.format(name))
        creator_keys = [
            'seqLength', 'numLayers', 'inputSize',
            'hiddenSize', 'miniBatch', 'device', 'seed'
        ]
        creator_args = {key: test_args[key] for key in creator_keys}

        print("Setting up...")
        control = control_creator(**creator_args)
        experim = experim_creator(**creator_args)

        # Precondition
        assertEqual(experim.inputs, control.inputs[:2])
        assertEqual(experim.params, control.params)

        print("Checking outputs...")
        control_out, control_hiddens = control.forward(*control.inputs)
        control_hx, control_cx = control_hiddens
        experim_out, experim_hiddens = experim.forward(*experim.inputs)
        experim_hx, experim_cx = experim_hiddens

        experim_padded = nn.utils.rnn.pad_sequence(experim_out).squeeze(-2)
        assertEqual(experim_padded, control_out)
        assertEqual(torch.cat(experim_hx, dim=1), control_hx)
        assertEqual(torch.cat(experim_cx, dim=1), control_cx)

        print("Checking grads...")
        assert control.backward_setup is not None
        assert experim.backward_setup is not None
        assert control.backward is not None
        assert experim.backward is not None
        control_backward_inputs = control.backward_setup(
            (control_out, control_hiddens), test_args['seed'])
        experim_backward_inputs = experim.backward_setup(
            (experim_out, experim_hiddens), test_args['seed'])

        control.backward(*control_backward_inputs)
        experim.backward(*experim_backward_inputs)

        control_grads = [p.grad for p in control.params]
        experim_grads = [p.grad for p in experim.params]
        assertEqual(experim_grads, control_grads)

        if test_args['verbose']:
            print(experim.forward.graph_for(*experim.inputs))
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lstm correctness')

    parser.add_argument('--seqLength', default='100', type=int)
    parser.add_argument('--numLayers', default='1', type=int)
    parser.add_argument('--inputSize', default='512', type=int)
    parser.add_argument('--hiddenSize', default='512', type=int)
    parser.add_argument('--miniBatch', default='64', type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--check_grad', default='True', type=bool)
    parser.add_argument('--variable_lstms', action='store_true')
    parser.add_argument('--seed', default='17', type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--rnns', nargs='*',
                        help='What to run. jit_premul, jit, etc')
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ['jit_premul', 'jit']
    print(args)

    if 'cuda' in args.device:
        assert torch.cuda.is_available()

    rnn_runners = get_nn_runners(*args.rnns)

    should_test_varlen_lstms = args.variable_lstms
    test_args = vars(args)
    del test_args['rnns']
    del test_args['variable_lstms']

    if should_test_varlen_lstms:
        test_vl_py(**test_args)

    for name, creator, context in rnn_runners:
        with context():
            print('testing {}...'.format(name))
            test_rnns(creator, pytorch_lstm_creator, **test_args)
