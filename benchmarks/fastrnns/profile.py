import argparse
import os
import subprocess
import sys
import time
import torch
import datetime

from .runner import get_rnn_runners

PY3 = sys.version_info >= (3, 0)


def run_rnn(name, rnn_creator, nloops=5,
            seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
            miniBatch=64, device='cuda', seed=None):
    def run_iter(modeldef):
        # Forward
        forward_output = modeldef.forward(*modeldef.inputs)

        # "loss computation" and backward
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)

        # "Update" parameters
        if modeldef.backward is not None:
            for param in modeldef.params:
                param.grad.data.zero_()
        torch.cuda.synchronize()

    assert device == 'cuda'
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)
    modeldef = rnn_creator(**creator_args)

    [run_iter(modeldef) for _ in range(nloops)]


def profile(rnns, sleep_between_seconds=1, nloops=5,
            internal_run=True,  # Unused, get rid of this TODO
            seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
            miniBatch=64, device='cuda', seed=None):
    params = dict(seqLength=seqLength, numLayers=numLayers,
                  inputSize=inputSize, hiddenSize=hiddenSize,
                  miniBatch=miniBatch, device=device, seed=seed)
    for name, creator, context in get_rnn_runners(*rnns):
        with context():
            run_rnn(name, creator, nloops, **params)
            time.sleep(sleep_between_seconds)


def system(command):
    """Returns (return-code, stdout, stderr)"""
    print('[system] {}'.format(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        output = output.decode("ascii")
        err = err.decode("ascii")
    return rc, output, err


def describe_sizes(**sizes):
    # seqLength, numLayers, inputSize, hiddenSize, miniBatch
    return 's{}-l{}-i{}-h{}-b{}'.format(
        sizes['seqLength'],
        sizes['numLayers'],
        sizes['inputSize'],
        sizes['hiddenSize'],
        sizes['miniBatch'],
    )


OUTPUT_DIR = '~/profout/'


def nvprof_output_filename(rnns, **params):
    rnn_tag = '-'.join(rnns)
    size_tag = describe_sizes(**params)
    date_tag = datetime.datetime.now().strftime("%m%d%y-%H%M")
    return '{}prof_{}_{}_{}.nvvp'.format(OUTPUT_DIR, rnn_tag,
                                         size_tag, date_tag)


def nvprof(cmd, outpath):
    return system('nvprof -o {} {}'.format(outpath, cmd))


def full_profile(rnns, **args):
    args['internal_run'] = True
    profile_args = []
    for k, v in args.items():
        profile_args.append('--{}={}'.format(k, v))
    profile_args.append('--rnns {}'.format(' '.join(rnns)))

    outpath = nvprof_output_filename(rnns, **args)

    cmd = '{} -m fastrnns.profile {}'.format(
        sys.executable, ' '.join(profile_args))
    rc, stdout, stderr = nvprof(cmd, outpath)
    if rc is not 0:
        raise RuntimeError('stderr: {}\nstdout: {}'.format(stderr, stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile RNNs')

    parser.add_argument('--seqLength', default='100', type=int)
    parser.add_argument('--numLayers', default='1', type=int)
    parser.add_argument('--inputSize', default='512', type=int)
    parser.add_argument('--hiddenSize', default='512', type=int)
    parser.add_argument('--miniBatch', default='64', type=int)
    parser.add_argument('--sleep_between_seconds', default='1', type=int)
    parser.add_argument('--nloops', default='5', type=int)

    parser.add_argument('--rnns', nargs='*',
                        help='What to run. cudnn, aten, jit, etc')

    # if internal_run, we actually run the rnns.
    # if not internal_run, we shell out to nvprof with internal_run=T
    parser.add_argument('--internal_run', default=False, type=bool,
                        help='Don\'t use this')
    args = parser.parse_args()
    if args.rnns is None:
        args.rnns = ['cudnn', 'aten', 'jit']
    print(args)

    if args.internal_run:
        profile(**vars(args))
    else:
        full_profile(**vars(args))
