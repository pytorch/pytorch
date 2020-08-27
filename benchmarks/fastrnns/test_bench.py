from __future__ import print_function
import pytest
import torch
from .runner import get_nn_runners

default_rnns = ['cudnn', 'aten', 'jit', 'jit_premul', 'jit_premul_bias', 'jit_simple',
                         'jit_multilayer', 'py']
default_cnns = ['resnet18', 'resnet18_jit', 'resnet50', 'resnet50_jit']
all_nets = default_rnns + default_cnns

def pytest_generate_tests(metafunc):
    # This creates lists of tests to generate, can be customized
    if metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize('net_name', all_nets, scope="class")
        metafunc.parametrize("executor_and_fuser", ["legacy-old"], scope="class")

def set_fuser(fuser_name):
    if fuser_name == 'te':
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == 'old':
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == 'none':
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

def set_executor(executor_name):
    if executor_name == 'profiling':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
    elif executor_name == 'simple':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(False)
    elif executor_name == 'legacy':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

@pytest.fixture(scope='class')
def modeldef(request, net_name, executor_and_fuser):
    executor, fuser = executor_and_fuser.split("-")
    set_executor(executor)
    set_fuser(fuser)
    # Given a 'net_name' provided by generate_tests, build the thing
    name, rnn_creator, context = get_nn_runners(net_name)[0]
    creator_args = creator_args = {
        'seqLength': 100, 'numLayers': 1,
        'inputSize': 512, 'hiddenSize': 512,
        'miniBatch': 64, 'device': 'cuda', 'seed': None
    }
    return rnn_creator(**creator_args)

def cuda_sync(func, *args, **kwargs):
    out = func(*args, **kwargs)
    torch.cuda.synchronize()
    return out

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    max_time=0.1,
    group="fastrnns",
)
class TestBenchNetwork:
    # See 'modeldef' fixture, which provides the things to benchmark
    def test_forward(self, modeldef, benchmark):
        forward_output = benchmark(cuda_sync, modeldef.forward, *modeldef.inputs)

    def test_backward(self, modeldef, benchmark):
        backward_input = modeldef.forward(*modeldef.inputs)
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(backward_input)

        if modeldef.backward is not None:
            benchmark(cuda_sync, modeldef.backward, *backward_input, retain_graph=True)

            for param in modeldef.params:
                assert param.grad is not None
                param.grad.data.zero_()
