import numpy as np
import os
import time
import tensor_engine
import torch

class BenchmarkBase(object):
    def __init__(self, mode, device):
        self.mode = mode
        self.device = device
        if mode == 'both':
            self.requires_grad = True
        elif mode == 'fwd':
            self.requires_grad = False
        else:
            raise ValueError('invalid mode: %s' % (mode))
        self.result_grad = None
        self.grad_variables = []

    def forward(self):
        '''do one step worth of computation
        '''
        raise ValueError('this method should be reimplemented by subclass')

    def check(self):
        np.testing.assert_allclose(
            self.reference(), self.numpy(self.forward(*self.inputs)), atol=1e-2)

    def config(self):
        '''returns an array for the current benchmark configs
        '''
        raise ValueError('this method should be reimplemented by subclass')

    def desc(self):
        '''return the description of the current benchmark
        '''
        config = self.config()
        config_str = '_'.join([str(x) for x in config])
        device = self.device
        if 'NNC_NUM_THREADS' in os.environ:
            num_threads_str = os.environ['NNC_NUM_THREADS']
            device += num_threads_str
        return '%s: %s_%s_%s_%s' % (self.engine.mode, self.module(), self.mode, device, config_str)

    @staticmethod
    def module():
        raise ValueError('this method should be reimplemented by subclass')

    def memory_workload(self):
        raise ValueError('this method should be reimplemented by subclass')

    def compute_workload(self):
        '''return the number of scalar operations it takes to finish the tensor op'''
        return None

    @staticmethod
    def default_configs():
        '''return a list of defualt configs for this benchmark'''
        raise ValueError('this method should be reimplemented by subclass')

    def is_supported(self):
        return True


class Benchmark(BenchmarkBase):
    def __init__(self, mode, device):
        super().__init__(mode, device)
        self.engine = tensor_engine.get_engine()
        self.engine.reset(device)

        # forward all member functions in self.engine to self
        for method in dir(self.engine):
            if not callable(getattr(self.engine, method)):
                continue
            # don't forward if this function is overriden here
            if hasattr(self, method):
                continue
            # don't forward if it is a internal function
            if method.startswith('_'):
                continue
            method_engine = getattr(self.engine, method)
            setattr(self, method, method_engine)

    def rand(self, shape, device=None, requires_grad=False):
        v = self.engine.rand(shape, device=device, requires_grad=requires_grad)
        if requires_grad:
            self.grad_variables.append(v)
        return v

    def nchw_rand(self, shape, device=None, requires_grad=False):
        v = self.engine.nchw_rand(shape, device=device, requires_grad=requires_grad)
        if requires_grad:
            self.grad_variables.append(v)
        return v


def run_benchmark(benchmark):
    warmups = 10
    if benchmark.device == 'cuda':
        iters = 1000
    else:
        iters = 10
    engine = tensor_engine.get_engine()

    if callable(getattr(benchmark, 'reference', None)):
        benchmark.check()
    else:
        print(f"Warning: no reference result for {benchmark.module()}")

    bm_jit = None
    for i in range(warmups + iters):
        if i == warmups:
            if benchmark.device == 'cuda':
                engine.sync_cuda()
            time_start = time.time()

        if i == 0 and benchmark.jit_mode == 'trace':
            bm_jit = torch.jit.trace(benchmark.forward, example_inputs=benchmark.inputs)
        if bm_jit:
            z = bm_jit(*benchmark.inputs)
        else:
            z = benchmark.forward(*benchmark.inputs)
        if benchmark.mode == 'both':
            if benchmark.result_grad is None:
                benchmark.result_grad = engine.rand_like(z)
            engine.backward([z], [benchmark.result_grad], benchmark.grad_variables)

    if benchmark.device == 'cuda':
        engine.sync_cuda()

    duration = time.time() - time_start
    iter_time = duration / iters
    memory_workload = benchmark.memory_workload()
    compute_workload = benchmark.compute_workload()

    msg = '%s: %.2f us, SOL %.2f GB/s, algorithmic %.2f GB/s' % (
        benchmark.desc(), iter_time * 1e6,
        memory_workload['sol'] / iter_time / 1e9,
        memory_workload['algorithmic'] / iter_time / 1e9,
    )
    if compute_workload is not None:
        msg += ', compute %.2f Gops/s' % (compute_workload / iter_time / 1e9)
    print(msg)


benchmark_classes = []

def register_benchmark_class(benchmark_cls):
    benchmark_classes.append(benchmark_cls)
