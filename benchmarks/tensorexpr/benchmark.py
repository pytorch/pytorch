import contextlib
import numpy as np
import os
import time
from . import tensor_engine
import torch
import json


class Benchmark(object):
    def __init__(self, mode, device):
        self.mode = mode
        self.deterministic = False
        self.device = device
        self.output_type = "stdout"
        if mode == "both":
            self.requires_grad = True
        elif mode == "fwd":
            self.requires_grad = False
        else:
            raise ValueError("invalid mode: %s" % (mode))
        self.result_grad = None
        self.grad_variables = []
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
            if method.startswith("_"):
                continue
            method_engine = getattr(self.engine, method)
            setattr(self, method, method_engine)

    def forward(self):
        """do one step worth of computation
        """
        raise ValueError("this method should be reimplemented by subclass")

    def check(self):
        if not self.deterministic:
            return
        np.testing.assert_allclose(
            self.reference(), self.numpy(self.compute()), atol=1e-2
        )

    def config(self):
        """returns an array for the current benchmark configs
        """
        raise ValueError("this method should be reimplemented by subclass")

    def desc(self):
        """return the description of the current benchmark
        """
        config = self.config()
        config_str = "_".join([str(x) for x in config])
        device = self.device
        if "NNC_NUM_THREADS" in os.environ:
            num_threads_str = os.environ["NNC_NUM_THREADS"]
            device += num_threads_str
        return "%s: %s_%s_%s_%s" % (
            self.engine.mode,
            self.module(),
            self.mode,
            device,
            config_str,
        )

    @staticmethod
    def module():
        raise ValueError("this method should be reimplemented by subclass")

    def memory_workload(self):
        raise ValueError("this method should be reimplemented by subclass")

    def compute_workload(self):
        """return the number of scalar operations it takes to finish the tensor op"""
        return None

    @staticmethod
    def default_configs():
        """return a list of defualt configs for this benchmark"""
        raise ValueError("this method should be reimplemented by subclass")

    def is_supported(self):
        return True

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

    def compute(self):
        if self.bm_jit:
            return self.bm_jit(*self.inputs)
        else:
            return self.forward(*self.inputs)

    def run(self, args):
        torch._C._jit_override_can_fuse_on_gpu(args.cuda_fuser == "old")
        torch._C._jit_set_texpr_fuser_enabled(args.cuda_fuser == "te")
        with cuda_pointwise_context(
            args.cuda_pointwise_loop_levels,
            args.cuda_pointwise_block_count,
            args.cuda_pointwise_block_size,
        ):
            return self.run_impl()

    def run_impl(self):
        warmups = 10
        if self.device == "cuda":
            iters = 1000
        else:
            iters = 10
        engine = tensor_engine.get_engine()

        self.bm_jit = None
        for i in range(warmups + iters):
            if i == warmups:
                if self.device == "cuda":
                    engine.sync_cuda()
                time_start = time.time()

            if i == 0:
                if self.jit_mode == "trace":
                    self.bm_jit = torch.jit.trace(
                        self.forward, example_inputs=self.inputs, check_trace=False
                    )
                if callable(getattr(self, "reference", None)):
                    self.check()
                else:
                    print("Warning: no reference result for ", self.module())
            z = self.compute()
            if self.mode == "both":
                if self.result_grad is None:
                    self.result_grad = engine.rand_like(z)
                engine.backward([z], [self.result_grad], self.grad_variables)

        if self.device == "cuda":
            engine.sync_cuda()

        duration = time.time() - time_start
        iter_time = duration / iters
        memory_workload = self.memory_workload()
        compute_workload = self.compute_workload()

        result_dict = {
            "desc": self.desc(),
            "us": iter_time * 1e6,
            "sol": memory_workload["sol"] / iter_time / 1e9,
            "algorithmic": memory_workload["algorithmic"] / iter_time / 1e9,
        }
        if compute_workload:
            result_dict["compute_workload"] = compute_workload / iter_time / 1e9
        self.dump_result(result_dict)

    def dump_result(self, result_dict):
        if self.output_type == "json":
            print(json.dumps(result_dict))
        elif self.output_type == "stdout":
            msg = "%s: %.2f us, SOL %.2f GB/s, algorithmic %.2f GB/s" % (
                result_dict["desc"],
                result_dict["us"],
                result_dict["sol"],
                result_dict["algorithmic"],
            )
            if "compute_workload" in result_dict:
                msg += ", compute %.2f Gops/s" % (
                    result_dict["compute_workload"] / iter_time / 1e9
                )
            print(msg)
        else:
            raise Exception("Unknown output_type " + self.output_type)


@contextlib.contextmanager
def cuda_pointwise_context(loop_levels, block_count, block_size):
    if loop_levels:
        old_loop_levels = torch._C._jit_get_te_cuda_pointwise_loop_levels()
        torch._C._jit_set_te_cuda_pointwise_loop_levels(loop_levels)
    if block_count:
        old_block_count = torch._C._jit_get_te_cuda_pointwise_block_count()
        torch._C._jit_set_te_cuda_pointwise_block_count(block_count)
    if block_size:
        old_block_size = torch._C._jit_get_te_cuda_pointwise_block_size()
        torch._C._jit_set_te_cuda_pointwise_block_size(block_size)

    yield

    if loop_levels:
        torch._C._jit_set_te_cuda_pointwise_loop_levels(old_loop_levels)
    if block_count:
        torch._C._jit_set_te_cuda_pointwise_block_count(old_block_count)
    if block_size:
        torch._C._jit_set_te_cuda_pointwise_block_size(old_block_size)


benchmark_classes = []


def register_benchmark_class(benchmark_cls):
    benchmark_classes.append(benchmark_cls)
