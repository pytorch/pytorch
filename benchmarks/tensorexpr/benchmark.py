import contextlib
import numpy as np
import os
import time
from . import tensor_engine
import torch
import json


class Benchmark:
    def __init__(self, mode, device, dtype):
        self.mode = mode
        self.deterministic = False
        self.device = device
        self.dtype = dtype
        self.output_type = "stdout"
        self.print_ir = False
        self.print_kernel = False
        if mode == "both":
            self.requires_grad = True
        elif mode == "fwd":
            self.requires_grad = False
        else:
            raise ValueError(f"invalid mode: {mode}")
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
        return f"{self.engine.mode}: {self.module()}_{self.mode}_{device}_{config_str}"

    @staticmethod
    def module():
        raise ValueError("this method should be reimplemented by subclass")

    def memory_workload(self):
        raise ValueError("this method should be reimplemented by subclass")

    def compute_workload(self):
        """return the number of scalar operations it takes to finish the tensor op"""
        return None

    @staticmethod
    def input_iterable():
        """A benchmark child class should return true if it utilizes the input iter arg"""
        return False

    def dtype_to_bytes(self) :
        return torch.tensor(0, dtype=self.dtype).element_size()

    @staticmethod
    def default_configs():
        """return a list of defualt configs for this benchmark"""
        raise ValueError("this method should be reimplemented by subclass")

    def is_supported(self):
        return True

    def rand(self, shape, device=None, dtype=None, requires_grad=False):
        v = self.engine.rand(shape, device=device, dtype=dtype, requires_grad=requires_grad)
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
        self.print_ir = args.print_ir
        if args.cuda_fuser == "old" :
            torch._C._jit_override_can_fuse_on_gpu(True)
            if args.print_kernel :
                os.environ['PYTORCH_FUSION_DEBUG'] = '1'
            return self.run_impl(True)
        elif args.cuda_fuser == "te" :
            torch._C._jit_set_texpr_fuser_enabled(True)
            with cuda_pointwise_context(
                args.cuda_pointwise_loop_levels,
                args.cuda_pointwise_block_count,
                args.cuda_pointwise_block_size,
            ):
                return self.run_impl(True)
        elif args.cuda_fuser == "nvf" :
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_bailout_depth(20)
            if args.print_kernel :
                os.environ['PYTORCH_CUDA_FUSER_DEBUG'] = '1'
            return self.run_impl(True)
        else :
            return self.run_impl(False)

    def run_impl(self, use_fuser):
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
                if self.jit_mode == "trace" and use_fuser :
                    self.bm_jit = torch.jit.trace(
                        self.forward, example_inputs=self.inputs, check_trace=False
                    )
                if callable(getattr(self, "reference", None)):
                    self.check()
                else:
                    print("Warning: no reference result for ", self.module())
            elif i == 1:
                # The fusion graph is visible after the first iter is executed
                if self.jit_mode == "trace" and use_fuser and self.print_ir :
                    print(self.bm_jit.graph_for(*self.inputs))
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
            "sol": memory_workload["sol"] * self.dtype_to_bytes() / iter_time / 1e9,
            "algorithmic": memory_workload["algorithmic"] * self.dtype_to_bytes() / iter_time / 1e9,
        }
        if compute_workload:
            result_dict["compute_workload"] = compute_workload / iter_time / 1e9
        self.dump_result(result_dict)

    def dump_result(self, result_dict):
        if self.output_type == "json":
            print(json.dumps(result_dict))
        elif self.output_type == "stdout":
            msg = "{}: {:.2f} us, SOL {:.2f} GB/s, algorithmic {:.2f} GB/s".format(
                result_dict["desc"],
                result_dict["us"],
                result_dict["sol"],
                result_dict["algorithmic"],
            )
            if "compute_workload" in result_dict:
                msg += f", compute {result_dict['compute_workload']:.2f} Gops/s"
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

    try:
        yield
    finally:
        if loop_levels:
            torch._C._jit_set_te_cuda_pointwise_loop_levels(old_loop_levels)
        if block_count:
            torch._C._jit_set_te_cuda_pointwise_block_count(old_block_count)
        if block_size:
            torch._C._jit_set_te_cuda_pointwise_block_size(old_block_size)

# Auxiliary class to facilitate dynamic input shape
class DynamicShape:
    r'''
    An Auxiliary class for dynamic shape benchmarks

    Pre-computes input with random shapes and also
    modifies the compute method so in each call the
    fuser sees a different input tensor shape
    '''

    # Number of random inputs in an instance
    SAMPLE_SIZE = 100

    def __init__(self, dynamic_range=1.2):
        self._input_samples = []
        self._input_sample_index = 0
        self._dynamic_range = 1. / dynamic_range if dynamic_range > 1.0 else dynamic_range
        self._enable_dynamic_shapes = True

    # Returns the input test case that current index points to
    @property
    def inputs(self):
        return self._input_samples[self._input_sample_index]

    # An inputs assignment actually adds a test case in the class buffer
    @inputs.setter
    def inputs(self, val):
        self._input_samples.append(val)

    # Runs normal compute while increment test case index
    def compute(self):
        super().compute()
        self._input_sample_index = (self._input_sample_index + 1) % self.SAMPLE_SIZE

    # Defined by benchmark, the benchmark needs to specify the input
    # tensor construction in this method, essentially the same way
    # a benchmark creates the inputs list in the initializer
    def instantiate_input(self):
        raise NotImplementedError

    # Instantiate random shaped inputs and start the benchmark run
    def run(self, args):
        # force disable dynamic shape from command line
        if args.no_dynamic_shape:
            self._enable_dynamic_shapes = False
        self.load_inputs()
        super().run(args)

    # pre-compute inputs so the creations of random tensors
    # do not add to the compute time
    def load_inputs(self):
        for i in range(self.SAMPLE_SIZE - 1):
            self.instantiate_input()

    # returns a randomized shape
    def rand_shape(self, shape):
        if not self._enable_dynamic_shapes:
            return shape
        ratios = np.random.uniform(self._dynamic_range, 1.0, len(shape))
        dyn_shape = list(
            np.multiply(shape, ratios).astype(int)
        )
        return dyn_shape


benchmark_classes = []


def register_benchmark_class(benchmark_cls):
    benchmark_classes.append(benchmark_cls)
