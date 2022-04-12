import time
from collections import namedtuple
from torch.utils import ThroughputBenchmark

NUM_LOOP_ITERS = 1000
BenchmarkConfig = namedtuple('BenchmarkConfig', 'num_warmup_iters num_iters')
ModuleConfig = namedtuple('ModuleConfig', 'pt_fn c2_op num_params graph_mode')

def ms_to_us(time_ms):
    return (time_ms * 1e3)

def secs_to_us(time_s):
    return (time_s * 1e6)

def secs_to_ms(time_s):
    return (time_s * 1e3)

def benchmark_using_throughput_benchmark(config, module):
    print("Benchmarking via ThroughputBenchmark")
    bench = ThroughputBenchmark(module.module)
    bench.add_input(*module.tensor_inputs)
    stats = bench.benchmark(1, config.num_warmup_iters, config.num_iters)
    return stats.latency_avg_ms / NUM_LOOP_ITERS

def benchmark_module(config, module, use_throughput_benchmark=False):
    if use_throughput_benchmark:
        return benchmark_using_throughput_benchmark(config, module)
    module.forward(config.num_warmup_iters)
    print("Running module for {} iterations".format(config.num_iters))
    start = time.time()
    module.forward(config.num_iters)
    end = time.time()
    time_elapsed_s = (end - start)
    return (secs_to_ms(time_elapsed_s) / config.num_iters / NUM_LOOP_ITERS)
