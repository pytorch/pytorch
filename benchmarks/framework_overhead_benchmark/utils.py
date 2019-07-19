from __future__ import absolute_import, division, print_function, unicode_literals
import time
from collections import namedtuple

NUM_PT_LOOP_ITERS = 1000
BenchmarkConfig = namedtuple('BenchmarkConfig', 'num_warmup_iters num_iters')
ModuleConfig = namedtuple('ModuleConfig', 'pt_fn num_params graph_mode')

def ms_to_us(time_ms):
    return (time_ms * 1e3)

def secs_to_us(time_s):
    return (time_s * 1e6)

def secs_to_ms(time_s):
    return (time_s * 1e3)

def benchmark_module(config, module):
    module.forward(config.num_warmup_iters)
    print("Running module for {} iterations".format(config.num_iters))
    start = time.time()
    module.forward(config.num_iters)
    end = time.time()
    time_elapsed_s = (end - start)
    return (secs_to_ms(time_elapsed_s) / config.num_iters / NUM_PT_LOOP_ITERS)
