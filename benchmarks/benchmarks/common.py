from __future__ import absolute_import, division, print_function

import random
import torch
import numpy


# Better not to use random numbers for benchmarks, but just in case,
# seed everything
random.seed(123123459)
torch.manual_seed(123123459)
numpy.random.seed(123123459)


class Benchmark(object):
    # asv auto-selects number of iterations, so benchmarks runs in between
    # goal_time/10 and goal_time (seconds)
    goal_time = 0.25
