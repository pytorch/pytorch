from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import importlib
import os
from benchmarks.operator_benchmark import benchmark_runner

if __name__ == "__main__":
    # TODO: current way of importing other tests are fragile, so we need to have a robust way
    for module in os.listdir(os.path.dirname(__file__)):
        if module == '__init__.py' or not module.endswith('_test.py'):
            continue
        importlib.import_module("benchmarks.operator_benchmark.ops." + module[:-3])
    benchmark_runner.main()
