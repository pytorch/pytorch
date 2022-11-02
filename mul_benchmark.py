#! /bin/bash/python3
# -*- coding: utf-8 -*-
import torch
import torch.utils.benchmark as benchmark

def generate_example(device, dtype):
    a = torch.ones(100, device=device, dtype=dtype)
    b = torch.ones(100, device=device, dtype=dtype)
    yield (a, b)

def main():
    test_case_generators = [
        generate_example('cpu', torch.float)
    ]
    results = []

    for test_case_generator in test_case_generators:
        for a, b in test_case_generator:
            # results.append(benchmark.Timer(
            #     # setup='import subprocess\nimport sys\nsubprocess.check_call(["conda activate pytorch"])',
            #     stmt='torch.mul(a, b)',
            #     globals={'a': a, 'b': b},
            #     sub_label='',
            #     description='torch.mul after',
            # ).blocked_autorange())

            results.append(benchmark.Timer(
                stmt='torch.mul(a, b)',
                # setup='import subprocess\nimport sys\nsubprocess.check_call(["conda activate codegen_task_before"])',
                globals={'a': a, 'b': b},
                sub_label='',
                description='torch.mul before',
            ).blocked_autorange())

    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

if __name__ == "__main__":
    main()
