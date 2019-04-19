from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from benchmarks.operator_benchmark import benchmark_core, benchmark_utils

"""Caffe2 performance microbenchmarks.

This module contains Caffe2-specific functionalities for performance
microbenchmarks.
"""


def Caffe2OperatorTestCase(test_name, op_type, input_shapes, op_args, run_mode):
    """Benchmark Tester function for Caffe2 framework.
    """
    idx = 0
    input_blobs = []
    for input in input_shapes:
        blob_name = 'input_' + test_name + str(input_shapes) + str(op_args) + str(idx)
        input_blobs.append(blob_name)
        # TODO: figure out the data type from operator schema/
        # or accept custom data type for more comprehensive coverage.
        # Also, consider a more complex range/distribution of numerical inputs.
        workspace.FeedBlob(blob_name, benchmark_utils.numpy_random_fp32(*input))
        idx += 1

    # TODO: consider reuse logic in Caffe2's Functional utility to get
    # these benefits
    # - Read operator schema to figure out if inplace enforcement is needed
    # for the operator and name the output blob appropriately.
    # - Also figure out the number of outputs from operator schema.
    op = core.CreateOperator(
        op_type, input_blobs, ['out'], **op_args
    )

    def benchmark_func(num_runs):
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise RuntimeError('Unable to run operator test case ' % test_name)

    benchmark_core.add_benchmark_tester("Caffe2", test_name, input_shapes, op_args, run_mode, benchmark_func)
