from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from caffe2.python import core, workspace

import numpy as np

import caffe2.test.benchmark.benchmark_core as bc
import caffe2.test.benchmark.benchmark_utils as bu

"""Caffe2 performance microbenchmarks.

This module contains Caffe2-specific functionalities for performance
microbenchmarks.
"""


Caffe2OperatorTestCase = namedtuple(
    "Caffe2OperatorTestCase",
    ["test_name", "op_type", "input_shapes", "op_args", "run_mode"])


@bc.benchmark_tester
def caffe2_tester(test_case):
    """Benchmark Tester function for Caffe2 framework.
    test_case is expected to be a Caffe2OperatorTestCase object. If not, the
    function will return False.
    It returns a function that contains the code to benchmarked
    (operator execution).
    """
    if type(test_case) is Caffe2OperatorTestCase:
        print("Running benchmark test case %s with caffe2" % (test_case.test_name))

        idx = 0
        input_blobs = []
        for input in test_case.input_shapes:
            blob_name = 'input_' + str(idx)
            input_blobs.append(blob_name)
            # TODO: figure out the data type from operator schema/
            # or accept custom data type for more comprehensive coverage.
            # Also, consider a more complex range/distribution of numerical inputs.
            workspace.FeedBlob(blob_name, bu.numpy_random_fp32(*input))
            idx += 1

        # TODO: consider reuse logic in Caffe2's Functional utility to get
        # these benefits
        # - Read operator schema to figure out if inplace enforcement is needed
        # for the operator and name the output blob appropriately.
        # - Also figure out the number of outputs from operator schema.
        op = core.CreateOperator(
            test_case.op_type, input_blobs, ['out'], **test_case.op_args
        )

        def benchmark_func():
            workspace.RunOperatorOnce(op)

        return benchmark_func
    return False
