from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
import unittest


class TestONNXWhile(serial.SerializedTestCase):
    @serial.given(
        condition=st.booleans(),
        max_trip_count=st.integers(0, 100),
        save_scopes=st.booleans(),
        disable_scopes=st.booleans(),
        seed=st.integers(0, 65535),
        **hu.gcs_cpu_only)
    def test_onnx_while_fibb(
            self, condition, max_trip_count, save_scopes, disable_scopes, seed, gc, dc):
        np.random.seed(seed)
        if disable_scopes:
            save_scopes = False

        # Create body net
        body_net = caffe2_pb2.NetDef()
        # Two loop carried dependencies: first and second
        body_net.external_input.extend(['i', 'cond', 'first', 'second'])
        body_net.external_output.extend(['cond_new', 'second', 'third', 'third'])
        add_op = core.CreateOperator(
            'Add',
            ['first', 'second'],
            ['third'],
        )
        print3 = core.CreateOperator(
            'Print',
            ['third'],
            [],
        )
        limit_const = core.CreateOperator(
            'ConstantFill',
            [],
            ['limit_const'],
            shape=[1],
            dtype=caffe2_pb2.TensorProto.FLOAT,
            value=100.0,
        )
        cond = core.CreateOperator(
            'LT',
            ['third', 'limit_const'],
            ['cond_new'],
        )
        body_net.op.extend([add_op, print3, limit_const, cond])

        while_op = core.CreateOperator(
            'ONNXWhile',
            ['max_trip_count', 'condition', 'first_init', 'second_init'],
            ['first_a', 'second_a', 'third_a'],
            body=body_net,
            has_cond=True,
            has_trip_count=True,
            save_scopes=save_scopes,
            disable_scopes=disable_scopes,
        )

        condition_arr = np.array(condition).astype(np.bool)
        max_trip_count_arr = np.array(max_trip_count).astype(np.int64)
        first_init = np.array([1]).astype(np.float32)
        second_init = np.array([1]).astype(np.float32)

        def ref(max_trip_count, condition, first_init, second_init):
            first = 1
            second = 1
            results = []
            if condition:
                for _ in range(max_trip_count):
                    third = first + second
                    first = second
                    second = third
                    results.append(third)
                    if third > 100:
                        break
            return (first, second, np.array(results).astype(np.float32))

        self.assertReferenceChecks(
            gc,
            while_op,
            [max_trip_count_arr, condition_arr, first_init, second_init],
            ref,
        )

if __name__ == "__main__":
    unittest.main()
