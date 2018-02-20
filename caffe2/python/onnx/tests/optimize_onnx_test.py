# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package onnx
# Module caffe2.python.onnx.tests.optimize_onnx_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tarfile
import tempfile
import unittest

from collections import namedtuple
from subprocess import Popen, PIPE
from six.moves.urllib.request import urlretrieve
import numpy as np

import onnx
from onnx import helper, ModelProto, TensorProto
from onnx.backend.test.runner import Runner
import caffe2.python.onnx.backend as c2

from caffe2.python.onnx.tests.test_utils import TestCase

class TestRoundtrip(TestCase):
    def _roundtrip(self, model_name):
        model_dir = Runner(c2)._prepare_model_data(
            namedtuple('dummy', ['model_name'])(model_name))

        pb_path = os.path.join(model_dir, 'model.pb')

        before_roundtrip = onnx.load(pb_path)

        with open(pb_path, 'rb') as pb:
            after_roundtrip = onnx.load_from_string(pb.read())

        assert onnx.helper.printable_graph(before_roundtrip.graph) \
            == onnx.helper.printable_graph(after_roundtrip.graph)

        with open(pb_path, 'rb') as pb:
            assert after_roundtrip.SerializeToString() == pb.read()

    # arbitrarily pick one relatively small model to sanity test with
    def test_squeezenet_v3(self):
        self._roundtrip('squeezenet-ir-version-3')

    # testing just to be sure that we no-op instead of breaking on an
    # older IR version.
    def test_squeezenet_v1(self):
        self._roundtrip('squeezenet-ir-version-1')

class TestOptimize(TestCase):
    def _optimized(self, graph):
        orig_model = helper.make_model(graph, producer_name='onnx-to-caffe2-test')
        orig_model_str = orig_model.SerializeToString()
        optimized_model_str = c2.Caffe2Backend.optimize_onnx(orig_model_str)
        optimized_model = ModelProto()
        optimized_model.ParseFromString(optimized_model_str)
        return optimized_model

    def test_nop_transpose(self):
        trans = helper.make_node("Transpose", ["X"], ["Y"], perm=[0,1])
        graph = helper.make_graph(
            [trans],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3))],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2))])
        optimized_model = self._optimized(graph)

        for node in optimized_model.graph.node:
            assert node.op_type != "Transpose"

    def test_fuse_transpose(self):
        trans1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1,0,2])
        trans2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[2,0,1])
        trans3 = helper.make_node("Transpose", ["Z"], ["A"], perm=[2,0,1])
        graph = helper.make_graph(
            [trans1, trans2, trans3],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, (4, 3, 2))])
        optimized_model = self._optimized(graph)

        assert len(list(optimized_model.graph.node)) == 1

    def test_fuse_transpose_into_gemm(self):
        trans1 = helper.make_node("Transpose", ["X"], ["A"], perm=[1,0])
        trans2 = helper.make_node("Transpose", ["Y"], ["B"], perm=[1,0])
        gemm = helper.make_node("Gemm", ["A", "B", "C"], ["Z"])
        graph = helper.make_graph(
            [trans1, trans2, gemm],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3)),
             helper.make_tensor_value_info("Y", TensorProto.FLOAT, (5, 2)),
             helper.make_tensor_value_info("C", TensorProto.FLOAT, (3, 5))],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (3, 5))])
        optimized_model = self._optimized(graph)

        assert len(list(optimized_model.graph.node)) == 1

if __name__ == '__main__':
    unittest.main()
