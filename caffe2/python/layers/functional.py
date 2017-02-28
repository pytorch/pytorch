from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema, scope, workspace
from caffe2.python.layers.layers import (
    ModelLayer,
)
import caffe2.proto.caffe2_pb2 as caffe2_pb2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Functional(ModelLayer):

    def __init__(self, model, input_record, num_outputs, function,
                 name='functional', **kwargs):
        super(Functional, self).__init__(model, name, input_record, **kwargs)
        self._function = function

        with scope.NameScope(self.name):
            self.output_schema = schema.NewRecord(
                model.net, schema.RawTuple(num_outputs))

        # Fake execution of the function to infer shapes and types automatically
        had_issues = False
        try:
            type_net = core.Net('_temp_type_and_shape_inference_net')
            schema.InitEmptyRecord(type_net, input_record, enforce_types=True)

            function(type_net, self.input_record, self.output_schema)
            (shapes, types) = workspace.InferShapesAndTypes([type_net], {})
            for i in range(num_outputs):
                blob = self.output_schema[i]()
                if blob not in types or blob not in shapes:
                    had_issues = True
                    continue
                # If batch dimension is not first - give up on shape
                # inference for that blob
                if shapes[blob][0] != 0:
                    had_issues = True
                    continue

                shape = tuple(shapes[blob][1:])

                # TODO(amalevich): Move it to some shared library
                dtype = None
                if types[blob] == caffe2_pb2.TensorProto.DOUBLE:
                    dtype = (np.float64, shape)
                elif types[blob] == caffe2_pb2.TensorProto.FLOAT:
                    dtype = (np.float32, shape)
                elif types[blob] == caffe2_pb2.TensorProto.INT32:
                    dtype = (np.int32, shape)
                elif types[blob] == caffe2_pb2.TensorProto.INT64:
                    dtype = (np.int64, shape)

                if dtype is not None:
                    self.output_schema[i].set_type(dtype)
        except TypeError as ex:
            had_issues = True
            logger.warning(str(ex))

        if had_issues:
            logger.warning(
                "Type inference had problems for layer: {}".format(self.name))

    def add_ops(self, net):
        self._function(net, self.input_record, self.output_schema)
