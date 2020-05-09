# @package functional
# Module caffe2.python.layers.functional
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
import six
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Functional(ModelLayer):

    def __init__(self, model, input_record, output_names_or_num, function,
                 name='functional', output_dtypes=None, tags=None, **kwargs):

        # allow coercion
        input_record = schema.as_record(input_record)

        super(Functional, self).__init__(model, name, input_record, tags=tags, **kwargs)
        self._function = function
        self._kwargs = kwargs
        return_struct = (
            isinstance(output_names_or_num, list) or
            (isinstance(output_names_or_num, six.integer_types) and
             output_names_or_num != 1)
        )

        with scope.NameScope(self.name, reset=True):
            if isinstance(output_names_or_num, int):
                struct_output_schema = schema.NewRecord(
                    model.net, schema.RawTuple(output_names_or_num))
            elif isinstance(output_names_or_num, schema.Field):
                self.output_schema = output_names_or_num.clone(keep_blobs=True)
                return
            else:
                if not isinstance(output_names_or_num, list):
                    output_names_or_num = [output_names_or_num]
                out_tuple = [(out, np.void) for out in output_names_or_num]
                struct_output_schema = schema.NewRecord(
                    model.net, schema.Struct(*out_tuple))

        num_outputs = len(struct_output_schema.field_blobs())

        # functional layer returns Struct if more than one outputs or output is
        # a list, otherwise Scalar
        if return_struct:
            self.output_schema = struct_output_schema
        else:
            self.output_schema = struct_output_schema[0]

        # If output_dtypes is provided, use it for output schema. Otherwise
        # the shape and type will be inferred.
        if output_dtypes is not None:
            if not isinstance(output_dtypes, list):
                output_dtypes = [output_dtypes] * num_outputs
            assert len(output_dtypes) == num_outputs
            for dtype, scalar in zip(output_dtypes,
                                     self.output_schema.all_scalars()):
                scalar.set_type(dtype)
            return

        # Fake execution of the function to infer shapes and types automatically
        had_issues = False
        try:
            type_net = core.Net('_temp_type_and_shape_inference_net')
            schema.InitEmptyRecord(type_net, input_record, enforce_types=True)

            function(type_net, self.input_record, self.output_schema, **kwargs)
            (shapes, types) = workspace.InferShapesAndTypes([type_net], {})
            for i in range(num_outputs):
                scalar_schema = (self.output_schema[i] if return_struct
                                 else self.output_schema)
                blob = scalar_schema()
                if blob not in types or blob not in shapes:
                    had_issues = True
                    continue
                if shapes[blob] == []:
                    # Scalar type
                    shape = tuple()
                elif shapes[blob][0] == 0:
                    shape = tuple(shapes[blob][1:])
                else:
                    logger.warning("unexpected shape: {}".format(shapes[blob]))
                    # If batch dimension is not first - give up on shape
                    # inference for that blob
                    had_issues = True
                    continue

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
                elif types[blob] == caffe2_pb2.TensorProto.FLOAT16:
                    dtype = (np.float16, shape)

                if dtype is not None:
                    scalar_schema.set_type(dtype)
        except TypeError as ex:
            had_issues = True
            logger.warning(str(ex))

        if had_issues:
            logger.warning(
                "Type inference had problems for layer: {}".format(self.name))

    def add_ops(self, net):
        self._function(
            net, self.input_record, self.output_schema, **(self._kwargs))
