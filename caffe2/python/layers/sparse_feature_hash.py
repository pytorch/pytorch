## @package sparse_feature_hash
# Module caffe2.python.layers.sparse_feature_hash
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema, core
from caffe2.python.layers.layers import (
    ModelLayer,
    IdList,
    IdScoreList,
)
from caffe2.python.layers.tags import (
    Tags
)

import numpy as np


class SparseFeatureHash(ModelLayer):

    def __init__(self, model, input_record, seed=0, modulo=None,
                 use_hashing=True, use_divide_mod=False, divisor=None, name='sparse_feature_hash', **kwargs):
        super(SparseFeatureHash, self).__init__(model, name, input_record, **kwargs)

        assert use_hashing + use_divide_mod < 2, "use_hashing and use_divide_mod cannot be set true at the same time."

        if use_divide_mod:
            assert divisor >= 1, 'Unexpected divisor: {}'.format(divisor)

            self.divisor = self.create_param(param_name='divisor',
                                             shape=[1],
                                             initializer=('GivenTensorInt64Fill', {'values': np.array([divisor])}),
                                             optimizer=model.NoOptim)

        self.seed = seed
        self.use_hashing = use_hashing
        self.use_divide_mod = use_divide_mod

        if schema.equal_schemas(input_record, IdList):
            self.modulo = modulo or self.extract_hash_size(input_record.items.metadata)
            metadata = schema.Metadata(
                categorical_limit=self.modulo,
                feature_specs=input_record.items.metadata.feature_specs,
                expected_value=input_record.items.metadata.expected_value
            )
            with core.NameScope(name):
                self.output_schema = schema.NewRecord(model.net, IdList)
            self.output_schema.items.set_metadata(metadata)

        elif schema.equal_schemas(input_record, IdScoreList):
            self.modulo = modulo or self.extract_hash_size(input_record.keys.metadata)
            metadata = schema.Metadata(
                categorical_limit=self.modulo,
                feature_specs=input_record.keys.metadata.feature_specs,
                expected_value=input_record.keys.metadata.expected_value
            )
            with core.NameScope(name):
                self.output_schema = schema.NewRecord(model.net, IdScoreList)
            self.output_schema.keys.set_metadata(metadata)

        else:
            assert False, "Input type must be one of (IdList, IdScoreList)"

        assert self.modulo >= 1, 'Unexpected modulo: {}'.format(self.modulo)
        if input_record.lengths.metadata:
            self.output_schema.lengths.set_metadata(input_record.lengths.metadata)

        # operators in this layer do not have CUDA implementation yet.
        # In addition, since the sparse feature keys that we are hashing are
        # typically on CPU originally, it makes sense to have this layer on CPU.
        self.tags.update([Tags.CPU_ONLY])

    def extract_hash_size(self, metadata):
        if metadata.feature_specs and metadata.feature_specs.desired_hash_size:
            return metadata.feature_specs.desired_hash_size
        elif metadata.categorical_limit is not None:
            return metadata.categorical_limit
        else:
            assert False, "desired_hash_size or categorical_limit must be set"

    def add_ops(self, net):
        net.Copy(
            self.input_record.lengths(),
            self.output_schema.lengths()
        )
        if schema.equal_schemas(self.output_schema, IdList):
            input_blob = self.input_record.items()
            output_blob = self.output_schema.items()
        elif schema.equal_schemas(self.output_schema, IdScoreList):
            input_blob = self.input_record.keys()
            output_blob = self.output_schema.keys()
            net.Copy(
                self.input_record.values(),
                self.output_schema.values()
            )
        else:
            raise NotImplementedError()

        if self.use_hashing:
            net.IndexHash(
                input_blob, output_blob, seed=self.seed, modulo=self.modulo
            )
        else:
            if self.use_divide_mod:
                quotient = net.Div([input_blob, self.divisor], [net.NextScopedBlob('quotient')])
                net.Mod(
                    quotient, output_blob, divisor=self.modulo, sign_follow_divisor=True
                )
            else:
                net.Mod(
                    input_blob, output_blob, divisor=self.modulo, sign_follow_divisor=True
                )
