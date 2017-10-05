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

## @package tags
# Module caffe2.python.layers.tags
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six

from caffe2.python import context


@context.define_context(allow_default=True)
class TagContext(object):
    """
    Scope driven way to provide tags to the layers.
    """

    def __init__(self, tags=None):
        # Tags is expected to be list to keep order of adding/removing things
        self.tags = tags or []

    def add_tags(self, tags):
        self.tags.extend(tags)

    def remove_tags(self, tags):
        assert self.tags[-len(tags):] == tags
        self.tags = self.tags[:-len(tags)]


class Tags(object):
    # TODO(amalevich): Tags might need to live in their own contexts, add this
    # split later
    EXCLUDE_FROM_TRAIN = 'exclude_from_train'
    EXCLUDE_FROM_EVAL = 'exclude_from_eval'
    EXCLUDE_FROM_PREDICTION = 'exclude_from_prediction'
    EXCLUDE_FROM_ACCUMULATE_PRED = 'exclude_from_accumulate_pred'
    PREPROCESSING = 'preprocessing'
    HANDLE_AS_SPARSE_LAYER = 'handle_as_sparse_layer'
    GRADIENT_FROM_PS = 'gradient_from_ps'
    PREFER_GPU = 'prefer_gpu'
    CPU_ONLY = 'cpu_only'

    # The following two tags are hints to **distributed training framework**.
    # The first tag is to determine a layer contains a sparse parameter and the
    # parameter should be sharded and operators on those parameters should be
    # done on distributed parameter servers. The second tag is to determine a
    # layer contains a sparse parameters among others, and that the parameters
    # should not be sharded (i.e. should be placed together on a node)
    SPARSE_SHARDED = 'sparse_sharded'
    SPARSE_DONT_SHARD = 'sparse_dont_shard'

    # In certain cases we want to have different schema for training and
    # prediction, as an example in prediction we might need to have only
    # subset of ids present in the orignal schema. This tag is one of the ways
    # to mark operators that will be removed from prediction and should
    # override schema for predictors.
    PREDICTION_SCHEMA = 'prediction_schema'

    def __init__(self, tags):
        if not isinstance(tags, list):
            tags = [tags]
        self.tags = tags

    def __enter__(self):
        TagContext.current().add_tags(self.tags)
        return self

    def __exit__(self, type, value, traceback):
        TagContext.current().remove_tags(self.tags)

    def __call__(self, func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


Tags.TRAIN_ONLY = [Tags.EXCLUDE_FROM_PREDICTION, Tags.EXCLUDE_FROM_EVAL,
                   Tags.EXCLUDE_FROM_ACCUMULATE_PRED]
Tags.EVAL_ONLY = [Tags.EXCLUDE_FROM_PREDICTION, Tags.EXCLUDE_FROM_TRAIN,
                  Tags.EXCLUDE_FROM_ACCUMULATE_PRED]
Tags.PREDICTION_ONLY = [Tags.EXCLUDE_FROM_TRAIN, Tags.EXCLUDE_FROM_EVAL,
                        Tags.EXCLUDE_FROM_ACCUMULATE_PRED]
