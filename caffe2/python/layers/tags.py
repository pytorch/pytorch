## @package tags
# Module caffe2.python.layers.tags
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
    TRAIN_ONLY = 'train_only'
    PREPROCESSING = 'preprocessing'

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
