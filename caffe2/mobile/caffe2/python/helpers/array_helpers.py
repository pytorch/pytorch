## @package arra_helpers
# Module caffe2.python.helpers.array_helpers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def concat(model, blobs_in, blob_out, **kwargs):
    """Depth Concat."""
    if kwargs.get('order') and kwargs.get('axis'):
        # The backend throws an error if both are given
        kwargs.pop('order')

    return model.net.Concat(
        blobs_in,
        [blob_out, "_" + blob_out + "_concat_dims"],
        **kwargs
    )[0]


def depth_concat(model, blobs_in, blob_out, **kwargs):
    """The old depth concat function - we should move to use concat."""
    print("DepthConcat is deprecated. use Concat instead.")
    return concat(blobs_in, blob_out, **kwargs)
