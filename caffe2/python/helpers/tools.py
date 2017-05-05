## @package tools
# Module caffe2.python.helpers.tools
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def image_input(
    model, blob_in, blob_out, order="NCHW", use_gpu_transform=False, **kwargs
):
    if order == "NCHW":
        if (use_gpu_transform):
            kwargs['use_gpu_transform'] = 1 if use_gpu_transform else 0
            # GPU transform will handle NHWC -> NCHW
            data, label = model.net.ImageInput(
                blob_in, [blob_out[0], blob_out[1]], **kwargs
            )
            pass
        else:
            data, label = model.net.ImageInput(
                blob_in, [blob_out[0] + '_nhwc', blob_out[1]], **kwargs
            )
            data = model.net.NHWC2NCHW(data, blob_out[0])
    else:
        data, label = model.net.ImageInput(blob_in, blob_out, **kwargs)
    return data, label


def video_input(model, blob_in, blob_out, **kwargs):
    data, label = model.net.VideoInput(blob_in, blob_out, **kwargs)
    return data, label
