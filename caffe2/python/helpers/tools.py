## @package tools
# Module caffe2.python.helpers.tools
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def image_input(
    model, blob_in, blob_out, order="NCHW", use_gpu_transform=False, **kwargs
):
    assert 'is_test' in kwargs, "Argument 'is_test' is required"
    if order == "NCHW":
        if (use_gpu_transform):
            kwargs['use_gpu_transform'] = 1 if use_gpu_transform else 0
            # GPU transform will handle NHWC -> NCHW
            outputs = model.net.ImageInput(blob_in, blob_out, **kwargs)
            pass
        else:
            outputs = model.net.ImageInput(
                blob_in, [blob_out[0] + '_nhwc'] + blob_out[1:], **kwargs
            )
            outputs_list = list(outputs)
            outputs_list[0] = model.net.NHWC2NCHW(outputs_list[0], blob_out[0])
            outputs = tuple(outputs_list)
    else:
        outputs = model.net.ImageInput(blob_in, blob_out, **kwargs)
    return outputs


def video_input(model, blob_in, blob_out, **kwargs):
    # size of outputs can vary depending on kwargs
    outputs = model.net.VideoInput(blob_in, blob_out, **kwargs)
    return outputs
