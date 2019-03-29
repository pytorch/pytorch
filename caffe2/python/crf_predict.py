from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from caffe2.python.crf import CRFWithLoss


def crf_update_predictions(model, crf_with_loss, classes):
    return apply_crf(
        model.param_init_net,
        model.net,
        crf_with_loss.transitions,
        classes,
        crf_with_loss.num_classes,
    )


def apply_crf(init_net, net, transitions, predictions, num_classes):
    padded_classes = CRFWithLoss.pad_predictions(
        predictions, init_net, net, num_classes
    )
    bestPath = net.ViterbiPath([padded_classes, transitions])
    new_padded_classes = net.SwapBestPath([padded_classes, bestPath])
    # Revert the effect of pad_predictions by removing the last two rows and
    # the last two columns
    new_classes = net.RemovePadding(
        [new_padded_classes], padding_width=1, end_padding_width=1
    )
    slice_starts = np.array([0, 0]).astype(np.int32)
    slice_ends = np.array([-1, -3]).astype(np.int32)
    slice_starts = net.GivenTensorIntFill([], shape=[2], values=slice_starts)
    slice_ends = net.GivenTensorIntFill([], shape=[2], values=slice_ends)
    new_classes = net.Slice([new_classes, slice_starts, slice_ends])
    return new_classes
