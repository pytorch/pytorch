from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2

ROI_CANONICAL_SCALE = 224  # default: 224
ROI_CANONICAL_LEVEL = 4  # default: 4
ROI_MAX_LEVEL = 5  # default: 5
ROI_MIN_LEVEL = 2  # default: 2
RPN_MAX_LEVEL = 6  # default: 6
RPN_MIN_LEVEL = 2  # default: 2
RPN_POST_NMS_TOP_N = 2000  # default: 2000


#
# Should match original Detectron code at
# https://github.com/facebookresearch/Detectron/blob/master/lib/ops/collect_and_distribute_fpn_rpn_proposals.py
#

def boxes_area(boxes):
    """Compute the area of an array of boxes."""
    w = (boxes[:, 2] - boxes[:, 0] + 1)
    h = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = w * h
    assert np.all(areas >= 0), 'Negative areas founds'
    return areas


def map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    s = np.sqrt(boxes_area(rois))
    s0 = ROI_CANONICAL_SCALE
    lvl0 = ROI_CANONICAL_LEVEL

    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls


def collect(inputs):
    post_nms_topN = RPN_POST_NMS_TOP_N
    k_max = RPN_MAX_LEVEL
    k_min = RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]

    # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
    # Combine predictions across all levels and retain the top scoring
    #
    # equivalent to Detectron code
    #   rois = np.concatenate([blob.data for blob in roi_inputs])
    #   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
    rois = np.concatenate(roi_inputs)
    scores = np.concatenate(score_inputs).squeeze()
    assert rois.shape[0] == scores.shape[0]
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]
    return rois


def distribute(rois, _, outputs):
    """To understand the output blob order see return value of
    roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    # equivalent to Detectron code
    #   lvl_min = cfg.FPN.ROI_MIN_LEVEL
    #   lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvl_min = ROI_MIN_LEVEL
    lvl_max = ROI_MAX_LEVEL
    lvls = map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    # equivalent to Detectron code
    #   outputs[0].reshape(rois.shape)
    #   outputs[0].data[...] = rois
    outputs[0] = rois

    # Create new roi blobs for each FPN level
    # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        # equivalent to Detectron code
        #   outputs[output_idx + 1].reshape(blob_roi_level.shape)
        #   outputs[output_idx + 1].data[...] = blob_roi_level
        outputs[output_idx + 1] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    # equivalent to Detectron code
    #   py_op_copy_blob(
    #       rois_idx_restore.astype(np.int32), outputs[-1])
    outputs[-1] = rois_idx_restore.astype(np.int32)


def collect_and_distribute_fpn_rpn_ref(*inputs):
    assert inputs
    num_rpn_lvls = RPN_MAX_LEVEL - RPN_MIN_LEVEL + 1
    assert len(inputs) == 2 * num_rpn_lvls
    N = inputs[0].shape[0]
    for i in range(num_rpn_lvls):
        assert len(inputs[i].shape) == 2
        assert inputs[i].shape[0] == N
        assert inputs[i].shape[1] == 5
    for i in range(num_rpn_lvls, 2 * num_rpn_lvls):
        assert len(inputs[i].shape) == 1
        assert inputs[i].shape[0] == N

    num_roi_lvls = ROI_MAX_LEVEL - ROI_MIN_LEVEL + 1
    outputs = (num_roi_lvls + 2) * [None]
    rois = collect(inputs)
    distribute(rois, None, outputs)

    return outputs


class TestCollectAndDistributeFpnRpnProposals(hu.HypothesisTestCase):
    def run_on_device(self, device_opts):
        np.random.seed(0)

        proposal_count = 5000
        input_names = []
        inputs = []

        for lvl in range(RPN_MIN_LEVEL, RPN_MAX_LEVEL + 1):
            rpn_roi = (
                ROI_CANONICAL_SCALE *
                np.random.rand(proposal_count, 5).astype(np.float32)
            )
            for i in range(proposal_count):
                # Make RoIs have positive area, since they
                # are in the format [[batch_idx, x0, y0, x1, y2], ...]
                rpn_roi[i][3] += rpn_roi[i][1]
                rpn_roi[i][4] += rpn_roi[i][2]
            input_names.append('rpn_rois_fpn{}'.format(lvl))
            inputs.append(rpn_roi)
        for lvl in range(RPN_MIN_LEVEL, RPN_MAX_LEVEL + 1):
            rpn_roi_score = np.random.rand(proposal_count).astype(np.float32)
            input_names.append('rpn_roi_probs_fpn{}'.format(lvl))
            inputs.append(rpn_roi_score)

        output_names = [
            'rois',
        ]
        for lvl in range(ROI_MIN_LEVEL, ROI_MAX_LEVEL + 1):
            output_names.append('rois_fpn{}'.format(lvl))
        output_names.append('rois_idx_restore')

        op = core.CreateOperator(
            'CollectAndDistributeFpnRpnProposals',
            input_names,
            output_names,
            arg=[
                utils.MakeArgument("roi_canonical_scale", ROI_CANONICAL_SCALE),
                utils.MakeArgument("roi_canonical_level", ROI_CANONICAL_LEVEL),
                utils.MakeArgument("roi_max_level", ROI_MAX_LEVEL),
                utils.MakeArgument("roi_min_level", ROI_MIN_LEVEL),
                utils.MakeArgument("rpn_max_level", RPN_MAX_LEVEL),
                utils.MakeArgument("rpn_min_level", RPN_MIN_LEVEL),
                utils.MakeArgument("post_nms_topN", RPN_POST_NMS_TOP_N),
            ],
            device_option=device_opts)

        self.assertReferenceChecks(
            device_option=device_opts,
            op=op,
            inputs=inputs,
            reference=collect_and_distribute_fpn_rpn_ref,
        )

    def test_cpu(self):
        device_opts_cpu = caffe2_pb2.DeviceOption()
        self.run_on_device(device_opts_cpu)


if __name__ == "__main__":
    unittest.main()
