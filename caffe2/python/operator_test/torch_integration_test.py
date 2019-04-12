from __future__ import absolute_import, division, print_function, unicode_literals

from caffe2.python import core, workspace
import torch
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from scipy.stats import norm
import unittest


def generate_rois(roi_counts, im_dims):
    assert len(roi_counts) == len(im_dims)
    all_rois = []
    for i, num_rois in enumerate(roi_counts):
        if num_rois == 0:
            continue
        # [batch_idx, x1, y1, x2, y2]
        rois = np.random.uniform(0, im_dims[i], size=(roi_counts[i], 5)).astype(
            np.float32
        )
        rois[:, 0] = i  # batch_idx
        # Swap (x1, x2) if x1 > x2
        rois[:, 1], rois[:, 3] = (
            np.minimum(rois[:, 1], rois[:, 3]),
            np.maximum(rois[:, 1], rois[:, 3]),
        )
        # Swap (y1, y2) if y1 > y2
        rois[:, 2], rois[:, 4] = (
            np.minimum(rois[:, 2], rois[:, 4]),
            np.maximum(rois[:, 2], rois[:, 4]),
        )
        all_rois.append(rois)
    if len(all_rois) > 0:
        return np.vstack(all_rois)
    return np.empty((0, 5)).astype(np.float32)


def generate_rois_rotated(roi_counts, im_dims):
    rois = generate_rois(roi_counts, im_dims)
    # [batch_id, ctr_x, ctr_y, w, h, angle]
    rotated_rois = np.empty((rois.shape[0], 6)).astype(np.float32)
    rotated_rois[:, 0] = rois[:, 0]  # batch_id
    rotated_rois[:, 1] = (rois[:, 1] + rois[:, 3]) / 2.0  # ctr_x = (x1 + x2) / 2
    rotated_rois[:, 2] = (rois[:, 2] + rois[:, 4]) / 2.0  # ctr_y = (y1 + y2) / 2
    rotated_rois[:, 3] = rois[:, 3] - rois[:, 1] + 1.0  # w = x2 - x1 + 1
    rotated_rois[:, 4] = rois[:, 4] - rois[:, 2] + 1.0  # h = y2 - y1 + 1
    rotated_rois[:, 5] = np.random.uniform(-90.0, 90.0)  # angle in degrees
    return rotated_rois


def create_bbox_transform_inputs(roi_counts, num_classes, rotated):
    batch_size = len(roi_counts)
    total_rois = sum(roi_counts)
    im_dims = np.random.randint(100, 600, batch_size)
    rois = (
        generate_rois_rotated(roi_counts, im_dims)
        if rotated
        else generate_rois(roi_counts, im_dims)
    )
    box_dim = 5 if rotated else 4
    deltas = np.random.randn(total_rois, box_dim * num_classes).astype(np.float32)
    im_info = np.zeros((batch_size, 3)).astype(np.float32)
    im_info[:, 0] = im_dims
    im_info[:, 1] = im_dims
    im_info[:, 2] = 1.0
    return rois, deltas, im_info


class TorchIntegration(hu.HypothesisTestCase):
    @given(
        roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10),
        num_classes=st.integers(1, 10),
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        **hu.gcs_cpu_only
    )
    def test_bbox_transform(
        self,
        roi_counts,
        num_classes,
        rotated,
        angle_bound_on,
        clip_angle_thresh,
        gc,
        dc,
    ):
        """
        Test with rois for multiple images in a batch
        """
        rois, deltas, im_info = create_bbox_transform_inputs(
            roi_counts, num_classes, rotated
        )

        def bbox_transform_ref():
            ref_op = core.CreateOperator(
                "BBoxTransform",
                ["rois", "deltas", "im_info"],
                ["box_out"],
                apply_scale=False,
                rotated=rotated,
                angle_bound_on=angle_bound_on,
                clip_angle_thresh=clip_angle_thresh,
            )
            workspace.FeedBlob("rois", rois)
            workspace.FeedBlob("deltas", deltas)
            workspace.FeedBlob("im_info", im_info)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("box_out")

        box_out = torch.tensor(bbox_transform_ref())
        a, b = torch.ops._caffe2.BBoxTransform(
            torch.tensor(rois),
            torch.tensor(deltas),
            torch.tensor(im_info),
            [1.0, 1.0, 1.0, 1.0],
            False,
            rotated,
            angle_bound_on,
            -90,
            90,
            clip_angle_thresh,
        )

        torch.testing.assert_allclose(box_out, a)

    @given(
        roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10),
        num_classes=st.integers(1, 10),
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        **hu.gcs_cpu_only
    )
    def test_box_with_nms_limits(
        self,
        roi_counts,
        num_classes,
        rotated,
        angle_bound_on,
        clip_angle_thresh,
        gc,
        dc,
    ):
        rotated = False  # FIXME remove this after rotation is supported
        rois, deltas, im_info = create_bbox_transform_inputs(
            roi_counts, num_classes, rotated
        )
        pred_bbox, batch_splits = [
            t.detach().numpy()
            for t in torch.ops._caffe2.BBoxTransform(
                torch.tensor(rois),
                torch.tensor(deltas),
                torch.tensor(im_info),
                [1.0, 1.0, 1.0, 1.0],
                False,
                rotated,
                angle_bound_on,
                -90,
                90,
                clip_angle_thresh,
            )
        ]
        class_prob = np.random.randn(sum(roi_counts), num_classes).astype(np.float32)
        score_thresh = 0.5
        nms_thresh = 0.5
        topk_per_image = sum(roi_counts) / 2

        def box_with_nms_limit_ref():
            input_blobs = ["class_prob", "pred_bbox", "batch_splits"]
            output_blobs = ["score_nms", "bbox_nms", "class_nms", "batch_splits_nms"]
            ref_op = core.CreateOperator(
                "BoxWithNMSLimit",
                input_blobs,
                output_blobs,
                score_thresh=float(score_thresh),
                nms=float(nms_thresh),
                detections_per_im=int(topk_per_image),
                soft_nms_enabled=False,
                soft_nms_method="linear",
                soft_nms_sigma=0.5,
                soft_nms_min_score_thres=0.001,
                rotated=rotated,
            )
            workspace.FeedBlob("class_prob", class_prob)
            workspace.FeedBlob("pred_bbox", pred_bbox)
            workspace.FeedBlob("batch_splits", batch_splits)
            workspace.RunOperatorOnce(ref_op)
            return (workspace.FetchBlob(b) for b in output_blobs)

        output_refs = box_with_nms_limit_ref()
        outputs = torch.ops._caffe2.BoxWithNMSLimit(
            torch.tensor(class_prob),
            torch.tensor(pred_bbox),
            torch.tensor(batch_splits),
            score_thresh=float(score_thresh),
            nms=float(nms_thresh),
            detections_per_im=int(topk_per_image),
            soft_nms_enabled=False,
            soft_nms_method="linear",
            soft_nms_sigma=0.5,
            soft_nms_min_score_thres=0.001,
            rotated=rotated,
        )

        for o, o_ref in zip(outputs, output_refs):
            torch.testing.assert_allclose(o, o_ref)

    @given(
        A=st.integers(min_value=4, max_value=4),
        H=st.integers(min_value=10, max_value=10),
        W=st.integers(min_value=8, max_value=8),
        img_count=st.integers(min_value=3, max_value=3),
    )
    def test_generate_proposals(self, A, H, W, img_count):
        scores = np.ones((img_count, A, H, W)).astype(np.float32)
        bbox_deltas = (
            np.linspace(0, 10, num=img_count * 4 * A * H * W)
            .reshape((img_count, 4 * A, H, W))
            .astype(np.float32)
        )
        im_info = np.ones((img_count, 3)).astype(np.float32) / 10
        anchors = np.ones((A, 4)).astype(np.float32)

        def generate_proposals_ref():
            ref_op = core.CreateOperator(
                "GenerateProposals",
                ["scores", "bbox_deltas", "im_info", "anchors"],
                ["rois", "rois_probs"],
                spatial_scale=2.0,
            )
            workspace.FeedBlob("scores", scores)
            workspace.FeedBlob("bbox_deltas", bbox_deltas)
            workspace.FeedBlob("im_info", im_info)
            workspace.FeedBlob("anchors", anchors)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("rois"), workspace.FetchBlob("rois_probs")

        rois, rois_probs = generate_proposals_ref()
        rois = torch.tensor(rois)
        rois_probs = torch.tensor(rois_probs)
        a, b = torch.ops._caffe2.GenerateProposals(
            torch.tensor(scores),
            torch.tensor(bbox_deltas),
            torch.tensor(im_info),
            torch.tensor(anchors),
            2.0,
            6000,
            300,
            0.7,
            16,
            True,
            -90,
            90,
            1.0,
        )
        torch.testing.assert_allclose(rois, a)
        torch.testing.assert_allclose(rois_probs, b)

    @given(
        bsz=st.integers(1, 5),
        seq_lens=st.integers(1, 6),
        emb_lens=st.integers(5, 10),
        hidden_size=st.integers(3, 7),
        num_layers=st.integers(1, 4),
        has_biases=st.booleans(),
        is_bidirectional=st.booleans(),
        batch_first=st.booleans(),
    )
    def test_inference_lstm(
        self,
        bsz,
        seq_lens,
        emb_lens,
        hidden_size,
        num_layers,
        has_biases,
        is_bidirectional,
        batch_first,
    ):
        num_directions = 2 if is_bidirectional else 1
        hx = np.zeros((num_layers * num_directions, bsz, hidden_size), dtype=np.float32)

        if batch_first:
            inputs = np.random.randn(bsz, seq_lens, emb_lens).astype(np.float32)
        else:
            inputs = np.random.randn(seq_lens, bsz, emb_lens).astype(np.float32)

        torch_lstm = torch.nn.LSTM(
            emb_lens,
            hidden_size,
            batch_first=batch_first,
            bidirectional=is_bidirectional,
            bias=has_biases,
            num_layers=num_layers,
        )

        def inference_lstm_ref():
            input_names = ["inputs", "hidden_0", "hidden_1"]
            workspace.FeedBlob("inputs", inputs)
            workspace.FeedBlob("hidden_0", hx)
            workspace.FeedBlob("hidden_1", hx)
            for i, param in enumerate(torch_lstm._flat_weights):
                input_names.append("param_{}".format(i))
                workspace.FeedBlob("param_{}".format(i), param.detach().numpy())

            ref_op = core.CreateOperator(
                "InferenceLSTM",
                input_names,
                ["output", "hidden", "cell"],
                num_layers=num_layers,
                has_biases=has_biases,
                batch_first=batch_first,
                bidirectional=is_bidirectional,
            )
            workspace.RunOperatorOnce(ref_op)
            return (
                workspace.FetchBlob("output"),
                workspace.FetchBlob("hidden"),
                workspace.FetchBlob("cell")
            )

        output, hidden, cell = inference_lstm_ref()
        output = torch.tensor(output)
        hidden = torch.tensor(hidden)
        cell = torch.tensor(cell)
        lstm_in = [
            torch.from_numpy(inputs),
            torch.from_numpy(hx),
            torch.from_numpy(hx),
        ] + [param.detach() for param in torch_lstm._flat_weights]

        a, b, c = torch.ops._caffe2.InferenceLSTM(
            lstm_in, num_layers, has_biases, batch_first, is_bidirectional
        )
        torch.testing.assert_allclose(output, a)
        torch.testing.assert_allclose(hidden, b)
        torch.testing.assert_allclose(cell, c)

    # Test case is using workspace.has_cuda_support and not workspace.has_gpu_support
    # to exclude it from HIP because tensor interop doesn't work for HIP tensors yet
    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    @given(
        A=st.integers(min_value=4, max_value=4),
        H=st.integers(min_value=10, max_value=10),
        W=st.integers(min_value=8, max_value=8),
        img_count=st.integers(min_value=3, max_value=3),
    )
    def test_generate_proposals_cuda(self, A, H, W, img_count):
        scores = np.ones((img_count, A, H, W)).astype(np.float32)
        bbox_deltas = (
            np.linspace(0, 10, num=img_count * 4 * A * H * W)
            .reshape((img_count, 4 * A, H, W))
            .astype(np.float32)
        )
        im_info = np.ones((img_count, 3)).astype(np.float32) / 10
        anchors = np.ones((A, 4)).astype(np.float32)

        def generate_proposals_ref():
            ref_op = core.CreateOperator(
                "GenerateProposals",
                ["scores", "bbox_deltas", "im_info", "anchors"],
                ["rois", "rois_probs"],
                spatial_scale=2.0,
            )
            workspace.FeedBlob("scores", scores)
            workspace.FeedBlob("bbox_deltas", bbox_deltas)
            workspace.FeedBlob("im_info", im_info)
            workspace.FeedBlob("anchors", anchors)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("rois"), workspace.FetchBlob("rois_probs")

        rois, rois_probs = generate_proposals_ref()
        rois = torch.tensor(rois)
        rois_probs = torch.tensor(rois_probs)
        a, b = torch.ops._caffe2.GenerateProposals(
            torch.tensor(scores).cuda(),
            torch.tensor(bbox_deltas).cuda(),
            torch.tensor(im_info).cuda(),
            torch.tensor(anchors).cuda(),
            2.0,
            6000,
            300,
            0.7,
            16,
            True,
            -90,
            90,
            1.0,
        )
        torch.testing.assert_allclose(rois, a.cpu())
        torch.testing.assert_allclose(rois_probs, b.cpu())

    @given(
        N=st.integers(min_value=1, max_value=2),
        C=st.integers(min_value=4, max_value=4),
        H=st.integers(min_value=10, max_value=10),
        W=st.integers(min_value=8, max_value=8),
    )
    def _test_roi_align(self, N, C, H, W, device):
        def rand_roi():
            return np.array(
                [
                    float(int(N * np.random.rand())),
                    0.5 * np.random.rand() * W,
                    0.5 * np.random.rand() * H,
                    (0.5 + 0.5 * np.random.rand()) * W,
                    (0.5 + 0.5 * np.random.rand()) * H,
                ]
            ).astype(np.float32)

        feature = np.random.randn(N, C, H, W).astype(np.float32)
        rois = np.array([rand_roi() for _ in range(10)])

        def roi_align_ref(_feature, _rois):
            ref_op = core.CreateOperator(
                "RoIAlign",
                ["feature", "rois"],
                ["roi_feature"],
                spatial_scale=1.0,
                pooled_h=3,
                pooled_w=3,
                sampling_ratio=0,
            )
            workspace.FeedBlob("feature", _feature)
            workspace.FeedBlob("rois", _rois)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("roi_feature")

        roi_feature_ref = roi_align_ref(feature, rois)
        roi_feature = torch.ops._caffe2.RoIAlign(
            torch.Tensor(feature).to(device),
            torch.Tensor(rois).to(device),
            order="NCHW",
            spatial_scale=1.0,
            pooled_h=3,
            pooled_w=3,
            sampling_ratio=0,
        )
        torch.testing.assert_allclose(roi_feature_ref, roi_feature.cpu())

    def test_roi_align_cpu(self):
        self._test_roi_align(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_roi_align_cuda(self):
        self._test_roi_align(device="cuda")

    @given(X=hu.tensor(),
           fast_gelu=st.booleans())
    def _test_gelu_op(self, X, fast_gelu, device):
        def _gelu_ref(_X):
            return (_X * norm.cdf(_X).astype(np.float32), )
        expected_output, = _gelu_ref(X)
        actual_output = torch.ops._caffe2.Gelu(torch.tensor(X), fast_gelu)

        rtol = 1e-3 if fast_gelu else 1e-4
        atol = 1e-5
        torch.testing.assert_allclose(
            expected_output, actual_output.cpu(), rtol=rtol, atol=atol)

    def test_gelu_op(self):
        self._test_gelu_op(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_gelu_op_cuda(self):
        self._test_gelu_op(device="cuda")


if __name__ == '__main__':
    unittest.main()
