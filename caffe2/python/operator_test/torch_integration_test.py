

import struct
import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import torch
from caffe2.python import core, workspace
from hypothesis import given, settings
from scipy.stats import norm


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


# Eigen/Python round 0.5 away from 0, Numpy rounds to even
round_to_nearest = np.vectorize(round)


def bytes_to_floats(byte_matrix):
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float32)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = struct.unpack("f", bytearray(byte_values))
    return floats


def floats_to_bytes(floats):
    byte_matrix = np.empty([np.shape(floats)[0], 4], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float32), (value, floats)
        as_bytes = struct.pack("f", value)
        # In Python3 bytes will be a list of int, in Python2 a list of string
        if isinstance(as_bytes[0], int):
            byte_matrix[i] = list(as_bytes)
        else:
            byte_matrix[i] = [ord(i) for i in as_bytes]
    return byte_matrix


def fused_rowwise_8bit_quantize_reference(data):
    minimum = np.min(data, axis=1, keepdims=True)
    maximum = np.max(data, axis=1, keepdims=True)
    span = maximum - minimum
    bias = minimum
    scale = span / 255.0
    inverse_scale = 255.0 / (span + 1e-8)
    quantized_data = round_to_nearest((data - bias) * inverse_scale)
    scale_bytes = floats_to_bytes(scale.reshape(-1))
    bias_bytes = floats_to_bytes(bias.reshape(-1))
    return np.concatenate([quantized_data, scale_bytes, bias_bytes], axis=1)


def fused_rowwise_8bit_quantize_dequantize_reference(data):
    fused_quantized = fused_rowwise_8bit_quantize_reference(data)
    scale = bytes_to_floats(fused_quantized[:, -8:-4].astype(np.uint8))
    bias = bytes_to_floats(fused_quantized[:, -4:].astype(np.uint8))
    quantized_data = fused_quantized[:, :-8]
    return quantized_data * scale + bias


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
            legacy_plus_one=True,
        )

        torch.testing.assert_allclose(box_out, a)

    @given(
        roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10),
        num_classes=st.integers(1, 10),
        rotated=st.booleans(),
        angle_bound_on=st.booleans(),
        clip_angle_thresh=st.sampled_from([-1.0, 1.0]),
        batch_splits_dtype=st.sampled_from([torch.float32, torch.int32]),
        **hu.gcs_cpu_only
    )
    def test_box_with_nms_limits(
        self,
        roi_counts,
        num_classes,
        rotated,
        angle_bound_on,
        clip_angle_thresh,
        batch_splits_dtype,
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
                legacy_plus_one=True,
            )
        ]
        class_prob = np.random.randn(sum(roi_counts), num_classes).astype(np.float32)
        score_thresh = 0.5
        nms_thresh = 0.5
        topk_per_image = sum(roi_counts) / 2

        def box_with_nms_limit_ref():
            input_blobs = ["class_prob", "pred_bbox", "batch_splits"]
            output_blobs = [
                "score_nms",
                "bbox_nms",
                "class_nms",
                "batch_splits_nms",
                "keeps_nms",
                "keeps_size_nms",
            ]
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
            torch.tensor(batch_splits, dtype=batch_splits_dtype),
            score_thresh=float(score_thresh),
            nms=float(nms_thresh),
            detections_per_im=int(topk_per_image),
            soft_nms_enabled=False,
            soft_nms_method="linear",
            soft_nms_sigma=0.5,
            soft_nms_min_score_thres=0.001,
            rotated=rotated,
            cls_agnostic_bbox_reg=False,
            input_boxes_include_bg_cls=True,
            output_classes_include_bg_cls=True,
            legacy_plus_one=True,
        )

        for o, o_ref in zip(outputs, output_refs):
            torch.testing.assert_allclose(o, o_ref)

    @given(
        dim_1=st.integers(min_value=10, max_value=10),
        dim_2=st.integers(min_value=3, max_value=3),
        dim_3=st.integers(min_value=2, max_value=2),
    )
    def test_sparse_to_dense_mask(self, dim_1, dim_2, dim_3):
        indices = np.array([i + 1 for i in range(dim_1)]).astype(np.int32)
        values = np.random.rand(dim_1, dim_2, dim_3).astype(np.float32)
        default_value = np.zeros((dim_2, dim_3)).astype(np.float32)
        mask = [2, 4, 9]

        def sparse_to_dense_mask_ref(return_presence_mask=False):
            ref_op = core.CreateOperator(
                "SparseToDenseMask",
                ["indices", "values", "default_value"],
                ["output", "presence_mask"],
                mask=mask,
                return_presence_mask=return_presence_mask,
            )
            workspace.FeedBlob("indices", indices)
            workspace.FeedBlob("values", values)
            workspace.FeedBlob("default_value", default_value)
            workspace.RunOperatorOnce(ref_op)

            if return_presence_mask:
                return (
                    workspace.FetchBlob("output"),
                    workspace.FetchBlob("presence_mask"),
                )

            return workspace.FetchBlob("output")

        # Testing return_presence_mask = False
        output = sparse_to_dense_mask_ref()
        output = torch.tensor(output)

        a, _ = torch.ops._caffe2.SparseToDenseMask(
            torch.tensor(indices),
            torch.tensor(values),
            torch.tensor(default_value),
            None,
            mask=mask,
        )

        torch.testing.assert_allclose(output, a)

        # Testing return_presence_mask = True
        output, presence_mask = sparse_to_dense_mask_ref(return_presence_mask=True)
        output = torch.tensor(output)
        presence_mask = torch.tensor(presence_mask)

        a, b = torch.ops._caffe2.SparseToDenseMask(
            torch.tensor(indices),
            torch.tensor(values),
            torch.tensor(default_value),
            None,
            mask=mask,
            return_presence_mask=True,
        )

        torch.testing.assert_allclose(output, a)
        torch.testing.assert_allclose(presence_mask, b)

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
            legacy_plus_one=True,
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
                workspace.FetchBlob("cell"),
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
            legacy_plus_one=True,
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
            torch.tensor(feature).to(device),
            torch.tensor(rois).to(device),
            order="NCHW",
            spatial_scale=1.0,
            pooled_h=3,
            pooled_w=3,
            sampling_ratio=0,
            aligned=False,
        )
        torch.testing.assert_allclose(roi_feature_ref, roi_feature.cpu())

    def test_roi_align_cpu(self):
        self._test_roi_align(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_roi_align_cuda(self):
        self._test_roi_align(device="cuda")

    @given(
        N=st.integers(min_value=1, max_value=2),
        C=st.integers(min_value=4, max_value=4),
        H=st.integers(min_value=10, max_value=10),
        W=st.integers(min_value=8, max_value=8),
    )
    def _test_roi_align_rotated(self, N, C, H, W, device):
        def rand_rotated_roi():
            return np.array(
                [
                    float(int(N * np.random.rand())),
                    np.random.rand() * W,
                    np.random.rand() * H,
                    np.random.rand() * W,
                    np.random.rand() * H,
                    np.random.rand() * 360 - 180,
                ]
            ).astype(np.float32)

        feature = np.random.randn(N, C, H, W).astype(np.float32)
        rois = np.array([rand_rotated_roi() for _ in range(10)])

        def roi_align_ref(_feature, _rois):
            ref_op = core.CreateOperator(
                "RoIAlignRotated",
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
        roi_feature = torch.ops._caffe2.RoIAlignRotated(
            torch.tensor(feature).to(device),
            torch.tensor(rois).to(device),
            order="NCHW",
            spatial_scale=1.0,
            pooled_h=3,
            pooled_w=3,
            sampling_ratio=0,
            aligned=False,
        )
        torch.testing.assert_allclose(roi_feature_ref, roi_feature.cpu())

    def test_roi_align_rotated_cpu(self):
        self._test_roi_align_rotated(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_roi_align_rotated_cuda(self):
        self._test_roi_align_rotated(device="cuda")

    @given(roi_counts=st.lists(st.integers(0, 5), min_size=1, max_size=10))
    def test_collect_and_distribute_fpn_rpn_proposals_op(self, roi_counts):
        batch_size = len(roi_counts)
        im_dims = np.random.randint(100, 600, batch_size)
        rpn_rois_and_scores = []
        for i in range(5):
            rpn_rois_and_scores.append(torch.tensor(generate_rois(roi_counts, im_dims)))
        for i in range(5):
            rpn_rois_and_scores.append(torch.rand(sum(roi_counts)))

        rois = torch.ops._caffe2.CollectRpnProposals(
            rpn_rois_and_scores,
            rpn_max_level=6,
            rpn_min_level=2,
            rpn_post_nms_topN=sum(roi_counts),
        )
        fpn_outputs = torch.ops._caffe2.DistributeFpnProposals(
            rois,
            roi_canonical_scale=224,
            roi_canonical_level=4,
            roi_max_level=5,
            roi_min_level=2,
            legacy_plus_one=True,
        )

        all_outputs = torch.ops._caffe2.CollectAndDistributeFpnRpnProposals(
            rpn_rois_and_scores,
            roi_canonical_scale=224,
            roi_canonical_level=4,
            roi_max_level=5,
            roi_min_level=2,
            rpn_max_level=6,
            rpn_min_level=2,
            rpn_post_nms_topN=sum(roi_counts),
            legacy_plus_one=True,
        )

        rois_fpn_list = fpn_outputs[:-1]
        rois_idx_restore_int32 = fpn_outputs[-1]

        # [rois] + fpn_outputs should be equal to all_outputs
        torch.testing.assert_allclose(rois, all_outputs[0])
        for x, y in zip(fpn_outputs, all_outputs[1:]):
            torch.testing.assert_allclose(x, y)

    @given(X=hu.tensor(), fast_gelu=st.booleans())
    def _test_gelu_op(self, X, fast_gelu, device):
        def _gelu_ref(_X):
            return (_X * norm.cdf(_X).astype(np.float32),)

        (expected_output,) = _gelu_ref(X)
        actual_output = torch.ops._caffe2.Gelu(torch.tensor(X), fast_gelu)

        rtol = 1e-3 if fast_gelu else 1e-4
        atol = 1e-5
        torch.testing.assert_allclose(
            expected_output, actual_output.cpu(), rtol=rtol, atol=atol
        )

    def test_gelu_op(self):
        self._test_gelu_op(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_gelu_op_cuda(self):
        self._test_gelu_op(device="cuda")

    @given(
        inputs=hu.lengths_tensor(
            dtype=np.float32, min_value=1, max_value=5, allow_empty=True
        )
    )
    def _test_lengths_op(self, inputs, ref_op_name, torch_op, device):
        data, lengths = inputs

        def _lengths_ref(X, Y):
            ref_op = core.CreateOperator(ref_op_name, ["X", "Y"], "out")
            workspace.FeedBlob("X", X)
            workspace.FeedBlob("Y", Y)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("out")

        expected_output = _lengths_ref(data, lengths)
        actual_output = torch_op(
            torch.tensor(data), torch.tensor(lengths, dtype=torch.int32)
        )

        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def _test_lengths_sum_op(self, device):
        self._test_lengths_op("LengthsSum", torch.ops._caffe2.LengthsSum, device)

    def test_lengths_sum_op(self):
        self._test_lengths_sum_op(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_lengths_sum_op_cuda(self):
        self._test_lengths_sum_op(device="cuda")

    def _test_lengths_mean_op(self, device):
        self._test_lengths_op("LengthsMean", torch.ops._caffe2.LengthsMean, device)

    def test_lengths_mean_op(self):
        self._test_lengths_mean_op(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_lengths_mean_op_cuda(self):
        self._test_lengths_mean_op(device="cuda")

    def _test_lengths_max_op(self, device):
        self._test_lengths_op("LengthsMax", torch.ops._caffe2.LengthsMax, device)

    def test_lengths_max_op(self):
        self._test_lengths_max_op(device="cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_lengths_max_op_cuda(self):
        self._test_lengths_max_op(device="cuda")

    def _test_resize_nearest_op(self, device):
        data = np.random.rand(1, 2, 3, 4).astype(np.float32)

        def _resize_nearest_ref(X):
            ref_op = core.CreateOperator(
                "ResizeNearest",
                ["X"],
                ["Y"],
                width_scale=2.0,
                height_scale=1.5,
                order="NCHW",
            )
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = _resize_nearest_ref(data)
        actual_output = torch.ops._caffe2.ResizeNearest(
            torch.tensor(data).to(device),
            order="NCHW",
            width_scale=2.0,
            height_scale=1.5,
        )

        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def test_resize_nearest_op_cpu(self):
        return self._test_resize_nearest_op("cpu")

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_resize_nearest_op_cuda(self):
        return self._test_resize_nearest_op("cuda")

    @given(input_data=hu.tensor(min_dim=2, max_dim=2))
    def test_Fused8BitRowwiseQuantizedToFloat(self, input_data):
        QuantizeOp = core.CreateOperator(
            "FloatToFused8BitRowwiseQuantized", ["input_data"], ["quantized_data"]
        )

        workspace.FeedBlob("input_data", input_data)
        workspace.RunOperatorOnce(QuantizeOp)

        quantized_data = workspace.FetchBlob("quantized_data")

        dequantized_data = torch.ops._caffe2.Fused8BitRowwiseQuantizedToFloat(
            torch.tensor(quantized_data)
        )

        reference = fused_rowwise_8bit_quantize_dequantize_reference(input_data)
        np.testing.assert_array_almost_equal(dequantized_data.numpy(), reference)

    @given(binary_input=st.booleans())
    def test_piecewise_linear_op(self, binary_input):
        if binary_input:
            num_dims = 1
        else:
            num_dims = 3
        data = np.random.rand(1024, num_dims).astype(np.float32)
        slopes = np.zeros(4 * num_dims).astype(np.float32)
        bounds = np.sort(
            np.random.rand(5, num_dims).astype(np.float32), axis=0
        ).flatten("F")
        intercepts = np.random.rand(4 * num_dims).astype(np.float32)

        def _piecewise_linear_ref(X):
            ref_op = core.CreateOperator(
                "PiecewiseLinearTransform",
                ["data", "bounds", "slopes", "intercepts"],
                ["calibrated"],
                binary=binary_input,
            )
            workspace.FeedBlob("data", X)
            workspace.FeedBlob("bounds", bounds)
            workspace.FeedBlob("slopes", slopes)
            workspace.FeedBlob("intercepts", intercepts)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("calibrated")

        expected_output = _piecewise_linear_ref(data)
        actual_output = torch.ops._caffe2.PiecewiseLinearTransform(
            torch.tensor(data),
            bounds.tolist(),
            slopes.tolist(),
            intercepts.tolist(),
            binary_input,
        )

        torch.testing.assert_allclose(torch.tensor(expected_output), actual_output)

    def test_alias_with_name_is_in_place(self):
        device = "cuda" if workspace.has_cuda_support else "cpu"
        x = torch.tensor([3., 42.]).to(device=device)
        y = torch.ops._caffe2.AliasWithName(x, "new_name")
        x[1] = 6
        torch.testing.assert_allclose(x, torch.tensor([3., 6.]).to(device=device))
        # y should also change because y is alias of x
        torch.testing.assert_allclose(y, torch.tensor([3., 6.]).to(device=device))

    @unittest.skipIf(not workspace.has_cuda_support, "No cuda support")
    def test_copy_between_cpu_and_gpu(self):
        x_cpu_ref = torch.tensor([1., 2., 3.])
        x_gpu_ref = x_cpu_ref.to("cuda")

        x_gpu = torch.ops._caffe2.CopyCPUToGPU(x_cpu_ref)
        torch.testing.assert_allclose(x_gpu, x_gpu_ref)
        x_cpu = torch.ops._caffe2.CopyGPUToCPU(x_gpu)
        torch.testing.assert_allclose(x_cpu, x_cpu_ref)

    def test_index_hash_op(self):
        data = np.random.randint(low=0, high=1000, size=(4, 4, 4))

        def _index_hash_ref(X):
            ref_op = core.CreateOperator("IndexHash", ["X"], ["Y"], seed=0, modulo=100)
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = _index_hash_ref(data)
        actual_output = torch.ops._caffe2.IndexHash(
            torch.tensor(data), seed=0, modulo=100
        )

        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def test_bucketize_op(self):
        data = np.random.rand(8, 10).astype(np.float32) * 1000
        boundaries = np.array([1, 10, 100, 1000, 100000]).astype(np.float32)

        def _bucketize_ref(X):
            ref_op = core.CreateOperator(
                "Bucketize", ["X"], ["Y"], boundaries=boundaries
            )
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = _bucketize_ref(data)
        actual_output = torch.ops._caffe2.Bucketize(torch.tensor(data), boundaries)
        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    @given(X=hu.tensor(), eps=st.floats(min_value=1e-4, max_value=1e-2))
    def test_logit(self, X, eps):
        def ref(X, eps):
            ref_op = core.CreateOperator("Logit", ["X"], ["Y"], eps=eps)
            workspace.FeedBlob("X", X)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = ref(X, eps)
        actual_output = torch.ops._caffe2.Logit(torch.tensor(X), eps)
        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def test_percentile(self):
        original_values = np.array([[3.0, 5.0, 3], [5.0, 1.0, 6.0]]).astype(np.float32)
        value_to_pct = np.array([[3, 0.2], [5, 0.5], [1, 0.3], [3, 0.6]]).astype(
            np.float32
        )
        lengths = np.array([2, 1, 1]).astype(np.int32)

        def _percentile_ref(original_values, value_to_pct, lengths):
            ref_op = core.CreateOperator(
                "Percentile", ["original_values", "value_to_pct", "lengths"], ["Y"]
            )
            workspace.FeedBlob("original_values", original_values)
            workspace.FeedBlob("value_to_pct", value_to_pct)
            workspace.FeedBlob("lengths", lengths)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = _percentile_ref(original_values, value_to_pct, lengths)
        actual_output = torch.ops._caffe2.Percentile(
            torch.tensor(original_values),
            torch.tensor(value_to_pct),
            torch.tensor(lengths),
        )
        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def test_batch_bucket_one_hot_op(self):
        data = np.array([[2, 3], [4, 1], [2, 5]]).astype(np.float32)
        lengths = np.array([2, 3]).astype(np.int32)
        boundaries = np.array([0.1, 2.5, 1, 3.1, 4.5]).astype(np.float32)

        def _batch_bucket_one_hot_ref(data, lengths, boundaries):
            ref_op = core.CreateOperator(
                "BatchBucketOneHot", ["data", "lengths", "boundaries"], ["Y"]
            )
            workspace.FeedBlob("data", data)
            workspace.FeedBlob("lengths", lengths)
            workspace.FeedBlob("boundaries", boundaries)
            workspace.RunOperatorOnce(ref_op)
            return workspace.FetchBlob("Y")

        expected_output = _batch_bucket_one_hot_ref(data, lengths, boundaries)
        actual_output = torch.ops._caffe2.BatchBucketOneHot(
            torch.tensor(data), torch.tensor(lengths), torch.tensor(boundaries)
        )
        torch.testing.assert_allclose(expected_output, actual_output.cpu())

    def test_gather_ranges_to_dense_op(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        ranges = np.array([[[2, 4]], [[0, 0]]])
        key = np.array([0, 1, 3, 2, 1, 0, 1, 0])
        lengths = np.array([4])
        min_observation = 2
        max_mismatched_ratio = 0.5
        max_empty_ratio = 1.0

        outputs_name = ["X_{}".format(i) for i in range(len(lengths))]
        ref_op = core.CreateOperator(
            "GatherRangesToDense",
            ["data", "ranges", "key"],
            outputs_name,
            lengths=lengths,
            min_observation=min_observation,
            max_mismatched_ratio=max_mismatched_ratio,
            max_empty_ratio=max_empty_ratio,
        )
        workspace.FeedBlob("data", data)
        workspace.FeedBlob("ranges", ranges)
        workspace.FeedBlob("key", key)
        workspace.RunOperatorOnce(ref_op)
        ref_outputs = []
        for output_name in outputs_name:
            ref_outputs.append(workspace.FetchBlob(output_name))

        outputs = torch.ops._caffe2.GatherRangesToDense(
            torch.from_numpy(data),
            torch.from_numpy(ranges),
            torch.from_numpy(key),
            lengths=lengths,
            min_observation=min_observation,
            max_mismatched_ratio=max_mismatched_ratio,
            max_empty_ratio=max_empty_ratio,
        )

        self.assertEqual(len(ref_outputs), len(outputs))
        for i in range(0, len(ref_outputs)):
            np.testing.assert_array_almost_equal(ref_outputs[i], outputs[i].numpy())

    @given(lengths_0=st.integers(1, 10), lengths_1=st.integers(1, 10))
    @settings(deadline=10000)
    def test_merge_id_lists(self, lengths_0, lengths_1):
        def _merge_id_lists(lengths, values):
            ref_op = core.CreateOperator(
                "MergeIdLists",
                ["lengths_0", "values_0", "lengths_1", "values_1"],
                ["merged_lengths", "merged_values"],
            )
            workspace.FeedBlob("lengths_0", lengths[0])
            workspace.FeedBlob("values_0", values[0])
            workspace.FeedBlob("lengths_1", lengths[1])
            workspace.FeedBlob("values_1", values[1])
            workspace.RunOperatorOnce(ref_op)
            return (
                workspace.FetchBlob("merged_lengths"),
                workspace.FetchBlob("merged_values"),
            )

        lengths = [
            np.array([lengths_0]).astype(np.int32),
            np.array([lengths_1]).astype(np.int32),
        ]
        values = [
            np.random.choice(np.arange(0, 10), size=lengths_0, replace=False).astype(
                np.int32
            ),
            np.random.choice(np.arange(10, 20), size=lengths_1, replace=False).astype(
                np.int32
            ),
        ]

        expected_merged_lengths, expected_merged_values = _merge_id_lists(
            lengths, values
        )
        output_merged_lengths, output_merged_values = torch.ops._caffe2.MergeIdLists(
            [
                torch.tensor(lengths[0]),
                torch.tensor(values[0]),
                torch.tensor(lengths[1]),
                torch.tensor(values[1]),
            ]
        )
        torch.testing.assert_allclose(expected_merged_lengths, output_merged_lengths)
        torch.testing.assert_allclose(expected_merged_values, output_merged_values)

    def test_learning_rate(self):
        base_lr = 0.05
        no_iter = torch.tensor([0])
        one_iter = torch.tensor([1])
        two_iter = torch.tensor([2])

        # Fixed policy
        self.assertEqual(
            base_lr,
            torch.ops._caffe2.LearningRate(
                iterations=no_iter, base_lr=base_lr, policy="fixed"
            ),
        )
        self.assertEqual(
            base_lr,
            torch.ops._caffe2.LearningRate(
                iterations=one_iter, base_lr=base_lr, policy="fixed"
            ),
        )

        # Step policy
        gamma = 0.99
        stepsize = 1

        self.assertEqual(
            base_lr,
            torch.ops._caffe2.LearningRate(
                iterations=no_iter,
                base_lr=base_lr,
                policy="step",
                stepsize=stepsize,
                gamma=gamma,
            ),
        )
        self.assertAlmostEqual(
            base_lr * (gamma ** (1.0 / stepsize)),
            torch.ops._caffe2.LearningRate(
                iterations=one_iter,
                base_lr=base_lr,
                policy="step",
                stepsize=stepsize,
                gamma=gamma,
            ),
        )
        self.assertAlmostEqual(
            base_lr * (gamma ** (2.0 / stepsize)),
            torch.ops._caffe2.LearningRate(
                iterations=two_iter,
                base_lr=base_lr,
                policy="step",
                stepsize=stepsize,
                gamma=gamma,
            ),
        )

    def test_pack_segments(self):
        s = torch.rand(3, 3, 3)
        lengths = torch.tensor([2, 1])
        packed_tensor, _ = torch.ops._caffe2.PackSegments(lengths, s)
        self.assertEqual(packed_tensor.numpy().shape, (2, 2, 3, 3))
        unpacked_tensor = torch.ops._caffe2.UnpackSegments(lengths, packed_tensor)
        torch.testing.assert_allclose(s, unpacked_tensor)


if __name__ == "__main__":
    unittest.main()
