from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st

from hypothesis import given

from caffe2.python import core

import numpy as np
import math

class CropAndResizeMethod:
    BILINEAR = 0
    NEAREST = 1

class TestCropAndResize(hu.HypothesisTestCase):

    @given(
        roi_counts = st.lists(st.integers(1, 5), min_size = 1, max_size = 3),
        image_channels = st.integers(1, 3),
        image_height = st.integers(1, 10),
        image_width = st.integers(1, 10),
        crop_height = st.integers(1, 15),
        crop_width = st.integers(1, 15),
        method = st.sampled_from([CropAndResizeMethod.BILINEAR, CropAndResizeMethod.NEAREST]),
        extrapolation_value = st.floats(-1.0, 1.0),
        **hu.gcs
    )
    def test_crop_and_resize(
            self,
            roi_counts,
            image_channels,
            image_height,
            image_width,
            crop_height,
            crop_width,
            method,
            extrapolation_value,
            gc,
            dc
    ):

        batch_size = len(roi_counts)

        image = np.random.rand(
            batch_size,
            image_channels,
            image_height,
            image_width).astype(np.float32)

        all_rois = []
        for i, num_rois in enumerate(roi_counts):
            if num_rois == 0:
                continue
            rois = np.random.uniform(0.0, 1.0, size=(roi_counts[i], 5)).astype(
                np.float32
            )
            rois[:, 0] = i
            all_rois.extend(rois)

        rois = np.asarray(all_rois).astype(np.float32)

        def crop_and_resize_ref(
                image,
                rois
        ):

            out_rois = np.zeros(
                shape = (rois.shape[0], image_channels, crop_height, crop_width),
                dtype = np.float32
            )

            for b, box in enumerate(rois):

                batch_index = int(box[0])

                if batch_index < 0 or batch_index > batch_size:
                    continue

                x1, y1, x2, y2 = box[1], box[2], box[3], box[4]

                width_scale = ((x2 - x1) * (image_width - 1) / (crop_width - 1)) if (crop_width > 1) else 0.0
                height_scale = ((y2 - y1) * (image_height - 1) / (crop_height - 1)) if (crop_height > 1) else 0.0

                for c in range(image_channels):

                    for y in range(crop_height):

                        in_y = ((y1 * (image_height - 1) + y * height_scale)) \
                                    if (crop_height > 1) else (0.5 * (y1 + y2) * (image_height - 1))

                        if in_y < 0 or in_y > image_height - 1:

                            for x in range(crop_width):
                                out_rois[b][c][y][x] = extrapolation_value
                            continue

                        if method == CropAndResizeMethod.BILINEAR:

                            top_y_index = int(math.floor(in_y))
                            bottom_y_index = int(math.ceil(in_y))

                            y_lerp = float(in_y - float(top_y_index))

                            for x in range(crop_width):

                                in_x = ((x1 * (image_width - 1) + x * width_scale)) \
                                    if (crop_width > 1) else (0.5 * (x1 + x2) * (image_width - 1))

                                if in_x < 0 or in_x > image_width - 1:
                                    out_rois[b][c][y][x] = extrapolation_value
                                    continue

                                left_x_index = int(math.floor(in_x))
                                right_x_index = int(math.ceil(in_x))

                                x_lerp = float(in_x - float(left_x_index))

                                top_left = image[batch_index][c][top_y_index][left_x_index]
                                top_right = image[batch_index][c][top_y_index][right_x_index]
                                bottom_left = image[batch_index][c][bottom_y_index][left_x_index]
                                bottom_right = image[batch_index][c][bottom_y_index][right_x_index]

                                top = top_left + (top_right - top_left) * x_lerp
                                bottom = bottom_left + (bottom_right - bottom_left) * x_lerp

                                out_rois[b][c][y][x] = top + (bottom - top) * y_lerp

                        else:

                            assert method == CropAndResizeMethod.NEAREST

                            for x in range(crop_width):

                                in_x = ((x1 * (image_width - 1) + x * width_scale)) \
                                    if (crop_width > 1) else (0.5 * (x1 + x2) * (image_width - 1))

                                if in_x < 0 or in_x > image_width - 1:
                                    out_rois[b][c][y][x] = extrapolation_value
                                    continue

                                nearest_x_index = int(round(in_x))
                                nearest_y_index = int(round(in_y))

                                out_rois[b][c][y][x] = image[batch_index][c][nearest_y_index][nearest_x_index]

            return (out_rois,)

        op = core.CreateOperator(
            "CropAndResize",
            ["image", "rois"],
            ["out_rois"],
            crop_height = crop_height,
            crop_width = crop_width,
            method = method,
            extrapolation_value = extrapolation_value
        )


        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[image, rois],
            reference=crop_and_resize_ref
        )

        self.assertDeviceChecks(
            dc,
            op,
            [image, rois],
            [0]
        )

        self.assertGradientChecks(
           gc,
           op,
           [image, rois],
           0,
           [0]
        )
