from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestPriorBoxOp(hu.HypothesisTestCase):
    @given(
        image_size=st.integers(100, 300),
        fm_size_ratio=st.sampled_from([0.5, 0.25, 0.125, 0.0625]),
        width=st.integers(10, 30),
        height=st.integers(10, 30),
        step_w = st.sampled_from([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        step_h = st.sampled_from([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        clip=st.booleans(),
        offset=st.sampled_from([0.0, 1.0]),
        **hu.gcs_cpu_only
    )
    def test_prior_box(
        self,
        image_size,
        fm_size_ratio,
        width,
        height,
        step_w,
        step_h,
        clip,
        offset,
        gc,
        dc,
    ):
        image = np.zeros([1, 1, image_size, image_size], dtype=np.float32)
        fm_size = int(image_size * fm_size_ratio)
        fm = np.zeros([1, 1, fm_size, fm_size], dtype=np.float32)

        def priors_ref(fm, image):
            image_height, image_width = image.shape[2:]
            fm_height, fm_width = fm.shape[2:]

            x = np.tile(np.arange(fm_width), (fm_height, 1))
            y = np.tile(np.arange(fm_height).reshape(fm_height, 1), fm_width)

            center_x = (x + offset) * step_w
            center_y = (y + offset) * step_h

            priors = np.zeros([fm_height, fm_width, 4])
            priors[:, :, 0] = (center_x - width / 2.0) / image_width
            priors[:, :, 1] = (center_y - height / 2.0) / image_height
            priors[:, :, 2] = (center_x + width / 2.0) / image_width
            priors[:, :, 3] = (center_y + height / 2.0) / image_height

            if clip:
                priors = np.clip(priors, 0.0, 1.0)

            return [priors.ravel()]


        op = core.CreateOperator(
            "PriorBox",
            ["fm", "image"],
            ["priors"],
            widths=[float(width)],
            heights=[float(height)],
            step_w=step_w,
            step_h=step_h,
            clip=clip,
            offset=offset,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[fm, image],
            reference=priors_ref,
        )

    @given(
        image_size=st.integers(100, 300),
        fm_size_ratio=st.sampled_from([0.5, 0.25, 0.125, 0.0625]),
        width=st.integers(10, 30),
        height=st.integers(10, 30),
        clip=st.booleans(),
        offset=st.sampled_from([0.0, 1.0]),
        **hu.gcs_cpu_only
    )
    def test_prior_box_zero_step(
        self,
        image_size,
        fm_size_ratio,
        width,
        height,
        clip,
        offset,
        gc,
        dc,
    ):
        image = np.zeros([1, 1, image_size, image_size], dtype=np.float32)
        fm_size = int(image_size * fm_size_ratio)
        fm = np.zeros([1, 1, fm_size, fm_size], dtype=np.float32)

        def priors_ref(fm, image):
            image_height, image_width = image.shape[2:]
            fm_height, fm_width = fm.shape[2:]

            step_w = float(image_width) / fm_width
            step_h = float(image_height) / fm_height

            x = np.tile(np.arange(fm_width), (fm_height, 1))
            y = np.tile(np.arange(fm_height).reshape(fm_height, 1), fm_width)

            center_x = (x + offset) * step_w
            center_y = (y + offset) * step_h

            priors = np.zeros([fm_height, fm_width, 4])
            priors[:, :, 0] = (center_x - width / 2.0) / image_width
            priors[:, :, 1] = (center_y - height / 2.0) / image_height
            priors[:, :, 2] = (center_x + width / 2.0) / image_width
            priors[:, :, 3] = (center_y + height / 2.0) / image_height

            if clip:
                priors = np.clip(priors, 0.0, 1.0)

            return [priors.ravel()]


        op = core.CreateOperator(
            "PriorBox",
            ["fm", "image"],
            ["priors"],
            widths=[float(width)],
            heights=[float(height)],
            clip=clip,
            offset=offset,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[fm, image],
            reference=priors_ref,
        )
