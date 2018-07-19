from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class RMACRegionsOpTest(hu.HypothesisTestCase):
    @given(
        n=st.integers(500, 500),
        h=st.integers(1, 10),
        w=st.integers(1, 10),
        scales=st.integers(1, 3),
        **hu.gcs
    )
    def test(self, n, h, w, scales, gc, dc):
        X = np.random.rand(n, 64, h, w).astype(np.float32)
        overlap = 0.4

        def ref_op(X):
            N, H, W = X.shape[0], X.shape[2], X.shape[3]

            # Possible regions for the long dimension
            steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
            minW = np.minimum(H, W)

            # steps(idx) regions for long dimension
            b = (np.maximum(H, W) - minW) / (steps - 1)
            idx = np.argmin(
                np.abs(((minW**2 - minW * b) / minW**2) - overlap)) + 1

            # Region overplus per dimension
            Wd = 0
            Hd = 0
            if H < W:
                Wd = idx
            elif H > W:
                Hd = idx

            regions_xywh = []
            for l in range(1, scales + 1):
                wl = np.floor(2 * minW / (l + 1))

                # Center coordinates
                if l + Wd - 1 > 0:
                    b = (W - wl) / (l + Wd - 1)
                else:
                    b = 0
                cenW = np.floor(b * np.arange(l - 1 + Wd + 1))

                # Center coordinates
                if l + Hd - 1 > 0:
                    b = (H - wl) / (l + Hd - 1)
                else:
                    b = 0
                cenH = np.floor(b * np.arange(l - 1 + Hd + 1))

                for i_ in cenW:
                    for j_ in cenH:
                        regions_xywh.append([i_, j_, wl, wl])

            # Round the regions. Careful with the borders!
            for i in range(len(regions_xywh)):
                for j in range(4):
                    regions_xywh[i][j] = int(round(regions_xywh[i][j]))
                if regions_xywh[i][0] + regions_xywh[i][2] > W:
                    regions_xywh[i][0] -= (
                        (regions_xywh[i][0] + regions_xywh[i][2]) - W
                    )
                if regions_xywh[i][1] + regions_xywh[i][3] > H:
                    regions_xywh[i][1] -= (
                        (regions_xywh[i][1] + regions_xywh[i][3]) - H
                    )
            # Filter out 0-sized regions
            regions_xywh = [r for r in regions_xywh if r[2] * r[3] > 0]

            # Convert to ROIPoolOp format: (batch_index x1 y1 x2 y2)
            regions = [
                [i, x, y, x + w - 1, y + h - 1]
                for i in np.arange(N) for x, y, w, h in regions_xywh
            ]
            return (np.array(regions).astype(np.float32), )

        op = core.CreateOperator(
            'RMACRegions',
            ['X'],
            ['RMAC_REGIONS'],
            scales=scales,
            overlap=overlap,
        )

        # Check against numpy reference
        self.assertReferenceChecks(gc, op, [X], ref_op)
