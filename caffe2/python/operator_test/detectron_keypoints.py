




try:
    import cv2
except ImportError:
    pass  # skip if opencv is not available
import numpy as np


# === copied from utils/keypoints.py as reference ===
_NUM_KEYPOINTS = -1  # cfg.KRCNN.NUM_KEYPOINTS
_INFERENCE_MIN_SIZE = 0  # cfg.KRCNN.INFERENCE_MIN_SIZE


def heatmaps_to_keypoints(maps, rois):
    """Extracts predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths).astype(int)
    heights_ceil = np.ceil(heights).astype(int)

    num_keypoints = np.maximum(maps.shape[1], _NUM_KEYPOINTS)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = _INFERENCE_MIN_SIZE

    xy_preds = np.zeros(
        (len(rois), 4, num_keypoints), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)

        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(num_keypoints):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds


def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores


def approx_heatmap_keypoint(heatmaps_in, bboxes_in):
    '''
Mask R-CNN uses bicubic upscaling before taking the maximum of the heat map
for keypoints. We are using bilinear upscaling, which means we can approximate
the maximum coordinate with the low dimension maximum coordinates. We would like
to avoid bicubic upscaling, because it is computationally expensive. Brown and
Lowe  (Invariant Features from Interest Point Groups, 2002) uses a method  for
fitting a 3D quadratic function to the local sample points to determine the
interpolated location of the maximum of scale space, and his experiments showed
that this provides a substantial improvement to matching and stability for
keypoint extraction. This approach uses the Taylor expansion (up to the
quadratic terms) of the scale-space function. It is equivalent with the Newton
method. This efficient method were used in many keypoint estimation algorithms
like SIFT, SURF etc...

The implementation of Newton methods with numerical analysis is straight forward
and super simple, though we need a linear solver.

    '''
    assert len(bboxes_in.shape) == 2
    N = bboxes_in.shape[0]
    assert bboxes_in.shape[1] == 4
    assert len(heatmaps_in.shape) == 4
    assert heatmaps_in.shape[0] == N
    keypoint_count = heatmaps_in.shape[1]
    heatmap_size = heatmaps_in.shape[2]
    assert heatmap_size >= 2
    assert heatmaps_in.shape[3] == heatmap_size

    keypoints_out = np.zeros((N, keypoint_count, 4))

    for k in range(N):
        x0, y0, x1, y1 = bboxes_in[k, :]
        xLen = np.maximum(x1 - x0, 1)
        yLen = np.maximum(y1 - y0, 1)
        softmax_map = scores_to_probs(heatmaps_in[k, :, :, :].copy())
        f = heatmaps_in[k]
        for j in range(keypoint_count):
            f = heatmaps_in[k][j]
            maxX = -1
            maxY = -1
            maxScore = -100.0
            maxProb = -100.0
            for y in range(heatmap_size):
                for x in range(heatmap_size):
                    score = f[y, x]
                    prob = softmax_map[j, y, x]
                    if maxX < 0 or maxScore < score:
                        maxScore = score
                        maxProb = prob
                        maxX = x
                        maxY = y

            # print(maxScore, maxX, maxY)
            # initialize fmax values of 3x3 grid
            # when 3x3 grid going out-of-bound, mirrowing around center
            fmax = [[0] * 3 for r in range(3)]
            for x in range(3):
                for y in range(3):
                    hm_x = x + maxX - 1
                    hm_y = y + maxY - 1
                    hm_x = hm_x - 2 * (hm_x >= heatmap_size) + 2 * (hm_x < 0)
                    hm_y = hm_y - 2 * (hm_y >= heatmap_size) + 2 * (hm_y < 0)
                    assert((hm_x < heatmap_size) and (hm_x >= 0))
                    assert((hm_y < heatmap_size) and (hm_y >= 0))
                    fmax[y][x] = f[hm_y][hm_x]

            # print("python fmax ", fmax)
            # b = -f'(0), A = f''(0) Hessian matrix
            b = [-(fmax[1][2] - fmax[1][0]) / 2, -
                 (fmax[2][1] - fmax[0][1]) / 2]
            A = [[fmax[1][0] - 2 * fmax[1][1] + fmax[1][2],
                  (fmax[2][2] - fmax[2][0] - fmax[0][2] + fmax[0][0]) / 4],
                 [(fmax[2][2] - fmax[2][0] - fmax[0][2] + fmax[0][0]) / 4,
                  fmax[0][1] - 2 * fmax[1][1] + fmax[2][1]]]
            # print("python A")
            # print(A)
            # solve Ax=b
            div = A[1][1] * A[0][0] - A[0][1] * A[1][0]
            if abs(div) < 0.0001:
                deltaX = 0
                deltaY = 0
                deltaScore = maxScore
            else:
                deltaY = (b[1] * A[0][0] - b[0] * A[1][0]) / div
                deltaX = (b[0] * A[1][1] - b[1] * A[0][1]) / div
                # clip delta if going out-of-range of 3x3 grid
                if abs(deltaX) > 1.5 or abs(deltaY) > 1.5:
                    scale = 1.5 / max(abs(deltaX), abs(deltaY))
                    deltaX *= scale
                    deltaY *= scale
                # score = f(0) + f'(0)*x + 1/2 * f''(0) * x^2
                #    = f(0) - b*x + 1/2*x*A*x
                deltaScore = (
                    fmax[1][1] - (b[0] * deltaX + b[1] * deltaY) +
                    0.5 * (deltaX * deltaX * A[0][0] +
                           deltaX * deltaY * A[1][0] +
                           deltaY * deltaX * A[0][1] +
                           deltaY * deltaY * A[1][1]))

            assert abs(deltaX) <= 1.5
            assert abs(deltaY) <= 1.5

            # final coordinates
            keypoints_out[k, j, :] = (
                x0 + (maxX + deltaX + .5) * xLen / heatmap_size,
                y0 + (maxY + deltaY + .5) * yLen / heatmap_size,
                deltaScore,
                maxProb,
            )

    keypoints_out = np.transpose(keypoints_out, [0, 2, 1])

    return keypoints_out
