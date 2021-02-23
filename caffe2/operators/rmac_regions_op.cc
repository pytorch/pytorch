#include "caffe2/operators/rmac_regions_op.h"

#include <float.h>

namespace caffe2 {

template <>
bool RMACRegionsOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(0); // Input tensor
                            // RoIs
  auto* output = Output(
      0,
      {0, 5},
      at::dtype<float>()); // [batch_id x1 y1 x2 y2] format of ROIPoolOp

  if (X.numel() == 0) {
    return true;
  }

  int batch_size = X.dim32(0);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int minW = std::min(H, W);

  // steps(idx) regions for long dimension
  int step = 0;
  if (W != H) {
    int min_step = 1;
    int max_step = 6;
    float cur_min = FLT_MAX;
    for (int idx = min_step; idx <= max_step; ++idx) {
      float b = (std::max(H, W) - minW) / (1.0 * idx);
      float val = std::abs((minW * minW - minW * b) / (minW * minW) - overlap_);
      if (val < cur_min) {
        step = idx;
        cur_min = val;
      }
    }
  }

  // Region overplus per dimension
  int Wd = (W > H) ? step : 0;
  int Hd = (H > W) ? step : 0;

  // Regions at each scale
  for (int l = 1; l <= scales_; ++l) {
    int region_size = 2 * minW / (l + 1);
    if (region_size == 0) {
      // Empty region.
      // Break early as further scales will also result in empty regions.
      break;
    }

    // Region coordinates
    float bw =
        (l + Wd - 1 > 0) ? ((W - region_size) / (1.0 * (l + Wd - 1))) : 0;
    float bh =
        (l + Hd - 1 > 0) ? ((H - region_size) / (1.0 * (l + Hd - 1))) : 0;

    int cur_rows = output->dim32(0);
    output->Extend((l + Wd) * (l + Hd), 50);
    auto* outputData = output->template mutable_data<float>() + cur_rows * 5;

    for (int i = 0; i < l + Wd; ++i) {
      for (int j = 0; j < l + Hd; ++j) {
        int x1 = bw * i;
        int y1 = bh * j;
        // Careful with the borders
        if (x1 + region_size > W) {
          x1 -= (x1 + region_size - W);
        }
        if (y1 + region_size > H) {
          y1 -= (y1 + region_size - H);
        }
        int x2 = x1 + region_size - 1;
        int y2 = y1 + region_size - 1;

        // Write region coordinates for batch 0
        *outputData++ = 0;
        *outputData++ = x1;
        *outputData++ = y1;
        *outputData++ = x2;
        *outputData++ = y2;
      }
    }
  }

  // Replicate regions for all items in batch
  int num_rois = output->dim32(0);
  output->Extend((batch_size - 1) * num_rois, 50);
  auto* outputData = output->template mutable_data<float>();
  for (int b = 1; b < batch_size; ++b) {
    // Copy all rois
    std::copy_n(outputData, num_rois * 5, outputData + b * num_rois * 5);
    // Override batch index
    for (int r = 0; r < num_rois; ++r) {
      outputData[(b * num_rois + r) * 5] = b;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(RMACRegions, RMACRegionsOp<CPUContext>);

OPERATOR_SCHEMA(RMACRegions)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes a fixed-grid of RMAC region coordinates at various levels
as described in https://arxiv.org/abs/1511.05879.
)DOC")
    .Arg("scales", "Number of scales to sample regions at.")
    .Arg("overlap", "Overlap between consecutive regions.")
    .Input(0, "X", "The input 4D tensor of shape NCHW.")
    .Output(
        0,
        "RMAC_REGIONS",
        "The output RMAC regions for all items in the batch. Tensor of shape "
        "(N x 5) following the ROIPoolOp format - each row is of the format "
        "(batch_index x1 y1 x2 y2) where x1, y1, x2, y2 are the region "
        "co-ordinates. Each region is repeated N times corresponding to each "
        "item in the batch.");

SHOULD_NOT_DO_GRADIENT(RMACRegions);

} // namespace caffe2
