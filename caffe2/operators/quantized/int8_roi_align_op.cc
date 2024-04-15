#include "caffe2/operators/quantized/int8_roi_align_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8RoIAlign, int8::Int8RoIAlignOp);

OPERATOR_SCHEMA(Int8RoIAlign)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask R-CNN.
)DOC")
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .Arg(
        "spatial_scale",
        "(float) default 1.0; Spatial scale of the input feature map X "
        "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
        "w.r.t. the input image.")
    .Arg("pooled_h", "(int) default 1; Pooled output Y's height.")
    .Arg("pooled_w", "(int) default 1; Pooled output Y's width.")
    .Arg(
        "sampling_ratio",
        "(int) default -1; number of sampling points in the interpolation grid "
        "used to compute the output value of each pooled output bin. If > 0, "
        "then exactly sampling_ratio x sampling_ratio grid points are used. If "
        "<= 0, then an adaptive number of grid points are used (computed as "
        "ceil(roi_width / pooled_w), and likewise for height).")
    .Input(0, "X", "4D Int8 Tensor feature map input of shape (N, C, H, W).")
    .Input(
        1,
        "RoIs",
        "2D input of shape (R, 4 or 5) specifying R RoIs "
        "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
        "coordinates are in the coordinate system of the input image. For "
        "inputs corresponding to a single image, batch index can be excluded "
        "to have just 4 columns.")
    .Output(
        0,
        "Y",
        "4D Int8 Tensor output of shape (R, C, pooled_h, pooled_w). "
        "The r-th batch element "
        "is a pooled feature map corresponding to the r-th RoI.");

} // namespace caffe2
