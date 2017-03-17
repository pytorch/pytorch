#include "caffe2/operators/resize_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool ResizeNearestOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  int output_width = W * width_scale_;
  int output_height = H * height_scale_;
  Y->Resize(N, C, output_height, output_width);

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < output_height; ++y) {
        const int in_y = std::min((int)(y / height_scale_), (H - 1));
        for (int x = 0; x < output_width; ++x) {
          const int in_x = std::min((int)(x / width_scale_), (W - 1));
          Ydata[output_width * y + x] = Xdata[in_y * W + in_x];
        }
      }
      Xdata += H * W;
      Ydata += output_width * output_height;
    }
  }

  return true;
}

namespace {

REGISTER_CPU_OPERATOR(ResizeNearest, ResizeNearestOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(ResizeNearest)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input using nearest neighbor
interpolation. The `width_scale` and `height_scale` arguments control the size
of the output, which is given by:
  output_width = floor(input_width * width_scale)
  output_height = floor(output_height * height_scale)
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

} // namespace
} // namespace caffe2
