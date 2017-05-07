#include "caffe2/operators/resize_op.h"

#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

void resizeNearest2x(
    int N,
    int C,
    int H,
    int W,
    const float* input,
    float* output) {
  const int outputH = H * 2;
  const int outputW = W * 2;
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < outputH; ++y) {
        const int y_in = y / 2;

#ifdef __ARM_NEON__
        int vecW = (W / 4) * 4; // round down
        int x = 0;
        for (; x < vecW; x += 4) {
          // load 0 1 2 3
          float32x4_t v = vld1q_f32(input + y_in * W + x);
          const int oidx = outputW * y + x * 2;
          float32x4x2_t v2 = {{v, v}};
          // store 00 11 22 33
          vst2q_f32(output + oidx + 0, v2);
        }

        // handle remainder
        for (; x < W; ++x) {
          const float v = input[y_in * W + x];
          const int oidx = outputW * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#else
        for (int x = 0; x < W; ++x) {
          const float v = input[y_in * W + x];
          const int oidx = outputW * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#endif
      }
      input += H * W;
      output += outputH * outputW;
    }
  }
}

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

  // Specialized implementation for fast 2x upsampling
  if (width_scale_ == 2.0 && height_scale_ == 2.0) {
    resizeNearest2x(N, C, H, W, Xdata, Ydata);
    return true;
  }

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
            interpolation. The `width_scale` and `height_scale` arguments
            control the size of the output, which is given by:
            output_width = floor(input_width * width_scale)
            output_height = floor(output_height * height_scale)
            )DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

} // namespace
} // namespace caffe2
