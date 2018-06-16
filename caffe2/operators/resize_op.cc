#include "caffe2/operators/resize_op.h"

#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

void resizeNearest2x(
    int batch_size,
    int num_channels,
    int input_height,
    int input_width,
    const float* input,
    float* output) {
  const int output_height = input_height * 2;
  const int output_width = input_width * 2;
  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = h / 2;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int vecW = (input_width / 4) * 4; // round down
        int w = 0;
        for (; w < vecW; w += 4) {
          // load 0 1 2 3
          float32x4_t v = vld1q_f32(input + in_h * input_width + w);
          const int oidx = output_width * h + w * 2;
          float32x4x2_t v2 = {{v, v}};
          // store 00 11 22 33
          vst2q_f32(output + oidx + 0, v2);
        }

        // handle remainder
        for (; w < input_width; ++w) {
          const float v = input[in_h * input_width + w];
          const int oidx = output_width * h + w * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#else
        for (int w = 0; w < input_width; ++w) {
          const float v = input[in_h * input_width + w];
          const int oidx = output_width * h + w * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#endif
      }
      input += input_height * input_width;
      output += output_height * output_width;
    }
  }
}

template <>
bool ResizeNearestOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  // Specialized implementation for fast 2x upsampling
  if (width_scale_ == 2.0 && height_scale_ == 2.0) {
    resizeNearest2x(
        batch_size, num_channels, input_height, input_width, Xdata, Ydata);
    return true;
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < output_height; ++h) {
        const int in_h = std::min((int)(h / height_scale_), (input_height - 1));
        for (int w = 0; w < output_width; ++w) {
          const int in_w = std::min((int)(w / width_scale_), (input_width - 1));
          Ydata[output_width * h + w] = Xdata[input_width * in_h + in_w];
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }

  return true;
}

template <>
bool ResizeNearestOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  const int batch_size = X.dim32(0), input_height = X.dim32(1),
            input_width = X.dim32(2), num_channels = X.dim32(3);
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, output_height, output_width, num_channels);

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int h = 0; h < output_height; ++h) {
      const int in_h = std::min((int)(h / height_scale_), (input_height - 1));
      for (int w = 0; w < output_width; ++w) {
        const int in_w = std::min((int)(w / width_scale_), (input_width - 1));
        for (int c = 0; c < num_channels; ++c) {
          const int y_idx =
              ((n * output_height + h) * output_width + w) * num_channels + c;
          const int x_idx =
              ((n * input_height + in_h) * input_width + in_w) * num_channels +
              c;
          Ydata[y_idx] = Xdata[x_idx];
        }
      }
    }
  }

  return true;
}

template <>
bool ResizeNearestOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
}

REGISTER_CPU_OPERATOR(ResizeNearest, ResizeNearestOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(ResizeNearest)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .Arg("order", "Order of the input, either NCHW or NHWC")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input using nearest neighbor
interpolation. The `width_scale` and `height_scale` arguments
control the size of the output, which is given by:
output_width = floor(input_width * width_scale)
output_height = floor(output_height * height_scale)
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

} // namespace caffe2
