#include "caffe2/operators/resize_op.h"

#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

#ifdef USE_MKLDNN
#include "caffe2/ideep/operators/operator_fallback_ideep.h"
#include "caffe2/ideep/utils/ideep_operator.h"
#endif

namespace caffe2 {

static void resizeNearestNCHW2x(
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
      for (int y = 0; y < output_height; ++y) {
        const int in_y = y / 2;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
        int vecW = (input_width / 4) * 4; // round down
        int x = 0;
        for (; x < vecW; x += 4) {
          // load 0 1 2 3
          float32x4_t v = vld1q_f32(input + in_y * input_width + x);
          const int oidx = output_width * y + x * 2;
          float32x4x2_t v2 = {{v, v}};
          // store 00 11 22 33
          vst2q_f32(output + oidx + 0, v2);
        }

        // handle remainder
        for (; x < input_width; ++x) {
          const float v = input[in_y * input_width + x];
          const int oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
#else
        for (int x = 0; x < input_width; ++x) {
          const float v = input[in_y * input_width + x];
          const int oidx = output_width * y + x * 2;
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

  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_width = input_width * width_scale_;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_height = input_height * height_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());

  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();

  // Specialized implementation for fast 2x upsampling
  if (width_scale_ == 2.0 && height_scale_ == 2.0) {
    resizeNearestNCHW2x(
        batch_size, num_channels, input_height, input_width, Xdata, Ydata);
    return true;
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < output_height; ++y) {
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
        for (int x = 0; x < output_width; ++x) {
          // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
          const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
          Ydata[output_width * y + x] = Xdata[input_width * in_y + in_x];
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

  const int batch_size = X.dim32(0), input_height = X.dim32(1),
            input_width = X.dim32(2), num_channels = X.dim32(3);
  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_width = input_width * width_scale_;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int output_height = input_height * height_scale_;

  const int output_width_stride = output_width * num_channels;
  const int input_width_stride = input_width * num_channels;

  auto* Y = Output(
      0,
      {batch_size, output_height, output_width, num_channels},
      at::dtype<float>());

  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int y = 0; y < output_height; ++y) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const int in_y = std::min((int)(y / height_scale_), (input_height - 1));
      for (int x = 0; x < output_width; ++x) {
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        const int in_x = std::min((int)(x / width_scale_), (input_width - 1));
        std::memcpy(
            &Ydata[output_width_stride * y + num_channels * x],
            &Xdata[input_width_stride * in_y + num_channels * in_x],
            num_channels * sizeof(float));
      }
    }
    Xdata += input_height * input_width_stride;
    Ydata += output_height * output_width_stride;
  }

  return true;
}

template <>
bool ResizeNearestOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}

template <>
bool ResizeNearestGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_height = dY.dim32(2), input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);
  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  auto* dX = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < input_height; ++y) {
        const int out_y =
            // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
            std::min((int)(y / height_scale_), (output_height - 1));
        for (int x = 0; x < input_width; ++x) {
          const int out_x =
              // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
              std::min((int)(x / width_scale_), (output_width - 1));
          dXdata[output_width * out_y + out_x] += dYdata[input_width * y + x];
        }
      }
      dYdata += input_height * input_width;
      dXdata += output_height * output_width;
    }
  }

  return true;
}

template <>
bool ResizeNearestGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), input_height = dY.dim32(1),
            input_width = dY.dim32(2), num_channels = dY.dim32(3);
  const int output_height = X.dim32(1);
  const int output_width = X.dim32(2);
  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    const float* scales_data = scales.data<float>();
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  auto* dX = Output(
      0,
      {batch_size, output_height, output_width, num_channels},
      at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

  const int output_width_stride = output_width * num_channels;
  const int input_width_stride = input_width * num_channels;

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int y = 0; y < input_height; ++y) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      const int out_y = std::min((int)(y / height_scale_), (output_height - 1));
      for (int x = 0; x < input_width; ++x) {
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        const int out_x = std::min((int)(x / width_scale_), (output_width - 1));

        float* dXdata_c0 =
            dXdata + output_width_stride * out_y + num_channels * out_x;
        const float* dYdata_c0 =
            dYdata + input_width_stride * y + num_channels * x;

        for (int c = 0; c < num_channels; ++c) {
          dXdata_c0[c] += dYdata_c0[c];
        }
      }
    }
    dYdata += input_height * input_width_stride;
    dXdata += output_height * output_width_stride;
  }

  return true;
}

template <>
bool ResizeNearestGradientOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}
REGISTER_CPU_OPERATOR(ResizeNearest, ResizeNearestOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    ResizeNearestGradient,
    ResizeNearestGradientOp<float, CPUContext>);

#ifdef USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    ResizeNearest,
    IDEEPFallbackOp<ResizeNearestOp<float, CPUContext>>);
#endif

// Input: X, output: Y
OPERATOR_SCHEMA(ResizeNearest)
    .NumInputs(1, 2)
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
    .Input(0, "X", "Input tensor")
    .Input(
        1,
        "scales", // the hack to support onnx spec
        "1D, 2-element, Scales tensor, [height_scale, width_scale]")
    .Output(0, "Y", "Output tensor")
    .InheritOnnxSchema("Upsample");

// Input: dY, output: dX
GRADIENT_OPERATOR_SCHEMA(ResizeNearestGradient)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension");

class GetResizeNearestGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (def_.input().size() == 2) {
      // this is a hack to support the second input as dynamic
      // width_scale and height_scale to align with onnx change
      return SingleGradientDef(
          "ResizeNearestGradient",
          "",
          vector<string>{GO(0), I(0), I(1)},
          vector<string>{GI(0)});
    }
    return SingleGradientDef(
        "ResizeNearestGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ResizeNearest, GetResizeNearestGradient);

} // namespace caffe2

using ResizeNearestOpFloatCPU =
    caffe2::ResizeNearestOp<float, caffe2::CPUContext>;

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    ResizeNearest,
    "_caffe2::ResizeNearest("
      "Tensor X, "
      "str order, "
      "float width_scale, "
      "float height_scale"
    ") -> (Tensor Y)",
    ResizeNearestOpFloatCPU);
// clang-format on
