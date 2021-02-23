#include "caffe2/operators/resize_3d_op.h"

#include "caffe2/utils/math.h"

#ifdef CAFFE2_USE_MKLDNN
#include "caffe2/ideep/operators/operator_fallback_ideep.h"
#include "caffe2/ideep/utils/ideep_operator.h"
#endif

namespace caffe2 {

void resizeNearest3DNCHW2x(
    int batch_size,
    int num_channels,
    int temporal_scale,
    int input_frames,
    int input_height,
    int input_width,
    const float* input,
    float* output) {
  const int output_frames = input_frames * temporal_scale;
  const int output_height = input_height * 2;
  const int output_width = input_width * 2;
  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int f = 0; f < output_frames; ++f ) {
        const int in_f = f / temporal_scale;
        for (int y = 0; y < output_height; ++y) {
          const int in_y = y / 2;

          for (int x = 0; x < input_width; ++x) {
            const float v =
              input[((in_f * input_height) + in_y) * input_width + x];
            const int oidx = y * output_width + x * 2;
            output[oidx + 0] = v;
            output[oidx + 1] = v;
          }
        }
        output += output_height * output_width;
      }
      input += input_frames * input_height * input_width;
    }
  }
}

template <>
bool ResizeNearest3DOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  const auto XDims = X.sizes();
  CAFFE_ENFORCE_EQ(5, XDims.size());

  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_frames = X.dim32(2), input_height = X.dim32(3),
            input_width = X.dim32(4);

  CAFFE_ENFORCE_EQ(InputSize(), 1);

  int output_frames = input_frames * temporal_scale_;
  int output_height = input_height * height_scale_;
  int output_width = input_width * width_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_frames, output_height, output_width},
      at::dtype<float>());

  const float* Xdata = X.data<float>();
  float* Ydata = Y->template mutable_data<float>();

  // Specialized implementation for fast 2x upsampling
  if (width_scale_ == 2.0 && height_scale_ == 2.0) {
    CAFFE_ENFORCE(temporal_scale_ == 1 || temporal_scale_ == 2,
      "temporal_scale must be either 1 or 2");

    resizeNearest3DNCHW2x(
        batch_size, num_channels, temporal_scale_, input_frames, input_height,
        input_width, Xdata, Ydata);
    return true;
  }

  CAFFE_THROW("Not implemented when height- and width scale are not 2");
}

template <>
bool ResizeNearest3DOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      CAFFE_THROW("Not implemented for storage order: ", order_);
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}

template <>
bool ResizeNearest3DGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& X = Input(1);

  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(5, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_frames = dY.dim32(2), input_height = dY.dim32(3),
            input_width = dY.dim32(4);

  const int output_frames = X.dim32(2);
  const int output_height = X.dim32(3);
  const int output_width = X.dim32(4);

  CAFFE_ENFORCE_EQ(InputSize(), 2);

  auto* dX = Output(
      0,
      {batch_size, num_channels, output_frames, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CPUContext>(
      dX->numel(), 0.0f, dX->template mutable_data<float>(), &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int f = 0; f < input_frames; ++f) {
        const int out_f =
          std::min((int)(f / temporal_scale_), output_frames - 1);
        for (int y = 0; y < input_height; ++y) {
          const int out_y =
              std::min((int)(y / height_scale_), (output_height - 1));
          for (int x = 0; x < input_width; ++x) {
            const int out_x =
                std::min((int)(x / width_scale_), (output_width - 1));
            dXdata[(out_f * output_height + out_y) * output_width + out_x] +=
              dYdata[(f * input_height + y) * input_width + x];
          }
        }
      }
      dYdata += input_frames * input_height * input_width;
      dXdata += output_frames * output_height * output_width;
    }
  }

  return true;
}

template <>
bool ResizeNearest3DGradientOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NHWC:
      CAFFE_THROW("Not implemented for storage order: ", order_);
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    default:
      CAFFE_THROW("Unknown Storage order: ", order_);
  }
}
REGISTER_CPU_OPERATOR(ResizeNearest3D, ResizeNearest3DOp<float, CPUContext>);
REGISTER_CPU_GRADIENT_OPERATOR(
    ResizeNearest3DGradient,
    ResizeNearest3DGradientOp<float, CPUContext>);

#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    ResizeNearest3D,
    IDEEPFallbackOp<ResizeNearest3DOp<float, CPUContext>>);
#endif

// Input: X, output: Y
OPERATOR_SCHEMA(ResizeNearest3D)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("temporal_scale", "Scale along temporal dimension")
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .SetDoc(R"DOC(
Resizes the spatial dimensions of the input tensor using nearest neighbor
interpolation. The `width_scale` and `height_scale` arguments
control the size of the output, which is given by:
output_width = floor(input_width * width_scale)
output_height = floor(output_height * height_scale)
Assumptions:
  - Only resize height and width
  - Both width_scale and height_scale scale are 2
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

// Input: dY, output: dX
GRADIENT_OPERATOR_SCHEMA(ResizeNearest3DGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("temporal_scale", "Scale along temporal dimension")
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension");

class GetResizeNearest3DGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeNearest3DGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ResizeNearest3D, GetResizeNearest3DGradient);

} // namespace caffe2

using ResizeNearest3DOpFloatCPU =
    caffe2::ResizeNearest3DOp<float, caffe2::CPUContext>;

// clang-format off
C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    ResizeNearest3D,
    "_caffe2::ResizeNearest3D("
      "Tensor X, "
      "str order, "
      "float temporal_scale, "
      "float width_scale, "
      "float height_scale"
    ") -> (Tensor Y)",
    ResizeNearest3DOpFloatCPU);
// clang-format on
