#include "caffe2/operators/resize_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool ResizeNearestGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  auto* dX = Output(0);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_height = dY.dim32(2), input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);
  dX->Resize(batch_size, num_channels, output_height, output_width);
  math::Set<float, CPUContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < num_channels; ++c) {
      for (int y = 0; y < input_height; ++y) {
        const int out_y =
            std::min((int)(y / height_scale_), (output_height - 1));
        for (int x = 0; x < input_width; ++x) {
          const int out_x =
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
  auto* dX = Output(0);

  const auto& inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), input_height = dY.dim32(1),
            input_width = dY.dim32(2), num_channels = dY.dim32(3);
  const int output_height = X.dim32(1);
  const int output_width = X.dim32(2);
  dX->Resize(batch_size, output_height, output_width, num_channels);
  math::Set<float, CPUContext>(
      dX->size(), 0.0f, dX->mutable_data<float>(), &context_);

  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int h = 0; h < input_height; ++h) {
      const int out_h = std::min((int)(h / height_scale_), (output_height - 1));
      for (int w = 0; w < input_width; ++w) {
        const int out_w = std::min((int)(w / width_scale_), (output_width - 1));
        for (int c = 0; c < num_channels; ++c) {
          const int dx_idx =
              ((n * output_height + out_h) * output_width + out_w) *
                  num_channels +
              c;
          const int dy_idx =
              ((n * input_height + h) * input_width + w) * num_channels + c;
          dXdata[dx_idx] += dYdata[dy_idx];
        }
      }
    }
  }
  return true;
}

template <>
bool ResizeNearestGradientOp<float, CPUContext>::RunOnDevice() {
  switch (order_) {
    case StorageOrder::NCHW:
      return RunOnDeviceWithOrderNCHW();
    case StorageOrder::NHWC:
      return RunOnDeviceWithOrderNHWC();
    default:
      CAFFE_THROW("Unknown storage order: ", order_);
  }
}

REGISTER_CPU_OPERATOR(
    ResizeNearestGradient,
    ResizeNearestGradientOp<float, CPUContext>);

// Input: dY, output: dX
OPERATOR_SCHEMA(ResizeNearestGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("width_scale", "Scale along width dimension")
    .Arg("height_scale", "Scale along height dimension")
    .Arg("order", "Order of the input, either NCHW or NHWC");

class GetResizeNearestGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeNearestGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ResizeNearest, GetResizeNearestGradient);
} // namespace caffe2
