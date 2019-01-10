// TODO: reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"

namespace caffe2 {

using std::min;
using std::max;

class LpPool {};

template <>
bool PoolOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int pooled_height = Y->dim32(2);
  int pooled_width = Y->dim32(3);

  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_[0] - pads_[0];
          int wstart = pw * stride_[1] - pads_[1];
          int hend = min(hstart + kernel_[0], height);
          int wend = min(wstart + kernel_[1], width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              Ydata[pool_index] += std::pow(std::abs(Xdata[input_index]), p);
            }
          }
          Ydata[pool_index] = std::pow(Ydata[pool_index], inv_p);
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);

  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_[0] - pads_[0];
        int wstart = pw * stride_[1] - pads_[1];
        int hend = min(hstart + kernel_[0], height);
        int wend = min(wstart + kernel_[1], width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] +=
                  std::pow(std::abs(Xdata[input_index + c]), p);
            }
          }
        }
        for (int c = 0; c < channels; ++c) {
          Ydata[pool_index + c] = std::pow(Ydata[pool_index + c], inv_p);
        }
      }
    }
    // Do offset.
    Xdata += X.size() / X.dim32(0);
    Ydata += Y->size() / Y->dim32(0);
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  const auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  float* dXdata = dX->mutable_data<float>();

  int channels = X.dim32(1);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(1));
  int height = X.dim32(2);
  int width = X.dim32(3);
  ConvPoolOpBase<CPUContext>::ComputePads({height, width});
  int pooled_height = dY.dim32(2);
  int pooled_width = dY.dim32(3);
  // The main loop
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_[0] - pads_[0];
          int wstart = pw * stride_[1] - pads_[1];
          int hend = min(hstart + kernel_[0], height);
          int wend = min(wstart + kernel_[1], width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          float scale = 1. / (hend - hstart) / (wend - wstart);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              // gradient of p-norm is x_j * |x_j|^{p-2} / |x|_p^{p-1}
              dXdata[h * width + w] += dYdata[ph * pooled_width + pw] *
                  Xdata[h * width + w] *
                  std::pow(std::abs(Xdata[h * width + w]), p - 2) /
                  std::pow(Ydata[ph * pooled_width + pw], p - 1);
            }
          }
        }
      }
      // offset
      dXdata += height * width;
      dYdata += pooled_height * pooled_width;
      Ydata += pooled_height * pooled_width;
      Xdata += height * width;
    }
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  const auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  // The main loop
  int height = X.dim32(1);
  int width = X.dim32(2);
  ConvPoolOpBase<CPUContext>::ComputePads({height, width});
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  int pooled_height = dY.dim32(1);
  int pooled_width = dY.dim32(2);
  int channels = X.dim32(3);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(3));
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_[0] - pads_[0];
        int wstart = pw * stride_[1] - pads_[1];
        int hend = min(hstart + kernel_[0], height);
        int wend = min(wstart + kernel_[1], width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float scale = 1. / (hend - hstart) / (wend - wstart);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            for (int c = 0; c < channels; ++c) {
              dXdata[(h * width + w) * channels + c] +=
                  dYdata[(ph * pooled_width + pw) * channels + c] *
                  Xdata[(h * width + w) * channels + c] *
                  std::pow(
                      std::abs(Xdata[(h * width + w) * channels + c]), p - 2) /
                  std::pow(
                      Ydata[(ph * pooled_width + pw) * channels + c], p - 1);
            }
          }
        }
      }
    }
    // offset
    dXdata += X.size() / X.dim32(0);
    dYdata += dY.size() / dY.dim32(0);
    Xdata += X.size() / X.dim32(0);
    Ydata += Y.size() / Y.dim32(0);
  }
  return true;
}

REGISTER_CPU_OPERATOR(LpPool, PoolOp<float, CPUContext, LpPool>);
REGISTER_CPU_OPERATOR(
    LpPoolGradient,
    PoolGradientOp<float, CPUContext, LpPool>);

OPERATOR_SCHEMA(LpPool)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
LpPool consumes an input blob X and applies L-p pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. L-p pooling consisting of taking the L-p norm of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
)DOC")
    .Input(
        0,
        "X",
        "Input data tensor from the previous operator; dimensions "
        "depend on whether the NCHW or NHWC operators are being used. For example, "
        "in the former, the input has size (N x C x H x W), where N is the batch "
        "size, C is the number of channels, and H and W are the height and the width "
        "of the data. The corresponding permutation of dimensions is used in the "
        "latter case. ")
    .Output(
        0,
        "Y",
        "Output data tensor from L-p pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.");

OPERATOR_SCHEMA(LpPoolGradient).NumInputs(3).NumOutputs(1);

class GetPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(LpPool, GetPoolGradient);
}
