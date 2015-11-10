#include "caffe2/operators/averagepool_op.h"

namespace caffe2 {

using std::max;
using std::min;

template <>
bool AveragePoolOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim(1));

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(
      Y->size(), 0, Ydata, &device_context_);
  // The main loop
  int channels = X.dim(1);
  int height = X.dim(2);
  int width = X.dim(3);
  int pooled_height = Y->dim(2);
  int pooled_width = Y->dim(3);
  for (int n = 0; n < X.dim(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h_ - pad_t_;
          int wstart = pw * stride_w_ - pad_l_;
          int hend = min(hstart + kernel_h_, height);
          int wend = min(wstart + kernel_w_, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              Ydata[pool_index] += Xdata[input_index];
            }
          }
          Ydata[pool_index] /= (hend - hstart) * (wend - wstart);
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
bool AveragePoolOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim(1);
  int width = X.dim(2);
  int channels = X.dim(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &device_context_);
  // The main loop
  int pooled_height = Y->dim(1);
  int pooled_width = Y->dim(2);
  for (int n = 0; n < X.dim(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h_ - pad_t_;
        int wstart = pw * stride_w_ - pad_l_;
        int hend = min(hstart + kernel_h_, height);
        int wend = min(wstart + kernel_w_, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] += Xdata[input_index + c];
            }
          }
        }
        float scale = 1. / (hend - hstart) / (wend - wstart);
        for (int c = 0; c < channels; ++c) {
          Ydata[pool_index + c] *= scale;
        }
      }
    }
    // Do offset.
    Xdata += X.size() / X.dim(0);
    Ydata += Y->size() / Y->dim(0);
  }
  return true;
}

template <>
bool AveragePoolGradientOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ReshapeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &device_context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  int channels = X.dim(1);
  CAFFE_CHECK_EQ(channels, dY.dim(1));
  int height = X.dim(2);
  int width = X.dim(3);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim(2);
  int pooled_width = dY.dim(3);
  // The main loop
  for (int n = 0; n < X.dim(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h_ - pad_t_;
          int wstart = pw * stride_w_ - pad_l_;
          int hend = min(hstart + kernel_h_, height);
          int wend = min(wstart + kernel_w_, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          float scale  = 1. / (hend - hstart) / (wend - wstart);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              dXdata[h * width + w] +=
                dYdata[ph * pooled_width + pw] * scale;
            }
          }
        }
      }
      // offset
      dXdata += height * width;
      dYdata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool AveragePoolGradientOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& dY = Input(1);
  CAFFE_CHECK_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ReshapeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &device_context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // The main loop
  int height = X.dim(1);
  int width = X.dim(2);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim(1);
  int pooled_width = dY.dim(2);
  int channels = X.dim(3);
  CAFFE_CHECK_EQ(channels, dY.dim(3));
  for (int n = 0; n < X.dim(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h_ - pad_t_;
        int wstart = pw * stride_w_ - pad_l_;
        int hend = min(hstart + kernel_h_, height);
        int wend = min(wstart + kernel_w_, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        float scale  = 1. / (hend - hstart) / (wend - wstart);
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            for (int c = 0; c < channels; ++c) {
              dXdata[(h * width + w) * channels + c] +=
                dYdata[(ph * pooled_width + pw) * channels + c] * scale;
            }
          }
        }
      }
    }
    // offset
    dXdata += X.size() / X.dim(0);
    dYdata += dY.size() / dY.dim(0);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(AveragePool, AveragePoolOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AveragePoolGradient,
                      AveragePoolGradientOp<float, CPUContext>);

struct GetAveragePoolGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    return SingleGradientDef(
        "AveragePoolGradient", "",
        vector<string>{I(def, 0), GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(AveragePool, GetAveragePoolGradient);

}  // namespace
}  // namespace caffe2
