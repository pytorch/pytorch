#include "caffe2/operators/maxpool_op.h"

namespace caffe2 {

using std::max;
using std::min;

template <>
bool MaxPoolOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Tensor<int, CPUContext>* index =
      OperatorBase::template Output<Tensor<int, CPUContext> >(1);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim(1));
  index->ReshapeLike(*Y);

  const float* Xdata = X.data();
  float* Ydata = Y->mutable_data();
  int* index_data = index->mutable_data();
  math::Set<float, CPUContext>(
      Y->size(), std::numeric_limits<float>::lowest(), Ydata, &device_context_);
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
              if (Xdata[input_index] > Ydata[pool_index]) {
                Ydata[pool_index] = Xdata[input_index];
                index_data[pool_index] = c * height * width + h * width + w;
              }
            }
          }
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
      index_data += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool MaxPoolOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Tensor<int, CPUContext>* index =
      OperatorBase::template Output<Tensor<int, CPUContext> >(1);
  int height = X.dim(1);
  int width = X.dim(2);
  int channels = X.dim(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);
  index->ReshapeLike(*Y);

  const float* Xdata = X.data();
  float* Ydata = Y->mutable_data();
  int* index_data = index->mutable_data();
  math::Set<float, CPUContext>(
      Y->size(), std::numeric_limits<float>::lowest(), Ydata, &device_context_);
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
        // compute max in range X[n, hstart:hend, wstart:wend, :]
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              if (Xdata[input_index + c] > Ydata[pool_index + c]) {
                Ydata[pool_index + c] = Xdata[input_index + c];
                index_data[pool_index + c] = input_index + c;
              }
            }
          }
        }
      }
    }
    // Do offset.
    Xdata += X.size() / X.dim(0);
    Ydata += Y->size() / Y->dim(0);
    index_data += Y->size() / Y->dim(0);
  }
  return true;
}

template <>
bool MaxPoolGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  const Tensor<int, CPUContext>& maxid =
      OperatorBase::template Input<Tensor<int, CPUContext> >(2);
  DCHECK_EQ(maxid.size(), dY.size());
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ReshapeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data(), &device_context_);
  const float* dYdata = dY.data();
  const int* maxid_data = maxid.data();
  float* dXdata = dX->mutable_data();
  // Since we have recorded all the indices, we just need to run a simple
  // assignment pass.
  const int single_input_size = X.size() / X.dim(0);
  const int single_output_size = dY.size() / dY.dim(0);
  for (int n = 0; n < dY.dim(0); ++n) {
    for (int i = 0; i < single_output_size; ++i) {
      // DCHECK_LT(maxid_data[i], single_input_size);
      dXdata[maxid_data[i]] += dYdata[i];
    }
    dXdata += single_input_size;
    maxid_data += single_output_size;
    dYdata += single_output_size;
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(MaxPool, MaxPoolOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(MaxPoolGradient, MaxPoolGradientOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
