// TODO: reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {
// These two classe are just used as template arguments passed to the PoolOp
// template to instantiate the different algorithms.
class AveragePool {};
class MaxPool {};
}  // namespace

template <>
bool PoolOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(
      Y->size(), 0, Ydata, &context_);
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
bool PoolOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);
  for (int n = 0; n < X.dim32(0); ++n) {
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
    Xdata += X.size() / X.dim32(0);
    Ydata += Y->size() / Y->dim32(0);
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  // Note that Input(1) is not needed in average pooling.
  auto& dY = Input(2);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  int channels = X.dim32(1);
  CHECK_EQ(channels, dY.dim32(1));
  int height = X.dim32(2);
  int width = X.dim32(3);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim32(2);
  int pooled_width = dY.dim32(3);
  // The main loop
  for (int n = 0; n < X.dim32(0); ++n) {
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
bool PoolGradientOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  // Note that Input(1) is not needed in average pooling.
  auto& dY = Input(2);
  CHECK_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &context_);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // The main loop
  int height = X.dim32(1);
  int width = X.dim32(2);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim32(1);
  int pooled_width = dY.dim32(2);
  int channels = X.dim32(3);
  CHECK_EQ(channels, dY.dim32(3));
  for (int n = 0; n < X.dim32(0); ++n) {
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
    dXdata += X.size() / X.dim32(0);
    dYdata += dY.size() / dY.dim32(0);
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(
      Y->size(), std::numeric_limits<float>::lowest(), Ydata, &context_);
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
              }
            }
          }
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
bool PoolOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);

  EigenMatrixMap<float> Ymat(
      Y->mutable_data<float>(), channels, Y->size() / channels);
  ConstEigenMatrixMap<float> Xmat(
      X.data<float>(), channels, X.size() / channels);
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);

  // The main loop
  // TODO: figure out the best location to put the omp parallel for command.
  #pragma omp parallel for
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * stride_h_ - pad_t_;
      int hend = min(hstart + kernel_h_, height);
      hstart = max(hstart, 0);
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w_ - pad_l_;
        int wend = min(wstart + kernel_w_, width);
        wstart = max(wstart, 0);
        // compute max in range X[n, hstart:hend, wstart:wend, :]
        auto Y_col = Ymat.col((n * pooled_height + ph) * pooled_width + pw);
        Y_col.setConstant(std::numeric_limits<float>::lowest());
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            Y_col = Y_col.cwiseMax(Xmat.col((n * height + h) * width + w));
          }
        }
      }
    }
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.size(), 0, dX->mutable_data<float>(), &context_);
  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  int channels = X.dim32(1);
  CHECK_EQ(channels, dY.dim32(1));
  int height = X.dim32(2);
  int width = X.dim32(3);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim32(2);
  int pooled_width = dY.dim32(3);
  for (int n = 0; n < X.dim32(0); ++n) {
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
              // OK here is a trick: this may multi-assign gradients.
              // which is not ideal.
              if (Xdata[input_index] == Ydata[pool_index]) {
                dXdata[input_index] += dYdata[pool_index];
              }
            }
          }
        }
      }
      // Do offset.
      Xdata += height * width;
      dXdata += height * width;
      Ydata += pooled_height * pooled_width;
      dYdata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolGradientOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  DCHECK_EQ(dY.ndim(), 4);
  auto* dX = Output(0);
  dX->ResizeLike(X);

  int channels = X.dim32(3);
  CHECK_EQ(channels, dY.dim32(3));
  ConstEigenArrayMap<float> Ymat(
      Y.data<float>(), channels, Y.size() / channels);
  ConstEigenArrayMap<float> dYmat(
      dY.data<float>(), channels, Y.size() / channels);
  ConstEigenArrayMap<float> Xmat(
      X.data<float>(), channels, X.size() / channels);
  EigenArrayMap<float> dXmat(
      dX->mutable_data<float>(), channels, X.size() / channels);
  dXmat.setZero();
  int height = X.dim32(1);
  int width = X.dim32(2);
  ConvPoolOpBase<CPUContext>::ComputePads(height, width);
  int pooled_height = dY.dim32(1);
  int pooled_width = dY.dim32(2);

  // The main loop
  // Do not do openmp here: the following for loops are looping over the pooled
  // output, so if one parallelizes the outer loops, race conditions could
  // happen in the inner loops.
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h_ - pad_t_;
        int wstart = pw * stride_w_ - pad_l_;
        int hend = min(hstart + kernel_h_, height);
        int wend = min(wstart + kernel_w_, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (n * pooled_height + ph) * pooled_width + pw;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (n * height + h) * width + w;
            dXmat.col(input_index) +=
                dYmat.col(pool_index) *
                (Xmat.col(input_index).cwiseEqual(Ymat.col(pool_index))
                    .cast<float>());
          }
        }
      }
    }
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(AveragePool, PoolOp<float, CPUContext, AveragePool>);
REGISTER_CPU_OPERATOR(AveragePoolGradient,
                      PoolGradientOp<float, CPUContext, AveragePool>);

OPERATOR_SCHEMA(AveragePool)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
AveragePool consumes an input blob X and applies average pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Average pooling consisting of averaging all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
  )DOC")
  .Input(0, "X", "Input data tensor from the previous operator; dimensions "
  "depend on whether the NCHW or NHWC operators are being used. For example, "
  "in the former, the input has size (N x C x H x W), where N is the batch "
  "size, C is the number of channels, and H and W are the height and the width "
  "of the data. The corresponding permutation of dimensions is used in the "
  "latter case. ")
  .Output(0, "Y", "Output data tensor from average pooling across the input "
  "tensor. Dimensions will vary based on various kernel, stride, and pad "
  "sizes.");

OPERATOR_SCHEMA(AveragePoolGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(MaxPool, PoolOp<float, CPUContext, MaxPool>);
REGISTER_CPU_OPERATOR(MaxPoolGradient,
                      PoolGradientOp<float, CPUContext, MaxPool>);

OPERATOR_SCHEMA(MaxPool)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
MaxPool consumes an input blob X and applies max pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Max pooling consisting of taking the maximumvalue of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
  )DOC")
  .Input(0, "X", "Input data tensor from the previous operator; dimensions "
  "depend on whether the NCHW or NHWC operators are being used. For example, "
  "in the former, the input has size (N x C x H x W), where N is the batch "
  "size, C is the number of channels, and H and W are the height and the width "
  "of the data. The corresponding permutation of dimensions is used in the "
  "latter case. ")
  .Output(0, "Y", "Output data tensor from max pooling across the input "
  "tensor. Dimensions will vary based on various kernel, stride, and pad "
  "sizes.");

OPERATOR_SCHEMA(MaxPoolGradient).NumInputs(3).NumOutputs(1);

class GetPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(AveragePool, GetPoolGradient);
REGISTER_GRADIENT(MaxPool, GetPoolGradient);

}  // namespace
}  // namespace caffe2
