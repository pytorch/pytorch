#include "caffe2/operators/pool_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {
// These two classe are just used as template arguments passed to the
// PoolGradientOp
// template to instantiate the different algorithms.
template <typename T>
class AveragePool {
 public:
  static void process_grad(
      const T& /*x_data*/,
      const T& /*y_data*/,
      const T& dy_data,
      const T& scale,
      T& dx_data) {
    dx_data += (scale * dy_data);
  }

  static void process_grad(
      const int y_col,
      const int x_col,
      const float scale,
      ConstEigenArrayMap<float>& /*x_data*/,
      ConstEigenArrayMap<float>& /*y_data*/,
      ConstEigenArrayMap<float>& dy_data,
      EigenArrayMap<float>& dx_data) {
    dx_data.col(x_col) += scale * dy_data.col(y_col);
  }
};

template <typename T>
class MaxPool {
 public:
  static void process_grad(
      const T& x_data,
      const T& y_data,
      const T& dy_data,
      const T& /*scale*/,
      T& dx_data) {
    if (x_data == y_data) {
      dx_data += dy_data;
    }
  }

  static void process_grad(
      const int y_col,
      const int x_col,
      const float /*scale*/,
      ConstEigenArrayMap<float>& x_data,
      ConstEigenArrayMap<float>& y_data,
      ConstEigenArrayMap<float>& dy_data,
      EigenArrayMap<float>& dx_data) {
    dx_data.col(x_col) +=
        dy_data.col(y_col) * (x_data.col(x_col)
                                  .cwiseEqual(y_data.col(y_col))
                                  .template cast<float>());
  }
};
}

template <typename T, class Context, typename PoolType>
bool PoolGradientOp<T, Context, PoolType>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  // TODO(Yangqing): Add shape checks.
  dX->ResizeLike(X);
  math::Set<float, CPUContext>(
      X.numel(), 0, dX->template mutable_data<float>(), &context_);
  const float* Xdata = X.template data<float>();
  const float* Ydata = Y.template data<float>();
  const float* dYdata = dY.template data<float>();
  float* dXdata = dX->template mutable_data<float>();
  int channels = X.dim32(1);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(1));
  int height = X.dim32(2);
  int width = kernel_.size() > 1 ? X.dim32(3) : 1;
  int depth = kernel_.size() > 2 ? X.dim32(4) : 1;
  vector<int> dims(X.sizes().begin() + 2, X.sizes().end());
  ConvPoolOpBase<CPUContext>::ComputePads(dims);
  int pooled_height = dY.dim32(2);
  int pooled_width = kernel_.size() > 1 ? dY.dim32(3) : 1;
  int pooled_depth = kernel_.size() > 2 ? dY.dim32(4) : 1;
  // The main loop
  switch (kernel_.size()) {
    case 1:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            float scale = 1. / (hend - hstart);
            for (int h = hstart; h < hend; ++h) {
              PoolType::process_grad(
                  Xdata[h], Ydata[ph], dYdata[ph], scale, dXdata[h]);
            }
          }
          // offset
          Xdata += height;
          dXdata += height;
          Ydata += pooled_height;
          dYdata += pooled_height;
        }
      }
      break;
    case 2:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            for (int pw = 0; pw < pooled_width; ++pw) {
              int wstart = pw * stride_w() - pad_l();
              int wend = min(wstart + kernel_w(), width);
              wstart = max(wstart, 0);
              float scale = 1. / (hend - hstart) / (wend - wstart);
              const int pooled_index = ph * pooled_width + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width + w;
                  PoolType::process_grad(
                      Xdata[index],
                      Ydata[pooled_index],
                      dYdata[pooled_index],
                      scale,
                      dXdata[index]);
                }
              }
            }
          }
          // offset
          Xdata += height * width;
          dXdata += height * width;
          Ydata += pooled_height * pooled_width;
          dYdata += pooled_height * pooled_width;
        }
      }
      break;
    case 3:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            for (int pw = 0; pw < pooled_width; ++pw) {
              int wstart = pw * stride_w() - pad_l();
              int wend = min(wstart + kernel_w(), width);
              wstart = max(wstart, 0);
              for (int pd = 0; pd < pooled_depth; ++pd) {
                int dstart = pd * stride_[2] - pads_[2];
                int dend = min(dstart + kernel_[2], depth);
                dstart = max(dstart, 0);
                float scale =
                    1. / (hend - hstart) / (wend - wstart) / (dend - dstart);
                const int pooled_index =
                    ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    for (int d = dstart; d < dend; ++d) {
                      const int index = h * width * depth + w * depth + d;
                      PoolType::process_grad(
                          Xdata[index],
                          Ydata[pooled_index],
                          dYdata[pooled_index],
                          scale,
                          dXdata[index]);
                    }
                  }
                }
              }
            }
          }
          // offset
          Xdata += height * width * depth;
          dXdata += height * width * depth;
          Ydata += pooled_height * pooled_width * pooled_depth;
          dYdata += pooled_height * pooled_width * pooled_depth;
        }
      }
      break;
    default:
      CAFFE_THROW("Unsupported pooling size");
      return false;
  }
  return true;
}

template <typename T, class Context, typename PoolType>
bool PoolGradientOp<T, Context, PoolType>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  DCHECK_EQ(dY.dim(), kernel_.size() + 2);
  auto* dX = Output(0);
  dX->ResizeLike(X);

  int channels = X.dim32(X.dim() - 1);
  CAFFE_ENFORCE_EQ(channels, dY.dim32(dY.dim() - 1));
  ConstEigenArrayMap<T> Ymat(
      Y.template data<float>(), channels, Y.numel() / channels);
  ConstEigenArrayMap<float> dYmat(
      dY.template data<float>(), channels, Y.numel() / channels);
  ConstEigenArrayMap<float> Xmat(
      X.template data<float>(), channels, X.numel() / channels);
  EigenArrayMap<float> dXmat(
      dX->template mutable_data<float>(), channels, X.numel() / channels);
  dXmat.setZero();
  int height = X.dim32(1);
  int width = kernel_.size() > 1 ? X.dim32(2) : 1;
  int depth = kernel_.size() > 2 ? X.dim32(3) : 1;
  vector<int> dims(X.sizes().begin() + 1, X.sizes().end() - 1);
  ConvPoolOpBase<CPUContext>::ComputePads(dims);
  int pooled_height = dY.dim32(1);
  int pooled_width = kernel_.size() > 1 ? dY.dim32(2) : 1;
  int pooled_depth = kernel_.size() > 2 ? dY.dim32(3) : 1;

  // The main loop
  // Do not do openmp here: the following for loops are looping over the pooled
  // output, so if one parallelizes the outer loops, race conditions could
  // happen in the inner loops.
  switch (kernel_.size()) {
    case 1:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          const int pool_index = n * pooled_height + ph;
          const float scale = 1. / (hend - hstart);
          for (int h = hstart; h < hend; ++h) {
            const int input_index = n * height + h;
            PoolType::process_grad(
                pool_index, input_index, scale, Xmat, Ymat, dYmat, dXmat);
          }
        }
      }
      break;
    case 2:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width; ++pw) {
            int wstart = pw * stride_w() - pad_l();
            int wend = min(wstart + kernel_w(), width);
            wstart = max(wstart, 0);
            const int pool_index = (n * pooled_height + ph) * pooled_width + pw;
            const float scale = 1. / (hend - hstart) / (wend - wstart);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int input_index = (n * height + h) * width + w;
                PoolType::process_grad(
                    pool_index, input_index, scale, Xmat, Ymat, dYmat, dXmat);
              }
            }
          }
        }
      }
      break;
    case 3:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width; ++pw) {
            int wstart = pw * stride_w() - pad_l();
            int wend = min(wstart + kernel_w(), width);
            wstart = max(wstart, 0);
            for (int pd = 0; pd < pooled_depth; ++pd) {
              int dstart = pd * stride_[2] - pads_[2];
              int dend = min(dstart + kernel_[2], depth);
              dstart = max(dstart, 0);
              const int pool_index =
                  ((n * pooled_height + ph) * pooled_width + pw) *
                      pooled_depth +
                  pd;
              const float scale =
                  1. / (hend - hstart) / (wend - wstart) / (dend - dstart);
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  for (int d = dstart; d < dend; ++d) {
                    const int input_index =
                        ((n * height + h) * width + w) * depth + d;
                    PoolType::process_grad(
                        pool_index,
                        input_index,
                        scale,
                        Xmat,
                        Ymat,
                        dYmat,
                        dXmat);
                  }
                }
              }
            }
          }
        }
      }
      break;
    default:
      CAFFE_THROW("Unsupported pooling size");
      return false;
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    AveragePoolGradient,
    PoolGradientOp<float, CPUContext, AveragePool<float>>);
OPERATOR_SCHEMA(AveragePoolGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool1DGradient,
    PoolGradientOp<float, CPUContext, AveragePool<float>>);
OPERATOR_SCHEMA(AveragePool1DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool2DGradient,
    PoolGradientOp<float, CPUContext, AveragePool<float>>);
OPERATOR_SCHEMA(AveragePool2DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool3DGradient,
    PoolGradientOp<float, CPUContext, AveragePool<float>>);
OPERATOR_SCHEMA(AveragePool3DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPoolGradient,
    PoolGradientOp<float, CPUContext, MaxPool<float>>);
OPERATOR_SCHEMA(MaxPoolGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool1DGradient,
    PoolGradientOp<float, CPUContext, MaxPool<float>>);
OPERATOR_SCHEMA(MaxPool1DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool2DGradient,
    PoolGradientOp<float, CPUContext, MaxPool<float>>);
OPERATOR_SCHEMA(MaxPool2DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool3DGradient,
    PoolGradientOp<float, CPUContext, MaxPool<float>>);
OPERATOR_SCHEMA(MaxPool3DGradient).NumInputs(3).NumOutputs(1);

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
REGISTER_GRADIENT(AveragePool, GetPoolGradient);
REGISTER_GRADIENT(AveragePool1D, GetPoolGradient);
REGISTER_GRADIENT(AveragePool2D, GetPoolGradient);
REGISTER_GRADIENT(AveragePool3D, GetPoolGradient);
REGISTER_GRADIENT(MaxPool, GetPoolGradient);
REGISTER_GRADIENT(MaxPool1D, GetPoolGradient);
REGISTER_GRADIENT(MaxPool2D, GetPoolGradient);
REGISTER_GRADIENT(MaxPool3D, GetPoolGradient);
}
