#include "caffe2/operators/pool_op.h"

#include <limits>
#include <string>
#include <type_traits>

#include "caffe2/operators/pool_op_util.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T, StorageOrder kOrder>
void ComputeAveragePool1D(
    int l,
    int r,
    int y,
    T scale,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeAveragePool1D<float, StorageOrder::NCHW>(
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = X_arr.col(0).segment(l, r - l).sum() * scale;
}

template <>
void ComputeAveragePool1D<float, StorageOrder::NHWC>(
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y) = X_arr.col(l);
  for (int i = l + 1; i < r; ++i) {
    Y_arr->col(y) += X_arr.col(i);
  }
  Y_arr->col(y) *= scale;
}

template <typename T, StorageOrder kOrder>
void ComputeAveragePool2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    T scale,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeAveragePool2D<float, StorageOrder::NCHW>(
    const int /* W */,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = X_arr.block(l, t, r - l, b - t).sum() * scale;
}

template <>
void ComputeAveragePool2D<float, StorageOrder::NHWC>(
    const int W,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y).setZero();
  for (int i = t; i < b; ++i) {
    for (int j = l; j < r; ++j) {
      Y_arr->col(y) += X_arr.col(i * W + j);
    }
  }
  Y_arr->col(y) *= scale;
}

template <typename T, StorageOrder kOrder>
void ComputeAveragePool3D(
    int H,
    int W,
    int p,
    int a,
    int t,
    int b,
    int l,
    int r,
    int y,
    T scale,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeAveragePool3D<float, StorageOrder::NCHW>(
    const int H,
    const int /* W */,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = 0;
  for (int i = p; i < a; ++i) {
    (*Y_arr)(y) += X_arr.block(l, i * H + t, r - l, b - t).sum();
  }
  (*Y_arr)(y) *= scale;
}

template <>
void ComputeAveragePool3D<float, StorageOrder::NHWC>(
    const int H,
    const int W,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y).setZero();
  for (int i = p; i < a; ++i) {
    for (int j = t; j < b; ++j) {
      for (int k = l; k < r; ++k) {
        Y_arr->col(y) += X_arr.col(i * H * W + j * W + k);
      }
    }
  }
  Y_arr->col(y) *= scale;
}

template <typename T, StorageOrder kOrder>
void RunAveragePool1D(
    const int N,
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
        : ConstEigenArrayMap<T>(X_ptr, C, X_size);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_size, 1)
        : EigenArrayMap<T>(Y_ptr, C, Y_size);
    for (int y = 0; y < Y_size; ++y) {
      const int l = std::max(y * stride - pad, 0);
      const int r = std::min(y * stride - pad + kernel, X_size);
      const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
      ComputeAveragePool1D<T, kOrder>(l, r, y, scale, X_arr, &Y_arr);
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunAveragePool2D(
    const int N,
    const int C,
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_W, Y_H)
        : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
    for (int h = 0; h < Y_H; ++h) {
      const int t = std::max(h * stride_h - pad_t, 0);
      const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
      for (int w = 0; w < Y_W; ++w) {
        const int l = std::max(w * stride_w - pad_l, 0);
        const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
        const int y = h * Y_W + w;
        const T scale = T(1) /
            static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                             : (b - t) * (r - l));
        ComputeAveragePool2D<T, kOrder>(
            X_W, t, b, l, r, y, scale, X_arr, &Y_arr);
      }
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunAveragePool3D(
    const int N,
    const int C,
    const int X_D,
    const int X_H,
    const int X_W,
    const int Y_D,
    const int Y_H,
    const int Y_W,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_p,
    const int pad_t,
    const int pad_l,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
        : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
    for (int d = 0; d < Y_D; ++d) {
      const int p = std::max(d * stride_d - pad_p, 0);
      const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);
      for (int h = 0; h < Y_H; ++h) {
        const int t = std::max(h * stride_h - pad_t, 0);
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int w = 0; w < Y_W; ++w) {
          const int l = std::max(w * stride_w - pad_l, 0);
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int y = d * Y_H * Y_W + h * Y_W + w;
          const T scale = T(1) /
              static_cast<T>(count_include_pad ? kernel_d * kernel_h * kernel_w
                                               : (a - p) * (b - t) * (r - l));
          ComputeAveragePool3D<T, kOrder>(
              X_H, X_W, p, a, t, b, l, r, y, scale, X_arr, &Y_arr);
        }
      }
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPool1D(
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeMaxPool1D<float, StorageOrder::NCHW>(
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = X_arr.col(0).segment(l, r - l).maxCoeff();
}

template <>
void ComputeMaxPool1D<float, StorageOrder::NHWC>(
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y) = X_arr.col(l);
  for (int i = l + 1; i < r; ++i) {
    Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i));
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPool2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeMaxPool2D<float, StorageOrder::NCHW>(
    const int /* W */,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = X_arr.block(l, t, r - l, b - t).maxCoeff();
}

template <>
void ComputeMaxPool2D<float, StorageOrder::NHWC>(
    const int W,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y).setConstant(std::numeric_limits<float>::lowest());
  for (int i = t; i < b; ++i) {
    for (int j = l; j < r; ++j) {
      Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i * W + j));
    }
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPool3D(
    int H,
    int W,
    int p,
    int a,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& X_arr,
    EigenArrayMap<T>* Y_arr);

template <>
void ComputeMaxPool3D<float, StorageOrder::NCHW>(
    const int H,
    const int /* W */,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  (*Y_arr)(y) = std::numeric_limits<float>::lowest();
  for (int i = p; i < a; ++i) {
    (*Y_arr)(y) = std::max(
        (*Y_arr)(y), X_arr.block(l, i * H + t, r - l, b - t).maxCoeff());
  }
}

template <>
void ComputeMaxPool3D<float, StorageOrder::NHWC>(
    const int H,
    const int W,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& X_arr,
    EigenArrayMap<float>* Y_arr) {
  Y_arr->col(y).setConstant(std::numeric_limits<float>::lowest());
  for (int i = p; i < a; ++i) {
    for (int j = t; j < b; ++j) {
      for (int k = l; k < r; ++k) {
        Y_arr->col(y) = Y_arr->col(y).max(X_arr.col(i * H * W + j * W + k));
      }
    }
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPool1D(
    const int N,
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
        : ConstEigenArrayMap<T>(X_ptr, C, X_size);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_size, 1)
        : EigenArrayMap<T>(Y_ptr, C, Y_size);
    for (int y = 0; y < Y_size; ++y) {
      const int l = std::max(y * stride - pad, 0);
      const int r = std::min(y * stride - pad + kernel, X_size);
      ComputeMaxPool1D<T, kOrder>(l, r, y, X_arr, &Y_arr);
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPool2D(
    const int N,
    const int C,
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_W, Y_H)
        : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
    for (int h = 0; h < Y_H; ++h) {
      const int t = std::max(h * stride_h - pad_t, 0);
      const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
      for (int w = 0; w < Y_W; ++w) {
        const int l = std::max(w * stride_w - pad_l, 0);
        const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
        const int y = h * Y_W + w;
        ComputeMaxPool2D<T, kOrder>(X_W, t, b, l, r, y, X_arr, &Y_arr);
      }
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}
template <typename T, StorageOrder kOrder>
void RunMaxPool3D(
    const int N,
    const int C,
    const int X_D,
    const int X_H,
    const int X_W,
    const int Y_D,
    const int Y_H,
    const int Y_W,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_p,
    const int pad_t,
    const int pad_l,
    const T* X,
    T* Y) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  const T* X_ptr = X;
  T* Y_ptr = Y;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    EigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
        : EigenArrayMap<T>(Y_ptr, C, Y_HxW);
    for (int d = 0; d < Y_D; ++d) {
      const int p = std::max(d * stride_d - pad_p, 0);
      const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);
      for (int h = 0; h < Y_H; ++h) {
        const int t = std::max(h * stride_h - pad_t, 0);
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int w = 0; w < Y_W; ++w) {
          const int l = std::max(w * stride_w - pad_l, 0);
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int y = d * Y_H * Y_W + h * Y_W + w;
          ComputeMaxPool3D<T, kOrder>(
              X_H, X_W, p, a, t, b, l, r, y, X_arr, &Y_arr);
        }
      }
    }
    X_ptr += X_stride;
    Y_ptr += Y_stride;
  }
}

} // namespace

template <>
template <>
bool AveragePoolFunctor<CPUContext>::
    GlobalPoolingForward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CPUContext* context) const {
  const std::array<int, 2> X_dims = {N * C, HxW};
  const std::array<int, 2> Y_dims = {N * C, 1};
  math::ReduceMean<float, CPUContext>(
      2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
  return true;
}

template <>
template <>
bool AveragePoolFunctor<CPUContext>::
    GlobalPoolingForward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CPUContext* context) const {
  math::Set<float, CPUContext>(N * C, 0.0f, Y, context);
  const float* X_ptr = X;
  float* Y_ptr = Y;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < HxW; ++j) {
      math::Add<float, CPUContext>(C, Y_ptr, X_ptr + j * C, Y_ptr, context);
    }
    X_ptr += HxW * C;
    Y_ptr += C;
  }
  math::Scale<float, float, CPUContext>(
      N * C, 1.0f / static_cast<float>(HxW), Y, Y, context);
  return true;
}

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD(T, kOrder)           \
  template <>                                                                \
  template <>                                                                \
  bool AveragePoolFunctor<CPUContext>::Forward<T, kOrder>(                   \
      const int N,                                                           \
      const int C,                                                           \
      const std::vector<int>& X_dims,                                        \
      const std::vector<int>& Y_dims,                                        \
      const std::vector<int>& kernel,                                        \
      const std::vector<int>& dilation,                                      \
      const std::vector<int>& stride,                                        \
      const std::vector<int>& pads,                                          \
      const T* X,                                                            \
      T* Y,                                                                  \
      CPUContext* /* context */) const {                                     \
    const int ndim = X_dims.size();                                          \
    switch (ndim) {                                                          \
      case 1: {                                                              \
        RunAveragePool1D<T, kOrder>(                                         \
            N,                                                               \
            C,                                                               \
            X_dims[0],                                                       \
            Y_dims[0],                                                       \
            kernel[0],                                                       \
            stride[0],                                                       \
            pads[0],                                                         \
            count_include_pad,                                               \
            X,                                                               \
            Y);                                                              \
        return true;                                                         \
      }                                                                      \
      case 2: {                                                              \
        if (std::is_same<T, float>::value && kOrder == StorageOrder::NCHW && \
            pool_op_util::IsNeon4x4p0s0Eligible(                             \
                X_dims[0],                                                   \
                X_dims[1],                                                   \
                Y_dims[0],                                                   \
                Y_dims[1],                                                   \
                kernel[0],                                                   \
                kernel[1],                                                   \
                stride[0],                                                   \
                stride[1],                                                   \
                pads[0],                                                     \
                pads[1],                                                     \
                pads[2],                                                     \
                pads[3],                                                     \
                dilation[0],                                                 \
                dilation[1],                                                 \
                X,                                                           \
                Y)) {                                                        \
          pool_op_util::RunNeonAveragePool4x4p0s0NCHW(                       \
              N, C, X_dims[0], X_dims[1], X, Y);                             \
        } else {                                                             \
          RunAveragePool2D<T, kOrder>(                                       \
              N,                                                             \
              C,                                                             \
              X_dims[0],                                                     \
              X_dims[1],                                                     \
              Y_dims[0],                                                     \
              Y_dims[1],                                                     \
              kernel[0],                                                     \
              kernel[1],                                                     \
              stride[0],                                                     \
              stride[1],                                                     \
              pads[0],                                                       \
              pads[1],                                                       \
              count_include_pad,                                             \
              X,                                                             \
              Y);                                                            \
        }                                                                    \
        return true;                                                         \
      }                                                                      \
      case 3: {                                                              \
        RunAveragePool3D<T, kOrder>(                                         \
            N,                                                               \
            C,                                                               \
            X_dims[0],                                                       \
            X_dims[1],                                                       \
            X_dims[2],                                                       \
            Y_dims[0],                                                       \
            Y_dims[1],                                                       \
            Y_dims[2],                                                       \
            kernel[0],                                                       \
            kernel[1],                                                       \
            kernel[2],                                                       \
            stride[0],                                                       \
            stride[1],                                                       \
            stride[2],                                                       \
            pads[0],                                                         \
            pads[1],                                                         \
            pads[2],                                                         \
            count_include_pad,                                               \
            X,                                                               \
            Y);                                                              \
        return true;                                                         \
      }                                                                      \
      default: {                                                             \
        CAFFE_THROW("Unsupported pooling dim: ", ndim);                      \
        return false;                                                        \
      }                                                                      \
    }                                                                        \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD

template <>
template <>
bool MaxPoolFunctor<CPUContext>::
    GlobalPoolingForward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CPUContext* context) const {
  const std::array<int, 2> X_dims = {N * C, HxW};
  const std::array<int, 2> Y_dims = {N * C, 1};
  math::ReduceMax<float, CPUContext>(
      2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
  return true;
}

template <>
template <>
bool MaxPoolFunctor<CPUContext>::
    GlobalPoolingForward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CPUContext* context) const {
  math::Set<float, CPUContext>(
      N * C, std::numeric_limits<float>::lowest(), Y, context);
  const float* X_ptr = X;
  float* Y_ptr = Y;
  for (int i = 0; i < N; ++i) {
    ConstEigenArrayMap<float> X_arr(X_ptr, C, HxW);
    EigenVectorArrayMap<float> Y_arr(Y_ptr, C);
    for (int j = 0; j < HxW; ++j) {
      Y_arr = Y_arr.max(X_arr.col(j));
    }
    X_ptr += HxW * C;
    Y_ptr += C;
  }
  return true;
}

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD(T, kOrder)                \
  template <>                                                                 \
  template <>                                                                 \
  bool MaxPoolFunctor<CPUContext>::Forward<T, kOrder>(                        \
      const int N,                                                            \
      const int C,                                                            \
      const std::vector<int>& X_dims,                                         \
      const std::vector<int>& Y_dims,                                         \
      const std::vector<int>& kernel,                                         \
      const std::vector<int>& dilation,                                       \
      const std::vector<int>& stride,                                         \
      const std::vector<int>& pads,                                           \
      const T* X,                                                             \
      T* Y,                                                                   \
      CPUContext* /* context */) const {                                      \
    const int ndim = X_dims.size();                                           \
    switch (ndim) {                                                           \
      case 1: {                                                               \
        RunMaxPool1D<T, kOrder>(                                              \
            N, C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y); \
        return true;                                                          \
      }                                                                       \
      case 2: {                                                               \
        if (std::is_same<T, float>::value && kOrder == StorageOrder::NCHW &&  \
            pool_op_util::IsNeon2x2p0s0Eligible(                              \
                X_dims[0],                                                    \
                X_dims[1],                                                    \
                Y_dims[0],                                                    \
                Y_dims[1],                                                    \
                kernel[0],                                                    \
                kernel[1],                                                    \
                stride[0],                                                    \
                stride[1],                                                    \
                pads[0],                                                      \
                pads[1],                                                      \
                pads[2],                                                      \
                pads[3],                                                      \
                dilation[0],                                                  \
                dilation[1],                                                  \
                X,                                                            \
                Y)) {                                                         \
          pool_op_util::RunNeonMaxPool2x2p0s0NCHW(                            \
              N, C, X_dims[0], X_dims[1], X, Y);                              \
        } else {                                                              \
          RunMaxPool2D<T, kOrder>(                                            \
              N,                                                              \
              C,                                                              \
              X_dims[0],                                                      \
              X_dims[1],                                                      \
              Y_dims[0],                                                      \
              Y_dims[1],                                                      \
              kernel[0],                                                      \
              kernel[1],                                                      \
              stride[0],                                                      \
              stride[1],                                                      \
              pads[0],                                                        \
              pads[1],                                                        \
              X,                                                              \
              Y);                                                             \
        }                                                                     \
        return true;                                                          \
      }                                                                       \
      case 3: {                                                               \
        RunMaxPool3D<T, kOrder>(                                              \
            N,                                                                \
            C,                                                                \
            X_dims[0],                                                        \
            X_dims[1],                                                        \
            X_dims[2],                                                        \
            Y_dims[0],                                                        \
            Y_dims[1],                                                        \
            Y_dims[2],                                                        \
            kernel[0],                                                        \
            kernel[1],                                                        \
            kernel[2],                                                        \
            stride[0],                                                        \
            stride[1],                                                        \
            stride[2],                                                        \
            pads[0],                                                          \
            pads[1],                                                          \
            pads[2],                                                          \
            X,                                                                \
            Y);                                                               \
        return true;                                                          \
      }                                                                       \
      default: {                                                              \
        CAFFE_THROW("Unsupported pooling dim: ", ndim);                       \
        return false;                                                         \
      }                                                                       \
    }                                                                         \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
constexpr char kAveragePoolDoc[] = R"DOC(
consumes an input blob and applies average pooling across the the blob according
to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists
of taking the average value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-0.2883434   0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
   [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
     0.9989975 ]
   [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
     0.33195344]
   [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
     0.2586746 ]
   [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
   [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

Y:
 [[[[0.7462672  0.83399826 0.2948959 ]
   [0.4843537  0.3506009  0.35500962]
   [0.9251013  0.19026303 0.13366827]]]]
```

</details>

)DOC";

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
constexpr char kMaxPoolDoc[] = R"DOC(
consumes an input blob and applies max pooling across the the blob according to
kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of
taking the maximum value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
   [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
     1.2798511e+00 -3.2214901e-01]
   [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
   [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
   [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
   [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
     7.2035044e-01  2.7642491e-01]]]]

Y:
 [[[[-0.28534958  1.108865    1.2798511 ]
   [ 1.6845188  -0.266332   -0.09642485]
   [ 2.4605224   0.13020347  0.72035044]]]]

```

</details>

)DOC";

std::function<void(OpSchema&)> AveragePoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    std::string doc = "AveragePool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
    // schema.Arg(
    //     "kernel", "*(type: int)* Size of the window to take an average
    //     over.");
    // schema.Arg("stride", "*(type: int)* Stride of the window.");
    // schema.Arg(
    //     "pad",
    //     "*(type: int)* Implicit zero padding to be added on both sides.");
    // schema.Arg(
    //     "dilation",
    //     "*(type: int)* Parameter that controls the stride of elements in the
    //     " "window.");
    // schema.Arg(
    //     "order",
    //     "*(type: string; default: 'NCHW')* Order of the blob dimensions.");
    // schema.Arg(
    //     "count_include_pad",
    //     "*(type: bool; default: False)* When True, will include the "
    //     "zero-padding in the averaging.");
  };
}

std::function<void(OpSchema&)> MaxPoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    std::string doc = "MaxPool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
    /*
    schema.Arg("kernel", "*(type: int)* Size of the window to take an average
    over."); schema.Arg("stride", "*(type: int)* Stride of the window.");
    schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both
    sides."); schema.Arg("dilation", "*(type: int)* Parameter that controls
    the stride of elements in the window."); schema.Arg("order", "*(type:
    string; default: 'NCHW')* Order of the blob dimensions.");
    */
  };
}
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    AveragePool,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(AveragePool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator(""))
    .InheritOnnxSchema();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    AveragePool1D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(AveragePool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("1D"))
    .InheritOnnxSchema("AveragePool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    AveragePool2D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(AveragePool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("2D"))
    .InheritOnnxSchema("AveragePool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    AveragePool3D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(AveragePool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("3D"))
    .InheritOnnxSchema("AveragePool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MaxPool,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MaxPool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator(""))
    .InheritOnnxSchema();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MaxPool1D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MaxPool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("1D"))
    .InheritOnnxSchema("MaxPool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MaxPool2D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MaxPool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("2D"))
    .InheritOnnxSchema("MaxPool");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MaxPool3D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MaxPool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("3D"))
    .InheritOnnxSchema("MaxPool");

} // namespace caffe2
