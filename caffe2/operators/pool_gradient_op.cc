#include "caffe2/operators/pool_op.h"

#include <string>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {

template <typename T, StorageOrder kOrder>
void ComputeAveragePoolGradient1D(
    int l,
    int r,
    int y,
    T scale,
    const ConstEigenArrayMap<T>& dY_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeAveragePoolGradient1D<float, StorageOrder::NCHW>(
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  dX_arr->col(0).segment(l, r - l) += dY_arr(y) * scale;
}

template <>
void ComputeAveragePoolGradient1D<float, StorageOrder::NHWC>(
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = l; i < r; ++i) {
    dX_arr->col(i) += dY_arr.col(y) * scale;
  }
}

template <typename T, StorageOrder kOrder>
void ComputeAveragePoolGradient2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    T scale,
    const ConstEigenArrayMap<T>& dY_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeAveragePoolGradient2D<float, StorageOrder::NCHW>(
    const int /* W */,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  dX_arr->block(l, t, r - l, b - t) += dY_arr(y) * scale;
}

template <>
void ComputeAveragePoolGradient2D<float, StorageOrder::NHWC>(
    const int W,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const float scale,
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = t; i < b; ++i) {
    for (int j = l; j < r; ++j) {
      dX_arr->col(i * W + j) += dY_arr.col(y) * scale;
    }
  }
}

template <typename T, StorageOrder kOrder>
void ComputeAveragePoolGradient3D(
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
    const ConstEigenArrayMap<T>& dY_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeAveragePoolGradient3D<float, StorageOrder::NCHW>(
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
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = p; i < a; ++i) {
    dX_arr->block(l, i * H + t, r - l, b - t) += dY_arr(y) * scale;
  }
}

template <>
void ComputeAveragePoolGradient3D<float, StorageOrder::NHWC>(
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
    const ConstEigenArrayMap<float>& dY_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = p; i < a; ++i) {
    for (int j = t; j < b; ++j) {
      for (int k = l; k < r; ++k) {
        dX_arr->col(i * H * W + j * W + k) += dY_arr.col(y) * scale;
      }
    }
  }
}

template <typename T, StorageOrder kOrder>
void RunAveragePoolGradient1D(
    const int N,
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const bool count_include_pad,
    const T* dY,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_size);
  const T* dY_ptr = dY;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_size, 1)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_size);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_size, 1)
        : EigenArrayMap<T>(dX_ptr, C, X_size);
    for (int y = 0; y < Y_size; ++y) {
      const int l = std::max(y * stride - pad, 0);
      const int r = std::min(y * stride - pad + kernel, X_size);
      const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
      ComputeAveragePoolGradient1D<T, kOrder>(l, r, y, scale, dY_arr, &dX_arr);
    }
    dY_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunAveragePoolGradient2D(
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
    const T* dY,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
  const T* dY_ptr = dY;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_H)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_W, X_H)
        : EigenArrayMap<T>(dX_ptr, C, X_HxW);
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
        ComputeAveragePoolGradient2D<T, kOrder>(
            X_W, t, b, l, r, y, scale, dY_arr, &dX_arr);
      }
    }
    dY_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunAveragePoolGradient3D(
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
    const T* dY,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
  const T* dY_ptr = dY;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_D * Y_H)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_W, X_D * X_H)
        : EigenArrayMap<T>(dX_ptr, C, X_HxW);
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
          ComputeAveragePoolGradient3D<T, kOrder>(
              X_H, X_W, p, a, t, b, l, r, y, scale, dY_arr, &dX_arr);
        }
      }
    }
    dY_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPoolGradient1D(
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& dY_arr,
    const ConstEigenArrayMap<T>& X_arr,
    const ConstEigenArrayMap<T>& Y_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeMaxPoolGradient1D<float, StorageOrder::NCHW>(
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  dX_arr->col(0).segment(l, r - l) +=
      (X_arr.col(0).segment(l, r - l) == Y_arr(y)).cast<float>() * dY_arr(y);
}

template <>
void ComputeMaxPoolGradient1D<float, StorageOrder::NHWC>(
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = l; i < r; ++i) {
    dX_arr->col(i) +=
        (X_arr.col(i) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPoolGradient2D(
    int W,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& dY_arr,
    const ConstEigenArrayMap<T>& X_arr,
    const ConstEigenArrayMap<T>& Y_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeMaxPoolGradient2D<float, StorageOrder::NCHW>(
    const int /* W */,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  dX_arr->block(l, t, r - l, b - t) +=
      (X_arr.block(l, t, r - l, b - t) == Y_arr(y)).cast<float>() * dY_arr(y);
}

template <>
void ComputeMaxPoolGradient2D<float, StorageOrder::NHWC>(
    const int W,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = t; i < b; ++i) {
    for (int j = l; j < r; ++j) {
      const int x = i * W + j;
      dX_arr->col(x) +=
          (X_arr.col(x) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
    }
  }
}

template <typename T, StorageOrder kOrder>
void ComputeMaxPoolGradient3D(
    int H,
    int W,
    int p,
    int a,
    int t,
    int b,
    int l,
    int r,
    int y,
    const ConstEigenArrayMap<T>& dY_arr,
    const ConstEigenArrayMap<T>& X_arr,
    const ConstEigenArrayMap<T>& Y_arr,
    EigenArrayMap<T>* dX_arr);

template <>
void ComputeMaxPoolGradient3D<float, StorageOrder::NCHW>(
    const int H,
    const int /* W */,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = p; i < a; ++i) {
    dX_arr->block(l, i * H + t, r - l, b - t) +=
        (X_arr.block(l, i * H + t, r - l, b - t) == Y_arr(y)).cast<float>() *
        dY_arr(y);
  }
}

template <>
void ComputeMaxPoolGradient3D<float, StorageOrder::NHWC>(
    const int H,
    const int W,
    const int p,
    const int a,
    const int t,
    const int b,
    const int l,
    const int r,
    const int y,
    const ConstEigenArrayMap<float>& dY_arr,
    const ConstEigenArrayMap<float>& X_arr,
    const ConstEigenArrayMap<float>& Y_arr,
    EigenArrayMap<float>* dX_arr) {
  for (int i = p; i < a; ++i) {
    for (int j = t; j < b; ++j) {
      for (int k = l; k < r; ++k) {
        const int x = i * H * W + j * W + k;
        dX_arr->col(x) +=
            (X_arr.col(x) == Y_arr.col(y)).cast<float>() * dY_arr.col(y);
      }
    }
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPoolGradient1D(
    const int N,
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_size : X_size * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_size : Y_size * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_size);
  const T* dY_ptr = dY;
  const T* X_ptr = X;
  const T* Y_ptr = Y;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_size, 1)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_size);
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_size, 1)
        : ConstEigenArrayMap<T>(X_ptr, C, X_size);
    ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(Y_ptr, Y_size, 1)
        : ConstEigenArrayMap<T>(Y_ptr, C, Y_size);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_size, 1)
        : EigenArrayMap<T>(dX_ptr, C, X_size);
    for (int y = 0; y < Y_size; ++y) {
      const int l = std::max(y * stride - pad, 0);
      const int r = std::min(y * stride - pad + kernel, X_size);
      ComputeMaxPoolGradient1D<T, kOrder>(
          l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
    }
    dY_ptr += Y_stride;
    X_ptr += X_stride;
    Y_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPoolGradient2D(
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
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
  const T* dY_ptr = dY;
  const T* X_ptr = X;
  const T* Y_ptr = Y;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_H)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(Y_ptr, Y_W, Y_H)
        : ConstEigenArrayMap<T>(Y_ptr, C, Y_HxW);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_W, X_H)
        : EigenArrayMap<T>(dX_ptr, C, X_HxW);
    for (int h = 0; h < Y_H; ++h) {
      const int t = std::max(h * stride_h - pad_t, 0);
      const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);
      for (int w = 0; w < Y_W; ++w) {
        const int l = std::max(w * stride_w - pad_l, 0);
        const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);
        const int y = h * Y_W + w;
        ComputeMaxPoolGradient2D<T, kOrder>(
            X_W, t, b, l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
      }
    }
    dY_ptr += Y_stride;
    X_ptr += X_stride;
    Y_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

template <typename T, StorageOrder kOrder>
void RunMaxPoolGradient3D(
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
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int batch_size = kOrder == StorageOrder::NCHW ? N * C : N;
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int X_stride = kOrder == StorageOrder::NCHW ? X_HxW : X_HxW * C;
  const int Y_stride = kOrder == StorageOrder::NCHW ? Y_HxW : Y_HxW * C;
  std::memset(dX, 0, sizeof(T) * N * C * X_HxW);
  const T* dY_ptr = dY;
  const T* X_ptr = X;
  const T* Y_ptr = Y;
  T* dX_ptr = dX;
  for (int i = 0; i < batch_size; ++i) {
    ConstEigenArrayMap<T> dY_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(dY_ptr, Y_W, Y_D * Y_H)
        : ConstEigenArrayMap<T>(dY_ptr, C, Y_HxW);
    ConstEigenArrayMap<T> X_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(X_ptr, X_W, X_D * X_H)
        : ConstEigenArrayMap<T>(X_ptr, C, X_HxW);
    ConstEigenArrayMap<T> Y_arr = kOrder == StorageOrder::NCHW
        ? ConstEigenArrayMap<T>(Y_ptr, Y_W, Y_D * Y_H)
        : ConstEigenArrayMap<T>(Y_ptr, C, Y_HxW);
    EigenArrayMap<T> dX_arr = kOrder == StorageOrder::NCHW
        ? EigenArrayMap<T>(dX_ptr, X_W, X_D * X_H)
        : EigenArrayMap<T>(dX_ptr, C, X_HxW);
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
          ComputeMaxPoolGradient3D<T, kOrder>(
              X_H, X_W, p, a, t, b, l, r, y, dY_arr, X_arr, Y_arr, &dX_arr);
        }
      }
    }
    dY_ptr += Y_stride;
    X_ptr += X_stride;
    Y_ptr += Y_stride;
    dX_ptr += X_stride;
  }
}

} // namespace

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_1D(T, kOrder) \
  template <>                                                          \
  template <>                                                          \
  bool AveragePoolFunctor<CPUContext>::Backward<T, kOrder, 1>(         \
      const int N,                                                     \
      const int C,                                                     \
      const std::array<int, 1>& X_dims,                                \
      const std::array<int, 1>& Y_dims,                                \
      const std::array<int, 1>& kernel,                                \
      const std::array<int, 1>& /* dilation */,                        \
      const std::array<int, 1>& stride,                                \
      const std::array<int, 2>& pads,                                  \
      const T* dY,                                                     \
      const T* /* X */,                                                \
      const T* /* Y */,                                                \
      T* dX,                                                           \
      CPUContext* /* context */) const {                               \
    RunAveragePoolGradient1D<T, kOrder>(                               \
        N,                                                             \
        C,                                                             \
        X_dims[0],                                                     \
        Y_dims[0],                                                     \
        kernel[0],                                                     \
        stride[0],                                                     \
        pads[0],                                                       \
        count_include_pad,                                             \
        dY,                                                            \
        dX);                                                           \
    return true;                                                       \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_1D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_1D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_1D

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_2D(T, kOrder) \
  template <>                                                          \
  template <>                                                          \
  bool AveragePoolFunctor<CPUContext>::Backward<T, kOrder, 2>(         \
      const int N,                                                     \
      const int C,                                                     \
      const std::array<int, 2>& X_dims,                                \
      const std::array<int, 2>& Y_dims,                                \
      const std::array<int, 2>& kernel,                                \
      const std::array<int, 2>& /* dilation */,                        \
      const std::array<int, 2>& stride,                                \
      const std::array<int, 4>& pads,                                  \
      const T* dY,                                                     \
      const T* /* X */,                                                \
      const T* /* Y */,                                                \
      T* dX,                                                           \
      CPUContext* /* context */) const {                               \
    RunAveragePoolGradient2D<T, kOrder>(                               \
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
        dY,                                                            \
        dX);                                                           \
    return true;                                                       \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_2D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_2D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_2D

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_3D(T, kOrder) \
  template <>                                                          \
  template <>                                                          \
  bool AveragePoolFunctor<CPUContext>::Backward<T, kOrder, 3>(         \
      const int N,                                                     \
      const int C,                                                     \
      const std::array<int, 3>& X_dims,                                \
      const std::array<int, 3>& Y_dims,                                \
      const std::array<int, 3>& kernel,                                \
      const std::array<int, 3>& /* dilation */,                        \
      const std::array<int, 3>& stride,                                \
      const std::array<int, 6>& pads,                                  \
      const T* dY,                                                     \
      const T* /* X */,                                                \
      const T* /* Y */,                                                \
      T* dX,                                                           \
      CPUContext* /* context */) const {                               \
    RunAveragePoolGradient3D<T, kOrder>(                               \
        N,                                                             \
        C,                                                             \
        X_dims[0],                                                     \
        X_dims[1],                                                     \
        X_dims[2],                                                     \
        Y_dims[0],                                                     \
        Y_dims[1],                                                     \
        Y_dims[2],                                                     \
        kernel[0],                                                     \
        kernel[1],                                                     \
        kernel[2],                                                     \
        stride[0],                                                     \
        stride[1],                                                     \
        stride[2],                                                     \
        pads[0],                                                       \
        pads[1],                                                       \
        pads[2],                                                       \
        count_include_pad,                                             \
        dY,                                                            \
        dX);                                                           \
    return true;                                                       \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_3D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_3D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_BACKWARD_3D

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_1D(T, kOrder) \
  template <>                                                      \
  template <>                                                      \
  bool MaxPoolFunctor<CPUContext>::Backward<T, kOrder, 1>(         \
      const int N,                                                 \
      const int C,                                                 \
      const std::array<int, 1>& X_dims,                            \
      const std::array<int, 1>& Y_dims,                            \
      const std::array<int, 1>& kernel,                            \
      const std::array<int, 1>& /* dilation */,                    \
      const std::array<int, 1>& stride,                            \
      const std::array<int, 2>& pads,                              \
      const T* dY,                                                 \
      const T* X,                                                  \
      const T* Y,                                                  \
      T* dX,                                                       \
      CPUContext* /* context */) const {                           \
    RunMaxPoolGradient1D<T, kOrder>(                               \
        N,                                                         \
        C,                                                         \
        X_dims[0],                                                 \
        Y_dims[0],                                                 \
        kernel[0],                                                 \
        stride[0],                                                 \
        pads[0],                                                   \
        dY,                                                        \
        X,                                                         \
        Y,                                                         \
        dX);                                                       \
    return true;                                                   \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_1D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_1D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_1D

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_2D(T, kOrder) \
  template <>                                                      \
  template <>                                                      \
  bool MaxPoolFunctor<CPUContext>::Backward<T, kOrder, 2>(         \
      const int N,                                                 \
      const int C,                                                 \
      const std::array<int, 2>& X_dims,                            \
      const std::array<int, 2>& Y_dims,                            \
      const std::array<int, 2>& kernel,                            \
      const std::array<int, 2>& /* dilation */,                    \
      const std::array<int, 2>& stride,                            \
      const std::array<int, 4>& pads,                              \
      const T* dY,                                                 \
      const T* X,                                                  \
      const T* Y,                                                  \
      T* dX,                                                       \
      CPUContext* /* context */) const {                           \
    RunMaxPoolGradient2D<T, kOrder>(                               \
        N,                                                         \
        C,                                                         \
        X_dims[0],                                                 \
        X_dims[1],                                                 \
        Y_dims[0],                                                 \
        Y_dims[1],                                                 \
        kernel[0],                                                 \
        kernel[1],                                                 \
        stride[0],                                                 \
        stride[1],                                                 \
        pads[0],                                                   \
        pads[1],                                                   \
        dY,                                                        \
        X,                                                         \
        Y,                                                         \
        dX);                                                       \
    return true;                                                   \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_2D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_2D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_2D

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_3D(T, kOrder) \
  template <>                                                      \
  template <>                                                      \
  bool MaxPoolFunctor<CPUContext>::Backward<T, kOrder, 3>(         \
      const int N,                                                 \
      const int C,                                                 \
      const std::array<int, 3>& X_dims,                            \
      const std::array<int, 3>& Y_dims,                            \
      const std::array<int, 3>& kernel,                            \
      const std::array<int, 3>& /* dilation */,                    \
      const std::array<int, 3>& stride,                            \
      const std::array<int, 6>& pads,                              \
      const T* dY,                                                 \
      const T* X,                                                  \
      const T* Y,                                                  \
      T* dX,                                                       \
      CPUContext* /* context */) const {                           \
    RunMaxPoolGradient3D<T, kOrder>(                               \
        N,                                                         \
        C,                                                         \
        X_dims[0],                                                 \
        X_dims[1],                                                 \
        X_dims[2],                                                 \
        Y_dims[0],                                                 \
        Y_dims[1],                                                 \
        Y_dims[2],                                                 \
        kernel[0],                                                 \
        kernel[1],                                                 \
        kernel[2],                                                 \
        stride[0],                                                 \
        stride[1],                                                 \
        stride[2],                                                 \
        pads[0],                                                   \
        pads[1],                                                   \
        pads[2],                                                   \
        dY,                                                        \
        X,                                                         \
        Y,                                                         \
        dX);                                                       \
    return true;                                                   \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_3D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_3D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_BACKWARD_3D

REGISTER_CPU_OPERATOR(
    AveragePoolGradient,
    PoolGradientOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(AveragePoolGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool1DGradient,
    PoolGradientOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(AveragePool1DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool2DGradient,
    PoolGradientOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(AveragePool2DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    AveragePool3DGradient,
    PoolGradientOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(AveragePool3DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPoolGradient,
    PoolGradientOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(MaxPoolGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool1DGradient,
    PoolGradientOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(MaxPool1DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool2DGradient,
    PoolGradientOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(MaxPool2DGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    MaxPool3DGradient,
    PoolGradientOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);
OPERATOR_SCHEMA(MaxPool3DGradient).NumInputs(3).NumOutputs(1);

namespace {

class GetPoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        std::vector<std::string>{I(0), O(0), GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(AveragePool, GetPoolGradient);
REGISTER_GRADIENT(AveragePool1D, GetPoolGradient);
REGISTER_GRADIENT(AveragePool2D, GetPoolGradient);
REGISTER_GRADIENT(AveragePool3D, GetPoolGradient);

REGISTER_GRADIENT(MaxPool, GetPoolGradient);
REGISTER_GRADIENT(MaxPool1D, GetPoolGradient);
REGISTER_GRADIENT(MaxPool2D, GetPoolGradient);
REGISTER_GRADIENT(MaxPool3D, GetPoolGradient);

} // namespace caffe2
