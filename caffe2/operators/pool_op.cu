#include "caffe2/operators/pool_op.h"

#include <c10/util/accumulate.h>
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

#include <array>
#include <functional>
#include <limits>
#include <numeric>

namespace caffe2 {

namespace {

template <typename T>
__global__ void AveragePool1DForwardNCHWCUDAKernel(
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int nc = blockIdx.x;
  const T* X_ptr = X + nc * X_size;
  T* Y_ptr = Y + nc * Y_size;
  for (int y = threadIdx.x; y < Y_size; y += blockDim.x) {
    const int x = y * stride - pad;
    const int l = max(x, 0);
    const int r = min(x + kernel, X_size);
    const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
    T sum = 0;
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      sum += __ldg(X_ptr + i);
#else
      sum += X_ptr[i];
#endif
    }
    Y_ptr[y] = sum * scale;
  }
}

template <typename T>
__global__ void AveragePool1DForwardNHWCCUDAKernel(
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const bool count_include_pad,
    const T* X,
    T* Y) {
  const int n = blockIdx.x / Y_size;
  const int y = blockIdx.x % Y_size;
  const int x = y * stride - pad;
  const int l = max(x, 0);
  const int r = min(x + kernel, X_size);
  const T scale = T(1) / static_cast<T>(count_include_pad ? kernel : r - l);
  const T* X_ptr = X + n * X_size * C;
  T* Y_ptr = Y + n * Y_size * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      sum += __ldg(X_ptr + i * C + c);
#else
      sum += X_ptr[i * C + c];
#endif
    }
    Y_ptr[y * C + c] = sum * scale;
  }
}

template <typename T>
__global__ void AveragePool2DForwardNCHWCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / Y_H;
  const int yh = blockIdx.x % Y_H;
  const T* X_ptr = X + nc * X_HxW;
  T* Y_ptr = Y + nc * Y_HxW;
  const int xh = yh * stride_h - pad_t;
  const int t = max(xh, 0);
  const int b = min(xh + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w - pad_l;
    const int l = max(xw, 0);
    const int r = min(xw + kernel_w, X_W);
    const T scale = T(1) /
        static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                         : (b - t) * (r - l));
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >= 350
        sum += __ldg(X_ptr + i * X_W + j);
#else
        sum += X_ptr[i * X_W + j];
#endif
      }
    }
    Y_ptr[yh * Y_W + yw] = sum * scale;
  }
}

template <typename T>
__global__ void AveragePool2DForwardNHWCCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int n = blockIdx.x / Y_HxW;
  const int y = blockIdx.x % Y_HxW;
  const int yh = y / Y_W;
  const int yw = y % Y_W;
  const int xh = yh * stride_h - pad_t;
  const int xw = yw * stride_w - pad_l;
  const int t = max(xh, 0);
  const int b = min(xh + kernel_h, X_H);
  const int l = max(xw, 0);
  const int r = min(xw + kernel_w, X_W);
  const T scale = T(1) /
      static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                       : (b - t) * (r - l));
  const T* X_ptr = X + n * X_HxW * C;
  T* Y_ptr = Y + n * Y_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >= 350
        sum += __ldg(X_ptr + (i * X_W + j) * C + c);
#else
        sum += X_ptr[(i * X_W + j) * C + c];
#endif
      }
    }
    Y_ptr[y * C + c] = sum * scale;
  }
}

template <typename T>
__global__ void AveragePool3DForwardNCHWCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int yy = blockIdx.x / Y_H;
  const int nc = yy / Y_D;
  const int yd = yy % Y_D;
  const int yh = blockIdx.x % Y_H;
  const T* X_ptr = X + nc * X_HxW;
  T* Y_ptr = Y + nc * Y_HxW;
  const int xd = yd * stride_d - pad_p;
  const int xh = yh * stride_h - pad_t;
  const int p = max(xd, 0);
  const int a = min(xd + kernel_d, X_D);
  const int t = max(xh, 0);
  const int b = min(xh + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w - pad_l;
    const int l = max(xw, 0);
    const int r = min(xw + kernel_w, X_W);
    const T scale = T(1) /
        static_cast<T>(count_include_pad ? kernel_d * kernel_h * kernel_w
                                         : (a - p) * (b - t) * (r - l));
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
#if __CUDA_ARCH__ >= 350
          sum += __ldg(X_ptr + (i * X_H + j) * X_W + k);
#else
          sum += X_ptr[(i * X_H + j) * X_W + k];
#endif
        }
      }
    }
    Y_ptr[(yd * Y_H + yh) * Y_W + yw] = sum * scale;
  }
}

template <typename T>
__global__ void AveragePool3DForwardNHWCCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int n = blockIdx.x / Y_HxW;
  const int y = blockIdx.x % Y_HxW;
  const int yy = y / Y_W;
  const int yd = yy / Y_H;
  const int yh = yy % Y_H;
  const int yw = y % Y_W;
  const int xd = yd * stride_d - pad_p;
  const int xh = yh * stride_h - pad_t;
  const int xw = yw * stride_w - pad_l;
  const int p = max(xd, 0);
  const int a = min(xd + kernel_d, X_D);
  const int t = max(xh, 0);
  const int b = min(xh + kernel_h, X_H);
  const int l = max(xw, 0);
  const int r = min(xw + kernel_w, X_W);
  const T scale = T(1) /
      static_cast<T>(count_include_pad ? kernel_d * kernel_h * kernel_w
                                       : (a - p) * (b - t) * (r - l));
  const T* X_ptr = X + n * X_HxW * C;
  T* Y_ptr = Y + n * Y_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
#if __CUDA_ARCH__ >= 350
          sum += __ldg(X_ptr + ((i * X_H + j) * X_W + k) * C + c);
#else
          sum += X_ptr[((i * X_H + j) * X_W + k) * C + c];
#endif
        }
      }
    }
    Y_ptr[y * C + c] = sum * scale;
  }
}

template <typename T>
__global__ void GlobalAveragePoolingBackwardNCHWCUDAKernel(
    const int K,
    const int HxW,
    const T scale,
    const T* dY,
    T* dX) {
  const int nc = blockIdx.x / K;
  const int block = blockIdx.x % K;
  const int x = threadIdx.x + block * CAFFE_CUDA_NUM_THREADS;
  if (x < HxW) {
#if __CUDA_ARCH__ >= 350
    dX[nc * HxW + x] = __ldg(dY + nc) * scale;
#else
    dX[nc * HxW + x] = dY[nc] * scale;
#endif
  }
}

template <typename T>
__global__ void GlobalAveragePoolingBackwardNHWCCUDAKernel(
    const int C,
    const int HxW,
    const T scale,
    const T* dY,
    T* dX) {
  const int n = blockIdx.x / HxW;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
#if __CUDA_ARCH__ >= 350
    dX[blockIdx.x * C + c] = __ldg(dY + n * C + c) * scale;
#else
    dX[blockIdx.x * C + c] = dY[n * C + c] * scale;
#endif
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool1DBackwardNCHWCUDAKernel(
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* dY,
    T* dX) {
  const int nc = blockIdx.x;
  const T* dY_ptr = dY + nc * Y_size;
  T* dX_ptr = dX + nc * X_size;
  for (int x = threadIdx.x; x < X_size; x += blockDim.x) {
    const int w = x + pad;
    const int l = w < kernel ? 0 : (w - kernel) / stride + 1;
    const int r = min(w / stride + 1, Y_size);
    T sum = 0;
    for (int i = l; i < r; ++i) {
      if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
        sum += __ldg(dY_ptr + i);
#else
        sum += dY_ptr[i];
#endif
      } else {
        const int xx = i * stride - pad;
        const int xl = max(xx, 0);
        const int xr = min(xx + kernel, X_size);
#if __CUDA_ARCH__ >= 350
        sum += __ldg(dY_ptr + i) / static_cast<T>(xr - xl);
#else
        sum += dY_ptr[i] / static_cast<T>(xr - xl);
#endif
      }
    }
    if (kCountIncludePad) {
      dX_ptr[x] = sum / static_cast<T>(kernel);
    } else {
      dX_ptr[x] = sum;
    }
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool1DBackwardNHWCCUDAKernel(
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* dY,
    T* dX) {
  const int n = blockIdx.x / X_size;
  const int x = blockIdx.x % X_size;
  const int w = x + pad;
  const int l = w < kernel ? 0 : (w - kernel) / stride + 1;
  const int r = min(w / stride + 1, Y_size);
  const T scale = T(1) / static_cast<T>(kernel);
  const T* dY_ptr = dY + n * Y_size * C;
  T* dX_ptr = dX + n * X_size * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = l; i < r; ++i) {
      if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
        sum += __ldg(dY_ptr + i * C + c);
#else
        sum += dY_ptr[i * C + c];
#endif
      } else {
        const int xx = i * stride - pad;
        const int xl = max(xx, 0);
        const int xr = min(xx + kernel, X_size);
#if __CUDA_ARCH__ >= 350
        sum += __ldg(dY_ptr + i * C + c) / static_cast<T>(xr - xl);
#else
        sum += dY_ptr[i * C + c] / static_cast<T>(xr - xl);
#endif
      }
    }
    if (kCountIncludePad) {
      dX_ptr[x * C + c] = sum * scale;
    } else {
      dX_ptr[x * C + c] = sum;
    }
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool2DBackwardNCHWCUDAKernel(
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
    T* dX) {
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / X_H;
  const int hh = blockIdx.x % X_H;
  const T* dY_ptr = dY + nc * Y_HxW;
  T* dX_ptr = dX + nc * X_HxW;
  const int h = hh + pad_t;
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  for (int ww = threadIdx.x; ww < X_W; ww += blockDim.x) {
    const int w = ww + pad_l;
    const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int r = min(w / stride_w + 1, Y_W);
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
        if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + i * Y_W + j);
#else
          sum += dY_ptr[i * Y_W + j];
#endif
        } else {
          const int xh = i * stride_h - pad_t;
          const int xw = j * stride_w - pad_l;
          const int xt = max(xh, 0);
          const int xb = min(xh + kernel_h, X_H);
          const int xl = max(xw, 0);
          const int xr = min(xw + kernel_w, X_W);
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + i * Y_W + j) /
              static_cast<T>((xb - xt) * (xr - xl));
#else
          sum += dY_ptr[i * Y_W + j] / static_cast<T>((xb - xt) * (xr - xl));
#endif
        }
      }
    }
    if (kCountIncludePad) {
      dX_ptr[hh * X_W + ww] = sum / static_cast<T>(kernel_h * kernel_w);
    } else {
      dX_ptr[hh * X_W + ww] = sum;
    }
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool2DBackwardNHWCCUDAKernel(
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
    T* dX) {
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int n = blockIdx.x / X_HxW;
  const int x = blockIdx.x % X_HxW;
  const int h = x / X_W + pad_t;
  const int w = x % X_W + pad_l;
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
  const int r = min(w / stride_w + 1, Y_W);
  const T scale = T(1) / static_cast<T>(kernel_h * kernel_w);
  const T* dY_ptr = dY + n * Y_HxW * C;
  T* dX_ptr = dX + n * X_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
        if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + (i * Y_W + j) * C + c);
#else
          sum += dY_ptr[(i * Y_W + j) * C + c];
#endif
        } else {
          const int xh = i * stride_h - pad_t;
          const int xw = j * stride_w - pad_l;
          const int xt = max(xh, 0);
          const int xb = min(xh + kernel_h, X_H);
          const int xl = max(xw, 0);
          const int xr = min(xw + kernel_w, X_W);
#if __CUDA_ARCH__ >= 350
          sum += __ldg(dY_ptr + (i * Y_W + j) * C + c) /
              static_cast<T>((xb - xt) * (xr - xl));
#else
          sum += dY_ptr[(i * Y_W + j) * C + c] /
              static_cast<T>((xb - xt) * (xr - xl));
#endif
        }
      }
    }
    if (kCountIncludePad) {
      dX_ptr[x * C + c] = sum * scale;
    } else {
      dX_ptr[x * C + c] = sum;
    }
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool3DBackwardNCHWCUDAKernel(
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
    T* dX) {
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int xx = blockIdx.x / X_H;
  const int nc = xx / X_D;
  const int dd = xx % X_D;
  const int hh = blockIdx.x % X_H;
  const T* dY_ptr = dY + nc * Y_HxW;
  T* dX_ptr = dX + nc * X_HxW;
  const int d = dd + pad_p;
  const int h = hh + pad_t;
  const int p = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
  const int a = min(d / stride_d + 1, Y_D);
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  for (int ww = threadIdx.x; ww < X_W; ww += blockDim.x) {
    const int w = ww + pad_l;
    const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int r = min(w / stride_w + 1, Y_W);
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
          if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
            sum += __ldg(dY_ptr + (i * Y_H + j) * Y_W + k);
#else
            sum += dY_ptr[(i * Y_H + j) * Y_W + k];
#endif
          } else {
            const int xd = i * stride_d - pad_p;
            const int xh = j * stride_h - pad_t;
            const int xw = k * stride_w - pad_l;
            const int xp = max(xd, 0);
            const int xa = min(xd + kernel_d, X_D);
            const int xt = max(xh, 0);
            const int xb = min(xh + kernel_h, X_H);
            const int xl = max(xw, 0);
            const int xr = min(xw + kernel_w, X_W);
#if __CUDA_ARCH__ >= 350
            sum += __ldg(dY_ptr + (i * Y_H + j) * Y_W + k) /
                static_cast<T>((xa - xp) * (xb - xt) * (xr - xl));
#else
            sum += dY_ptr[(i * Y_H + j) * Y_W + k] /
                static_cast<T>((xa - xp) * (xb - xt) * (xr - xl));
#endif
          }
        }
      }
    }
    if (kCountIncludePad) {
      dX_ptr[(dd * X_H + hh) * X_W + ww] =
          sum / static_cast<T>(kernel_d * kernel_h * kernel_w);
    } else {
      dX_ptr[(dd * X_H + hh) * X_W + ww] = sum;
    }
  }
}

template <typename T, bool kCountIncludePad>
__global__ void AveragePool3DBackwardNHWCCUDAKernel(
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
    T* dX) {
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int n = blockIdx.x / X_HxW;
  const int x = blockIdx.x % X_HxW;
  const int xx = x / X_W;
  const int d = xx / X_H + pad_p;
  const int h = xx % X_H + pad_t;
  const int w = x % X_W + pad_l;
  const int p = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
  const int a = min(d / stride_d + 1, Y_D);
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
  const int r = min(w / stride_w + 1, Y_W);
  const T scale = T(1) / static_cast<T>(kernel_d * kernel_h * kernel_w);
  const T* dY_ptr = dY + n * Y_HxW * C;
  T* dX_ptr = dX + n * X_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
          if (kCountIncludePad) {
#if __CUDA_ARCH__ >= 350
            sum += __ldg(dY_ptr + ((i * Y_H + j) * Y_W + k) * C + c);
#else
            sum += dY_ptr[((i * Y_H + j) * Y_W + k) * C + c];
#endif
          } else {
            const int xd = i * stride_d - pad_p;
            const int xh = j * stride_h - pad_t;
            const int xw = k * stride_w - pad_l;
            const int xp = max(xd, 0);
            const int xa = min(xd + kernel_d, X_D);
            const int xt = max(xh, 0);
            const int xb = min(xh + kernel_h, X_H);
            const int xl = max(xw, 0);
            const int xr = min(xw + kernel_w, X_W);
#if __CUDA_ARCH__ >= 350
            sum += __ldg(dY_ptr + ((i * Y_H + j) * Y_W + k) * C + c) /
                static_cast<T>((xa - xp) * (xb - xt) * (xr - xl));
#else
            sum += dY_ptr[((i * Y_H + j) * Y_W + k) * C + c] /
                static_cast<T>((xa - xp) * (xb - xt) * (xr - xl));
#endif
          }
        }
      }
    }
    if (kCountIncludePad) {
      dX_ptr[x * C + c] = sum * scale;
    } else {
      dX_ptr[x * C + c] = sum;
    }
  }
}

} // namespace

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::
    GlobalPoolingForward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CUDAContext* context) const {
  const std::array<int, 2> X_dims = {N * C, HxW};
  const std::array<int, 2> Y_dims = {N * C, 1};
  math::ReduceMean<float, CUDAContext>(
      2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
  return true;
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::
    GlobalPoolingForward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CUDAContext* context) const {
  if (ones.numel() != HxW) {
    ones.Resize(HxW);
    math::Set<float, CUDAContext>(
        HxW, 1.0f, ones.mutable_data<float>(), context);
  }
  math::GemmStridedBatched<float, CUDAContext>(
      CblasTrans,
      CblasNoTrans,
      N,
      C,
      1,
      HxW,
      1.0f / static_cast<float>(HxW),
      X,
      HxW * C,
      ones.data<float>(),
      0,
      0.0f,
      Y,
      C,
      context);
  return true;
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::Forward<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* X,
    float* Y,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  switch (ndim) {
    case 1: {
      const int num_blocks = N * C;
      AveragePool1DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      const int num_blocks = N * C * Y_dims[0];
      AveragePool2DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      const int num_blocks = N * C * Y_dims[0] * Y_dims[1];
      AveragePool3DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::Forward<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* X,
    float* Y,
    CUDAContext* context) const {
  // Each CUDA block handles one point, one thread per channel.
  const int ndim = X_dims.size();
  const auto Y_HxW = c10::multiply_integers(Y_dims.cbegin(), Y_dims.cend());
  switch (ndim) {
    case 1: {
      AveragePool1DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      AveragePool2DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      AveragePool3DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              count_include_pad,
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::
    GlobalPoolingBackward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* dY,
        const float* /* X */,
        const float* /* Y */,
        float* dX,
        CUDAContext* context) const {
  const float scale = 1.0f / static_cast<float>(HxW);
  const int K = (HxW + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  GlobalAveragePoolingBackwardNCHWCUDAKernel<float>
      <<<N * C * K, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          K, HxW, scale, dY, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::
    GlobalPoolingBackward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* dY,
        const float* /* X */,
        const float* /* Y */,
        float* dX,
        CUDAContext* context) const {
  const float scale = 1.0f / static_cast<float>(HxW);
  GlobalAveragePoolingBackwardNHWCCUDAKernel<float>
      <<<N * HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          C, HxW, scale, dY, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

#define DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(                       \
    cond, Func, T, num_blocks, threads_per_block, cuda_stream, ...)         \
  do {                                                                      \
    if (cond) {                                                             \
      Func<T, true>                                                         \
          <<<num_blocks, threads_per_block, 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    } else {                                                                \
      Func<T, false>                                                        \
          <<<num_blocks, threads_per_block, 0, cuda_stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
    }                                                                       \
  } while (false)

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::Backward<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* dY,
    const float* /* X */,
    const float* /* Y */,
    float* dX,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  switch (ndim) {
    case 1: {
      const int num_blocks = N * C;
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool1DBackwardNCHWCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          X_dims[0],
          Y_dims[0],
          kernel[0],
          stride[0],
          pads[0],
          dY,
          dX);
      return true;
    }
    case 2: {
      const int num_blocks = N * C * X_dims[0];
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool2DBackwardNCHWCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          X_dims[0],
          X_dims[1],
          Y_dims[0],
          Y_dims[1],
          kernel[0],
          kernel[1],
          stride[0],
          stride[1],
          pads[0],
          pads[1],
          dY,
          dX);
      return true;
    }
    case 3: {
      const int num_blocks = N * C * X_dims[0] * X_dims[1];
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool3DBackwardNCHWCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          X_dims[0],
          X_dims[1],
          X_dims[2],
          Y_dims[0],
          Y_dims[1],
          Y_dims[2],
          kernel[0],
          kernel[1],
          kernel[2],
          stride[0],
          stride[1],
          stride[2],
          pads[0],
          pads[1],
          pads[2],
          dY,
          dX);
      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool AveragePoolFunctor<CUDAContext>::Backward<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* dY,
    const float* /* X */,
    const float* /* Y */,
    float* dX,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  const auto X_HxW = c10::multiply_integers(X_dims.cbegin(), X_dims.cend());
  const int num_blocks = N * X_HxW;
  switch (ndim) {
    case 1: {
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool1DBackwardNHWCCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          C,
          X_dims[0],
          Y_dims[0],
          kernel[0],
          stride[0],
          pads[0],
          dY,
          dX);
      return true;
    }
    case 2: {
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool2DBackwardNHWCCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          C,
          X_dims[0],
          X_dims[1],
          Y_dims[0],
          Y_dims[1],
          kernel[0],
          kernel[1],
          stride[0],
          stride[1],
          pads[0],
          pads[1],
          dY,
          dX);
      return true;
    }
    case 3: {
      DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1(
          count_include_pad,
          AveragePool3DBackwardNHWCCUDAKernel,
          float,
          num_blocks,
          CAFFE_CUDA_NUM_THREADS,
          context->cuda_stream(),
          C,
          X_dims[0],
          X_dims[1],
          X_dims[2],
          Y_dims[0],
          Y_dims[1],
          Y_dims[2],
          kernel[0],
          kernel[1],
          kernel[2],
          stride[0],
          stride[1],
          stride[2],
          pads[0],
          pads[1],
          pads[2],
          dY,
          dX);
      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

#undef DISPATCH_KERNEL_FUNCTION_BY_BOOL_WITH_TYPE_1

namespace {

template <typename T>
__global__ void MaxPool1DForwardNCHWCUDAKernel(
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* X,
    T* Y) {
  const int nc = blockIdx.x;
  const T* X_ptr = X + nc * X_size;
  T* Y_ptr = Y + nc * Y_size;
  for (int y = threadIdx.x; y < Y_size; y += blockDim.x) {
    const int x = y * stride;
    const int l = max(x - pad, 0);
    const int r = min(x - pad + kernel, X_size);
    T val = std::numeric_limits<T>::lowest();
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      val = max(val, __ldg(X_ptr + i));
#else
      val = max(val, X_ptr[i]);
#endif
    }
    Y_ptr[y] = val;
  }
}

template <typename T>
__global__ void MaxPool1DForwardNHWCCUDAKernel(
    const int C,
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* X,
    T* Y) {
  const int n = blockIdx.x / Y_size;
  const int y = blockIdx.x % Y_size;
  const int x = y * stride;
  const int l = max(x - pad, 0);
  const int r = min(x - pad + kernel, X_size);
  const T* X_ptr = X + n * X_size * C;
  T* Y_ptr = Y + n * Y_size * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T val = std::numeric_limits<T>::lowest();
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      val = max(val, __ldg(X_ptr + i * C + c));
#else
      val = max(val, X_ptr[i * C + c]);
#endif
    }
    Y_ptr[y * C + c] = val;
  }
}

template <typename T>
__global__ void MaxPool2DForwardNCHWCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / Y_H;
  const int yh = blockIdx.x % Y_H;
  const T* X_ptr = X + nc * X_HxW;
  T* Y_ptr = Y + nc * Y_HxW;
  const int xh = yh * stride_h;
  const int t = max(xh - pad_t, 0);
  const int b = min(xh - pad_t + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w;
    const int l = max(xw - pad_l, 0);
    const int r = min(xw - pad_l + kernel_w, X_W);
    T val = std::numeric_limits<T>::lowest();
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >= 350
        val = max(val, __ldg(X_ptr + i * X_W + j));
#else
        val = max(val, X_ptr[i * X_W + j]);
#endif
      }
    }
    Y_ptr[yh * Y_W + yw] = val;
  }
}

template <typename T>
__global__ void MaxPool2DForwardNHWCCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int n = blockIdx.x / Y_HxW;
  const int y = blockIdx.x % Y_HxW;
  const int yh = y / Y_W;
  const int yw = y % Y_W;
  const int xh = yh * stride_h;
  const int xw = yw * stride_w;
  const int t = max(xh - pad_t, 0);
  const int b = min(xh - pad_t + kernel_h, X_H);
  const int l = max(xw - pad_l, 0);
  const int r = min(xw - pad_l + kernel_w, X_W);
  const T* X_ptr = X + n * X_HxW * C;
  T* Y_ptr = Y + n * Y_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T val = std::numeric_limits<T>::lowest();
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
#if __CUDA_ARCH__ >= 350
        val = max(val, __ldg(X_ptr + (i * X_W + j) * C + c));
#else
        val = max(val, X_ptr[(i * X_W + j) * C + c]);
#endif
      }
    }
    Y_ptr[y * C + c] = val;
  }
}

template <typename T>
__global__ void MaxPool3DForwardNCHWCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int yy = blockIdx.x / Y_H;
  const int nc = yy / Y_D;
  const int yd = yy % Y_D;
  const int yh = blockIdx.x % Y_H;
  const T* X_ptr = X + nc * X_HxW;
  T* Y_ptr = Y + nc * Y_HxW;
  const int xd = yd * stride_d;
  const int xh = yh * stride_h;
  const int p = max(xd - pad_p, 0);
  const int a = min(xd - pad_p + kernel_d, X_D);
  const int t = max(xh - pad_t, 0);
  const int b = min(xh - pad_t + kernel_h, X_H);
  for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x) {
    const int xw = yw * stride_w;
    const int l = max(xw - pad_l, 0);
    const int r = min(xw - pad_l + kernel_w, X_W);
    T val = std::numeric_limits<T>::lowest();
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
#if __CUDA_ARCH__ >= 350
          val = max(val, __ldg(X_ptr + (i * X_H + j) * X_W + k));
#else
          val = max(val, X_ptr[(i * X_H + j) * X_W + k]);
#endif
        }
      }
    }
    Y_ptr[(yd * Y_H + yh) * Y_W + yw] = val;
  }
}

template <typename T>
__global__ void MaxPool3DForwardNHWCCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int n = blockIdx.x / Y_HxW;
  const int y = blockIdx.x % Y_HxW;
  const int yy = y / Y_W;
  const int yw = y % Y_W;
  const int yh = yy % Y_H;
  const int yd = yy / Y_H;
  const int xd = yd * stride_d;
  const int xh = yh * stride_h;
  const int xw = yw * stride_w;
  const int p = max(xd - pad_p, 0);
  const int a = min(xd - pad_p + kernel_d, X_D);
  const int t = max(xh - pad_t, 0);
  const int b = min(xh - pad_t + kernel_h, X_H);
  const int l = max(xw - pad_l, 0);
  const int r = min(xw - pad_l + kernel_w, X_W);
  const T* X_ptr = X + n * X_HxW * C;
  T* Y_ptr = Y + n * Y_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T val = std::numeric_limits<T>::lowest();
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
#if __CUDA_ARCH__ >= 350
          val = max(val, __ldg(X_ptr + ((i * X_H + j) * X_W + k) * C + c));
#else
          val = max(val, X_ptr[((i * X_H + j) * X_W + k) * C + c]);
#endif
        }
      }
    }
    Y_ptr[y * C + c] = val;
  }
}

template <typename T>
__global__ void GlobalMaxPoolingBackwardNCHWCUDAKernel(
    const int K,
    const int HxW,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int nc = blockIdx.x / K;
  const int block = blockIdx.x % K;
  const int x = threadIdx.x + block * CAFFE_CUDA_NUM_THREADS;
  if (x < HxW) {
#if __CUDA_ARCH__ >= 350
    dX[nc * HxW + x] =
        (__ldg(X + nc * HxW + x) == __ldg(Y + nc)) ? __ldg(dY + nc) : T(0);
#else
    dX[nc * HxW + x] = (X[nc * HxW + x] == Y[nc]) ? dY[nc] : T(0);
#endif
  }
}

template <typename T>
__global__ void GlobalMaxPoolingBackwardNHWCCUDAKernel(
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int n = blockIdx.x / HxW;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
#if __CUDA_ARCH__ >= 350
    dX[blockIdx.x * C + c] =
        (__ldg(X + blockIdx.x * C + c) == __ldg(Y + n * C + c))
        ? __ldg(dY + n * C + c)
        : T(0);
#else
    dX[blockIdx.x * C + c] =
        (X[blockIdx.x * C + c] == Y[n * C + c]) ? dY[n * C + c] : T(0);
#endif
  }
}

template <typename T>
__global__ void MaxPool1DBackwardNCHWCUDAKernel(
    const int X_size,
    const int Y_size,
    const int kernel,
    const int stride,
    const int pad,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  const int nc = blockIdx.x;
  const T* dY_ptr = dY + nc * Y_size;
  const T* X_ptr = X + nc * X_size;
  const T* Y_ptr = Y + nc * Y_size;
  T* dX_ptr = dX + nc * X_size;
  for (int x = threadIdx.x; x < X_size; x += blockDim.x) {
    const int w = x + pad;
    const int l = w < kernel ? 0 : (w - kernel) / stride + 1;
    const int r = min(w / stride + 1, Y_size);
    T sum = 0;
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      if (__ldg(X_ptr + x) == __ldg(Y_ptr + i)) {
        sum += __ldg(dY_ptr + i);
      }
#else
      if (X_ptr[x] == Y_ptr[i]) {
        sum += dY_ptr[i];
      }
#endif
    }
    dX_ptr[x] = sum;
  }
}

template <typename T>
__global__ void MaxPool1DBackwardNHWCCUDAKernel(
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
  const int n = blockIdx.x / X_size;
  const int x = blockIdx.x % X_size;
  const int w = x + pad;
  const int l = w < kernel ? 0 : (w - kernel) / stride + 1;
  const int r = min(w / stride + 1, Y_size);
  const T* dY_ptr = dY + n * Y_size * C;
  const T* X_ptr = X + n * X_size * C;
  const T* Y_ptr = Y + n * Y_size * C;
  T* dX_ptr = dX + n * X_size * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = l; i < r; ++i) {
#if __CUDA_ARCH__ >= 350
      if (__ldg(X_ptr + x * C + c) == __ldg(Y_ptr + i * C + c)) {
        sum += __ldg(dY_ptr + i * C + c);
      }
#else
      if (X_ptr[x * C + c] == Y_ptr[i * C + c]) {
        sum += dY_ptr[i * C + c];
      }
#endif
    }
    dX_ptr[x * C + c] = sum;
  }
}

template <typename T>
__global__ void MaxPool2DBackwardNCHWCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int nc = blockIdx.x / X_H;
  const int xh = blockIdx.x % X_H;
  const T* dY_ptr = dY + nc * Y_HxW;
  const T* X_ptr = X + nc * X_HxW;
  const T* Y_ptr = Y + nc * Y_HxW;
  T* dX_ptr = dX + nc * X_HxW;
  const int h = xh + pad_t;
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  for (int xw = threadIdx.x; xw < X_W; xw += blockDim.x) {
    const int w = xw + pad_l;
    const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int r = min(w / stride_w + 1, Y_W);
    const int x = xh * X_W + xw;
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
        const int y = i * Y_W + j;
#if __CUDA_ARCH__ >= 350
        if (__ldg(X_ptr + x) == __ldg(Y_ptr + y)) {
          sum += __ldg(dY_ptr + y);
        }
#else
        if (X_ptr[x] == Y_ptr[y]) {
          sum += dY_ptr[y];
        }
#endif
      }
    }
    dX_ptr[x] = sum;
  }
}

template <typename T>
__global__ void MaxPool2DBackwardNHWCCUDAKernel(
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
  const int X_HxW = X_H * X_W;
  const int Y_HxW = Y_H * Y_W;
  const int n = blockIdx.x / X_HxW;
  const int x = blockIdx.x % X_HxW;
  const int h = x / X_W + pad_t;
  const int w = x % X_W + pad_l;
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
  const int r = min(w / stride_w + 1, Y_W);
  const T* dY_ptr = dY + n * Y_HxW * C;
  const T* X_ptr = X + n * X_HxW * C;
  const T* Y_ptr = Y + n * Y_HxW * C;
  T* dX_ptr = dX + n * X_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = t; i < b; ++i) {
      for (int j = l; j < r; ++j) {
        const int y = i * Y_W + j;
#if __CUDA_ARCH__ >= 350
        if (__ldg(X_ptr + x * C + c) == __ldg(Y_ptr + y * C + c)) {
          sum += __ldg(dY_ptr + y * C + c);
        }
#else
        if (X_ptr[x * C + c] == Y_ptr[y * C + c]) {
          sum += dY_ptr[y * C + c];
        }
#endif
      }
    }
    dX_ptr[x * C + c] = sum;
  }
}

template <typename T>
__global__ void MaxPool3DBackwardNCHWCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int xx = blockIdx.x / X_H;
  const int nc = xx / X_D;
  const int xd = xx % X_D;
  const int xh = blockIdx.x % X_H;
  const T* dY_ptr = dY + nc * Y_HxW;
  const T* X_ptr = X + nc * X_HxW;
  const T* Y_ptr = Y + nc * Y_HxW;
  T* dX_ptr = dX + nc * X_HxW;
  const int d = xd + pad_p;
  const int h = xh + pad_t;
  const int p = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
  const int a = min(d / stride_d + 1, Y_D);
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  for (int xw = threadIdx.x; xw < X_W; xw += blockDim.x) {
    const int w = xw + pad_l;
    const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int r = min(w / stride_w + 1, Y_W);
    const int x = (xd * X_H + xh) * X_W + xw;
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
          const int y = (i * Y_H + j) * Y_W + k;
#if __CUDA_ARCH__ >= 350
          if (__ldg(X_ptr + x) == __ldg(Y_ptr + y)) {
            sum += __ldg(dY_ptr + y);
          }
#else
          if (X_ptr[x] == Y_ptr[y]) {
            sum += dY_ptr[y];
          }
#endif
        }
      }
    }
    dX_ptr[x] = sum;
  }
}

template <typename T>
__global__ void MaxPool3DBackwardNHWCCUDAKernel(
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
  const int X_HxW = X_D * X_H * X_W;
  const int Y_HxW = Y_D * Y_H * Y_W;
  const int n = blockIdx.x / X_HxW;
  const int x = blockIdx.x % X_HxW;
  const int xx = x / X_W;
  const int d = xx / X_H + pad_p;
  const int h = xx % X_H + pad_t;
  const int w = x % X_W + pad_l;
  const int p = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
  const int a = min(d / stride_d + 1, Y_D);
  const int t = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
  const int b = min(h / stride_h + 1, Y_H);
  const int l = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
  const int r = min(w / stride_w + 1, Y_W);
  const T* dY_ptr = dY + n * Y_HxW * C;
  const T* X_ptr = X + n * X_HxW * C;
  const T* Y_ptr = Y + n * Y_HxW * C;
  T* dX_ptr = dX + n * X_HxW * C;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    T sum = 0;
    for (int i = p; i < a; ++i) {
      for (int j = t; j < b; ++j) {
        for (int k = l; k < r; ++k) {
          const int y = (i * Y_H + j) * Y_W + k;
#if __CUDA_ARCH__ >= 350
          if (__ldg(X_ptr + x * C + c) == __ldg(Y_ptr + y * C + c)) {
            sum += __ldg(dY_ptr + y * C + c);
          }
#else
          if (X_ptr[x * C + c] == Y_ptr[y * C + c]) {
            sum += dY_ptr[y * C + c];
          }
#endif
        }
      }
    }
    dX_ptr[x * C + c] = sum;
  }
}

} // namespace

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::
    GlobalPoolingForward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CUDAContext* context) const {
  const std::array<int, 2> X_dims = {N * C, HxW};
  const std::array<int, 2> Y_dims = {N * C, 1};
  math::ReduceMax<float, CUDAContext>(
      2, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
  return true;
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::
    GlobalPoolingForward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* X,
        float* Y,
        CUDAContext* context) const {
  const std::array<int, 3> X_dims = {N, HxW, C};
  const std::array<int, 3> Y_dims = {N, 1, C};
  math::ReduceMax<float, CUDAContext>(
      3, X_dims.data(), Y_dims.data(), 1.0f, X, Y, context);
  return true;
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::Forward<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* X,
    float* Y,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  switch (ndim) {
    case 1: {
      const int num_blocks = N * C;
      MaxPool1DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      const int num_blocks = N * C * Y_dims[0];
      MaxPool2DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      const int num_blocks = N * C * Y_dims[0] * Y_dims[1];
      MaxPool3DForwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::Forward<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* X,
    float* Y,
    CUDAContext* context) const {
  // Each CUDA block handles one point, one thread per channel.
  const int ndim = X_dims.size();
  const auto Y_HxW = c10::multiply_integers(Y_dims.cbegin(), Y_dims.cend());
  switch (ndim) {
    case 1: {
      MaxPool1DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      MaxPool2DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      MaxPool3DForwardNHWCCUDAKernel<float>
          <<<N * Y_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              X,
              Y);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::
    GlobalPoolingBackward<float, StorageOrder::NCHW>(
        const int N,
        const int C,
        const int HxW,
        const float* dY,
        const float* X,
        const float* Y,
        float* dX,
        CUDAContext* context) const {
  const int K = (HxW + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
  GlobalMaxPoolingBackwardNCHWCUDAKernel<float>
      <<<N * C * K, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          K, HxW, dY, X, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::
    GlobalPoolingBackward<float, StorageOrder::NHWC>(
        const int N,
        const int C,
        const int HxW,
        const float* dY,
        const float* X,
        const float* Y,
        float* dX,
        CUDAContext* context) const {
  GlobalMaxPoolingBackwardNHWCCUDAKernel<float>
      <<<N * HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
          C, HxW, dY, X, Y, dX);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::Backward<float, StorageOrder::NCHW>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* dY,
    const float* X,
    const float* Y,
    float* dX,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  switch (ndim) {
    case 1: {
      const int num_blocks = N * C;
      MaxPool1DBackwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      const int num_blocks = N * C * X_dims[0];
      MaxPool2DBackwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      const int num_blocks = N * C * X_dims[0] * X_dims[1];
      MaxPool3DBackwardNCHWCUDAKernel<float>
          <<<num_blocks, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

template <>
template <>
bool MaxPoolFunctor<CUDAContext>::Backward<float, StorageOrder::NHWC>(
    const int N,
    const int C,
    const std::vector<int>& X_dims,
    const std::vector<int>& Y_dims,
    const std::vector<int>& kernel,
    const std::vector<int>& /* dilation */,
    const std::vector<int>& stride,
    const std::vector<int>& pads,
    const float* dY,
    const float* X,
    const float* Y,
    float* dX,
    CUDAContext* context) const {
  const int ndim = X_dims.size();
  const auto X_HxW = c10::multiply_integers(X_dims.cbegin(), X_dims.cend());
  switch (ndim) {
    case 1: {
      MaxPool1DBackwardNHWCCUDAKernel<float>
          <<<N * X_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              Y_dims[0],
              kernel[0],
              stride[0],
              pads[0],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 2: {
      MaxPool2DBackwardNHWCCUDAKernel<float>
          <<<N * X_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              Y_dims[0],
              Y_dims[1],
              kernel[0],
              kernel[1],
              stride[0],
              stride[1],
              pads[0],
              pads[1],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    case 3: {
      MaxPool3DBackwardNHWCCUDAKernel<float>
          <<<N * X_HxW, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
              C,
              X_dims[0],
              X_dims[1],
              X_dims[2],
              Y_dims[0],
              Y_dims[1],
              Y_dims[2],
              kernel[0],
              kernel[1],
              kernel[2],
              stride[0],
              stride[1],
              stride[2],
              pads[0],
              pads[1],
              pads[2],
              dY,
              X,
              Y,
              dX);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      return true;
    }
    default: {
      CAFFE_THROW("Unsupported pooling dim: ", ndim);
      return false;
    }
  }
}

REGISTER_CUDA_OPERATOR(
    AveragePool,
    PoolOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AveragePoolGradient,
    PoolGradientOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    AveragePool1D,
    PoolOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AveragePool1DGradient,
    PoolGradientOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    AveragePool2D,
    PoolOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AveragePool2DGradient,
    PoolGradientOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    AveragePool3D,
    PoolOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    AveragePool3DGradient,
    PoolGradientOp<float, CUDAContext, AveragePoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    MaxPool,
    PoolOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    MaxPoolGradient,
    PoolGradientOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    MaxPool1D,
    PoolOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    MaxPool1DGradient,
    PoolGradientOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    MaxPool2D,
    PoolOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    MaxPool2DGradient,
    PoolGradientOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);

REGISTER_CUDA_OPERATOR(
    MaxPool3D,
    PoolOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    MaxPool3DGradient,
    PoolGradientOp<float, CUDAContext, MaxPoolFunctor<CUDAContext>>);

} // namespace caffe2
