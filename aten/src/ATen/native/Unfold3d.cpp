#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif // AT_MKL_ENABLED()

namespace at {
namespace native {

namespace {

bool IsAGeZeroAndALtB(int64_t a, int64_t b) {
  return static_cast<uint64_t>(a) < static_cast<uint64_t>(b);
}

template <typename T>
void MatCopy(int64_t M, int64_t N, int64_t lda, int64_t ldb, const T* A, T* B) {
  for (int64_t i = 0; i < M; ++i) {
    std::memcpy(B + i * ldb, A + i * lda, N * sizeof(T));
  }
}

template <typename T>
void MatCopy(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const T* A,
    T* B) {
  for (int64_t i = 0; i < M; ++i) {
    const T* A_ptr = A + i * lda;
    T* B_ptr = B + i * ldb;
    for (int64_t j = 0; j < N; ++j) {
      B_ptr[j * strideb] = A_ptr[j * stridea];
    }
  }
}

// Y += X
template <typename T>
void MatAdd(int64_t M, int64_t N, int64_t ldx, int64_t ldy, const T* X, T* Y) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      Y[i * ldy + j] += X[i * ldx + j];
    }
  }
}

// Y += X
template <typename T>
void MatAdd(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t stridex,
    int64_t ldy,
    int64_t stridey,
    const T* X,
    T* Y) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      Y[i * ldy + j * stridey] += X[i * ldx + j * stridex];
    }
  }
}

#if AT_MKL_ENABLED()

template <>
void MatCopy<float>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t ldb,
    const float* A,
    float* B) {
  mkl_somatcopy('R', 'N', M, N, 1.0f, A, lda, B, ldb);
}

template <>
void MatCopy<double>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t ldb,
    const double* A,
    double* B) {
  mkl_domatcopy('R', 'N', M, N, 1.0, A, lda, B, ldb);
}

template <>
void MatCopy<float>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const float* A,
    float* B) {
  mkl_somatcopy2('R', 'N', M, N, 1.0f, A, lda, stridea, B, ldb, strideb);
}

template <>
void MatCopy<double>(
    int64_t M,
    int64_t N,
    int64_t lda,
    int64_t stridea,
    int64_t ldb,
    int64_t strideb,
    const double* A,
    double* B) {
  mkl_domatcopy2('R', 'N', M, N, 1.0, A, lda, stridea, B, ldb, strideb);
}

template <>
void MatAdd<float>(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t ldy,
    const float* X,
    float* Y) {
  mkl_somatadd('R', 'N', 'N', M, N, 1.0f, X, ldx, 1.0f, Y, ldy, Y, ldy);
}

template <>
void MatAdd<double>(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t ldy,
    const double* X,
    double* Y) {
  mkl_domatadd('R', 'N', 'N', M, N, 1.0, X, ldx, 1.0, Y, ldy, Y, ldy);
}

template <>
void MatAdd(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t stridex,
    int64_t ldy,
    int64_t stridey,
    const float* X,
    float* Y) {
  for (int64_t i = 0; i < M; ++i) {
    cblas_saxpy(N, 1.0f, X + i * ldx, stridex, Y + i * ldy, stridey);
  }
}

template <>
void MatAdd(
    int64_t M,
    int64_t N,
    int64_t ldx,
    int64_t stridex,
    int64_t ldy,
    int64_t stridey,
    const double* X,
    double* Y) {
  for (int64_t i = 0; i < M; ++i) {
    cblas_daxpy(N, 1.0, X + i * ldx, stridex, Y + i * ldy, stridey);
  }
}

#endif // AT_MKL_ENABLED()

template <typename T>
void Unfold3dZeroPaddingCopyKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    const T* src,
    T* dst) {
  const int64_t n = C * kernel_d * kernel_h * kernel_w;
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  at::parallel_for(0, n, 0, [=](int64_t begin, int64_t end) {
    for (int64_t p = begin; p < end; ++p) {
      int64_t c = p;
      const int64_t kw = c % kernel_w;
      c /= kernel_w;
      const int64_t kh = c % kernel_h;
      c /= kernel_h;
      const int64_t kd = c % kernel_d;
      c /= kernel_d;
      for (int64_t yd = 0; yd < Y_D; ++yd) {
        const int64_t xd = yd * stride_d + kd;
        const T* src_ptr = src + c * X_size + xd * X_H * X_W + kh * X_W + kw;
        T* dst_ptr = dst + p * Y_size + yd * Y_H * Y_W;
        if (stride_w == 1) {
          MatCopy<T>(Y_H, Y_W, stride_h * X_W, Y_W, src_ptr, dst_ptr);
        } else {
          MatCopy<T>(
              Y_H, Y_W, stride_h * X_W, stride_w, Y_W, 1, src_ptr, dst_ptr);
        }
      }
    }
  });
}

template <typename T>
void Unfold3dCopyKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    const T* src,
    T* dst) {
  if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
    Unfold3dZeroPaddingCopyKernelImpl<T>(
        C,
        X_D,
        X_H,
        X_W,
        Y_D,
        Y_H,
        Y_W,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        src,
        dst);
    return;
  }

  const int64_t n = C * kernel_d * kernel_h * kernel_w;
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  at::parallel_for(0, n, 0, [=](int64_t begin, int64_t end) {
    for (int64_t p = begin; p < end; ++p) {
      int64_t c = p;
      const int64_t kw = c % kernel_w;
      c /= kernel_w;
      const int64_t kh = c % kernel_h;
      c /= kernel_h;
      const int64_t kd = c % kernel_d;
      c /= kernel_d;
      const T* src_ptr = src + c * X_size;
      T* dst_ptr = dst + p * Y_size;
      for (int64_t yd = 0; yd < Y_D; ++yd) {
        const int64_t xd = yd * stride_d - pad_d + kd;
        if (!IsAGeZeroAndALtB(xd, X_D)) {
          std::memset(dst_ptr + yd * Y_H * Y_W, 0, Y_H * Y_W * sizeof(T));
          continue;
        }
        for (int64_t yh = 0; yh < Y_H; ++yh) {
          const int64_t xh = yh * stride_h - pad_h + kh;
          if (!IsAGeZeroAndALtB(xh, X_H)) {
            std::memset(
                dst_ptr + yd * Y_H * Y_W + yh * Y_W, 0, Y_W * sizeof(T));
            continue;
          }
          for (int64_t yw = 0; yw < Y_W; ++yw) {
            const int64_t xw = yw * stride_w - pad_w + kw;
            dst_ptr[yd * Y_H * Y_W + yh * Y_W + yw] = IsAGeZeroAndALtB(xw, X_W)
                ? src_ptr[xd * X_H * X_W + xh * X_W + xw]
                : T(0);
          }
        }
      }
    }
  });
}

template <typename T>
void Unfold3dZeroPaddingAccKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    const T* src,
    T* dst) {
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  const int64_t kernel_size = kernel_d * kernel_h * kernel_w;
  at::parallel_for(0, C, 0, [=](int64_t begin, int64_t end) {
    std::memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
    for (int64_t c = begin; c < end; ++c) {
      for (int64_t kd = 0; kd < kernel_d; ++kd) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
          for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t p =
                c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
            for (int64_t yd = 0; yd < Y_D; ++yd) {
              const int64_t xd = yd * stride_d + kd;
              const T* src_ptr = src + p * Y_size + yd * Y_H * Y_W;
              T* dst_ptr = dst + c * X_size + xd * X_H * X_W + kh * X_W + kw;
              if (stride_w == 1) {
                MatAdd<T>(Y_H, Y_W, Y_W, stride_h * X_W, src_ptr, dst_ptr);
              } else {
                MatAdd<T>(
                    Y_H,
                    Y_W,
                    Y_W,
                    1,
                    stride_h * X_W,
                    stride_w,
                    src_ptr,
                    dst_ptr);
              }
            }
          }
        }
      }
    }
  });
}

template <typename T>
void Unfold3dAccKernelImpl(
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    const T* src,
    T* dst) {
  if (pad_d == 0 && pad_h == 0 && pad_w == 0) {
    Unfold3dZeroPaddingAccKernelImpl<T>(
        C,
        X_D,
        X_H,
        X_W,
        Y_D,
        Y_H,
        Y_W,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        src,
        dst);
    return;
  }
  const int64_t X_size = X_D * X_H * X_W;
  const int64_t Y_size = Y_D * Y_H * Y_W;
  const int64_t kernel_size = kernel_d * kernel_h * kernel_w;
  at::parallel_for(0, C, 0, [=](int64_t begin, int64_t end) {
    std::memset(dst + begin * X_size, 0, (end - begin) * X_size * sizeof(T));
    for (int64_t c = begin; c < end; ++c) {
      T* dst_ptr = dst + c * X_size;
      for (int64_t kd = 0; kd < kernel_d; ++kd) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
          for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t p =
                c * kernel_size + kd * kernel_h * kernel_w + kh * kernel_w + kw;
            const T* src_ptr = src + p * Y_size;
            for (int64_t yd = 0; yd < Y_D; ++yd) {
              const int64_t xd = yd * stride_d - pad_d + kd;
              if (!IsAGeZeroAndALtB(xd, X_D)) {
                continue;
              }
              for (int64_t yh = 0; yh < Y_H; ++yh) {
                const int64_t xh = yh * stride_h - pad_h + kh;
                if (!IsAGeZeroAndALtB(xh, X_H)) {
                  continue;
                }
                for (int64_t yw = 0; yw < Y_W; ++yw) {
                  const int64_t xw = yw * stride_w - pad_w + kw;
                  if (IsAGeZeroAndALtB(xw, X_W)) {
                    dst_ptr[xd * X_H * X_W + xh * X_W + xw] +=
                        src_ptr[yd * Y_H * Y_W + yh * Y_W + yw];
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

} // namespace

void Unfold3dCopyCPU(
    const Tensor& src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    Tensor* dst) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      src.scalar_type(),
      "Unfold3dCopyCPU",
      [=, &src]() {
        Unfold3dCopyKernelImpl<scalar_t>(
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            src.data_ptr<scalar_t>(),
            dst->data_ptr<scalar_t>());
      });
}

void Unfold3dAccCPU(
    const Tensor& src,
    int64_t C,
    int64_t X_D,
    int64_t X_H,
    int64_t X_W,
    int64_t Y_D,
    int64_t Y_H,
    int64_t Y_W,
    int64_t kernel_d,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_d,
    int64_t pad_h,
    int64_t pad_w,
    Tensor* dst) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      src.scalar_type(),
      "Unfold3dAccCPU",
      [=, &src]() {
        Unfold3dAccKernelImpl<scalar_t>(
            C,
            X_D,
            X_H,
            X_W,
            Y_D,
            Y_H,
            Y_W,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            src.data_ptr<scalar_t>(),
            dst->data_ptr<scalar_t>());
      });
}

} // namespace native
} // namespace at
