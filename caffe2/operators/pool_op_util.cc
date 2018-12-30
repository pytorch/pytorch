#include "caffe2/operators/pool_op_util.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {
namespace pool_op_util {

namespace {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

// Vectorizes 4x4p0s0 averge pooling for ARM NEON
void AvgPoolNeon4x4p0s0Plane(
    int inputH,
    int inputW,
    const float* input,
    float* output) {
  constexpr int kKernelHeight = 4;
  constexpr int kKernelWidth = 4;
  constexpr float kDiv = (1.0f / ((float)kKernelHeight * (float)kKernelWidth));

  // Handle portion that can be unrolled by 4
  constexpr int kUnroll = 4;
  constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
  constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

  if (inputW % kLoadCols == 0) {
    //
    // Manually unroll by 4 (kUnroll)
    //

    for (int h = 0; h < inputH; h += kKernelHeight) {
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
      const float* curInput = input + h * inputW;

      for (int w = 0; w < inputW; w += kLoadCols) {
        float32x4_t out = {};

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 0);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 1);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 2);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 3);
        }
        curInput += kLoadSizeFloat;

        out = vmulq_f32(out, vdupq_n_f32(kDiv));
        vst1q_f32_aligned(&outputRow[w / kKernelWidth], out);
      }
    }
  } else {
    //
    // Not unrolled
    //

    for (int h = 0; h < inputH; h += kKernelHeight) {
      const float* inputRow = input + h * inputW;
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

      for (int w = 0; w < inputW; w += kKernelWidth) {
        const float* curInput = inputRow + w;

        float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
        float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
        float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
        float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
        float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3) * kDiv;
        outputRow[w / kKernelWidth] = v0;
      }
    }
  }
}

// Vectorizes 2x2p0s0 averge pooling for ARM NEON
void MaxPoolNeon2x2p0s0Plane(
    int inputH,
    int inputW,
    const float* input,
    float* output) {
  constexpr int kKernelHeight = 2;
  constexpr int kKernelWidth = 2;

  // Handle portion that can be unrolled by 4
  constexpr int kUnroll = 4;
  constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
  constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

  if (inputW % kLoadCols == 0) {
    for (int h = 0; h < inputH; h += kKernelHeight) {
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
      const float* curInput = input + h * inputW;

      for (int w = 0; w < inputW; w += kLoadCols) {
        float32x2_t hmax_0, hmax_1, hmax_2, hmax_3;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_0 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_1 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_3 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;

        float32x4_t out_0 = vcombine_f32(hmax_0, hmax_1);
        float32x4_t out_1 = vcombine_f32(hmax_2, hmax_3);
        vst1q_f32_aligned(&outputRow[w / kKernelWidth + 0], out_0);
        vst1q_f32_aligned(&outputRow[w / kKernelWidth + 4], out_1);
      }
    }
  } else {
    // Not unrolled
    for (int h = 0; h < inputH; h += kKernelHeight) {
      const float* inputRow = input + h * inputW;
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

      for (int w = 0; w < inputW; w += kKernelWidth * 2) {
        const float* curInput = inputRow + w;
        float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
        float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
        float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
        float32x2_t hmax = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        vst1_f32(&outputRow[w / kKernelWidth], hmax);
      }
    }
  }
}

#endif

} // namespace

bool IsNeon4x4p0s0Eligible(
    const int input_h,
    const int input_w,
    const int output_h,
    const int output_w,
    const int kh,
    const int kw,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int dilation_h,
    const int dilation_w,
    const float* X,
    float* Y) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  // Use this kernel only if:
  //   1. Kernel size is 4x4
  //   2. Stride is 4x4
  //   3. Padding is 0
  //   4. Dilation is 1
  //   5. Output width and height are even divisors of input width
  //   6. Input width and height are divisible by 4 (should be implied by all of
  //      the above, but just check again)
  // Input and output pointers are aligned by float32x4_t
  const bool kernel_ok = (kh == 4) && (kw == 4);
  const bool stride_ok = (stride_h == 4) && (stride_w == 4);
  const bool pad_ok =
      (pad_t == 0) && (pad_l == 0) && (pad_b == 0) && (pad_r == 0);
  const bool dilation_ok = (dilation_h == 1) && (dilation_w == 1);
  const bool output_ok = (input_h % output_h == 0) && (input_w % output_w == 0);
  const bool input_ok = (input_w % 4 == 0) && (input_h % 4 == 0);
  const bool align_ok = isPointerAligned(X, sizeof(float32x4_t)) &&
      isPointerAligned(Y, sizeof(float32x4_t));
  return kernel_ok && stride_ok && pad_ok && dilation_ok && output_ok &&
      input_ok && align_ok;
#else
  (void)input_h;
  (void)input_w;
  (void)output_h;
  (void)output_w;
  (void)kh;
  (void)kw;
  (void)stride_h;
  (void)stride_w;
  (void)pad_t;
  (void)pad_l;
  (void)pad_b;
  (void)pad_r;
  (void)dilation_h;
  (void)dilation_w;
  (void)X;
  (void)Y;
  return false;
#endif
}

bool IsNeon2x2p0s0Eligible(
    const int input_h,
    const int input_w,
    const int output_h,
    const int output_w,
    const int kh,
    const int kw,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int dilation_h,
    const int dilation_w,
    const float* X,
    float* Y) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  // Use this kernel only if:
  //   1. Kernel size is 2x2
  //   2. Stride is 2x2
  //   3. Padding is 0
  //   4. Dilation is 1
  //   5. Output width and height are even divisors of input width
  //   6. Input width and height are divisible by 4 (should be implied b all of
  //      the above, but just check again)
  // Input and output pointers are aligned by float32x4_t
  const bool kernel_ok = (kh == 2) && (kw == 2);
  const bool stride_ok = (stride_h == 2) && (stride_w == 2);
  const bool pad_ok =
      (pad_t == 0) && (pad_l == 0) && (pad_b == 0) && (pad_r == 0);
  const bool dilation_ok = (dilation_h == 1) && (dilation_w == 1);
  const bool output_ok = (input_h % output_h == 0) && (input_w % output_w == 0);
  const bool input_ok = (input_w % 4 == 0) && (input_h % 4 == 0);
  const bool align_ok = isPointerAligned(X, sizeof(float32x4_t)) &&
      isPointerAligned(Y, sizeof(float32x4_t));
  return kernel_ok && stride_ok && pad_ok && dilation_ok && output_ok &&
      input_ok && align_ok;
#else
  (void)input_h;
  (void)input_w;
  (void)output_h;
  (void)output_w;
  (void)kh;
  (void)kw;
  (void)stride_h;
  (void)stride_w;
  (void)pad_t;
  (void)pad_l;
  (void)pad_b;
  (void)pad_r;
  (void)dilation_h;
  (void)dilation_w;
  (void)X;
  (void)Y;
  return false;
#endif
}

void RunNeonAveragePool4x4p0s0NCHW(
    int N,
    int C,
    int H,
    int W,
    const float* X,
    float* Y) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  const int X_stride = H * W;
  const int Y_stride = (H / 4) * (W / 4);
  const float* X_ptr = X;
  float* Y_ptr = Y;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      AvgPoolNeon4x4p0s0Plane(H, W, X_ptr, Y_ptr);
      X_ptr += X_stride;
      Y_ptr += Y_stride;
    }
  }
#else
  (void)N;
  (void)C;
  (void)H;
  (void)W;
  (void)X;
  (void)Y;
#endif
}

void RunNeonMaxPool2x2p0s0NCHW(
    int N,
    int C,
    int H,
    int W,
    const float* X,
    float* Y) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  const int X_stride = H * W;
  const int Y_stride = (H / 2) * (W / 2);
  const float* X_ptr = X;
  float* Y_ptr = Y;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      MaxPoolNeon2x2p0s0Plane(H, W, X_ptr, Y_ptr);
      X_ptr += X_stride;
      Y_ptr += Y_stride;
    }
  }
#else
  (void)N;
  (void)C;
  (void)H;
  (void)W;
  (void)X;
  (void)Y;
#endif
}

#define CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_1D(T, kCountIncludePad)    \
  template <>                                                          \
  void RunAveragePool1D<T, StorageOrder::NCHW, kCountIncludePad>(      \
      const int N,                                                     \
      const int C,                                                     \
      const int X_size,                                                \
      const int Y_size,                                                \
      const int kernel,                                                \
      const int stride,                                                \
      const int pad,                                                   \
      const T* X,                                                      \
      T* Y) {                                                          \
    const T* X_ptr = X;                                                \
    T* Y_ptr = Y;                                                      \
    for (int n = 0; n < N; ++n) {                                      \
      ConstEigenArrayMap<T> X_arr(X_ptr, X_size, C);                   \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_size, C);                        \
      for (int i = 0; i < Y_size; ++i) {                               \
        const int l = std::max(i * stride - pad, 0);                   \
        const int r = std::min(i * stride - pad + kernel, X_size);     \
        if (kCountIncludePad) {                                        \
          Y_arr.row(i) = X_arr.block(l, 0, r - l, C).colwise().sum();  \
        } else {                                                       \
          Y_arr.row(i) = X_arr.block(l, 0, r - l, C).colwise().mean(); \
        }                                                              \
      }                                                                \
      if (kCountIncludePad) {                                          \
        Y_arr /= static_cast<T>(kernel);                               \
      }                                                                \
      X_ptr += C * X_size;                                             \
      Y_ptr += C * Y_size;                                             \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  void RunAveragePool1D<T, StorageOrder::NHWC, kCountIncludePad>(      \
      const int N,                                                     \
      const int C,                                                     \
      const int X_size,                                                \
      const int Y_size,                                                \
      const int kernel,                                                \
      const int stride,                                                \
      const int pad,                                                   \
      const T* X,                                                      \
      T* Y) {                                                          \
    const T* X_ptr = X;                                                \
    T* Y_ptr = Y;                                                      \
    for (int n = 0; n < N; ++n) {                                      \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_size);                   \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_size);                        \
      for (int i = 0; i < Y_size; ++i) {                               \
        const int l = std::max(i * stride - pad, 0);                   \
        const int r = std::min(i * stride - pad + kernel, X_size);     \
        Y_arr.col(i) = X_arr.col(l);                                   \
        for (int j = l + 1; j < r; ++j) {                              \
          Y_arr.col(i) += X_arr.col(j);                                \
        }                                                              \
        if (!kCountIncludePad) {                                       \
          Y_arr.col(i) /= static_cast<T>(r - l);                       \
        }                                                              \
      }                                                                \
      if (kCountIncludePad) {                                          \
        Y_arr /= static_cast<T>(kernel);                               \
      }                                                                \
      X_ptr += C * X_size;                                             \
      Y_ptr += C * Y_size;                                             \
    }                                                                  \
  }
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_1D(float, true)
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_1D(float, false)
#undef CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_1D

#define CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_2D(T, kCountIncludePad)         \
  template <>                                                               \
  void RunAveragePool2D<T, StorageOrder::NCHW, kCountIncludePad>(           \
      const int N,                                                          \
      const int C,                                                          \
      const int X_H,                                                        \
      const int X_W,                                                        \
      const int Y_H,                                                        \
      const int Y_W,                                                        \
      const int kernel_h,                                                   \
      const int kernel_w,                                                   \
      const int stride_h,                                                   \
      const int stride_w,                                                   \
      const int pad_t,                                                      \
      const int pad_l,                                                      \
      const T* X,                                                           \
      T* Y) {                                                               \
    const int X_HxW = X_H * X_W;                                            \
    const int Y_HxW = Y_H * Y_W;                                            \
    std::memset(Y, 0, sizeof(T) * N * C * Y_HxW);                           \
    const T* X_ptr = X;                                                     \
    T* Y_ptr = Y;                                                           \
    for (int n = 0; n < N; ++n) {                                           \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_HxW, C);                              \
      for (int h = 0; h < Y_H; ++h) {                                       \
        const int t = std::max(h * stride_h - pad_t, 0);                    \
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);       \
        for (int w = 0; w < Y_W; ++w) {                                     \
          const int l = std::max(w * stride_w - pad_l, 0);                  \
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);     \
          const int row = h * Y_W + w;                                      \
          for (int i = t; i < b; ++i) {                                     \
            Y_arr.row(row) +=                                               \
                ConstEigenOuterStridedArrayMap<T>(                          \
                    X_ptr + i * X_W + l, r - l, C, EigenOuterStride(X_HxW)) \
                    .colwise()                                              \
                    .sum();                                                 \
          }                                                                 \
          if (!kCountIncludePad) {                                          \
            Y_arr.row(row) /= static_cast<T>((b - t) * (r - l));            \
          }                                                                 \
        }                                                                   \
      }                                                                     \
      if (kCountIncludePad) {                                               \
        Y_arr /= static_cast<T>(kernel_h * kernel_w);                       \
      }                                                                     \
      X_ptr += C * X_HxW;                                                   \
      Y_ptr += C * Y_HxW;                                                   \
    }                                                                       \
  }                                                                         \
  template <>                                                               \
  void RunAveragePool2D<T, StorageOrder::NHWC, kCountIncludePad>(           \
      const int N,                                                          \
      const int C,                                                          \
      const int X_H,                                                        \
      const int X_W,                                                        \
      const int Y_H,                                                        \
      const int Y_W,                                                        \
      const int kernel_h,                                                   \
      const int kernel_w,                                                   \
      const int stride_h,                                                   \
      const int stride_w,                                                   \
      const int pad_t,                                                      \
      const int pad_l,                                                      \
      const T* X,                                                           \
      T* Y) {                                                               \
    const int X_HxW = X_H * X_W;                                            \
    const int Y_HxW = Y_H * Y_W;                                            \
    std::memset(Y, 0, sizeof(T) * N * C * Y_HxW);                           \
    const T* X_ptr = X;                                                     \
    T* Y_ptr = Y;                                                           \
    for (int n = 0; n < N; ++n) {                                           \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_HxW);                         \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_HxW);                              \
      for (int h = 0; h < Y_H; ++h) {                                       \
        const int t = std::max(h * stride_h - pad_t, 0);                    \
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);       \
        for (int w = 0; w < Y_W; ++w) {                                     \
          const int l = std::max(w * stride_w - pad_l, 0);                  \
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);     \
          const int col = h * Y_W + w;                                      \
          for (int i = t; i < b; ++i) {                                     \
            for (int j = l; j < r; ++j) {                                   \
              Y_arr.col(col) += X_arr.col(i * X_W + j);                     \
            }                                                               \
          }                                                                 \
          if (!kCountIncludePad) {                                          \
            Y_arr.col(col) /= static_cast<T>((b - t) * (r - l));            \
          }                                                                 \
        }                                                                   \
      }                                                                     \
      if (kCountIncludePad) {                                               \
        Y_arr /= static_cast<T>(kernel_h * kernel_w);                       \
      }                                                                     \
      X_ptr += C * X_HxW;                                                   \
      Y_ptr += C * Y_HxW;                                                   \
    }                                                                       \
  }
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_2D(float, true)
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_2D(float, false)
#undef CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_2D

#define CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_3D(T, kCountIncludePad)          \
  template <>                                                                \
  void RunAveragePool3D<T, StorageOrder::NCHW, kCountIncludePad>(            \
      const int N,                                                           \
      const int C,                                                           \
      const int X_D,                                                         \
      const int X_H,                                                         \
      const int X_W,                                                         \
      const int Y_D,                                                         \
      const int Y_H,                                                         \
      const int Y_W,                                                         \
      const int kernel_d,                                                    \
      const int kernel_h,                                                    \
      const int kernel_w,                                                    \
      const int stride_d,                                                    \
      const int stride_h,                                                    \
      const int stride_w,                                                    \
      const int pad_p,                                                       \
      const int pad_t,                                                       \
      const int pad_l,                                                       \
      const T* X,                                                            \
      T* Y) {                                                                \
    const int X_HxW = X_D * X_H * X_W;                                       \
    const int Y_HxW = Y_D * Y_H * Y_W;                                       \
    std::memset(Y, 0, sizeof(T) * N * C * Y_HxW);                            \
    const T* X_ptr = X;                                                      \
    T* Y_ptr = Y;                                                            \
    for (int n = 0; n < N; ++n) {                                            \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_HxW, C);                               \
      for (int d = 0; d < Y_D; ++d) {                                        \
        const int p = std::max(d * stride_d - pad_p, 0);                     \
        const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);        \
        for (int h = 0; h < Y_H; ++h) {                                      \
          const int t = std::max(h * stride_h - pad_t, 0);                   \
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);      \
          for (int w = 0; w < Y_W; ++w) {                                    \
            const int l = std::max(w * stride_w - pad_l, 0);                 \
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);    \
            const int row = d * Y_H * Y_W + h * Y_W + w;                     \
            for (int i = p; i < a; ++i) {                                    \
              for (int j = t; j < b; ++j) {                                  \
                const int offset = i * X_H * X_W + j * X_W + l;              \
                Y_arr.row(row) +=                                            \
                    ConstEigenOuterStridedArrayMap<T>(                       \
                        X_ptr + offset, r - l, C, EigenOuterStride(X_HxW))   \
                        .colwise()                                           \
                        .sum();                                              \
              }                                                              \
            }                                                                \
            if (!kCountIncludePad) {                                         \
              Y_arr.row(row) /= static_cast<T>((a - p) * (b - t) * (r - l)); \
            }                                                                \
          }                                                                  \
        }                                                                    \
      }                                                                      \
      if (kCountIncludePad) {                                                \
        Y_arr /= static_cast<T>(kernel_d * kernel_h * kernel_w);             \
      }                                                                      \
      X_ptr += C * X_HxW;                                                    \
      Y_ptr += C * Y_HxW;                                                    \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void RunAveragePool3D<T, StorageOrder::NHWC, kCountIncludePad>(            \
      const int N,                                                           \
      const int C,                                                           \
      const int X_D,                                                         \
      const int X_H,                                                         \
      const int X_W,                                                         \
      const int Y_D,                                                         \
      const int Y_H,                                                         \
      const int Y_W,                                                         \
      const int kernel_d,                                                    \
      const int kernel_h,                                                    \
      const int kernel_w,                                                    \
      const int stride_d,                                                    \
      const int stride_h,                                                    \
      const int stride_w,                                                    \
      const int pad_p,                                                       \
      const int pad_t,                                                       \
      const int pad_l,                                                       \
      const T* X,                                                            \
      T* Y) {                                                                \
    const int X_HxW = X_D * X_H * X_W;                                       \
    const int Y_HxW = Y_D * Y_H * Y_W;                                       \
    std::memset(Y, 0, sizeof(T) * N * C * Y_HxW);                            \
    const T* X_ptr = X;                                                      \
    T* Y_ptr = Y;                                                            \
    for (int n = 0; n < N; ++n) {                                            \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_HxW);                          \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_HxW);                               \
      for (int d = 0; d < Y_D; ++d) {                                        \
        const int p = std::max(d * stride_d - pad_p, 0);                     \
        const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);        \
        for (int h = 0; h < Y_H; ++h) {                                      \
          const int t = std::max(h * stride_h - pad_t, 0);                   \
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);      \
          for (int w = 0; w < Y_W; ++w) {                                    \
            const int l = std::max(w * stride_w - pad_l, 0);                 \
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);    \
            const int col = d * Y_H * Y_W + h * Y_W + w;                     \
            for (int i = p; i < a; ++i) {                                    \
              for (int j = t; j < b; ++j) {                                  \
                for (int k = l; k < r; ++k) {                                \
                  Y_arr.col(col) += X_arr.col(i * X_H * X_W + j * X_W + k);  \
                }                                                            \
              }                                                              \
            }                                                                \
            if (!kCountIncludePad) {                                         \
              Y_arr.col(col) /= static_cast<T>((a - p) * (b - t) * (r - l)); \
            }                                                                \
          }                                                                  \
        }                                                                    \
      }                                                                      \
      if (kCountIncludePad) {                                                \
        Y_arr /= static_cast<T>(kernel_d * kernel_h * kernel_w);             \
      }                                                                      \
      X_ptr += C * X_HxW;                                                    \
      Y_ptr += C * Y_HxW;                                                    \
    }                                                                        \
  }
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_3D(float, true)
CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_3D(float, false)
#undef CAFFE2_SPECIALIZED_RUN_AVERAGE_POOL_3D

#define CAFFE2_SPECIALIZED_RUN_MAX_POOL_1D(T)                            \
  template <>                                                            \
  void RunMaxPool1D<T, StorageOrder::NCHW>(                              \
      const int N,                                                       \
      const int C,                                                       \
      const int X_size,                                                  \
      const int Y_size,                                                  \
      const int kernel,                                                  \
      const int stride,                                                  \
      const int pad,                                                     \
      const T* X,                                                        \
      T* Y) {                                                            \
    const T* X_ptr = X;                                                  \
    T* Y_ptr = Y;                                                        \
    for (int n = 0; n < N; ++n) {                                        \
      ConstEigenArrayMap<T> X_arr(X_ptr, X_size, C);                     \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_size, C);                          \
      for (int i = 0; i < Y_size; ++i) {                                 \
        const int l = std::max(i * stride - pad, 0);                     \
        const int r = std::min(i * stride - pad + kernel, X_size);       \
        Y_arr.row(i) = X_arr.block(l, 0, r - l, C).colwise().maxCoeff(); \
      }                                                                  \
      X_ptr += C * X_size;                                               \
      Y_ptr += C * Y_size;                                               \
    }                                                                    \
  }                                                                      \
  template <>                                                            \
  void RunMaxPool1D<T, StorageOrder::NHWC>(                              \
      const int N,                                                       \
      const int C,                                                       \
      const int X_size,                                                  \
      const int Y_size,                                                  \
      const int kernel,                                                  \
      const int stride,                                                  \
      const int pad,                                                     \
      const T* X,                                                        \
      T* Y) {                                                            \
    const T* X_ptr = X;                                                  \
    T* Y_ptr = Y;                                                        \
    for (int n = 0; n < N; ++n) {                                        \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_size);                     \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_size);                          \
      for (int i = 0; i < Y_size; ++i) {                                 \
        const int l = std::max(i * stride - pad, 0);                     \
        const int r = std::min(i * stride - pad + kernel, X_size);       \
        Y_arr.col(i) = X_arr.col(l);                                     \
        for (int j = l + 1; j < r; ++j) {                                \
          Y_arr.col(i) = Y_arr.col(i).max(X_arr.col(j));                 \
        }                                                                \
      }                                                                  \
      X_ptr += C * X_size;                                               \
      Y_ptr += C * Y_size;                                               \
    }                                                                    \
  }
CAFFE2_SPECIALIZED_RUN_MAX_POOL_1D(float)
#undef CAFFE2_SPECIALIZED_RUN_MAX_POOL_1D

#define CAFFE2_SPECIALIZED_RUN_MAX_POOL_2D(T)                               \
  template <>                                                               \
  void RunMaxPool2D<T, StorageOrder::NCHW>(                                 \
      const int N,                                                          \
      const int C,                                                          \
      const int X_H,                                                        \
      const int X_W,                                                        \
      const int Y_H,                                                        \
      const int Y_W,                                                        \
      const int kernel_h,                                                   \
      const int kernel_w,                                                   \
      const int stride_h,                                                   \
      const int stride_w,                                                   \
      const int pad_t,                                                      \
      const int pad_l,                                                      \
      const T* X,                                                           \
      T* Y) {                                                               \
    const int X_HxW = X_H * X_W;                                            \
    const int Y_HxW = Y_H * Y_W;                                            \
    EigenVectorArrayMap<T>(Y, N * C * Y_HxW)                                \
        .setConstant(std::numeric_limits<T>::lowest());                     \
    const T* X_ptr = X;                                                     \
    T* Y_ptr = Y;                                                           \
    for (int n = 0; n < N; ++n) {                                           \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_HxW, C);                              \
      for (int h = 0; h < Y_H; ++h) {                                       \
        const int t = std::max(h * stride_h - pad_t, 0);                    \
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);       \
        for (int w = 0; w < Y_W; ++w) {                                     \
          const int l = std::max(w * stride_w - pad_l, 0);                  \
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);     \
          const int row = h * Y_W + w;                                      \
          for (int i = t; i < b; ++i) {                                     \
            Y_arr.row(row) = Y_arr.row(row).max(                            \
                ConstEigenOuterStridedArrayMap<T>(                          \
                    X_ptr + i * X_W + l, r - l, C, EigenOuterStride(X_HxW)) \
                    .colwise()                                              \
                    .maxCoeff());                                           \
          }                                                                 \
        }                                                                   \
      }                                                                     \
      X_ptr += C * X_HxW;                                                   \
      Y_ptr += C * Y_HxW;                                                   \
    }                                                                       \
  }                                                                         \
  template <>                                                               \
  void RunMaxPool2D<T, StorageOrder::NHWC>(                                 \
      const int N,                                                          \
      const int C,                                                          \
      const int X_H,                                                        \
      const int X_W,                                                        \
      const int Y_H,                                                        \
      const int Y_W,                                                        \
      const int kernel_h,                                                   \
      const int kernel_w,                                                   \
      const int stride_h,                                                   \
      const int stride_w,                                                   \
      const int pad_t,                                                      \
      const int pad_l,                                                      \
      const T* X,                                                           \
      T* Y) {                                                               \
    const int X_HxW = X_H * X_W;                                            \
    const int Y_HxW = Y_H * Y_W;                                            \
    EigenVectorArrayMap<T>(Y, N * C * Y_HxW)                                \
        .setConstant(std::numeric_limits<T>::lowest());                     \
    const T* X_ptr = X;                                                     \
    T* Y_ptr = Y;                                                           \
    for (int n = 0; n < N; ++n) {                                           \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_HxW);                         \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_HxW);                              \
      for (int h = 0; h < Y_H; ++h) {                                       \
        const int t = std::max(h * stride_h - pad_t, 0);                    \
        const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);       \
        for (int w = 0; w < Y_W; ++w) {                                     \
          const int l = std::max(w * stride_w - pad_l, 0);                  \
          const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);     \
          const int col = h * Y_W + w;                                      \
          for (int i = t; i < b; ++i) {                                     \
            for (int j = l; j < r; ++j) {                                   \
              Y_arr.col(col) = Y_arr.col(col).max(X_arr.col(i * X_W + j));  \
            }                                                               \
          }                                                                 \
        }                                                                   \
      }                                                                     \
      X_ptr += C * X_HxW;                                                   \
      Y_ptr += C * Y_HxW;                                                   \
    }                                                                       \
  }
CAFFE2_SPECIALIZED_RUN_MAX_POOL_2D(float)
#undef CAFFE2_SPECIALIZED_RUN_MAX_POOL_2D

#define CAFFE2_SPECIALIZED_RUN_MAX_POOL_3D(T)                              \
  template <>                                                              \
  void RunMaxPool3D<T, StorageOrder::NCHW>(                                \
      const int N,                                                         \
      const int C,                                                         \
      const int X_D,                                                       \
      const int X_H,                                                       \
      const int X_W,                                                       \
      const int Y_D,                                                       \
      const int Y_H,                                                       \
      const int Y_W,                                                       \
      const int kernel_d,                                                  \
      const int kernel_h,                                                  \
      const int kernel_w,                                                  \
      const int stride_d,                                                  \
      const int stride_h,                                                  \
      const int stride_w,                                                  \
      const int pad_p,                                                     \
      const int pad_t,                                                     \
      const int pad_l,                                                     \
      const T* X,                                                          \
      T* Y) {                                                              \
    const int X_HxW = X_D * X_H * X_W;                                     \
    const int Y_HxW = Y_D * Y_H * Y_W;                                     \
    EigenVectorArrayMap<T>(Y, N * C * Y_HxW)                               \
        .setConstant(std::numeric_limits<T>::lowest());                    \
    const T* X_ptr = X;                                                    \
    T* Y_ptr = Y;                                                          \
    for (int n = 0; n < N; ++n) {                                          \
      EigenArrayMap<T> Y_arr(Y_ptr, Y_HxW, C);                             \
      for (int d = 0; d < Y_D; ++d) {                                      \
        const int p = std::max(d * stride_d - pad_p, 0);                   \
        const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);      \
        for (int h = 0; h < Y_H; ++h) {                                    \
          const int t = std::max(h * stride_h - pad_t, 0);                 \
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);    \
          for (int w = 0; w < Y_W; ++w) {                                  \
            const int l = std::max(w * stride_w - pad_l, 0);               \
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);  \
            const int row = d * Y_H * Y_W + h * Y_W + w;                   \
            for (int i = p; i < a; ++i) {                                  \
              for (int j = t; j < b; ++j) {                                \
                const int offset = i * X_H * X_W + j * X_W + l;            \
                Y_arr.row(row) = Y_arr.row(row).max(                       \
                    ConstEigenOuterStridedArrayMap<T>(                     \
                        X_ptr + offset, r - l, C, EigenOuterStride(X_HxW)) \
                        .colwise()                                         \
                        .maxCoeff());                                      \
              }                                                            \
            }                                                              \
          }                                                                \
        }                                                                  \
      }                                                                    \
      X_ptr += C * X_HxW;                                                  \
      Y_ptr += C * Y_HxW;                                                  \
    }                                                                      \
  }                                                                        \
  template <>                                                              \
  void RunMaxPool3D<T, StorageOrder::NHWC>(                                \
      const int N,                                                         \
      const int C,                                                         \
      const int X_D,                                                       \
      const int X_H,                                                       \
      const int X_W,                                                       \
      const int Y_D,                                                       \
      const int Y_H,                                                       \
      const int Y_W,                                                       \
      const int kernel_d,                                                  \
      const int kernel_h,                                                  \
      const int kernel_w,                                                  \
      const int stride_d,                                                  \
      const int stride_h,                                                  \
      const int stride_w,                                                  \
      const int pad_p,                                                     \
      const int pad_t,                                                     \
      const int pad_l,                                                     \
      const T* X,                                                          \
      T* Y) {                                                              \
    const int X_HxW = X_D * X_H * X_W;                                     \
    const int Y_HxW = Y_D * Y_H * Y_W;                                     \
    EigenVectorArrayMap<T>(Y, N * C * Y_HxW)                               \
        .setConstant(std::numeric_limits<T>::lowest());                    \
    const T* X_ptr = X;                                                    \
    T* Y_ptr = Y;                                                          \
    for (int n = 0; n < N; ++n) {                                          \
      ConstEigenArrayMap<T> X_arr(X_ptr, C, X_HxW);                        \
      EigenArrayMap<T> Y_arr(Y_ptr, C, Y_HxW);                             \
      for (int d = 0; d < Y_D; ++d) {                                      \
        const int p = std::max(d * stride_d - pad_p, 0);                   \
        const int a = std::min(d * stride_d - pad_p + kernel_d, X_D);      \
        for (int h = 0; h < Y_H; ++h) {                                    \
          const int t = std::max(h * stride_h - pad_t, 0);                 \
          const int b = std::min(h * stride_h - pad_t + kernel_h, X_H);    \
          for (int w = 0; w < Y_W; ++w) {                                  \
            const int l = std::max(w * stride_w - pad_l, 0);               \
            const int r = std::min(w * stride_w - pad_l + kernel_w, X_W);  \
            const int col = d * Y_H * Y_W + h * Y_W + w;                   \
            for (int i = p; i < a; ++i) {                                  \
              for (int j = t; j < b; ++j) {                                \
                for (int k = l; k < r; ++k) {                              \
                  Y_arr.col(col) = Y_arr.col(col).max(                     \
                      X_arr.col(i * X_H * X_W + j * X_W + k));             \
                }                                                          \
              }                                                            \
            }                                                              \
          }                                                                \
        }                                                                  \
      }                                                                    \
      X_ptr += C * X_HxW;                                                  \
      Y_ptr += C * Y_HxW;                                                  \
    }                                                                      \
  }
CAFFE2_SPECIALIZED_RUN_MAX_POOL_3D(float)
#undef CAFFE2_SPECIALIZED_RUN_MAX_POOL_3D

} // namespace pool_op_util
} // namespace caffe2
