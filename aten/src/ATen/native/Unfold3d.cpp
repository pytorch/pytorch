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

// Kernel for fast unfold+copy
// Borrowed from Theano
// Authors: Arjun Jain, Frederic Bastien, Jan Schluter, Nicolas Ballas
template <typename scalar_t>
static void unfolded3d_acc(
    scalar_t* finput_data,
    scalar_t* input_data,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  int64_t n = n_input_plane * input_height * input_width * input_depth;
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    int64_t line_index_offset = start;
    int64_t line_seg_len = (end - start);

    int64_t w = line_index_offset % input_width + pW;
    int64_t h_index = line_index_offset / input_width;
    int64_t h = h_index % input_height + pH;
    int64_t d_index = h_index / input_height;
    int64_t d = d_index % input_depth + pT;
    int64_t c = d_index / input_depth;

    int64_t outputHW = output_height * output_width;
    int64_t outputDHW = output_depth * outputHW;
    int64_t kHkW = kH * kW;
    int64_t kTkHkW = kT * kHkW;

    int64_t coeff_d_col = outputHW - dT * kHkW * outputDHW;
    int64_t coeff_h_col = output_width - dH * kW * outputDHW;
    int64_t coeff_w_col = (1 - dW * outputDHW);

    int64_t count = 0;
    while (count < line_seg_len) {
      // compute the start and end of the output
      int64_t w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;
      int64_t w_col_tmp = w / dW + 1;
      int64_t w_col_end = w_col_tmp < output_width ? w_col_tmp : output_width;

      int64_t h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;
      int64_t h_col_tmp = h / dH + 1;
      int64_t h_col_end = h_col_tmp < output_height ? h_col_tmp : output_height;

      int64_t d_col_start = (d < kT) ? 0 : (d - kT) / dT + 1;
      int64_t d_col_tmp = d / dT + 1;
      int64_t d_col_end = d_col_tmp < output_depth ? d_col_tmp : output_depth;

      scalar_t val = 0;
      int64_t offset = (c * kTkHkW + d * kHkW + h * kW + w) * outputDHW;

      int64_t offset_w_col_start = w_col_start * coeff_w_col;
      int64_t offset_d_col_start = d_col_start * coeff_d_col;
      int64_t offset_h_col_start = h_col_start * coeff_h_col;
      int64_t offset_w_col = offset_w_col_start + offset;
      int64_t offset_d_col;
      int64_t offset_h_col;
      int64_t w_col, d_col, h_col;
      for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
        offset_d_col = offset_d_col_start + offset_w_col;
        for (d_col = d_col_start; d_col < d_col_end; ++d_col) {
          offset_h_col = offset_h_col_start + offset_d_col;
          for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            val += finput_data[offset_h_col];
            offset_h_col += coeff_h_col;
          }
          offset_d_col += coeff_d_col;
        }
        offset_w_col += coeff_w_col;
      }

      input_data[line_index_offset + count] = val;
      count++;

      if (count < line_seg_len) {
        if (w - pW + 1 == input_width) {
          w = pW;
          if (h - pH + 1 == input_height) {
            h = pH;
            if (d - pT + 1 == input_depth) {
              d = pT;
              c++;
            } else
              d++;
          } else
            h++;
        } else
          w++;
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

void unfolded3d_acc_kernel_cpu(
    Tensor& finput,
    Tensor& input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded3d_acc_cpu", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();
        unfolded3d_acc(
            finput_data,
            input_data,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            pT,
            pH,
            pW,
            n_input_plane,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
      });
}

} // namespace native
} // namespace at
