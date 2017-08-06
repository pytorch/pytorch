// Copyright 2004-present Facebook. All Rights Reserved.

#import "data_conversion.h"
#import "metal_convolution.h"

#import <stdio.h>
#import <string.h>

void memcvt_F32_F16(float32_t* dst, const float16_t* src, size_t n) {
  int i = 0;
#if defined(__ARM_NEON__)
  for (; i < 4 * (n / 4); i += 4) {
    *((float32x4_t*)&dst[i]) = vcvt_f32_f16(*((float16x4_t*)&src[i]));
  }
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

void memcvt_F16_F32(float16_t* dst, const float32_t* src, size_t n) {
  int i = 0;
#if defined(__ARM_NEON__)
  for (; i < 4 * (n / 4); i += 4) {
    *((float16x4_t*)&dst[i]) = vcvt_f16_f32(*((float32x4_t*)&src[i]));
  }
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <typename T1, typename T2>
void memcpycvt(T1* dst, const T2* src, size_t n);

template <>
void memcpycvt(float* dst, const float* src, size_t n) {
  memcpy(dst, src, n * sizeof(float));
}

template <>
void memcpycvt(float16_t* dst, const float* src, size_t n) {
  parallelize<float16_t, float, memcvt_F16_F32, 2>(
      dst, src, n, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
}

template <>
void memcpycvt(float* dst, const float16_t* src, size_t n) {
  parallelize<float, float16_t, memcvt_F32_F16, 2>(
      dst, src, n, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
}

template <typename filter_type>
bool reformatKernelImage(
    const float* input_data,
    filter_type* output_data,
    int          kernels,
    int          input_kernels,
    int          kernel_offset,
    int          kernel_stride,
    int          channels,
    int          width,
    int          height,
    bool         transposed) {
  const int aligned_kernel_stride = kernel_stride <= 2 ? kernel_stride : 4 * ((kernel_stride + 3) / 4);

  if (output_data) {
    filter_type* buffer = output_data;

    for (int ks = 0; ks < kernels / kernel_stride; ks++) {
      for (int kb = 0; kb < kernel_stride; kb++) {
        int k = kernel_offset + kernel_stride * ks + kb;

        for (int c = 0; c < channels; c++) {
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              buffer[aligned_kernel_stride * (width * height * (ks * channels + c) + y * width + x) + kb] =
                  transposed
                      ? input_data
                            [(c * input_kernels + k) * width * height + (height - 1 - y) * height + (width - 1 - x)]
                      : input_data[(k * channels + c) * width * height + y * width + x];
            }
          }
        }
      }
    }
    return true;
  }
  return false;
}

template bool reformatKernelImage<float16_t>(
    const float* input_data,
    float16_t*   output_data,
    int          kernels,
    int          input_kernels,
    int          kernel_offset,
    int          kernel_stride,
    int          channels,
    int          width,
    int          height,
    bool         transposed);

template bool reformatKernelImage<float>(
    const float* input_data,
    float*       output_data,
    int          kernels,
    int          input_kernels,
    int          kernel_offset,
    int          kernel_stride,
    int          channels,
    int          width,
    int          height,
    bool         transposed);

template <typename filter_type>
filter_type* reformatKernelImage(
    const float*                        input_data,
    int                                 kernels,
    int                                 input_kernels,
    int                                 kernel_offset,
    int                                 kernel_stride,
    int                                 channels,
    int                                 width,
    int                                 height,
    bool                                transposed,
    std::function<filter_type*(size_t)> allocator) {
  const int aligned_kernel_stride = kernel_stride <= 2 ? kernel_stride : 4 * ((kernel_stride + 3) / 4);

  const int buffer_size = aligned_kernel_stride * (kernels / kernel_stride) * channels * width * height;

  filter_type* output_data = allocator(sizeof(filter_type) * buffer_size);

  reformatKernelImage(
      input_data,
      output_data,
      kernels,
      input_kernels,
      kernel_offset,
      kernel_stride,
      channels,
      width,
      height,
      transposed);

  return output_data;
}
template float16_t* reformatKernelImage<float16_t>(
    const float*                      input_data,
    int                               kernels,
    int                               input_kernels,
    int                               kernel_offset,
    int                               kernel_stride,
    int                               channels,
    int                               width,
    int                               height,
    bool                              transposed,
    std::function<float16_t*(size_t)> allocator);

template float* reformatKernelImage<float>(
    const float*                  input_data,
    int                           kernels,
    int                           input_kernels,
    int                           kernel_offset,
    int                           kernel_stride,
    int                           channels,
    int                           width,
    int                           height,
    bool                          transposed,
    std::function<float*(size_t)> allocator);

template <typename filter_type>
bool reformatKernelImage(
    const float*                     input_data,
    int                              kernels,
    int                              kernel_channels,
    int                              kernel_width,
    int                              kernel_height,
    bool                             transposed,
    std::function<filter_type*(int)> allocator) {
  int kernels_per_convolution = kernels;

  if (!calculate_kernels_per_convolution(kernels_per_convolution)) {
    return false;
  }

  const int convolutions = kernels / kernels_per_convolution;

  const int kernel_stride = kernels_per_convolution;

  const int aligned_kernel_stride = kernel_stride <= 2 ? kernel_stride : 4 * ((kernel_stride + 3) / 4);

  const int chunk_size = aligned_kernel_stride * (kernels_per_convolution / kernel_stride) * kernel_channels *
                         kernel_width * kernel_height;

  // This will allocate more memory than needed, but I couldn't find a better solution
  filter_type* output_data = allocator((aligned_kernel_stride + kernel_stride - 1) / kernel_stride);

  for (int c = 0; c < convolutions; c++) {
    if (!reformatKernelImage(
            input_data,
            output_data + c * chunk_size,
            kernels_per_convolution,
            kernels,
            c * kernels_per_convolution,
            kernel_stride,
            kernel_channels,
            kernel_width,
            kernel_height,
            transposed))
      return false;
  }

  return true;
}

template bool reformatKernelImage<float16_t>(
    const float*                   input_data,
    int                            kernels,
    int                            kernel_channels,
    int                            kernel_width,
    int                            kernel_height,
    bool                           transposed,
    std::function<float16_t*(int)> allocator);

template bool reformatKernelImage<float>(
    const float*               input_data,
    int                        kernels,
    int                        kernel_channels,
    int                        kernel_width,
    int                        kernel_height,
    bool                       transposed,
    std::function<float*(int)> allocator);

template <typename out_buffer_type>
out_buffer_type* reformatInputImage(
    const float*                            data,
    int                                     channels,
    int                                     width,
    int                                     height,
    std::function<out_buffer_type*(size_t)> allocator) {
  const int buffer_size = channels * width * height;

  out_buffer_type* output_data = allocator(sizeof(out_buffer_type) * buffer_size);

  if (output_data) {
    memcpycvt(output_data, data, buffer_size);
  }

  return output_data;
}

template float16_t* reformatInputImage<float16_t>(
    const float*                      data,
    int                               channels,
    int                               width,
    int                               height,
    std::function<float16_t*(size_t)> allocator);

template float* reformatInputImage<float>(
    const float*                  data,
    int                           channels,
    int                           width,
    int                           height,
    std::function<float*(size_t)> allocator);
