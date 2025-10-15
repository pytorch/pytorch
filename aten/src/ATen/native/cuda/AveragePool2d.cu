#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#endif

namespace at::native {
namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

__device__ inline int max(int a, int b) {
  return a >= b ? a : b;
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame(const int nthreads,
    const scalar_t* const bottom_data, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      continue;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame_nhwc(const int nthreads,
    const scalar_t* const bottom_data, const int64_t channels,
    const int64_t height, const int64_t width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      continue;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + n * channels * height * width + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[(h * width + w) * channels];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void avg_pool2d_backward_out_cuda_frame(const index_t nthreads, const scalar_t* const top_diff,
    const int64_t channels, const int64_t height,
    const int64_t width, const int64_t pooled_height, const int64_t pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override,
    bool count_include_pad, bool use_divisor) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);

        if (hstart >= hend || wstart >= wend) {
          continue;
        }

        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if(count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_diff_slice[ph * pooled_width + pw] / divide_factor;
      }
    }
    bottom_diff[index] = static_cast<scalar_t>(gradient);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void avg_pool2d_backward_out_cuda_frame_nhwc(const index_t nthreads,
    const scalar_t* const top_diff,
    const int64_t channels, const int64_t height,
    const int64_t width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override,
    bool count_include_pad, bool use_divisor) {
  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const int c = index % channels;
    const int w = (index / channels) % width;
    const int h = (index / channels / width) % height;
    const int n = index / channels / width / height;

    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_diff_slice = top_diff + n * channels * pooled_height * pooled_width + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);

        if (hstart >= hend || wstart >= wend) {
          continue;
        }

        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if(count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_diff_slice[(ph * pooled_width + pw) * channels] / divide_factor;
      }
    }
    bottom_diff[index] = static_cast<scalar_t>(gradient);
  }
}

} // anonymous namespace

TORCH_IMPL_FUNC(avg_pool2d_out_cuda)
(const Tensor& input_,
 int64_t kH_,
 int64_t kW_,
 int64_t dH_,
 int64_t dW_,
 int64_t padH_,
 int64_t padW_,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameGPU("avg_pool2d_out_cuda", {output_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kH_);
  const int kW = safe_downcast<int, int64_t>(kW_);

  const int dH = safe_downcast<int, int64_t>(dH_);
  const int dW = safe_downcast<int, int64_t>(dW_);

  const int padH = safe_downcast<int, int64_t>(padH_);
  const int padW = safe_downcast<int, int64_t>(padW_);

  /* sizes */
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const auto memory_format = input_.suggest_memory_format();

  Tensor input = input_.contiguous(memory_format);

  const auto count = safe_downcast<int32_t, int64_t>(output.numel());
  const uint32_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = ceil_div<uint32_t>(count, num_threads);

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

  if (count != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
      "avg_pool2d_out_cuda_frame",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *output_data = output.mutable_data_ptr<scalar_t>();
        const scalar_t *input_data = input.const_data_ptr<scalar_t>();

        switch (memory_format){
          case MemoryFormat::ChannelsLast: {
            output.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast);
            avg_pool2d_out_cuda_frame_nhwc<scalar_t, accscalar_t>
                <<<num_blocks,
                   num_threads,
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    count,
                    input_data,
                    nInputPlane,
                    inputHeight,
                    inputWidth,
                    outputHeight,
                    outputWidth,
                    kH,
                    kW,
                    dH,
                    dW,
                    padH,
                    padW,
                    output_data,
                    divisor_override_value,
                    count_include_pad,
                    use_divisor);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            break;
          }
          case MemoryFormat::Contiguous: {
            avg_pool2d_out_cuda_frame<scalar_t, accscalar_t>
                <<<num_blocks,
                   num_threads,
                   0,
                   at::cuda::getCurrentCUDAStream()>>>(
                    count,
                    input_data,
                    nInputPlane,
                    inputHeight,
                    inputWidth,
                    outputHeight,
                    outputWidth,
                    kH,
                    kW,
                    dH,
                    dW,
                    padH,
                    padW,
                    output_data,
                    divisor_override_value,
                    count_include_pad,
                    use_divisor);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            break;
          }
          default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      }
    );
  }
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_cuda) (
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override,
  const Tensor& gradInput
) {
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("avg_pool2d_backward_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const auto memory_format = input_.suggest_memory_format();
  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);


  const auto count = input.numel();
  if (count == 0) {
    return;
  }

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool gesm10x = properties->major >= 10;
  int double_threads = 1024;
  if (gesm10x) {
    double_threads = 768;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "avg_pool2d_backward_out_cuda_frame",
    [&] {
      const uint32_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, std::is_same<scalar_t, double>::value ? double_threads : 1024);
      const uint32_t num_blocks = ceil_div<uint32_t>(count, num_threads);

      using accscalar_t = acc_type<scalar_t, true>;

      const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
      scalar_t *gradInput_data = gradInput.mutable_data_ptr<scalar_t>();

      AT_DISPATCH_INDEX_TYPES(
        at::native::canUse32BitIndexMath(input, INT_MAX) ? ScalarType::Int : ScalarType::Long,
        "avg_pool2d_backward_out_cuda_frame_launcher",
        [&] {
              switch (memory_format) {

                case MemoryFormat::ChannelsLast: {
                  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::ChannelsLast);
                  avg_pool2d_backward_out_cuda_frame_nhwc<scalar_t, accscalar_t, index_t>
                    <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      count,
                      gradOutput_data,
                      nInputPlane,
                      inputHeight, inputWidth,
                      outputHeight, outputWidth,
                      kH, kW,
                      dH, dW,
                      padH, padW,
                      gradInput_data,
                      divisor_override_value,
                      count_include_pad, use_divisor);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  break;
                }
                case MemoryFormat::Contiguous: {
                  avg_pool2d_backward_out_cuda_frame<scalar_t, accscalar_t, index_t>
                    <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      count,
                      gradOutput_data,
                      nInputPlane,
                      inputHeight, inputWidth,
                      outputHeight, outputWidth,
                      kH, kW,
                      dH, dW,
                      padH, padW,
                      gradInput_data,
                      divisor_override_value,
                      count_include_pad, use_divisor);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  break;
                }
                default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
              }
            });
        });
}

} // namespace at::native
