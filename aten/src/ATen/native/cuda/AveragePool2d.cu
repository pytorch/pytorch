#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

#define CUDA_MAX_THREADS 1024
#define BLOCK_STRIDE 2

namespace at {
namespace native {
namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

__device__ inline int max(int a, int b) {
  return a >= b ? a : b;
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_out_cuda_frame(const int nthreads,
    const scalar_t* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
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
    // This is to make sure it gives exactly the same results as NHWC below, 
    // div vs mul reciprocal could have 10^-5 difference for half precision. 
    accscalar_t mul_factor = accscalar_t(1.0) / divide_factor; 
    top_data[index] = ScalarConvert<accscalar_t, scalar_t>::to(aveval * mul_factor);
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void avg_pool2d_out_cuda_frame_nhwc(
    const scalar_t* bottom_data, const int nbatch, 
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int in_stride_n, const int in_stride_c, 
    const int in_stride_h, const int in_stride_w,
    const int kernel_stride_C, const int kernel_size_C,
    scalar_t* top_data, 
    const int divisor_override, 
    const bool count_include_pad, 
    const bool use_divisor) {
  // reserved for future use
  const int dilation_h = 1; 
  const int dilation_w = 1; 

  extern __shared__ int smem[];
  accscalar_t *out_cached = reinterpret_cast<accscalar_t*>(smem);

  // flattening cta for pre-computation & smem initialization;
  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  // use shared memory to store temporary output value. This is simply to
  // reduce register usage.
  for (int i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = accscalar_t(0.0); 
  }

  __syncthreads();

  int batch_id = blockIdx.x % nbatch; 
  int channel_id = blockIdx.x / nbatch; 
  int channel_offset = threadIdx.x + channel_id * blockDim.x; 

  top_data = top_data + batch_id * pooled_height * pooled_width * channels;
  bottom_data = bottom_data + batch_id * in_stride_n;

  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C*blockDim.x];

  int oH = (pooled_height + gridDim.z-1) / gridDim.z;
  int oW = (pooled_width + gridDim.y-1) / gridDim.y;
  int ostartH = threadIdx.z + blockIdx.z*oH;
  int oendH = ::min(ostartH+oH, pooled_height);
  int ostartW = threadIdx.y + blockIdx.y*oW;
  int oendW = ::min(ostartW+oW, pooled_width);

  for (int oh = ostartH; oh < oendH; oh+=blockDim.z) {
    int hstart = oh * stride_h - pad_h;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height + pad_h);
    for (int ow = ostartW; ow < oendW; ow+=blockDim.y) {
      int wstart = ow * stride_w - pad_w;
      int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width + pad_w);

      // pool_size if count_include_pad
      const int pool_size = (hend - hstart) * (wend - wstart); 
      while(hstart < 0)
        hstart += dilation_h;
      while(wstart < 0)
        wstart += dilation_w;
      hend = min(hend, height);
      wend = min(wend, width);

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
      // avoid division in loops
      accscalar_t mul_factor = accscalar_t(1.0) / divide_factor;

      for (int ih = hstart; ih < hend; ih++) {
        for (int iw = wstart; iw < wend; iw++) {
          int cached_index = threadIdx.x; 
          const scalar_t *ptr_input = bottom_data + ih * in_stride_h + iw * in_stride_w;
          for(int c = channel_offset; c < channels; c+= blockDim.x*kernel_stride_C) {
            out_cached[cached_index] += ptr_input[c*in_stride_c];
            cached_index += blockDim.x; 
          }
        }
      }
      scalar_t *ptr_output_data = top_data + (oh * pooled_width + ow) * channels;

      int cached_index = threadIdx.x; 
      for(int c = channel_offset; c < channels; c+= blockDim.x*kernel_stride_C) {
        ptr_output_data[c] = scalar_cast<scalar_t>(out_cached[cached_index] * mul_factor);
        out_cached[cached_index] = accscalar_t(0.0); 
        cached_index += blockDim.x; 
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_backward_out_cuda_frame(const int nthreads, const scalar_t* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override, 
    bool count_include_pad, bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
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
    bottom_diff[index] = ScalarConvert<accscalar_t, scalar_t>::to(gradient);
  }
}

void avg_pool2d_out_cuda_template(
  Tensor& output,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input_, "input_", 2 };

  checkAllSameGPU("avg_pool2d_out_cuda", {output_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast){
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, 1, 1,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous(memory_format);

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  output.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0; 

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "avg_pool2d_out_cuda_frame",
    [&] {
      AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "avg_pool2d_out_cuda_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *input_data = input.data_ptr<scalar_t>();

        switch (memory_format){
          case MemoryFormat::ChannelsLast: {
            const int64_t in_stride_n = input.stride(-4);
            const int64_t in_stride_c = input.stride(-3);
            const int64_t in_stride_h = input.stride(-2);
            const int64_t in_stride_w = input.stride(-1);

            const int max_threads = std::min<int>(
                at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
            int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
            int block_x = std::min<int>(
                maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), at::cuda::warp_size()));
            int block_y = std::min<int>(
                maxThreadsDim[1], std::min<int>(lastPow2(outputWidth), max_threads / block_x));
            int block_z = std::min<int>(
                maxThreadsDim[2], std::min<int>(lastPow2(outputHeight), max_threads / block_x / block_y));
            block_x = std::min<int>(
                maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), max_threads / block_y / block_z));
            const dim3 block(block_x, block_y, block_z);

            int kernel_stride_C = cuda::ATenCeilDiv(
                safe_downcast<int, int64_t>(nInputPlane), block_x * 4); 
            int kernel_size_C = cuda::ATenCeilDiv(
                safe_downcast<int, int64_t>(nInputPlane), block_x * kernel_stride_C); 

            int grid_x = nbatch*kernel_stride_C;
            int grid_y = std::min<int>(
                at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
                cuda::ATenCeilDiv(safe_downcast<int, int64_t>(outputWidth), block_y*BLOCK_STRIDE));
            int grid_z = std::min<int>(
                at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
                cuda::ATenCeilDiv(safe_downcast<int, int64_t>(outputHeight), block_z*BLOCK_STRIDE));
            const dim3 grid(grid_x, grid_y, grid_z);

            size_t shmem_size = (kernel_size_C * block_x*block_y*block_z) * sizeof(accscalar_t);
            AT_ASSERT(shmem_size <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock); 

            avg_pool2d_out_cuda_frame_nhwc<scalar_t, accscalar_t>
            <<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                input_data, nbatch,
                nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                kH, kW, dH, dW, padH, padW,
                in_stride_n, in_stride_c,
                in_stride_h, in_stride_w,
                kernel_stride_C, kernel_size_C,
                output_data,
                divisor_override_value,
                count_include_pad,
                use_divisor);
            break;
          }
          case MemoryFormat::Contiguous: {
            const int32_t count = safe_downcast<int32_t, int64_t>(output.numel());
            const uint32_t  num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
            const uint32_t num_blocks = cuda::ATenCeilDiv<uint32_t>(count, num_threads);

            avg_pool2d_out_cuda_frame<scalar_t, accscalar_t>
              <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                input_data,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                output_data,
                divisor_override_value,
                count_include_pad, use_divisor);
            break; 
          }
          default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous"); 
        }
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());

  if (input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

Tensor& avg_pool2d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("avg_pool2d_backward_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const Tensor input = input_.contiguous();
  const Tensor gradOutput = gradOutput_.contiguous();

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  avg_pool2d_backward_shape_check(
    input_,
    gradOutput_,
    nbatch,
    kH, kW, dH, dW, padH, padW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  gradInput.resize_as_(input);

  const int32_t count =  safe_downcast<int32_t, int64_t>(input.numel());
  const uint32_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = cuda::ATenCeilDiv<uint32_t>(count, num_threads);

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0; 

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "avg_pool2d_backward_out_cuda_frame",
    [&] {
      AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "avg_pool2d_backward_out_cuda_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();

        avg_pool2d_backward_out_cuda_frame<scalar_t, accscalar_t>
            <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              count,
              gradOutput_data,
              nbatch,
              nInputPlane,
              inputHeight, inputWidth,
              outputHeight, outputWidth,
              kH, kW,
              dH, dW,
              padH, padW,
              gradInput_data,
              divisor_override_value, 
              count_include_pad, use_divisor);
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());

  return gradInput;
}

} // namespace

Tensor& avg_pool2d_out_cuda(
  Tensor& output,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  avg_pool2d_out_cuda_template(
   output,
   input,
   kernel_size,
   stride,
   padding,
   ceil_mode,
   count_include_pad,
   divisor_override);
  return output;
}

Tensor avg_pool2d_cuda(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  Tensor output = at::empty({0}, input.options());
  avg_pool2d_out_cuda_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return output;
}

Tensor& avg_pool2d_backward_out_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  avg_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return gradInput;
}

Tensor avg_pool2d_backward_cuda(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  avg_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override);
  return gradInput;
}

} // at::native
} // at
