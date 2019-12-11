#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at {
namespace native {
namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit

#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

// kernels borrowed from Caffe
template <typename scalar_t, typename accscalar_t>
__global__ void max_pool_forward_nchw(const int nthreads, const scalar_t* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    accscalar_t maxval = at::numeric_limits<accscalar_t>::lower_bound(); // -Infinity
    int maxidx = hstart * width + wstart;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        scalar_t val = bottom_data[h * width + w];
        if ((ScalarConvert<scalar_t, accscalar_t>::to(val) > maxval) || THCNumerics<scalar_t>::isnan(val)) {
          maxidx = h * width + w;
          maxval = ScalarConvert<scalar_t, accscalar_t>::to(val);
        }
      }
    }
    top_data[index] = ScalarConvert<scalar_t, accscalar_t>::to(maxval);
    top_mask[index] = maxidx;
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_forward_nhwc(const scalar_t* bottom_data,
                                   const int channels, const int height,
                                   const int width, const int pooled_height, const int pooled_width,
                                   const int kernel_h, const int kernel_w, const int stride_h,
                                   const int stride_w, const int pad_h, const int pad_w,
                                   const int dilation_h, const int dilation_w,
                                   const int in_stride_c, const int in_stride_h, const int in_stride_w,
                                   scalar_t* top_data, int64_t* top_mask) {
  extern __shared__ int smem[];
  int *out_mask_cached = smem;
  scalar_t *out_cached = reinterpret_cast<scalar_t*>(&out_mask_cached[channels*blockDim.y*blockDim.z]);

  // flattening cta for pre-computation & smem initialization;
  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  // use shared memory to store temporary output value. This is simply to
  // reduce register usage.
  for (int i = thread_id; i < channels*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = at::numeric_limits<scalar_t>::lower_bound();
    out_mask_cached[i] = 0;
  }

  __syncthreads();

  top_data = top_data + blockIdx.x * pooled_height * pooled_width * channels;
  top_mask = top_mask + blockIdx.x * pooled_height * pooled_width * channels;
  bottom_data = bottom_data + blockIdx.x * channels * height * width;

  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * channels];
  out_mask_cached = &out_mask_cached[(threadIdx.z * blockDim.y + threadIdx.y) * channels];

  int oH = (pooled_height + gridDim.z-1) / gridDim.z;
  int oW = (pooled_width + gridDim.y-1) / gridDim.y;
  int ostartH = threadIdx.z + blockIdx.z*oH;
  int oendH = ::min(ostartH+oH, pooled_height);
  int ostartW = threadIdx.y + blockIdx.y*oW;
  int oendW = ::min(ostartW+oW, pooled_width);

  for (int oh = ostartH; oh < oendH; oh+=blockDim.z) {
    for (int ow = ostartW; ow < oendW; ow+=blockDim.y) {
      int hstart = oh * stride_h - pad_h;
      int wstart = ow * stride_w - pad_w;
      int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
      int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
      while(hstart < 0)
        hstart += dilation_h;
      while(wstart < 0)
        wstart += dilation_w;
      for (int ih = hstart; ih < hend; ih++) {
        for (int iw = wstart; iw < wend; iw++) {
          const scalar_t *ptr_input = bottom_data + ih * in_stride_h + iw * in_stride_w;
          for(int c = threadIdx.x; c < channels; c+= blockDim.x) {
            scalar_t val = ptr_input[c];
            if ((scalar_cast<accscalar_t>(val) > out_cached[c]) || THCNumerics<scalar_t>::isnan(val)) {
              out_cached[c] = scalar_cast<accscalar_t>(val);
              out_mask_cached[c] = ih * width + iw;
            }
          }
        }
      }
      scalar_t *ptr_output_data = top_data + (oh * pooled_width + ow) * channels;
      int64_t *ptr_output_mask = top_mask + (oh * pooled_width + ow) * channels;
      for(int c = threadIdx.x; c < channels; c+= blockDim.x) {
        ptr_output_data[c] = out_cached[c];
        ptr_output_mask[c] = out_mask_cached[c];
        out_cached[c] = at::numeric_limits<scalar_t>::lower_bound();
        out_mask_cached[c] = 0;
      }
    }
  }
}


static const int BLOCK_THREADS = 256;

template <typename scalar_t, typename accscalar_t>
#if defined (__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 4)
#else
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 8)
#endif
__global__ void max_pool_backward_nchw(const int nthreads, const scalar_t* top_diff,
    const int64_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    scalar_t* bottom_diff) {
    CUDA_KERNEL_LOOP(index, height*width) {
    int h = index/width;
    int w = index - h * width;
    int phstart = p_start(h, pad_h, kernel_h, dilation_h, stride_h);
    int phend = p_end(h, pad_h, pooled_height, stride_h);
    int pwstart = p_start(w, pad_w, kernel_w, dilation_w, stride_w);
    int pwend = p_end(w, pad_w, pooled_width, stride_w);
    for (int n = blockIdx.y; n < num; n += gridDim.y)
       for (int c = blockIdx.z; c < channels; c+= gridDim.z) {

        accscalar_t gradient = accscalar_t(0);
        int offset = (n * channels + c) * pooled_height * pooled_width;
        top_diff += offset;
        top_mask += offset;
        if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (top_mask[ph * pooled_width + pw] == h * width + w) {
              gradient += ScalarConvert<scalar_t, accscalar_t>::to(top_diff[ph * pooled_width + pw]);
            }
          }
        }
        } else {
            if (top_mask[phstart * pooled_width + pwstart] == h * width + w) {
              gradient += ScalarConvert<scalar_t, accscalar_t>::to(top_diff[phstart * pooled_width + pwstart]);
            }
        }
        bottom_diff[(n*channels+c)*height*width+index] = ScalarConvert<accscalar_t, scalar_t>::to(gradient);
      }
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_backward_nhwc(const int nthreads, const scalar_t* top_diff,
                                    const int64_t* top_mask, const int num, const int channels,
                                    const int height, const int width, const int pooled_height,
                                    const int pooled_width, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w,
                                    const int out_stride_c, const int out_stride_h, const int out_stride_w,
                                    const int in_stride_c, const int in_stride_h, const int in_stride_w,
                                    scalar_t* bottom_diff) {
  extern __shared__ int smem[];
  scalar_t *out_cached = reinterpret_cast<scalar_t*>(smem);

  int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  int block_size = blockDim.x * blockDim.y * blockDim.z;

  for (int i = thread_id; i < channels*blockDim.y*blockDim.z; i+= block_size) {
    out_cached[i] = scalar_t(0.0);
  }
  __syncthreads();
  out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * channels];

  bottom_diff = bottom_diff + blockIdx.x * height * width * channels;
  top_mask = top_mask + blockIdx.x * pooled_height * pooled_width * channels;
  top_diff = top_diff + blockIdx.x * pooled_height * pooled_width * channels;

  int iH = (height + gridDim.z-1) / gridDim.z;
  int iW = (width + gridDim.y-1) / gridDim.y;
  int istartH = threadIdx.z + blockIdx.z*iH;
  int iendH = ::min(istartH+iH, height);
  int istartW = threadIdx.y + blockIdx.y*iW;
  int iendW = ::min(istartW+iW, width);

  for (int ih = istartH; ih < iendH; ih+=blockDim.z) {
    for (int iw = istartW; iw < iendW; iw+=blockDim.y) {
      int phstart = p_start(ih, pad_h, kernel_h, dilation_h, stride_h);
      int phend = p_end(ih, pad_h, pooled_height, stride_h);
      int pwstart = p_start(iw, pad_w, kernel_w, dilation_w, stride_w);
      int pwend = p_end(iw, pad_w, pooled_width, stride_w);
      if ((phstart + 1 != phend) || (pwstart + 1 != pwend)) {
        for(int oh = phstart; oh < phend; ++oh) {
          for(int ow = pwstart; ow < pwend; ++ow) {
            const int64_t* ptr_top_mask = top_mask + oh * out_stride_h + ow * out_stride_w;
            for (int c = threadIdx.x; c < channels; c += blockDim.x) {
              if (ptr_top_mask[c] == ih * width + iw) {
                out_cached[c] += scalar_cast<scalar_t>(top_diff[oh * out_stride_h + ow * out_stride_w + c]);
              }
            }
          }
        }
        scalar_t *ptr_bottom_diff = bottom_diff + (ih * width + iw) * channels;
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
          ptr_bottom_diff[c] = out_cached[c];
          out_cached[c] = scalar_t(0.0);
        }
      } else {
        const int64_t* ptr_top_mask = top_mask + phstart * out_stride_h + pwstart * out_stride_w;
        scalar_t *ptr_bottom_diff = bottom_diff + (ih * width + iw) * channels;
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
          if (ptr_top_mask[c] == ih * width + iw) {
            ptr_bottom_diff[c] = scalar_cast<scalar_t>(top_diff[phstart * out_stride_h + pwstart * out_stride_w + c]);
          }
        }
      }
    }
  }
}

void max_pool2d_with_indices_out_cuda_template(
           Tensor& output,
           Tensor& indices,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {output_arg, indices_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous(memory_format);

  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  output.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  indices.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  const int count = safe_downcast<int, int64_t>(output.numel());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data_ptr<scalar_t>();
      scalar_t *input_data = input.data_ptr<scalar_t>();
      int64_t *indices_data = indices.data_ptr<int64_t>();

      switch (memory_format) {
        case MemoryFormat::ChannelsLast: {
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
          int grid_x = nbatch;
          int grid_y = std::min<int>(
              at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
              cuda::ATenCeilDiv(safe_downcast<int, int64_t>(outputWidth), block_y*BLOCK_STRIDE));
          int grid_z = std::min<int>(
              at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
              cuda::ATenCeilDiv(safe_downcast<int, int64_t>(outputHeight), block_z*BLOCK_STRIDE));
          const dim3 grid(grid_x, grid_y, grid_z);

          max_pool_forward_nhwc<scalar_t, scalar_t>
          <<<grid, block, nInputPlane * block_y * block_z * (sizeof(int) + sizeof(scalar_t)), at::cuda::getCurrentCUDAStream()>>>(
              input_data,
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  in_stride_c, in_stride_h, in_stride_w,
                  output_data, indices_data);
          break;
        }
        case MemoryFormat::Contiguous: {
          const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                           BLOCK_THREADS);
          max_pool_forward_nchw<scalar_t, scalar_t>
              <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              count, input_data,
                  nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  output_data, indices_data);
          break;
        }
        default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
    }
  );

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "max_pool2d_with_indices_out_cuda_frame failed with error code ",
     cudaGetLastError());

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

void max_pool2d_with_indices_backward_out_cuda_template(
           Tensor& gradInput,
           const Tensor& gradOutput_,
           const Tensor& input_,
           const Tensor& indices,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }
  const Tensor input = input_.contiguous(memory_format);

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    input_,
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    /*cuda=*/ true);

  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t out_stride_c = gradOutput.stride(-3);
  const int64_t out_stride_h = gradOutput.stride(-2);
  const int64_t out_stride_w = gradOutput.stride(-1);

  gradInput.resize_as_(input);
  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  int64_t count = input.numel();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
      scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
      int64_t *indices_data = indices.data_ptr<int64_t>();

      switch (memory_format) {
        case MemoryFormat::ChannelsLast: {
          const int max_threads = std::min<int>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
          int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
          int block_x = std::min<int>(
              maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), at::cuda::warp_size()));
          int block_y = std::min<int>(
              maxThreadsDim[1], std::min<int>(lastPow2(inputWidth), max_threads / block_x));
          int block_z = std::min<int>(
              maxThreadsDim[2], std::min<int>(lastPow2(inputHeight), max_threads / block_x / block_y));
          block_x = std::min<int>(
              maxThreadsDim[0], std::min<int>(lastPow2(nInputPlane), max_threads / block_y / block_z));
          const dim3 block(block_x, block_y, block_z);
          int grid_x = nbatch;
          int grid_y = std::min<int>(
              at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
              cuda::ATenCeilDiv(safe_downcast<int, int64_t>(inputWidth), block_y*BLOCK_STRIDE));
          int grid_z = std::min<int>(
              at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
              cuda::ATenCeilDiv(safe_downcast<int, int64_t>(inputHeight), block_z*BLOCK_STRIDE));
          const dim3 grid(grid_x, grid_y, grid_z);

          max_pool_backward_nhwc<scalar_t, accscalar_t>
          <<<grid, block, nInputPlane * block_y * block_z * sizeof(scalar_t), at::cuda::getCurrentCUDAStream()>>>(
              count,
                  gradOutput_data,
                  indices_data,
                  nbatch,
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  out_stride_c, out_stride_h, out_stride_w,
                  in_stride_c, in_stride_h, in_stride_w,
                  gradInput_data);
          break;
        }
        case MemoryFormat::Contiguous: {
          int imgcount = inputWidth * inputHeight;
          dim3 grid;
          const int blocks = (imgcount + BLOCK_THREADS - 1) / BLOCK_THREADS;
          grid.x = blocks;
          grid.y = nbatch;
          uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
          if (maxGridY < grid.y) grid.y = maxGridY;
          grid.z = nInputPlane;
          uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
          if (maxGridZ < grid.z) grid.z = maxGridZ;

          max_pool_backward_nchw<scalar_t, accscalar_t>
          <<<grid, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
              count,
                  gradOutput_data,
                  indices_data,
                  nbatch,
                  nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                  kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                  gradInput_data);
          break;
        }
        default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
    }
  );

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
    "fractional_max_pool2d_backward_out_cuda failed with error code ",
    cudaGetLastError());
}

} // namespace

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out_cuda(
  Tensor& output,
  Tensor& indices,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  max_pool2d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices_cuda(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  max_pool2d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool2d_with_indices_backward_out_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  max_pool2d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

Tensor max_pool2d_with_indices_backward_cuda(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_pool2d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

} // at::native
} // at
