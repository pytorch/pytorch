#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/LaunchUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <ATen/native/AdaptivePooling.h>

#include <algorithm>
#include <cfloat>
#include <cmath>

#define START_IND(a,b,c) ((int64_t)((a / b) * c + ((a % b) * c) / b))
#define END_IND(a,b,c) (1 + ((int64_t)(a + 1) * c - 1) / b)

#define START_IND_INT(a,b,c) ((a * c) / b)
#define END_IND_INT(a,b,c) (((a + 1) * c + b - 1) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit
#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

namespace at::native {

namespace {

  // 4d tensor B x D x H x W
  // All kernels view batch dim B and feature dim D as collapsed.

  /*
   * Description:
   *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
   *    4D input, 4D output
   */
   template <typename scalar_t>
  __global__ void adaptive_average_pool(const scalar_t *input, scalar_t *output,
                          int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int64_t istrideD, int64_t istrideH, int64_t istrideW)
  {
    using opmath_t = at::opmath_type<scalar_t>;
    // iterators on output pixels
    int oh, ow;

    // select input/output plane based on thread/block ID
    int o_plane = blockIdx.x;
    int i_plane = o_plane;

    output = output + o_plane*osizeH*osizeW;
    input = input + i_plane*istrideD;

    int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
    int oendH = osizeH;
    const int ostepH = blockDim.y*gridDim.y;

    int ostartW = threadIdx.x;
    int oendW = osizeW;
    const int ostepW = blockDim.x;

    // For all output pixels...
    for(oh = ostartH; oh < oendH; oh += ostepH) {

      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for(ow = ostartW; ow < oendW; ow += ostepW) {

        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        // Compute the average pooling over corresponding input pixels
        const scalar_t *ptr_input = input + istartH*istrideH + istartW*istrideW;
        scalar_t *ptr_output = output + oh*osizeW + ow;
        opmath_t sum = static_cast<opmath_t>(0);
        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            scalar_t val = ptr_input[iw*istrideW];
            sum += val;
          }
          ptr_input += istrideH; // next input line
        }
        // Update output
        *ptr_output = sum / kH / kW;
      }
    }
  }

  /*
   * Description:
   *    this function computes the gradInput from gradOutput
   */
   template <typename T>
  __global__ void adaptive_average_gradinput(
    T *gradInput, const T *gradOutput,
    int isizeH, int isizeW, int osizeH, int osizeW
  )
  {
    // iterators on input pixels
    int ih, iw;

    // select input/output plane based on thread/block ID
    int i_plane = blockIdx.x;
    int o_plane = i_plane;

    gradOutput = gradOutput + o_plane*osizeH*osizeW;
    gradInput = gradInput + i_plane*isizeH*isizeW;

    int istartH = blockDim.y*blockIdx.y + threadIdx.y;
    int iendH = isizeH;
    int istepH = blockDim.y*gridDim.y;

    int istartW = threadIdx.x;
    int iendW = isizeW;
    int istepW = blockDim.x;

    // compute gradInput
    for(ih = istartH; ih < iendH; ih += istepH) {

      int ostartH = START_IND(ih, isizeH, osizeH);
      int oendH   = END_IND(ih, isizeH, osizeH);

      for(iw = istartW; iw < iendW; iw += istepW) {

        int ostartW = START_IND(iw, isizeW, osizeW);
        int oendW   = END_IND(iw, isizeW, osizeW);

        // Compute the gradients over corresponding output pixels
        T *ptr_gradInput = gradInput + ih*isizeW + iw;

        int oh, ow;
        for(oh = ostartH; oh < oendH; ++oh) {
          int kH = START_IND(oh, osizeH, isizeH) - END_IND(oh, osizeH, isizeH);
          for(ow = ostartW; ow < oendW; ++ow) {
            int kW = START_IND(ow, osizeW, isizeW) - END_IND(ow, osizeW, isizeW);
            T grad_delta = gradOutput[ow + oh*osizeW] / kH / kW;
            *ptr_gradInput += grad_delta;
          }
        }
      }
    }
  }

  /*
   * Description:
   *    this function computes the gradInput from gradOutput
   *    (uses atomic add)
   */
   template <typename T>
  __global__ void atomic_adaptive_average_gradinput(
    T *gradInput, const T *gradOutput,
    int isizeH, int isizeW, int osizeH, int osizeW
  )
  {
    // iterators on output indices
    int oh, ow;

    // select input/output plane based on thread/block ID
    int o_plane = blockIdx.x;
    int i_plane = o_plane;

    gradOutput = gradOutput + o_plane*osizeW*osizeH;
    gradInput = gradInput + i_plane*isizeW*isizeH;

    int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
    int oendH = osizeH;
    int ostepH = blockDim.y*gridDim.y;

    int ostartW = threadIdx.x;
    int oendW = osizeW;
    int ostepW = blockDim.x;

    // For all output pixels...
    for(oh = ostartH; oh < oendH; oh += ostepH) {

      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for(ow = ostartW; ow < oendW; ow += ostepW) {

        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        // Compute the gradients for over corresponding input pixels
        T *ptr_gradInput = gradInput + istartH*isizeW + istartW;
        const T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
        T grad_delta = *ptr_gradOutput / kW / kH;

        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            // atomic add since different threads could update same variable
            gpuAtomicAddNoReturn(&(ptr_gradInput[iw]), grad_delta);
          }
          ptr_gradInput += isizeW; // next input line
        }
      }
    }
  }

  /*
   * Description:
   *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
   *    NHWC layout for both input and output tensor
   *    4D input, 4D output
   */
   template <typename index_t, typename scalar_t>
  C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
  __global__ void adaptive_average_pool_nhwc(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                          int sizeB, int sizeC,
                          int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int kernel_stride_C, int kernel_size_C,
                          index_t istrideB, index_t istrideC,
                          index_t istrideH, index_t istrideW)
  {
    using opmath_t = at::opmath_type<scalar_t>;
    extern __shared__ int smem[];
    opmath_t *out_cached = reinterpret_cast<opmath_t*>(smem);

    // flattening cta for pre-computation & smem initialization;
    int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int block_size = blockDim.x * blockDim.y * blockDim.z;

    // use shared memory to store temporary output value. This is simply to
    // reduce register usage.
    for (index_t i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
      out_cached[i] = opmath_t(0.0);
    }

    __syncthreads();

    // each CTA handles a portion of a single slice on batch dimension;
    int batch_id = blockIdx.x % sizeB;
    int channel_id = blockIdx.x / sizeB;
    int channel_offset = threadIdx.x + channel_id * blockDim.x;

    // each CTA handles a single slice on batch dimension;
    // We use gridDim.x to handle striding on C as well.
    output = output + batch_id * osizeH * osizeW * sizeC;
    input = input + batch_id * istrideB;

    // split out_cached and exclusively it assigned to each thread;
    out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * kernel_size_C * blockDim.x];

    // iterate on output H & W.
    // Each CTA handles a consecutive H & W section (TILE); Do NOT stride CTA on
    // tile so there's a better chance to hit L1 cache.
    index_t oH = (osizeH + gridDim.z-1) / gridDim.z;
    index_t oW = (osizeW + gridDim.y-1) / gridDim.y;
    index_t ostartH = threadIdx.z + blockIdx.z*oH;
    index_t oendH = ::min(ostartH+oH, osizeH);
    index_t ostartW = threadIdx.y + blockIdx.y*oW;
    index_t oendW = ::min(ostartW+oW, osizeW);

    // Stride for threads, each warp can reuse L1 as they go. So theoretically
    // better chance to survive cache eviction.
    for (int oh = ostartH; oh < oendH; oh+=blockDim.z) {
      int istartH = START_IND_INT(oh, osizeH, isizeH);
      int iendH = END_IND_INT(oh, osizeH, isizeH);
      for (int ow = ostartW; ow < oendW; ow+=blockDim.y) {
        int istartW = START_IND_INT(ow, osizeW, isizeW);
        int iendW = END_IND_INT(ow, osizeW, isizeW);
        scalar_t factor = scalar_t(1.0) / ((iendH-istartH) * (iendW-istartW));

        // loop on input: hierarchy h->w->c, use shared memory here hopefully
        // would not stall global memory read;
        for (index_t ih = istartH; ih < iendH; ih++) {
          for (index_t iw = istartW; iw < iendW; iw++) {
            int cached_index = threadIdx.x;
            const scalar_t *ptr_input = input + ih*istrideH + iw*istrideW;
            for (index_t c = channel_offset;
                 c < sizeC;
                 c += blockDim.x*kernel_stride_C) {
              out_cached[cached_index] += ptr_input[c*istrideC];
              cached_index += blockDim.x;
            }
          }
        }
        scalar_t *ptr_output = output + (oh * osizeW + ow) * sizeC;

        int cached_index = threadIdx.x;
        // write accumulated output to global memory;
        for (index_t c = channel_offset;
             c < sizeC;
             c += blockDim.x*kernel_stride_C) {
          // This causes numerical issueptr when unit test with NCHW kernel;
          // switch to could verify the correctness;
          // output[c] = out_cached[c] / (iendH-istartH) / (iendW-istartW);
          ptr_output[c] = out_cached[cached_index] * factor;
          out_cached[cached_index] = opmath_t(0.0);
          cached_index += blockDim.x;
        }
        // no need to __syncthreads() since out_cached is not shared.
      }
    }
  }

  /*
   * Description:
   *    this function computes the gradInput from gradOutput
   *    NHWC layout for both input and output tensor
   *    4D input, 4D output
   */
   template <typename index_t, typename scalar_t>
  C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
  __global__ void adaptive_average_gradinput_nhwc(scalar_t* __restrict__ gradInput, const scalar_t* __restrict__ gradOutput,
                          int sizeB, int sizeC,
                          int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int kernel_stride_C, int kernel_size_C,
                          index_t ostrideB, index_t ostrideC,
                          index_t ostrideH, index_t ostrideW)
  {
    extern __shared__ int smem[];
    index_t *ostartW_cached = smem;
    index_t *oendW_cached = &ostartW_cached[isizeW];

    // be careful with alignment, in case scalar_t is fp16, we want to assign
    // int pointers first.
    scalar_t *r_kW_cached = reinterpret_cast<scalar_t*>(&oendW_cached[isizeW]);
    scalar_t *r_kH_cached = &r_kW_cached[osizeW];
    scalar_t *out_cached = &r_kH_cached[osizeH];

    // flattening cta for pre-computation & smem initialization;
    int thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int block_size = blockDim.x * blockDim.y * blockDim.z;

    // Precompute output start/end index per input index on width dimension;
    // Not doing this for height dimension, as that's our out-most loop.
    for (index_t i = thread_id; i < isizeW; i+= block_size) {
      ostartW_cached[i] = START_IND_INT(i, isizeW, osizeW);
      oendW_cached[i] = END_IND_INT(i, isizeW, osizeW);
    }

    // Precompute pooling height/weight factor for each output element;
    // This is used to weight output gradient when accumulate them on input
    // gradient.
    // Technically we don't have to compute it for the whole `osizeH`, since
    // each cta only covers a consecutive portion of the entire output. But it's
    // not going to save us from code divergence, and shared memory save is not
    // an issue neither, so just leave it as is for now.
    for (index_t i = thread_id; i < osizeH; i+= block_size) {
      r_kH_cached[i] = scalar_t(1.0) / (END_IND_INT(i, osizeH, isizeH) - START_IND_INT(i, osizeH, isizeH));
    }
    for (index_t i = thread_id; i < osizeW; i+= block_size) {
      r_kW_cached[i] = scalar_t(1.0) / (END_IND_INT(i, osizeW, isizeW) - START_IND_INT(i, osizeW, isizeW));
    }

    // each CTA handles a portion of a single slice on batch dimension;
    int batch_id = blockIdx.x % sizeB;
    int channel_id = blockIdx.x / sizeB;
    int channel_offset = threadIdx.x + channel_id * blockDim.x;

    // use shared memory to store temporary output value. This is simply to
    // reduce register usage.
    for (index_t i = thread_id; i < kernel_size_C*blockDim.x*blockDim.y*blockDim.z; i+= block_size) {
      out_cached[i] = scalar_t(0.0);
    }

    __syncthreads();

    // each CTA handles a portion of a single slice on batch dimension;
    // We use gridDim.x to handle striding on C as well.
    gradInput = gradInput + batch_id * isizeH * isizeW * sizeC;
    gradOutput = gradOutput + batch_id * ostrideB;

    // split out_cached and exclusively it assigned to each thread;
    out_cached = &out_cached[(threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x * kernel_size_C];

    // iterate on input H & W.
    // Each CTA handles a consecutive H & W section (TILE); Do NOT stride CTA on
    // tile so there's a better chance to hit L1 cache.
    index_t iH = (isizeH + gridDim.z-1) / gridDim.z;
    index_t iW = (isizeW + gridDim.y-1) / gridDim.y;
    index_t istartH = threadIdx.z + blockIdx.z*iH;
    index_t iendH = ::min(istartH+iH, isizeH);
    index_t istartW = threadIdx.y + blockIdx.y*iW;
    index_t iendW = ::min(istartW+iW, isizeW);

    // Stride for threads, each warp can reuse L1 as they go. So theoretically
    // better chance to survive cache eviction.
    for (index_t ih = istartH; ih < iendH; ih+=blockDim.z) {
      index_t ostartH = START_IND_INT(ih, isizeH, osizeH);
      index_t oendH = END_IND_INT(ih, isizeH, osizeH);
      for (index_t iw = istartW; iw < iendW; iw+=blockDim.y) {
        // loop on output: hierarchy h->w->c, so we could reuse weight factor f
        // because it remains the same for given oh & ow
        for(index_t oh = ostartH; oh < oendH; ++oh) {
          for(index_t ow = ostartW_cached[iw]; ow < oendW_cached[iw]; ++ow) {
            scalar_t f = r_kW_cached[ow] * r_kH_cached[oh];
            const scalar_t* ptr_gradOutput = gradOutput + oh*ostrideH + ow*ostrideW;
            int cached_index = threadIdx.x;
            for (index_t c = channel_offset;
                 c < sizeC;
                 c += blockDim.x*kernel_stride_C) {
              out_cached[cached_index] += ptr_gradOutput[c*ostrideC] * f;
              cached_index += blockDim.x;
            }
          }
        }
        scalar_t *ptr_gradInput = gradInput + (ih * isizeW + iw) * sizeC;
        int cached_index = threadIdx.x;
        // write accumulated gradIput to global memory;
        for (index_t c = channel_offset;
             c < sizeC;
             c += blockDim.x*kernel_stride_C) {
          ptr_gradInput[c] = out_cached[cached_index];
          out_cached[cached_index] = scalar_t(0.0);
          cached_index += blockDim.x;
        }
        // no need to __syncthreads() since out_cached is not shared.
      }
    }
  }

  // 4d tensor B x D x H x W

  void adaptive_avg_pool2d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size)
  {
    TensorArg input_arg{ input, "input", 1 },
              output_arg{ output, "output", 2 };
    checkAllSameGPU(__func__, {input_arg, output_arg});

    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
    int64_t ndim = input.dim();
    TORCH_CHECK((ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ", input.sizes());
    for (const auto i : {-2, -1}) {
      TORCH_CHECK(input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i + ndim, " being "
        "empty");
    }

    Tensor input_ = input;
    switch (input.suggest_memory_format()) {
      case at::MemoryFormat::ChannelsLast: {
        // special case for tensor memory format in channels_last
        TORCH_CHECK(input.ndimension() == 4,
                    "adaptive_avg_pool2d(): Expected 4D tensor, but got ",
                    input.sizes());

        int sizeB = input_.size(0);
        int sizeC = input_.size(1);
        int isizeH = input_.size(2);
        int isizeW = input_.size(3);

        int64_t istrideB = input_.stride(0);
        int64_t istrideC = input_.stride(1);
        int64_t istrideH = input_.stride(2);
        int64_t istrideW = input_.stride(3);

        int osizeH = output_size[0];
        int osizeW = output_size[1];
        // preserve channels_last stride on output tensor;
        if (!output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
          // TODO: modify this after resize_ added `memory_format` tag
          output.resize_({sizeB, sizeC, osizeH, osizeW}).as_strided_({sizeB, sizeC, osizeH, osizeW}, {sizeC*osizeH*osizeW, 1, osizeW*sizeC, sizeC});
        }

        if (output.numel() == 0) {
          return;
        }

        const int max_threads = std::min<int>(
            at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
        int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
        int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
        size_t sharedMemPerBlock = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;

        // Launch kernel on output tensor elements. Logic behind launch config:
        // output tensor size NCHW, strides NHWC;
        // Launch on:
        // N -> grid.x
        // H -> grid.z * block.z
        // W -> grid.y * block.y
        // C -> block.x
        // encourage larger block_y & block_z for better cache hit while maintain
        // reasonable block_x for coalesced memory access;
        int block_x = std::min<int>(
            maxThreadsDim[0], std::min<int>(lastPow2(sizeC), at::cuda::warp_size()));
        int block_y = std::min<int>(
            maxThreadsDim[1], std::min<int>(lastPow2(osizeW), max_threads / block_x));
        int block_z = std::min<int>(
            maxThreadsDim[2], std::min<int>(lastPow2(osizeH), max_threads / block_x / block_y));
        block_x = std::min<int>(
            maxThreadsDim[0], std::min<int>(lastPow2(sizeC), max_threads / block_y / block_z));
        const dim3 block(block_x, block_y, block_z);
        int kernel_stride_C = ceil_div(sizeC, block_x * 4);
        int kernel_size_C = ceil_div(sizeC, block_x * kernel_stride_C);

        // Do NOT clip grid_x, striding on Batch dimension is not in the kernel,
        // although it could be easily implemented given current kernel.
        int grid_x = sizeB*kernel_stride_C;
        // it's OK to clip grid_y & grid_z, as we block the two dimensions in the kernel;
        int grid_y = std::min<int>(
            maxGridSize[1], ceil_div(osizeW, block_y*BLOCK_STRIDE));
        int grid_z = std::min<int>(
            maxGridSize[2], ceil_div(osizeH, block_z*BLOCK_STRIDE));
        const dim3 grid(grid_x, grid_y, grid_z);


        // we are dealing with packed tensor here. max index is the same as numel.
        // TODO: to really support input tensor large enought to go beyond int32,
        // we will need to restrict out shared memory usage and adjust the launch
        // config;
        AT_ASSERT(input_.numel() < std::numeric_limits<int32_t>::max());
        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
            input_.scalar_type(), "adaptive_avg_pool2d_nhwc_cuda", [&] {
              using opmath_t = at::opmath_type<scalar_t>;
              size_t shmem_size = (kernel_size_C * block_x * block_y * block_z) * sizeof(opmath_t);
              AT_ASSERT(shmem_size <= sharedMemPerBlock);
              adaptive_average_pool_nhwc<int32_t><<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>> (
                input_.const_data_ptr<scalar_t>(),
                output.mutable_data_ptr<scalar_t>(),
                sizeB, sizeC, isizeH, isizeW, osizeH, osizeW,
                kernel_stride_C, kernel_size_C,
                istrideB, istrideC, istrideH, istrideW);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          );
        break;
      }
      case at::MemoryFormat::Contiguous: {
        int64_t grid_x = input.size(-3);
        if (input.ndimension() == 4) {
           input_ = input.contiguous();
           grid_x *= input_.size(-4);
        }
        int64_t sizeD  = input_.size(-3);
        int64_t isizeH = input_.size(-2);
        int64_t isizeW = input_.size(-1);

        int64_t istrideD = input_.stride(-3);
        int64_t istrideH = input_.stride(-2);
        int64_t istrideW = input_.stride(-1);

        int64_t osizeH = output_size[0];
        int64_t osizeW = output_size[1];
        if (input.ndimension() == 4) {
           output.resize_({input_.size(-4), sizeD, osizeH, osizeW});
        } else {
           output.resize_({sizeD, osizeH, osizeW});
        }
        if (output.numel() == 0) {
          return;
        }

        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
            input_.scalar_type(), "adaptive_avg_pool2d_cuda", [&] {
              const scalar_t *input_data = input_.const_data_ptr<scalar_t>();
              scalar_t *output_data = output.mutable_data_ptr<scalar_t>();

              // cuda blocks & threads:
              int blocksH = std::max<int64_t>((int)(16L / sizeD), 1);
              dim3 blocks(grid_x, blocksH);
              dim3 threads(32, 8);

              // run averagepool kernel
              adaptive_average_pool <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
                input_data, output_data,
                isizeH, isizeW, osizeH, osizeW,
                istrideD, istrideH, istrideW);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          );
        break;
      }
      default:
        TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
    }
  }

  void adaptive_avg_pool2d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input)
  {
    TensorArg grad_input_arg{ gradInput, "gradInput", 1 },
              grad_output_arg{ gradOutput_, "gradOutput_", 2 },
              input_arg{ input, "input", 3 };

    adaptive_pool_empty_output_check(gradOutput_, "adaptive_avg_pool2d_backward");

    checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg, input_arg});

    switch (input.suggest_memory_format()) {
      case at::MemoryFormat::ChannelsLast: {
        // special case for tensor memory format in channels_last
        TORCH_CHECK(input.ndimension() == 4,
                    "adaptive_avg_pool2d_backward_cuda(): Expected 4D tensor, but got ", input.ndimension());

        int sizeB = input.size(0);
        int sizeC = input.size(1);
        int isizeH = input.size(2);
        int isizeW = input.size(3);

        Tensor gradOutput = gradOutput_;

        int64_t ostrideB = gradOutput.stride(0);
        int64_t ostrideC = gradOutput.stride(1);
        int64_t ostrideH = gradOutput.stride(2);
        int64_t ostrideW = gradOutput.stride(3);

        int osizeH = gradOutput.size(-2);
        int osizeW = gradOutput.size(-1);

        // preserve channels_last stride on input tensor;
        if (!gradInput.is_contiguous(at::MemoryFormat::ChannelsLast)) {
          gradInput.as_strided_(
              {sizeB, sizeC, isizeH, isizeW},
              {sizeC*isizeH*isizeW, 1, isizeW*sizeC, sizeC});
        }

        int max_threads = std::min<int>(
            at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
        int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
        int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
        size_t sharedMemPerBlock = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;

        // Launch kernel on input tensor elements. Logic behind launch config:
        // input tensor size NCHW, strides NHWC;
        // Launch on:
        // N(C) -> grid.x (striding on C to reduce sh_mem usage)
        // H    -> grid.z * block.z
        // W    -> grid.y * block.y
        // C    -> block.x
        // encourage larger block_y & block_z for better cache hit while maintain
        // reasonable block_x for coalesced memory access;
        bool done = false;
        do {
          int block_x = std::max<int>(std::min<int>(
              maxThreadsDim[0], std::min<int>(lastPow2(sizeC), at::cuda::warp_size())), 1);
          int block_y = std::max<int>(std::min<int>(
              maxThreadsDim[1], std::min<int>(lastPow2(isizeW), max_threads / block_x)), 1);
          int block_z = std::max<int>(std::min<int>(
              maxThreadsDim[2], std::min<int>(lastPow2(isizeH), max_threads / block_x / block_y)), 1);
          block_x = std::max<int>(std::min<int>(
              maxThreadsDim[0], std::min<int>(lastPow2(sizeC), max_threads / block_y / block_z)), 1);
          const dim3 block(block_x, block_y, block_z);
          int kernel_stride_C = ceil_div(sizeC, block_x * 4);
          int kernel_size_C = ceil_div(sizeC, block_x * kernel_stride_C);

          // Do NOT clip grid_x, striding on Batch dimension is not in the kernel,
          // although it could be easily implemented given current kernel.
          int grid_x = sizeB*kernel_stride_C;
          // it's OK to clip grid_y & grid_z, as we block the two dimensions in the kernel;
          int grid_y = std::min<int>(
              maxGridSize[1], ceil_div(isizeW, block_y*BLOCK_STRIDE));
          int grid_z = std::min<int>(
              maxGridSize[2], ceil_div(isizeH, block_z*BLOCK_STRIDE));
          const dim3 grid(grid_x, grid_y, grid_z);

          // we are dealing with packed tensor here. max index is the same as numel.
          // TODO: to really support input tensor large enought to go beyond int32,
          // we will need to restrict out shared memory usage and adjust the launch
          // config;
          AT_ASSERT(input.numel() < std::numeric_limits<int32_t>::max());
          AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
              input.scalar_type(), "adaptive_avg_pool2d_backward_nhwc_cuda", [&] {
                size_t shmem_size = (kernel_size_C * block_x * block_y * block_z + osizeH + osizeW) * sizeof(scalar_t) + 2 * isizeW * sizeof(int32_t);
                if (shmem_size <= sharedMemPerBlock) {
                  adaptive_average_gradinput_nhwc<int32_t><<<grid, block, shmem_size, at::cuda::getCurrentCUDAStream()>>> (
                    gradInput.mutable_data_ptr<scalar_t>(),
                    gradOutput.const_data_ptr<scalar_t>(),
                    sizeB, sizeC, isizeH, isizeW, osizeH, osizeW,
                    kernel_stride_C, kernel_size_C,
                    ostrideB, ostrideC, ostrideH, ostrideW);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                  done = true;
                } else {
                  TORCH_WARN_ONCE("Requested shmem_size exceeds sharedMemPerBlock limit! Reducing max_threads...");
                  max_threads /= 2;
                }
              }
            );
        } while (!done && max_threads);
        if (!done) {
          TORCH_INTERNAL_ASSERT(false, "Couldn't reduce launch bounds to accomodate sharedMemPerBlock limit");
        }
        break;
      }
      case at::MemoryFormat::Contiguous: {
        bool atomic = true; // suboptimal, but without atomic it doesn't pass the tests

        Tensor gradOutput = gradOutput_.contiguous();

        int64_t sizeD  = input.size(-3);
        int64_t isizeH = input.size(-2);
        int64_t isizeW = input.size(-1);

        int64_t osizeH = gradOutput.size(-2);
        int64_t osizeW = gradOutput.size(-1);

        int64_t grid_x = sizeD;
        if (input.ndimension() == 4) grid_x *= input.size(-4);

          //bool atomic = (isizeW%osizeW != 0) || (isizeH%osizeH != 0);
        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
            input.scalar_type(), "adaptive_avg_pool2d_backward_cuda", [&] {
              const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
              scalar_t *gradInput_data = gradInput.mutable_data_ptr<scalar_t>();

              // cuda blocks & threads:
              int blocksH = std::max((int)(16L / sizeD), 1);
              dim3 blocks(grid_x, blocksH);
              dim3 threads(32, 8);

              if(atomic)
              {
                // run updateGradInput kernel, accumulate gradients atomically
                atomic_adaptive_average_gradinput <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
                  gradInput_data, gradOutput_data,
                  isizeH, isizeW, osizeH, osizeW);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
              else
              {
                // run updateGradInput kernel
                adaptive_average_gradinput <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
                  gradInput_data, gradOutput_data,
                  isizeH, isizeW, osizeH, osizeW);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            }
          );
        break;
      }
      default:
        TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");

    }
  }

} // namespace

  Tensor& adaptive_avg_pool2d_out_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output)
  {
    adaptive_avg_pool2d_out_cuda_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d_cuda(
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    auto output = at::empty({0}, input.options());
    adaptive_avg_pool2d_out_cuda_template(
      output, input, output_size);
    return output;
  }

  Tensor& adaptive_avg_pool2d_backward_out_cuda(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input)
  {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
    globalContext().alertNotDeterministic("adaptive_avg_pool2d_backward_out_cuda");
    gradInput.resize_as_(input);
    if (gradInput.numel() != 0) {
      adaptive_avg_pool2d_backward_out_cuda_template(
        gradInput, gradOutput, input);
    }
    return gradInput;
  }

  Tensor adaptive_avg_pool2d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input)
  {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic because of atomicAdd usage
    globalContext().alertNotDeterministic("adaptive_avg_pool2d_backward_cuda");
    auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    if (gradInput.numel() != 0) {
      adaptive_avg_pool2d_backward_out_cuda_template(
        gradInput, gradOutput, input);
    }
    return gradInput;
  }

} // namespace at::native

#undef BLOCK_STRIDE
#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
