#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/Utils.h"
#include "c10/util/Exception.h"
#include <THC/THCGeneral.h>
#include "THC/THCNumerics.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>

#define START_IND(a,b,c) (int)std::floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)std::ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit


namespace at {
namespace native {

namespace {

  // 4d tensor B x D x H x W
  // All kernels view batch dim B and feature dim D as collapsed.

  /*
   * Description:
   *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
   *    4D input, 4D output
   */
   template <typename T>
  __global__ void adaptiveaveragepool(T *input, T *output,
                          int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int64_t istrideD, int64_t istrideH, int64_t istrideW)
  {
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
        T *ptr_input = input + istartH*istrideH + istartW*istrideW;
        T *ptr_output = output + oh*osizeW + ow;
        T sum = ScalarConvert<int, T>::to(0);
        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            T val = ptr_input[iw*istrideW];
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
  __global__ void adaptiveaveragegradinput(
    T *gradInput, T *gradOutput,
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
  __global__ void atomicadaptiveaveragegradinput(
    T *gradInput, T *gradOutput,
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
        T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
        T grad_delta = *ptr_gradOutput / kW / kH;

        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            // atomic add since different threads could update same variable
            atomicAdd(&(ptr_gradInput[iw]), grad_delta);
          }
          ptr_gradInput += isizeW; // next input line
        }
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
    checkAllSameGPU("cudnn_adaptive_avg_pooling2d", {input_arg, output_arg});

    for (int64_t i = 0; i < input.ndimension(); i++) {
      AT_CHECK(input.size(i) > 0,
        "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i, " being "
        "empty");
    }

    AT_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
    Tensor input_ = input;
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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_.type(), "adaptive_avg_pool2d", [&] {
          scalar_t *input_data = input_.data<scalar_t>();
          scalar_t *output_data = output.data<scalar_t>();

          // cuda blocks & threads:
          int blocksH = std::max<int64_t>((int)(16L / sizeD), 1);
          dim3 blocks(grid_x, blocksH);
          dim3 threads(32, 8);

          // run averagepool kernel
          adaptiveaveragepool <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
            input_data, output_data,
            isizeH, isizeW, osizeH, osizeW,
            istrideD, istrideH, istrideW);
          }
      );
    THCudaCheck(cudaGetLastError());
  }

  void adaptive_avg_pool2d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input)
  {
    TensorArg grad_input_arg{ gradInput, "gradInput", 1 },
              grad_output_arg{ gradOutput_, "gradOutput_", 2 },
              input_arg{ input, "input", 3 };
    checkAllSameGPU("cudnn_adaptive_avg_pooling2d_out",
                    {grad_input_arg, grad_output_arg, input_arg});

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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "adaptive_avg_pool2d_backward", [&] {
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
          scalar_t *gradInput_data = gradInput.data<scalar_t>();

          // cuda blocks & threads:
          int blocksH = std::max((int)(16L / sizeD), 1);
          dim3 blocks(grid_x, blocksH);
          dim3 threads(32, 8);

          if(atomic)
          {
            // run updateGradInput kernel, accumulate gradients atomically
            atomicadaptiveaveragegradinput <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
              gradInput_data, gradOutput_data,
              isizeH, isizeW, osizeH, osizeW);
          }
          else
          {
            // run updateGradInput kernel
            adaptiveaveragegradinput <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>> (
              gradInput_data, gradOutput_data,
              isizeH, isizeW, osizeH, osizeW);
          }
        }
      );
    THCudaCheck(cudaGetLastError());
  }

} // namespace

  Tensor& adaptive_avg_pool2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size)
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
    gradInput.resize_as_(input);
    adaptive_avg_pool2d_backward_out_cuda_template(
      gradInput, gradOutput, input);
    return gradInput;
  }

  Tensor adaptive_avg_pool2d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input)
  {
    auto gradInput = at::zeros_like(input);
    adaptive_avg_pool2d_backward_out_cuda_template(
      gradInput, gradOutput, input);
    return gradInput;
  }

} // at::native
} // at

#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
