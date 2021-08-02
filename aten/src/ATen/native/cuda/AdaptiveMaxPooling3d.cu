#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCNumerics.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>


namespace at {
namespace native {

namespace {

__device__ inline int start_index(int a, int b, int c) {
  return (int)std::floor((float)(a * c) / b);
}

__device__ inline int end_index(int a, int b, int c) {
  return (int)std::ceil((float)((a + 1) * c) / b);
}

// 5d tensor B x D x T x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
 template <typename T>
__global__ void adaptivemaxpool(
                        T *input, T *output, int64_t *indices,
                        int isizeT, int isizeH, int isizeW,
                        int osizeT, int osizeH, int osizeW,
                        int64_t istrideD,
                        int64_t istrideT, int64_t istrideH, int64_t istrideW,
                        int64_t offsetZ)
{
  // iterators on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;     // output frame/time
  int d = o_plane / osizeT;  // slice/feature

  // input frame/time ramge is fixed.
  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  T *input_dt = input + d*istrideD + istartT*istrideT;
  // output offset by slice/feature and frame/time
  T *output_dt = output + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/time
  int64_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int istartH = start_index(oh, osizeH, isizeH);
    int iendH   = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int istartW = start_index(ow, osizeW, isizeW);
      int iendW   = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      T *ptr_input = input_dt + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output_dt + oh*osizeW + ow;
      int64_t *ptr_ind = indices_dt + oh*osizeW + ow;
      int64_t argmax = istartT*isizeH*isizeW + istartH*isizeW + istartW;
      T max = at::numeric_limits<T>::lower_bound(); // -Infinity

      int it, ih, iw;
      for(it = 0; it < kT; ++it) {
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            T val = ptr_input[ih*istrideH + iw*istrideW];
            if ((val > max) || THCNumerics<T>::isnan(val)) {
              max = val;
              argmax = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + iw+istartW;
            }
          }
        }
        ptr_input += istrideT;   // next input frame
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind = argmax;
    }
  }
}

template <typename scalar_t>
void adaptivemaxpool_loop(
                        scalar_t *input_data,
                        scalar_t *output_data,
                        int64_t *indices_data,
                        int64_t totalZ,
                        int isizeT, int isizeH, int isizeW,
                        int osizeT, int osizeH, int osizeW,
                        int64_t istrideD,
                        int64_t istrideT, int64_t istrideH, int64_t istrideW)
{
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    adaptivemaxpool<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input_data, output_data, indices_data, isizeT, isizeH, isizeW,
      osizeT, osizeH, osizeW, istrideD, istrideT, istrideH, istrideW, offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    totalZ -= 65535;
    offsetZ += 65535;
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    Assumes that input size can be perfectly divided by output size, i.e.
 *    each input pixel can only be argmax of one output pixel.
 */
 template <typename T>
__global__ void adaptivemaxgradinput(
  T *gradInput, T *gradOutput, int64_t *indices,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW,
  int64_t offsetZ
)
{
  // iterators on output pixels
  int oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  int d = o_plane / osizeT;     // output slice/feature

  // gradInput offset by slice/feature
  T *gradInput_d = gradInput + d*isizeT*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/otme
  T *gradOutput_dt = gradOutput + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/otme
  int64_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {
    for(ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      T *ptr_gradOutput = gradOutput_dt + oh*osizeW + ow;
      int64_t *ptr_ind = indices_dt + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput;
      int argmax = (*ptr_ind);
      gradInput_d[argmax] += grad_delta;
    }
  }
}

template <typename scalar_t>
void adaptivemaxgradinput_loop(
  scalar_t *gradInput_data,
  scalar_t *gradOutput_data,
  int64_t *indices_data,
  int64_t totalZ,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW)
{
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    adaptivemaxgradinput<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      gradInput_data, gradOutput_data, indices_data,
      isizeT, isizeH, isizeW, osizeT, osizeH, osizeW, offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    Uses atomic add.
 */
 template <typename T>
__global__ void atomicadaptivemaxgradinput(
  T *gradInput, T *gradOutput, int64_t *indices,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW,
  int64_t offsetZ
)
{
  // iterators on output pixels
  int oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  int d = o_plane / osizeT;     // output slice/feature

  // gradInput offset by slice/feature
  T *gradInput_d = gradInput + d*isizeT*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/otme
  T *gradOutput_dt = gradOutput + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/otme
  int64_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {
    for(ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      T *ptr_gradOutput = gradOutput_dt + oh*osizeW + ow;
      int64_t *ptr_ind = indices_dt + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput;
      int64_t argmax = (*ptr_ind);
      gpuAtomicAddNoReturn(&(gradInput_d[argmax]), grad_delta);
    }
  }
}

template <typename scalar_t>
void atomicadaptivemaxgradinput_loop(
  scalar_t *gradInput_data,
  scalar_t *gradOutput_data,
  int64_t *indices_data,
  int64_t totalZ,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW)
{
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    atomicadaptivemaxgradinput<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      gradInput_data, gradOutput_data, indices_data,
      isizeT, isizeH, isizeW, osizeT, osizeH, osizeW, offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    totalZ -= 65535;
    offsetZ += 65535;
  }
}
} // namespace

// 5d tensor B x D x T x H x W

TORCH_IMPL_FUNC(adaptive_max_pool3d_out_cuda)
(const Tensor& input,
 IntArrayRef output_size,
 const Tensor& output,
 const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(
      __func__, {output_arg, indices_arg, input_arg});

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input_ = input.ndimension() == 4 ? input : input.contiguous();

  if (input_.ndimension() == 4) {
    sizeD = input_.size(0);
    isizeT = input_.size(1);
    isizeH = input_.size(2);
    isizeW = input_.size(3);

    istrideD = input_.stride(0);
    istrideT = input_.stride(1);
    istrideH = input_.stride(2);
    istrideW = input_.stride(3);

    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input_.size(0);
    sizeD = input_.size(1);
    isizeT = input_.size(2);
    isizeH = input_.size(3);
    isizeW = input_.size(4);

    istrideD = input_.stride(1);
    istrideT = input_.stride(2);
    istrideH = input_.stride(3);
    istrideW = input_.stride(4);

    totalZ = sizeB * sizeD * osizeT;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "adaptive_max_pool3d_cuda", [&] {
        scalar_t* input_data = input_.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();
        int64_t* indices_data = indices.data_ptr<int64_t>();

        adaptivemaxpool_loop(
            input_data,
            output_data,
            indices_data,
            totalZ,
            isizeT,
            isizeH,
            isizeW,
            osizeT,
            osizeH,
            osizeW,
            istrideD,
            istrideT,
            istrideH,
            istrideW);
      });
}

TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_cuda)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__,
      {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  const Tensor gradOutput_ = gradOutput.contiguous();

  gradInput.zero_();

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t osizeT, osizeH, osizeW;
  int64_t totalZ;

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    osizeT = gradOutput_.size(1);
    osizeH = gradOutput_.size(2);
    osizeW = gradOutput_.size(3);
  } else {
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    osizeT = gradOutput_.size(2);
    osizeH = gradOutput_.size(3);
    osizeW = gradOutput_.size(4);
  }

  bool atomic = (isizeW % osizeW != 0) || (isizeH % osizeH != 0) ||
      (isizeT % osizeT != 0);

  if (input.ndimension() == 4) {
    totalZ = sizeD * osizeT;
  } else {
    int sizeB = input.size(0);
    totalZ = sizeB * sizeD * osizeT;
  }

  if (atomic) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool3d_backward_cuda",
        [&] {
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput_.data_ptr<scalar_t>();
          int64_t* indices_data = indices.data_ptr<int64_t>();

          atomicadaptivemaxgradinput_loop(
              gradInput_data,
              gradOutput_data,
              indices_data,
              totalZ,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool3d_backward_cuda",
        [&] {
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput_.data_ptr<scalar_t>();
          int64_t* indices_data = indices.data_ptr<int64_t>();

          adaptivemaxgradinput_loop(
              gradInput_data,
              gradOutput_data,
              indices_data,
              totalZ,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  }
 }
} // at::native
} // at
