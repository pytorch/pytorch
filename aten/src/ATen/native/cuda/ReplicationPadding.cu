#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/Utils.h"
#include "c10/util/Exception.h"
#include <THC/THCGeneral.h>
#include "THC/THCNumerics.cuh"
#include "THC/THCDeviceUtils.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>


namespace at {
namespace native {
__host__ __device__ __forceinline__ int imin(int a, int b) {
  return a > b ? b : a;
}

__host__ __device__ __forceinline__ int imax(int a, int b) {
  return a > b ? a : b;
}

__host__ __device__ __forceinline__ int iabs(int a) {
  return a >= 0 ? a : -a;
}

namespace {
template <typename scalar_t>
__global__ void replication_pad_forward_kernel(
    PackedTensorAccessor<scalar_t, 3> input,
    PackedTensorAccessor<scalar_t, 3> output,
    int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= output.size(2)) {
    return;
  }
  int outputPointX = outputPointId % output.size(2);

  int iStartX = imax(0, -padL);
  int oStartX = imax(0, padL);

  int inputPointX = imin(imax(padL, outputPointX), input.size(2) + padL - 1) - oStartX + iStartX;

  scalar_t valueToCopy = input[batch][plane][inputPointX];
  output[batch][plane][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor<scalar_t, 3> gradInput,
    PackedTensorAccessor<scalar_t, 3> gradOutput,
    int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= gradOutput.size(2)) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.size(2);

  int iStartX = imax(0, -padL);
  int oStartX = imax(0, padL);

  int inputPointX = imin(imax(padL, outputPointX), gradInput.size(2) + padL - 1) - oStartX + iStartX;

  scalar_t valueToCopy = gradOutput[batch][plane][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointX], valueToCopy);
}

void replication_pad1d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  AT_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int planeDim = 0;
  int dimw = 1;
  int numBatch = 1;

  int numInputDims = input.ndimension();
  AT_CHECK(input.numel() > 0 && (numInputDims == 2 || numInputDims == 3),
      "2D or 3D (batch mode) tensor expected for input")

    if (numInputDims == 3) {
      numBatch = input.size(0);
      planeDim++;
      dimw++;
    }

  int numPlanes = input.size(planeDim);
  int inputW = input.size(dimw);
  int outputW  = inputW + padL + padR;

  AT_CHECK(outputW >= 1,
      "input (W: ", inputW, ")is too small."
      " Calculated output W: ", outputW);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "replication_pad1d", [&] {


      if (numInputDims == 2) {
        output.resize_({numPlanes, outputW});
        auto input_ = input.reshape({1, input.size(0), input.size(1)});
        auto output_ = output.reshape({1, output.size(0), output.size(1)});
        auto devInput = input_.packed_accessor<scalar_t, 3>();
        auto devOutput = output_.packed_accessor<scalar_t, 3>();

        int outputPlaneSize = devOutput.size(2);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
          devOutput.size(1),
          devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel <<<gridSize, blockSize, 0,
          at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput, padL, padR);
      } else {
        output.resize_({numBatch, numPlanes, outputW});
        auto devInput = input.packed_accessor<scalar_t, 3>();
        auto devOutput = output.packed_accessor<scalar_t, 3>();

        int outputPlaneSize = devOutput.size(2);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel <<<gridSize, blockSize, 0,
           at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput, padL, padR);
      }
    }
  );
  THCudaCheck(cudaGetLastError());
}

void replication_pad1d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{

  AT_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  AT_CHECK(at::cuda::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int planeDim = 0;
  int dimw = 1;

  int numInputDims = input.ndimension();
  if (numInputDims == 3) {
    planeDim++;
    dimw++;
  }
  int iwidth = input.size(dimw);
  int owidth  = iwidth + padL + padR;

  AT_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));

  gradInput.resize_as_(input);
  gradInput.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "replication_pad1d_backward", [&] {

      auto gradInput_ = gradInput;
      auto gradOutput_ = gradOutput;
      if (numInputDims == 2) {
        gradInput_ = gradInput.reshape({1, gradInput.size(0),
          gradInput.size(1)});
        gradOutput_ = gradOutput.reshape({1, gradOutput.size(0),
          gradOutput.size(1)});
      }
      auto devGradInput = gradInput_.packed_accessor<scalar_t, 3>();
      auto devGradOutput = gradOutput_.packed_accessor<scalar_t, 3>();

      int outputPlaneSize = devGradOutput.size(2);
      dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
        devGradOutput.size(1),
        devGradOutput.size(0));
      dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

      replication_pad_backward_kernel <<<gridSize, blockSize, 0,
        at::cuda::getCurrentCUDAStream()>>>(devGradInput, devGradOutput,
            padL, padR);
    }
  );
  THCudaCheck(cudaGetLastError());
}
} // namespace

Tensor& replication_pad1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad1d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad1d_cuda(
    at::Tensor const& input,
    IntList paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad1d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad1d_backward_out_cuda(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  gradInput.resize_as_(input);
  replication_pad1d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad1d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  auto gradInput = at::zeros_like(input);
  replication_pad1d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

} // at::native
} // at
