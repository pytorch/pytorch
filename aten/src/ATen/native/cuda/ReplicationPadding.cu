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

namespace {
template <typename scalar_t>
__global__ void replication_pad_forward_kernel1d(
    PackedTensorAccessor64<scalar_t, 3> input,
    PackedTensorAccessor64<scalar_t, 3> output,
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
    PackedTensorAccessor64<scalar_t, 3> gradInput,
    PackedTensorAccessor64<scalar_t, 3> gradOutput,
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

template <typename scalar_t>
__global__ void replication_pad_forward_kernel2d(
    PackedTensorAccessor64<scalar_t, 4> input,
    PackedTensorAccessor64<scalar_t, 4> output,
    int padT, int padB, int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= output.size(2) * output.size(3)) {
    return;
  }
  int outputPointX = outputPointId % output.size(3);
  int outputPointY = outputPointId / output.size(3);

  int iStartX = imax(0, -padL);
  int iStartY = imax(0, -padT);
  int oStartX = imax(0, padL);
  int oStartY = imax(0, padT);

  int inputPointX = imin(imax(padL, outputPointX), input.size(3) + padL - 1) - oStartX + iStartX;
  int inputPointY = imin(imax(padT, outputPointY), input.size(2) + padT - 1) - oStartY + iStartY;

  scalar_t valueToCopy = input[batch][plane][inputPointY][inputPointX];
  output[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor64<scalar_t, 4> gradInput,
    PackedTensorAccessor64<scalar_t, 4> gradOutput,
    int padT, int padB, int padL, int padR) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= gradOutput.size(2) * gradOutput.size(3)) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.size(3);
  int outputPointY = outputPointId / gradOutput.size(3);

  int iStartX = imax(0, -padL);
  int iStartY = imax(0, -padT);
  int oStartX = imax(0, padL);
  int oStartY = imax(0, padT);

  int inputPointX = imin(imax(padL, outputPointX), gradInput.size(3) + padL - 1) - oStartX + iStartX;
  int inputPointY = imin(imax(padT, outputPointY), gradInput.size(2) + padT - 1) - oStartY + iStartY;

  scalar_t valueToCopy = gradOutput[batch][plane][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointY][inputPointX], valueToCopy);
}

template <typename scalar_t>
__global__ void replication_pad_forward_kernel3d(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int pfront, int pback, int ptop, int pbottom, int pleft, int pright) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= (output.size(2) * output.size(3) *
        output.size(4))) {
    return;
  }
  int outputPointX = outputPointId % output.size(4);
  int outputPointY = (outputPointId / output.size(4)) % output.size(3);
  int outputPointZ = outputPointId / (output.size(3) * output.size(4));

  int iStartX = imax(0, -pleft);
  int iStartY = imax(0, -ptop);
  int iStartZ = imax(0, -pfront);
  int oStartX = imax(0, pleft);
  int oStartY = imax(0, ptop);
  int oStartZ = imax(0, pfront);

  int inputPointX = imin(imax(pleft, outputPointX),
      input.size(4) + pleft - 1) - oStartX + iStartX;
  int inputPointY = imin(imax(ptop, outputPointY),
      input.size(3) + ptop - 1) - oStartY + iStartY;
  int inputPointZ = imin(imax(pfront, outputPointZ),
      input.size(2) + pfront - 1) - oStartZ + iStartZ;

  scalar_t valueToCopy =
    input[batch][plane][inputPointZ][inputPointY][inputPointX];
  output[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor64<scalar_t, 5> gradInput,
    PackedTensorAccessor64<scalar_t, 5> gradOutput,
    int pfront, int pback, int ptop, int pbottom, int pleft, int pright) {
  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  if (outputPointId >= (gradOutput.size(2) * gradOutput.size(3) *
        gradOutput.size(4))) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.size(4);
  int outputPointY = (outputPointId / gradOutput.size(4)) %
    gradOutput.size(3);
  int outputPointZ = outputPointId / (gradOutput.size(3) *
      gradOutput.size(4));

  int iStartX = imax(0, -pleft);
  int iStartY = imax(0, -ptop);
  int iStartZ = imax(0, -pfront);
  int oStartX = imax(0, pleft);
  int oStartY = imax(0, ptop);
  int oStartZ = imax(0, pfront);

  int inputPointX = imin(imax(pleft, outputPointX),
      gradInput.size(4) + pleft - 1) - oStartX + iStartX;
  int inputPointY = imin(imax(ptop, outputPointY),
      gradInput.size(3) + ptop - 1) - oStartY + iStartY;
  int inputPointZ = imin(imax(pfront, outputPointZ),
      gradInput.size(2) + pfront - 1) - oStartZ + iStartZ;

  scalar_t valueToCopy =
    gradOutput[batch][plane][outputPointZ][outputPointY][outputPointX];
  atomicAdd(&gradInput[batch][plane][inputPointZ][inputPointY][inputPointX],
      valueToCopy);
}

void replication_pad1d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 2, "padding Size is expected to be 2");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int planeDim = 0;
  int dimw = 1;
  int numBatch = 1;

  int numInputDims = input.ndimension();
  TORCH_CHECK(input.numel() > 0 && (numInputDims == 2 || numInputDims == 3),
      "2D or 3D (batch mode) tensor expected for input")

    if (numInputDims == 3) {
      numBatch = input.size(0);
      planeDim++;
      dimw++;
    }

  int numPlanes = input.size(planeDim);
  int inputW = input.size(dimw);
  int outputW  = inputW + padL + padR;

  TORCH_CHECK(outputW >= 1,
      "input (W: ", inputW, ")is too small."
      " Calculated output W: ", outputW);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad1d_cuda", [&] {


      if (numInputDims == 2) {
        output.resize_({numPlanes, outputW});
        auto input_ = input.unsqueeze(0);
        auto output_ = output.unsqueeze(0);
        auto devInput = input_.packed_accessor64<scalar_t, 3>();
        auto devOutput = output_.packed_accessor64<scalar_t, 3>();

        int outputPlaneSize = devOutput.size(2);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel1d <<<gridSize, blockSize, 0,
          at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput, padL, padR);
      } else {
        output.resize_({numBatch, numPlanes, outputW});
        auto devInput = input.packed_accessor64<scalar_t, 3>();
        auto devOutput = output.packed_accessor64<scalar_t, 3>();

        int outputPlaneSize = devOutput.size(2);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel1d <<<gridSize, blockSize, 0,
           at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput, padL, padR);
      }
      }
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

void replication_pad1d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{

  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 2, "padding Size is expected to be 2");

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

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));

  gradInput.resize_as_(input);
  gradInput.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad1d_backward_cuda", [&] {

      auto gradInput_ = gradInput;
      auto gradOutput_ = gradOutput;
      if (numInputDims == 2) {
      gradInput_ = gradInput.unsqueeze(0);
      gradOutput_ = gradOutput.unsqueeze(0);
      }
      auto devGradInput = gradInput_.packed_accessor64<scalar_t, 3>();
      auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 3>();

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
  AT_CUDA_CHECK(cudaGetLastError());
}

void replication_pad2d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 4, "padding Size is expected to be 4");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int padT = paddingSize[2];
  int padB = paddingSize[3];
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int numInputDims = input.dim();
  TORCH_CHECK(input.numel() && (numInputDims == 3 || numInputDims == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input, but got: ",
      input)

  if (numInputDims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  int numPlanes = input.size(planeDim);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);
  int outputH = inputH + padT + padB;
  int outputW  = inputW + padL + padR;

  TORCH_CHECK(outputW >= 1 || outputH >= 1,
      "input (H: ", inputH, ", W: ", inputW, ") is too small."
      " Calculated output H: ", outputH, " W: ", outputW);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad2d_cuda", [&] {


      if (numInputDims == 3) {
        output.resize_({numPlanes, outputH, outputW});
        auto input_ = input.unsqueeze(0);
        auto output_ = output.unsqueeze(0);
        auto devInput = input_.packed_accessor64<scalar_t, 4>();
        auto devOutput = output_.packed_accessor64<scalar_t, 4>();

        int outputPlaneSize = devOutput.size(2) * devOutput.size(3);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel2d <<<gridSize, blockSize, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            devInput, devOutput, padT, padB, padL, padR);
      } else {
        output.resize_({numBatch, numPlanes, outputH, outputW});
        auto devInput = input.packed_accessor64<scalar_t, 4>();
        auto devOutput = output.packed_accessor64<scalar_t, 4>();

        int outputPlaneSize = devOutput.size(2) * devOutput.size(3);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel2d <<<gridSize, blockSize, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput,
                                           padT, padB, padL, padR);
      }
      }
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

void replication_pad2d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{

  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");
  TORCH_CHECK(paddingSize.size() == 4, "padding Size is expected to be 4");

  int padL = paddingSize[0];
  int padR = paddingSize[1];
  int padT = paddingSize[2];
  int padB = paddingSize[3];
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = input.dim();
  if (numInputDims == 4) {
    planeDim++;
    dimh++;
    dimw++;
  }
  int iheight = input.size(dimh);
  int iwidth = input.size(dimw);
  int oheight = iheight + padT + padB;
  int owidth  = iwidth + padL + padR;

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));

  gradInput.resize_as_(input);
  gradInput.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad2d_backward_cuda", [&] {

        auto gradInput_ = gradInput;
        auto gradOutput_ = gradOutput;
        if (numInputDims == 3) {
          gradInput_ = gradInput.unsqueeze(0);
          gradOutput_ = gradOutput.unsqueeze(0);
        }
        auto devGradInput = gradInput_.packed_accessor64<scalar_t, 4>();
        auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 4>();

        int outputPlaneSize = devGradOutput.size(2) * devGradOutput.size(3);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
          devGradOutput.size(1),
          devGradOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);
        replication_pad_backward_kernel <<<gridSize, blockSize, 0,
        at::cuda::getCurrentCUDAStream()>>>(devGradInput, devGradOutput,
          padT, padB, padL, padR);
      }
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

static inline void shapeCheck3d(
    const Tensor& input,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback) {
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  int numInputDims = input.dim();

  TORCH_CHECK(input.numel() && (numInputDims == 4 || numInputDims == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ", input);

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int numPlanes = input.size(planeDim);
  int idepth = input.size(dimd);
  int iheight = input.size(dimh);
  int iwidth = input.size(dimw);
  int odepth = idepth + pfront + pback;
  int oheight = iheight + ptop + pbottom;
  int owidth  = iwidth + pleft + pright;
  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

}

static inline void shapeAndGradOutputCheck3d(
    const Tensor& input,
    const Tensor& gradOutput,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback) {
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  int numInputDims = input.dim();

  TORCH_CHECK(input.numel() && (numInputDims == 4 || numInputDims == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ", input);

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int numPlanes = input.size(planeDim);
  int idepth = input.size(dimd);
  int iheight = input.size(dimh);
  int iwidth = input.size(dimw);
  int odepth = idepth + pfront + pback;
  int oheight = iheight + ptop + pbottom;
  int owidth  = iwidth + pleft + pright;
  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");

  TORCH_CHECK(numPlanes == gradOutput.size(planeDim),
      "gradOutput width unexpected. Expected: ", numPlanes, ", Got: ",
      gradOutput.size(planeDim));
  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));
  TORCH_CHECK(odepth == gradOutput.size(dimd),
      "gradOutput depth unexpected. Expected: ", odepth, ", Got: ",
      gradOutput.size(dimd));
}

void replication_pad3d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 6, "padding Size is expected to be 6");
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  shapeCheck3d(input, pleft, pright, ptop,
      pbottom, pfront, pback);

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  int numBatch = 1;

  int numInputDims = input.dim();

  if (numInputDims == 5) {
    numBatch = input.size(0);
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int numPlanes = input.size(planeDim);
  int inputD = input.size(dimd);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);
  int outputD = inputD + pfront + pback;
  int outputH = inputH + ptop + pbottom;
  int outputW  = inputW + pleft + pright;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad3d_cuda", [&] {

      if (numInputDims == 4) {
        output.resize_({numPlanes, outputD, outputH, outputW});
        auto input_ = input.unsqueeze(0);
        auto output_ = output.unsqueeze(0);
        auto devInput = input_.packed_accessor64<scalar_t, 5>();
        auto devOutput = output_.packed_accessor64<scalar_t, 5>();

        int outputPlaneSize = devOutput.size(2) * devOutput.size(3) *
        devOutput.size(4);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel3d <<<gridSize, blockSize, 0,
        at::cuda::getCurrentCUDAStream()>>>(
            devInput, devOutput, pfront, pback, ptop, pbottom, pleft, pright);
      } else {
        output.resize_({numBatch, numPlanes, outputD, outputH, outputW});
        auto devInput = input.packed_accessor64<scalar_t, 5>();
        auto devOutput = output.packed_accessor64<scalar_t, 5>();

        int outputPlaneSize = devOutput.size(2) * devOutput.size(3) *
          devOutput.size(4);
        dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.size(1),
            devOutput.size(0));
        dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

        replication_pad_forward_kernel3d <<<gridSize, blockSize, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
                                           devInput, devOutput, pfront, pback, ptop, pbottom, pleft, pright);
      }
      }
  );
  AT_CUDA_CHECK(cudaGetLastError());
}

void replication_pad3d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 6, "padding Size is expected to be 6");
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  shapeAndGradOutputCheck3d(input, gradOutput, pleft, pright, ptop,
      pbottom, pfront, pback);

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  int numInputDims = input.dim();
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  gradInput.resize_as_(input);
  gradInput.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "replication_pad3d_backward_cuda", [&] {

      auto gradInput_ = gradInput;
      auto gradOutput_ = gradOutput;
      if (numInputDims == 4) {
        gradInput_ = gradInput.unsqueeze(0);
        gradOutput_ = gradOutput.unsqueeze(0);
      }
      auto devGradInput = gradInput_.packed_accessor64<scalar_t, 5>();
      auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 5>();

      int outputPlaneSize = devGradOutput.size(2) * devGradOutput.size(3) *
      devGradOutput.size(4);
      dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
          devGradOutput.size(1),
          devGradOutput.size(0));
      dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

      replication_pad_backward_kernel <<<gridSize, blockSize, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
                                          devGradInput, devGradOutput, pfront, pback, ptop, pbottom, pleft, pright);
      }
  );
  AT_CUDA_CHECK(cudaGetLastError());
}
} // namespace

Tensor& replication_pad1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  replication_pad1d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad1d_cuda(
    const Tensor& input,
    IntArrayRef paddingSize)
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
    IntArrayRef paddingSize)
{
  replication_pad1d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad1d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad1d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor& replication_pad2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  replication_pad2d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad2d_cuda(
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad2d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad2d_backward_out_cuda(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  replication_pad2d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad2d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor& replication_pad3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  replication_pad3d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad3d_cuda(
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad3d_out_cuda_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad3d_backward_out_cuda(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  replication_pad3d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad3d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

} // at::native
} // at
