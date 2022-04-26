#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#endif

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
    const int padL,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;
  if (outputPointId >= output.size(2)) {
    return;
  }
  const int outputPointX = outputPointId % output.size(2);

  const int iStartX = imax(0, -padL);
  const int oStartX = imax(0, padL);

  const int inputPointX = imin(imax(padL, outputPointX), input.size(2) + padL - 1) - oStartX + iStartX;

  scalar_t valueToCopy = input[batch][plane][inputPointX];
  output[batch][plane][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor64<scalar_t, 3> gradInput,
    PackedTensorAccessor64<scalar_t, 3> gradOutput,
    const int padL,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;
  if (outputPointId >= gradOutput.size(2)) {
    return;
  }
  const int outputPointX = outputPointId % gradOutput.size(2);

  const int iStartX = imax(0, -padL);
  const int oStartX = imax(0, padL);

  const int inputPointX = imin(imax(padL, outputPointX), gradInput.size(2) + padL - 1) - oStartX + iStartX;

  scalar_t valueToCopy = gradOutput[batch][plane][outputPointX];
  gpuAtomicAddNoReturn(&gradInput[batch][plane][inputPointX], valueToCopy);
}

template <typename scalar_t>
__global__ void replication_pad_forward_kernel2d(
    PackedTensorAccessor64<scalar_t, 4> input,
    PackedTensorAccessor64<scalar_t, 4> output,
    const int padT,
    const int padL,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;
  if (outputPointId >= output.size(2) * output.size(3)) {
    return;
  }
  const int outputPointX = outputPointId % output.size(3);
  const int outputPointY = outputPointId / output.size(3);

  const int iStartX = imax(0, -padL);
  const int iStartY = imax(0, -padT);
  const int oStartX = imax(0, padL);
  const int oStartY = imax(0, padT);

  const int inputPointX = imin(imax(padL, outputPointX), input.size(3) + padL - 1) - oStartX + iStartX;
  const int inputPointY = imin(imax(padT, outputPointY), input.size(2) + padT - 1) - oStartY + iStartY;

  scalar_t valueToCopy = input[batch][plane][inputPointY][inputPointX];
  output[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor64<scalar_t, 4> gradInput,
    PackedTensorAccessor64<scalar_t, 4> gradOutput,
    const int padT,
    const int padL,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;
  if (outputPointId >= gradOutput.size(2) * gradOutput.size(3)) {
    return;
  }
  const int outputPointX = outputPointId % gradOutput.size(3);
  const int outputPointY = outputPointId / gradOutput.size(3);

  const int iStartX = imax(0, -padL);
  const int iStartY = imax(0, -padT);
  const int oStartX = imax(0, padL);
  const int oStartY = imax(0, padT);

  const int inputPointX = imin(imax(padL, outputPointX), gradInput.size(3) + padL - 1) - oStartX + iStartX;
  const int inputPointY = imin(imax(padT, outputPointY), gradInput.size(2) + padT - 1) - oStartY + iStartY;

  scalar_t valueToCopy = gradOutput[batch][plane][outputPointY][outputPointX];
  gpuAtomicAddNoReturn(&gradInput[batch][plane][inputPointY][inputPointX], valueToCopy);
}

template <typename scalar_t>
__global__ void replication_pad_forward_kernel3d(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    const int pfront,
    const int ptop,
    const int pleft,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;
  if (outputPointId >= (output.size(2) * output.size(3) *
        output.size(4))) {
    return;
  }
  const int outputPointX = outputPointId % output.size(4);
  const int outputPointY = (outputPointId / output.size(4)) % output.size(3);
  const int outputPointZ = outputPointId / (output.size(3) * output.size(4));

  const int iStartX = imax(0, -pleft);
  const int iStartY = imax(0, -ptop);
  const int iStartZ = imax(0, -pfront);
  const int oStartX = imax(0, pleft);
  const int oStartY = imax(0, ptop);
  const int oStartZ = imax(0, pfront);

  const int inputPointX = imin(imax(pleft, outputPointX),
      input.size(4) + pleft - 1) - oStartX + iStartX;
  const int inputPointY = imin(imax(ptop, outputPointY),
      input.size(3) + ptop - 1) - oStartY + iStartY;
  const int inputPointZ = imin(imax(pfront, outputPointZ),
      input.size(2) + pfront - 1) - oStartZ + iStartZ;

  scalar_t valueToCopy =
    input[batch][plane][inputPointZ][inputPointY][inputPointX];
  output[batch][plane][outputPointZ][outputPointY][outputPointX] = valueToCopy;
}

template <typename scalar_t>
__global__ void replication_pad_backward_kernel(
    PackedTensorAccessor64<scalar_t, 5> gradInput,
    PackedTensorAccessor64<scalar_t, 5> gradOutput,
    const int pfront,
    const int ptop,
    const int pleft,
    const int y_shift,
    const int z_shift) {
  const int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  const int plane = blockIdx.y + y_shift;
  const int batch = blockIdx.z + z_shift;

  if (outputPointId >= (gradOutput.size(2) * gradOutput.size(3) *
        gradOutput.size(4))) {
    return;
  }
  const int outputPointX = outputPointId % gradOutput.size(4);
  const int outputPointY = (outputPointId / gradOutput.size(4)) %
    gradOutput.size(3);
  const int outputPointZ = outputPointId / (gradOutput.size(3) *
      gradOutput.size(4));

  const int iStartX = imax(0, -pleft);
  const int iStartY = imax(0, -ptop);
  const int iStartZ = imax(0, -pfront);
  const int oStartX = imax(0, pleft);
  const int oStartY = imax(0, ptop);
  const int oStartZ = imax(0, pfront);

  const int inputPointX = imin(imax(pleft, outputPointX),
      gradInput.size(4) + pleft - 1) - oStartX + iStartX;
  const int inputPointY = imin(imax(ptop, outputPointY),
      gradInput.size(3) + ptop - 1) - oStartY + iStartY;
  const int inputPointZ = imin(imax(pfront, outputPointZ),
      gradInput.size(2) + pfront - 1) - oStartZ + iStartZ;

  scalar_t valueToCopy =
    gradOutput[batch][plane][outputPointZ][outputPointY][outputPointX];
  gpuAtomicAddNoReturn(&gradInput[batch][plane][inputPointZ][inputPointY][inputPointX],
      valueToCopy);
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

  const auto padL = paddingSize[0];
  const auto padR = paddingSize[1];
  const auto padT = paddingSize[2];
  const auto padB = paddingSize[3];
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = input.dim();
  if (numInputDims == 4) {
    planeDim++;
    dimh++;
    dimw++;
  }
  const auto iheight = input.size(dimh);
  const auto iwidth = input.size(dimw);
  const auto oheight = iheight + padT + padB;
  const auto owidth  = iwidth + padL + padR;

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));

  gradInput.resize_as_(input);
  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
      input.scalar_type(), "replication_pad2d_backward_cuda", [&] {

        auto gradInput_ = gradInput;
        auto gradOutput_ = gradOutput;
        if (numInputDims == 3) {
          gradInput_ = gradInput.unsqueeze(0);
          gradOutput_ = gradOutput.unsqueeze(0);
        }
        auto devGradInput = gradInput_.packed_accessor64<scalar_t, 4>();
        auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 4>();

        int64_t outputPlaneSize = devGradOutput.size(2) * devGradOutput.size(3);
        int64_t size1 = devGradOutput.size(1);
        int64_t size0 = devGradOutput.size(0);

        for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
          int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
          for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
            int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));

            dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
            dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

            replication_pad_backward_kernel <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
              devGradInput, devGradOutput, padT, padL, block_y, block_z);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      }
  );
}

static inline void shapeCheck3d(
    const Tensor& input,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback) {
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  int numInputDims = input.dim();

  bool valid_dims = input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  TORCH_CHECK(
       (numInputDims == 4 && input.size(0) != 0 && valid_dims) ||
       (numInputDims == 5 && valid_dims && input.size(4) != 0),
       "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
       input.sizes());

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

  const int idepth = input.size(dimd);
  const int iheight = input.size(dimh);
  const int iwidth = input.size(dimw);
  const int odepth = idepth + pfront + pback;
  const int oheight = iheight + ptop + pbottom;
  const int owidth  = iwidth + pleft + pright;
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

  bool valid_dims = input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  TORCH_CHECK(
      (numInputDims == 4 && valid_dims) ||
      (numInputDims == 5 && valid_dims && input.size(4) != 0),
      "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

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

void replication_pad3d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 6, "padding Size is expected to be 6");
  const auto pleft = paddingSize[0];
  const auto pright = paddingSize[1];
  const auto ptop = paddingSize[2];
  const auto pbottom = paddingSize[3];
  const auto pfront = paddingSize[4];
  const auto pback = paddingSize[5];
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
  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
    input.scalar_type(), "replication_pad3d_backward_cuda", [&] {
      auto gradInput_ = gradInput;
      auto gradOutput_ = gradOutput;
      if (numInputDims == 4) {
        gradInput_ = gradInput.unsqueeze(0);
        gradOutput_ = gradOutput.unsqueeze(0);
      }
      auto devGradInput = gradInput_.packed_accessor64<scalar_t, 5>();
      auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 5>();

      const int64_t outputPlaneSize = devGradOutput.size(2) * devGradOutput.size(3) * devGradOutput.size(4);
      const int64_t size1 = devGradOutput.size(1);
      const int64_t size0 = devGradOutput.size(0);

      for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
        int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
          int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));

          dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
          dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

          replication_pad_backward_kernel <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
                    devGradInput, devGradOutput, pfront, ptop, pleft, block_y, block_z);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
}
} // namespace

TORCH_IMPL_FUNC(replication_pad1d_out_cuda) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");

  int64_t padL = paddingSize[0];
  int64_t padR = paddingSize[1];
  constexpr int64_t planeDim = -2;
  constexpr int64_t dimw = -1;

  int numInputDims = input.ndimension();

  int64_t numPlanes = input.size(planeDim);
  int64_t inputW = input.size(dimw);
  int64_t outputW  = output.size(dimw);

  if (input.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
    input.scalar_type(), "replication_pad1d_cuda", [&] {
      at::Tensor input_ = input;
      at::Tensor output_ = output;
      if (numInputDims == 2) {
        input_ = input.unsqueeze(0);
        output_ = output.unsqueeze(0);
      }

      auto devInput = input_.packed_accessor64<scalar_t, 3>();
      auto devOutput = output_.packed_accessor64<scalar_t, 3>();

      int64_t outputPlaneSize = devOutput.size(2);
      int64_t size1 = devOutput.size(1);
      int64_t size0 = devOutput.size(0);

      for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
        int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
          int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));

          dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
          dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

          replication_pad_forward_kernel1d <<<gridSize, blockSize, 0,
            at::cuda::getCurrentCUDAStream()>>>(devInput, devOutput, padL, block_y, block_z);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cuda) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef paddingSize,
  const Tensor& gradInput
) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad1d_backward_cuda");

  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(gradOutput),
      "output gradient tensor must fit into 32-bit index math");

  const int padL = paddingSize[0];
  int planeDim = 0;
  int dimw = 1;

  int numInputDims = input.ndimension();
  if (numInputDims == 3) {
    planeDim++;
    dimw++;
  }
  int iwidth = input.size(dimw);

  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
      input.scalar_type(), "replication_pad1d_backward_cuda", [&] {

      auto gradInput_ = gradInput;
      auto gradOutput_ = gradOutput;
      if (numInputDims == 2) {
        gradInput_ = gradInput.unsqueeze(0);
        gradOutput_ = gradOutput.unsqueeze(0);
      }
      auto devGradInput = gradInput_.packed_accessor64<scalar_t, 3>();
      auto devGradOutput = gradOutput_.packed_accessor64<scalar_t, 3>();

      int64_t outputPlaneSize = devGradOutput.size(2);
      int64_t size1 = devGradOutput.size(1);
      int64_t size0 = devGradOutput.size(0);

      for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
        int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
          int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));

          dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
          dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

          replication_pad_backward_kernel <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
            devGradInput, devGradOutput, padL, block_y, block_z);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
  });
}

TORCH_IMPL_FUNC(replication_pad2d_out_cuda) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  if (input.numel() == 0) {
    return;
  }
  const auto padL = paddingSize[0];
  // const auto padR = paddingSize[1]; // This padding is ignored here
  const auto padT = paddingSize[2];
  // const auto padB = paddingSize[3]; // This padding is ignored here
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
    input.scalar_type(), "replication_pad2d_cuda", [&] {
      at::Tensor input_ = input;
      at::Tensor output_ = output;
      if (input.dim() == 3) {
        input_ = input.unsqueeze(0);
        output_ = output.unsqueeze(0);
      }
      auto devInput = input_.packed_accessor64<scalar_t, 4>();
      auto devOutput = output_.packed_accessor64<scalar_t, 4>();
      int64_t outputPlaneSize = devOutput.size(2) * devOutput.size(3);
      int64_t size1 = devOutput.size(1);
      int64_t size0 = devOutput.size(0);
      for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
        int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
          int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));
          dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
          dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);
          replication_pad_forward_kernel2d <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
              devInput, devOutput, padT, padL, block_y, block_z);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
}

Tensor& replication_pad2d_backward_out_cuda(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad2d_backward_out_cuda");
  replication_pad2d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad2d_backward_cuda");
  auto gradInput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad2d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}


TORCH_IMPL_FUNC(replication_pad3d_out_cuda) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  const auto pleft = paddingSize[0];
  // const auto pright = paddingSize[1]; // Ignored here
  const auto ptop = paddingSize[2];
  // const auto pbottom = paddingSize[3]; // Ignored here
  const auto pfront = paddingSize[4];
  // const auto pback = paddingSize[5]; // Ignored here

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

  const auto numPlanes = input.size(planeDim);
  const auto inputD = input.size(dimd);
  const auto inputH = input.size(dimh);
  const auto inputW = input.size(dimw);
  const auto outputD = output.size(dimd);
  const auto outputH = output.size(dimh);
  const auto outputW = output.size(dimw);

  if (input.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(kHalf,
    input.scalar_type(), "replication_pad3d_cuda", [&] {
      at::Tensor input_ = input;
      at::Tensor output_ = output;
      if (numInputDims == 4) {
        input_ = input.unsqueeze(0);
        output_ = output.unsqueeze(0);
      }

      auto devInput = input_.packed_accessor64<scalar_t, 5>();
      auto devOutput = output_.packed_accessor64<scalar_t, 5>();

      const int64_t outputPlaneSize = devOutput.size(2) * devOutput.size(3) * devOutput.size(4);
      const int64_t size1 = devOutput.size(1);
      const int64_t size0 = devOutput.size(0);

      for (int64_t block_y = 0; block_y < size1; block_y += 65535) {
        int64_t block_y_size = std::min(size1 - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size0; block_z += 65535) {
          int64_t block_z_size = std::min(size0 - block_z, static_cast<int64_t>(65535));

          dim3 gridSize(ceil_div(outputPlaneSize, static_cast<int64_t>(256)), block_y_size, block_z_size);
          dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

          replication_pad_forward_kernel3d <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
              devInput, devOutput, pfront, ptop, pleft, block_y, block_z);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
}

Tensor& replication_pad3d_backward_out_cuda(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad3d_backward_out_cuda");
  replication_pad3d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cuda(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad3d_backward_cuda");
  auto gradInput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad3d_backward_out_cuda_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

} // at::native
} // at
