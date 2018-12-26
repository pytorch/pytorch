#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/detail/TensorInfo.cuh"
#include "ATen/cuda/detail/KernelUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/Utils.h"
#include "c10/util/Exception.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at {
namespace native {

using namespace at::cuda::detail;

namespace {

template <typename scalar_t, typename accscalar_t>
__device__ inline int get_interval(accscalar_t sample,
                                  int index,
                                  int inputSize,
                                  int outputSize,
                                  int poolSize) {
  accscalar_t alpha = (accscalar_t)(inputSize - poolSize) / (accscalar_t) (outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return (int) ((index + sample) * alpha) - (int) (sample * alpha);
  }
}

template <typename scalar_t, typename index_t>
__device__ inline scalar_t* get_ref_by_coord(TensorInfo<scalar_t, index_t> tensor_info,
                                             int ndims,
                                             index_t batch,
                                             index_t plane,
                                             index_t H,
                                             index_t W) {
  index_t offset = 0;
  if(ndims == 3) {
    offset = plane * tensor_info.strides[0] +
             H * tensor_info.strides[1] +
             W * tensor_info.strides[2];
  } else {
    offset = batch * tensor_info.strides[0] +
             plane * tensor_info.strides[1] +
             H * tensor_info.strides[2] +
             W * tensor_info.strides[3];
  }
  return tensor_info.data + offset;
}

template <typename scalar_t>
__global__ void fractional_max_pool2d_out_frame(
  TensorInfo<scalar_t, int> input,
  TensorInfo<scalar_t, int> output,
  TensorInfo<int64_t, int> indices,
  TensorInfo<scalar_t, int> samples,
  IntList pool_size,
  int PoolSizeWStatic) {

  using accscalar_t = at::acc_type<scalar_t, true>;

  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  int dim_output = output.dims;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.sizes[dim_output] * output.sizes[dim_output]) {
    int outputW = ourOutputPoint % output.sizes[dim_output];
    int outputH = ourOutputPoint / output.sizes[dim_output];

    int poolW = get_interval<scalar_t,
                             accscalar_t
                             >(static_cast<accscalar_t>(*(samples.data +
                                                      batch * samples.strides[0]
                                                      + plane * samples.strides[1])),
                               outputW,
                               input.sizes[dim_output],
                               output.sizes[dim_output],
                               poolSizeW);
    int poolH = get_interval<scalar_t,
                             accscalar_t
                             >(static_cast<accscalar_t>(*(samples.data +
                                  batch * samples.strides[0] +
                                  plane * samples.strides[1] + 1)),
                               outputH,
                               input.sizes[dim_output - 1],
                               output.sizes[dim_output - 1],
                               poolSizeH);

    scalar_t maxVal = at::numeric_limits<scalar_t>::lowest();
    int maxIndex = -1;

    for (int h = poolH; h < poolH + poolSizeH; ++h) {
      if (PoolSizeWStatic == -1) {
        for (int w = poolW; w < poolW + poolSizeW; ++w) {
          scalar_t val = *get_ref_by_coord<scalar_t,
                                           int>(input, dim_output,
                                             batch, plane, h, w);
          // for consistency with THNN, favor the first max
          if (val > maxVal) {
            maxIndex = h * input.sizes[dim_output] + w;
            maxVal = val;
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < PoolSizeWStatic; ++i) {
          int w = i + poolW;
          scalar_t val = *get_ref_by_coord<scalar_t,
                                           int>(input, dim_output,
                                             batch, plane, h, w);
          // for consistency with THNN, favor the first max
          if (val > maxVal) {
            maxIndex = h * input.sizes[3] + w;
            maxVal = val;
          }
        }
      }
    }

    assert(maxVal != at::numeric_limits<scalar_t>::lowest());
    assert(maxIndex != -1);

    int idx_offset = outputW * output.strides[dim_output] +
                     outputH * output.strides[dim_output - 1] +
                     plane * output.strides[dim_output - 2] +
                     (dim_output == 3 ? 0 : batch * output.strides[0]);
    *(indices.data + idx_offset) = maxIndex;
    *(output.data + idx_offset) = maxVal;
  }
}

template <typename scalar_t>
__global__ void fractional_max_pool2d_backward_out_frame(
  TensorInfo<scalar_t, int> gradInput,
  TensorInfo<scalar_t, int> gradOutput,
  TensorInfo<int64_t, int> indices) {
  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  int dim_output = gradOutput.dims;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.sizes[dim_output - 1] *
    gradOutput.sizes[dim_output]) {
    int outputW = ourOutputPoint % gradOutput.sizes[dim_output];
    int outputH = ourOutputPoint / gradOutput.sizes[dim_output];

    int index = *get_ref_by_coord<int64_t,
                                  int>(indices, dim_output,
                                    batch, plane, outputH, outputW);
    assert(index >= 0);
    int inputW = index % gradInput.sizes[dim_output];
    int inputH = index / gradInput.sizes[dim_output];
    assert(inputH < gradInput.sizes[dim_output - 1]);

    atomicAdd(get_ref_by_coord<scalar_t,
                               int>(gradInput, dim_output,
                                 batch, plane, inputH, inputW),
              *get_ref_by_coord<scalar_t,
                                int>(gradOutput, dim_output,
                                  batch, plane, inputH, inputW));
  }
}

void fractional_max_pool2d_out_cuda_template(
  Tensor & output,
  Tensor& indices,
  const Tensor& input,
  IntList pool_size,
  IntList output_size,
  const Tensor& randomSamples) {
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int64_t numBatch = 1;

  int numInputDims = input.ndimension();
  for (int64_t i = 0; i < numInputDims; i++) {
     AT_CHECK(input.size(i) > 0,
       "fractional_max_pool2d(): expected input to have non-empty spatial dimensions, "
       "but input has sizes ", input.sizes(), " with dimension ", i, " being "
       "empty");
   }

   AT_CHECK((numInputDims == 3 || numInputDims == 4),
     "non-empty 3D or 4D (batch mode) tensor expected for input");

  if (numInputDims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(dimh);
  int64_t inputW = input.size(dimw);

  int64_t outputH = output_size[0];
  int64_t outputW = output_size[1];
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  AT_CHECK(outputH + poolSizeH - 1 <= inputH,
             "fractional_max_pool2d(): pool_size height ", poolSizeH,
             " too large relative to input height ", inputH);
  AT_CHECK(outputW + poolSizeW - 1 <= inputW,
           "pool_size width ", poolSizeW,
           " too large relative to input width ", inputW);

  if (numInputDims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW});
    indices.resize_({numBatch, numPlanes, outputH, outputW});
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = output.size(numInputDims - 1) *
    output.size(numInputDims);
  dim3 grid((outputPlaneSize + 127) / 128,
            input.size(numInputDims - 2),
            numInputDims == 3 ? 1 : input.size(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  int POOL_W = (poolSizeW <= 7 && poolSizeW >= 2) ? poolSizeW : -1;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(),
    "fractional_max_pool2d_out_frame",
    [&] {
      fractional_max_pool2d_out_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(output),
          getTensorInfo<int64_t, int>(indices),
          getTensorInfo<scalar_t, int>(randomSamples),
          pool_size,
          POOL_W);
        }
      );
}

void fractional_max_pool2d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntList pool_size,
  IntList output_size,
  const Tensor& indices)
{
  int dimh = 1;
  int dimw = 2;

  int64_t numInputDims = input.ndimension();
  if (numInputDims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t inputH = input.size(dimh);
  int64_t inputW = input.size(dimw);

  int64_t outputH = output_size[0];
  int64_t outputW = output_size[1];
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  AT_CHECK(outputH == gradOutput.size(dimh),
           "fractional_max_pool2d(): gradOutput height unexpected");
  AT_CHECK(outputW == gradOutput.size(dimw),
           "fractional_max_pool2d(): gradOutput width unexpected");

  /* resize */
  gradInput = at::zeros_like(input);

  int dim_output = gradOutput.ndimension();

  /* backprop */
  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = gradOutput.size(dim_output - 1) *
    gradOutput.size(dim_output);
  dim3 grid((outputPlaneSize + 127) / 128,
            gradInput.size(dim_output - 2),
            gradInput.size(dim_output - 3));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOutput.type(),
  "fractional_max_pool2d_backward_out_frame",
  [&] {
    fractional_max_pool2d_backward_out_frame<scalar_t>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        getTensorInfo<scalar_t, int>(gradInput),
        getTensorInfo<scalar_t, int>(gradOutput),
        getTensorInfo<int64_t, int>(indices));
      }
    );
}

}// namespace

std::tuple<Tensor&, Tensor&> fractional_max_pool2d_out_cuda(
  at::Tensor& output,
  at::Tensor& indices,
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& randomSamples)
{
  fractional_max_pool2d_out_cuda_template(
    output,
    indices,
    input,
    pool_size,
    output_size,
    randomSamples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> fractional_max_pool2d_cuda(
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& randomSamples)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  fractional_max_pool2d_out_cuda_template(
    output,
    indices,
    input,
    pool_size,
    output_size,
    randomSamples);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& fractional_max_pool2d_backward_out_cuda(
  at::Tensor& gradInput,
  at::Tensor const& gradOutput_,
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& indices)
{
  gradInput.resize_as_(input);
  fractional_max_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    pool_size,
    output_size,
    indices);
  return gradInput;
}

Tensor fractional_max_pool2d_backward_cuda(
  at::Tensor const& gradOutput_,
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& indices)
{
  Tensor gradInput = at::zeros_like(input);
  fractional_max_pool2d_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    pool_size,
    output_size,
    indices);
  return gradInput;
}

}// at::native
}// at
