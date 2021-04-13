#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCNumerics.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at {
namespace native {

using namespace at::cuda::detail;

namespace {

template <typename scalar_t, typename accscalar_t>
__device__ inline int get_interval(accscalar_t sample,
  int index, int inputSize, int outputSize, int poolSize) {
  accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
    static_cast<accscalar_t>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int>((index + sample) * alpha) -
      static_cast<int>(sample * alpha);
  }
}

template <typename scalar_t>
__global__ void fractional_max_pool2d_out_cuda_frame(
  PackedTensorAccessor<scalar_t, 4> output,
  PackedTensorAccessor<int64_t, 4> indices,
  PackedTensorAccessor<scalar_t, 4> input,
  PackedTensorAccessor<scalar_t, 3> samples,
  int poolSizeH, int poolSizeW) {

  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < output.size(2) * output.size(3)) {
    int outputW = ourOutputPoint % output.size(3);
    int outputH = ourOutputPoint / output.size(3);

    int poolW = get_interval<scalar_t, accscalar_t>(
      static_cast<accscalar_t>(samples[batch][plane][0]),
        outputW, input.size(3), output.size(3), poolSizeW);
    int poolH = get_interval<scalar_t, accscalar_t>(
      static_cast<accscalar_t>(samples[batch][plane][1]),
        outputH, input.size(2), output.size(2), poolSizeH);

    scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
    int maxIndex = poolH * input.size(3) + poolW;

    for (int h = poolH; h < poolH + poolSizeH; ++h) {
      if (poolSizeW < 2 || poolSizeW > 7) {
        for (int w = poolW; w < poolW + poolSizeW; ++w) {
          scalar_t val = input[batch][plane][h][w];
          // for consistency with THNN, favor the first max
          if (val > maxVal || THCNumerics<scalar_t>::isnan(val)) {
            maxIndex = h * input.size(3) + w;
            maxVal = val;
          }
        }
      } else {
        for (int i = 0; i < poolSizeW; ++i) {
          int w = i + poolW;
          scalar_t val = input[batch][plane][h][w];
          // for consistency with THNN, favor the first max
          if (val > maxVal || THCNumerics<scalar_t>::isnan(val)) {
            maxIndex = h * input.size(3) + w;
            maxVal = val;
          }
        }
      }
    }

    indices[batch][plane][outputH][outputW] = maxIndex;
    output[batch][plane][outputH][outputW] = maxVal;
  }
}

template <typename scalar_t>
__global__ void fractional_max_pool2d_backward_out_cuda_frame(
  PackedTensorAccessor<scalar_t, 4> gradInput,
  PackedTensorAccessor<scalar_t, 4> gradOutput,
  PackedTensorAccessor<int64_t, 4> indices) {
  // Output (h, w) point that this thread is responsible for
  int ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.size(2) *
    gradOutput.size(3)) {
    int outputW = ourOutputPoint % gradOutput.size(3);
    int outputH = ourOutputPoint / gradOutput.size(3);

    int index = indices[batch][plane][outputH][outputW];
    assert(index >= 0);
    int inputW = index % gradInput.size(3);
    int inputH = index / gradInput.size(3);
    assert(inputH < gradInput.size(2));

    gpuAtomicAdd(
      &gradInput[batch][plane][inputH][inputW],
      gradOutput[batch][plane][outputH][outputW]
    );
  }
}

void fractional_max_pool2d_out_cuda_template(
  Tensor & output,
  Tensor& indices,
  const Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const Tensor& randomSamples) {
  TORCH_CHECK(
      pool_size.size() == 2,
      "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
  TORCH_CHECK(
      output_size.size() == 2,
      "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int ndims = input.ndimension();
  TORCH_CHECK(input.numel() > 0,
    "fractional_max_pool2d(): expected input to have non-empty ",
    "spatial dimensions.");

  TORCH_CHECK((ndims == 3 || ndims == 4),
     "non-empty 3D or 4D (batch mode) tensor expected for input");

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];
  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];

  TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
             "fractional_max_pool2d(): pool_size height ", poolSizeH,
             " too large relative to input height ", inputH);
  TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
           "pool_size width ", poolSizeW,
           " too large relative to input width ", inputW);

  if (ndims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW});
    indices.resize_({numBatch, numPlanes, outputH, outputW});
  }

  auto output_ = output;
  auto input_ = input;
  auto indices_ = indices;

  if(ndims == 3) {
    output_ = output_.reshape({1, numPlanes, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputH, outputW});
    input_ = input_.reshape({1, input.size(0), input.size(1), input.size(2)});
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = output_.size(2) *
    output_.size(3);
  dim3 grid((outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
            input_.size(1),
            input_.size(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "fractional_max_pool2d_out_cuda_frame",
    [&] {
      auto devInput = input_.packed_accessor<scalar_t, 4>();
      auto devOutput = output_.packed_accessor<scalar_t, 4>();
      auto devIndices = indices_.packed_accessor<int64_t, 4>();
      auto devSamples = randomSamples.packed_accessor<scalar_t, 3>();
      fractional_max_pool2d_out_cuda_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          devOutput, devIndices, devInput, devSamples,
          poolSizeH, poolSizeW);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
       }
     );
}

void fractional_max_pool2d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef pool_size /* unused */,
  IntArrayRef output_size,
  const Tensor& indices)
{
  int dimh = 1;
  int dimw = 2;

  int ndims = input.ndimension();
  if (ndims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];

  TORCH_CHECK(outputH == gradOutput.size(dimh),
           "fractional_max_pool2d(): gradOutput height unexpected");
  TORCH_CHECK(outputW == gradOutput.size(dimw),
           "fractional_max_pool2d(): gradOutput width unexpected");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput;
  auto indices_ = indices;

  if(ndims == 3) {
    gradInput_ = gradInput_.reshape({1, input.size(0), inputH, inputW});
    gradOutput_ = gradOutput_.reshape({1, gradOutput.size(0), outputH, outputW});
    indices_ = indices_.reshape({1, indices_.size(0), outputH, outputW});
  }

  /* backprop */
  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = gradOutput_.size(2) *
    gradOutput_.size(3);
  dim3 grid((outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
            gradInput_.size(1),
            gradInput_.size(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  auto devIndices = indices.packed_accessor<int64_t, 4>();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradOutput.scalar_type(),
    "fractional_max_pool2d_backward_out_cuda_frame",
    [&] {
      auto devGradInput = gradInput_.packed_accessor<scalar_t, 4>();
      auto devGradOutput = gradOutput_.packed_accessor<scalar_t, 4>();
      fractional_max_pool2d_backward_out_cuda_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        devGradInput, devGradOutput, devIndices);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  );
}

}// namespace

std::tuple<Tensor&, Tensor&> fractional_max_pool2d_out_cuda(
    const at::Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const at::Tensor& randomSamples,
    at::Tensor& output,
    at::Tensor& indices) {
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
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples)
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

Tensor& fractional_max_pool2d_backward_out_cuda(const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices,
  at::Tensor& gradInput)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("fractional_max_pool2d_backward_out_cuda");
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
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("fractional_max_pool2d_backward_cuda");
  Tensor gradInput = at::empty({0}, input.options());
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
