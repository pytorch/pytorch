#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <THC/THCAtomics.cuh>

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at {
namespace native {

using namespace at::cuda::detail;

namespace {

template <typename scalar_t, typename accscalar_t>
__device__ inline int64_t get_intervals(
  accscalar_t sample,
  int64_t index,
  int64_t inputSize,
  int64_t outputSize,
  int64_t poolSize) {
    accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
    if (index == outputSize - 1) {
      return inputSize - poolSize;
    } else {
      return static_cast<int64_t>((index + sample) * alpha) - \
        static_cast<int64_t>(sample * alpha);
    }
  }

template <typename scalar_t>
__global__ void fractional_max_pool3d_out_frame(
  PackedTensorAccessor64<scalar_t, 5> input,
  PackedTensorAccessor64<scalar_t, 5> output,
  PackedTensorAccessor64<int64_t, 5> indices,
  PackedTensorAccessor64<scalar_t, 3> samples,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {
    using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
    // Output (t, h, w) point that this thread is responsible for
    int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t plane = blockIdx.y;
    int64_t batch = blockIdx.z;
    // Each thread generates a specific output point
    if (ourOutputPoint < output.size(2) * output.size(3) *
      output.size(4)){
      int64_t outputT = ourOutputPoint / (output.size(3) *
                    output.size(4));
      int64_t outputH = (ourOutputPoint / output.size(4)) %
                    output.size(3);
      int64_t outputW = ourOutputPoint % output.size(4);

      int64_t poolT = get_intervals<scalar_t,accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][0]),
        outputT, input.size(2), output.size(2), poolSizeT);
      int64_t poolH = get_intervals<scalar_t, accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][1]),
        outputH, input.size(3), output.size(3), poolSizeH);
      int64_t poolW = get_intervals<scalar_t, accscalar_t>(
        static_cast<accscalar_t>(samples[batch][plane][2]),
        outputW, input.size(4), output.size(4), poolSizeW);

      scalar_t maxVal = at::numeric_limits<scalar_t>::lowest();
      int64_t maxIndex = -1;

      for(int64_t t = poolT; t < poolT + poolSizeT; ++ t) {
        for (int64_t h = poolH; h < poolH + poolSizeH; ++h) {
          if(poolSizeW < 2 || poolSizeW > 7) {
            for (int64_t w = poolW; w < poolW + poolSizeW; ++w) {
              scalar_t val = input[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal) {
                maxIndex = t * input.size(3) *
                  input.size(4) + h * input.size(4) + w;
                maxVal = val;
              }
            }
          } else {
            for (int64_t i = 0; i < poolSizeW; ++i) {
              int64_t w = i + poolW;
              scalar_t val = input[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal) {
                maxIndex = t * input.size(3) * input.size(4) +
                  h * input.size(4) + w;
                maxVal = val;
              }
            }
          }
        }
      }

      assert(maxVal != at::numeric_limits<scalar_t>::lowest());
      assert(maxIndex != -1);

      indices[batch][plane][outputT][outputH][outputW] = maxIndex;
      output[batch][plane][outputT][outputH][outputW] = maxVal;
    }
  }

template <typename scalar_t>
__global__ void fractional_max_pool3d_backward_out_frame(
  PackedTensorAccessor64<scalar_t, 5> gradInput,
  PackedTensorAccessor64<scalar_t, 5> gradOutput,
  PackedTensorAccessor64<int64_t, 5> indices) {
  // Output (h, w) point that this thread is responsible for
  int64_t ourOutputPoint = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t plane = blockIdx.y;
  int64_t batch = blockIdx.z;

  // Each thread generates a specific output point
  if (ourOutputPoint < gradOutput.size(2) *
    gradOutput.size(3) * gradOutput.size(4)) {
    int64_t outputW = ourOutputPoint % gradOutput.size(4);
    int64_t outputH = (ourOutputPoint / gradOutput.size(4)) %
                      gradOutput.size(3);
    int64_t outputT = ourOutputPoint / (gradOutput.size(3) *
                      gradOutput.size(4));

    int64_t index = indices[batch][plane][outputT][outputH][outputW];
    assert(index >= 0);
    int64_t inputW = index % gradInput.size(4);
    int64_t inputH = (index / gradInput.size(4)) %
      gradInput.size(3);
    int64_t inputT = index / (gradInput.size(3) *
      gradInput.size(4));
    assert(inputT < gradInput.size(2));

    gpuAtomicAdd(
      &gradInput[batch][plane][inputT][inputH][inputW],
      gradOutput[batch][plane][outputT][outputH][outputW]
      );
    }
  }

void fractional_max_pool3d_out_cuda_template(
  Tensor& output,
  Tensor& indices,
  const Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const Tensor& randomSamples) {
    int64_t planeDim = 0;
    int64_t dimt = 1;
    int64_t dimh = 2;
    int64_t dimw = 3;
    int64_t numBatch = 1;

    int64_t outputT = output_size[0];
    int64_t outputH = output_size[1];
    int64_t outputW = output_size[2];
    int64_t poolSizeT = pool_size[0];
    int64_t poolSizeH = pool_size[1];
    int64_t poolSizeW = pool_size[2];

    int64_t ndims = input.ndimension();
    TORCH_CHECK(
      input.numel() != 0 && (ndims == 4 || ndims == 5),
      "fractional_max_pool3d_out_cuda_template(): ",
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ",
      ndims);

    if (ndims == 5) {
      numBatch = input.size(0);
      planeDim++;
      dimt++;
      dimh++;
      dimw++;
    }

    /* sizes */
    int64_t numPlanes = input.size(planeDim);
    int64_t inputT = input.size(dimt);
    int64_t inputH = input.size(dimh);
    int64_t inputW = input.size(dimw);

    TORCH_CHECK(
      outputT + poolSizeT - 1 < inputT,
      "fractional_max_pool3d_out_cuda_template(): ",
      "pool time (", poolSizeT, ") too large relative to input time (",
      inputT, ")");
    TORCH_CHECK(
      outputH + poolSizeH - 1 < inputH,
      "fractional_max_pool3d_out_cuda_template(): ",
      "pool height (", poolSizeH, ") too large relative to input height (",
      inputH, ")");
    TORCH_CHECK(
      outputW + poolSizeW - 1 < inputW,
      "fractional_max_pool3d_out_cuda_template(): ",
      "pool width (", poolSizeW, ") too large relative to input width (",
      inputW, ")");

    if (ndims == 4) {
      /* resize output */
      output.resize_({numPlanes, outputT, outputH, outputW});
      /* indices will contain the locations for each output point */
      indices.resize_({numPlanes, outputT, outputH, outputW});
    } else {
      /* resize output */
      output.resize_({numBatch, numPlanes, outputT, outputH, outputW});
      /* indices will contain the locations for each output point */
      indices.resize_({numBatch, numPlanes, outputT, outputH, outputW});
    }

    auto output_ = output;
    auto indices_ = indices;
    auto input_ = input;
    if(ndims == 4) {
      output_ = output_.reshape({1, numPlanes, outputT, outputH, outputW});
      indices_ = indices_.reshape({1, numPlanes, outputT, outputH, outputW});
      input_ = input_.reshape({1, numPlanes, inputT, inputH, inputW});
    }

    // block is limited to 4 warps
    // grid handles overflow per each plane
    int64_t outputPlaneSize = output_.size(2) *
      output_.size(3) * output_.size(4);
    dim3 grid(
      (outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
      input_.size(1),
      input_.size(0));
    dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(),
      "fractional_max_pool3d_out_frame",
      [&]{
        fractional_max_pool3d_out_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          input_.packed_accessor64<scalar_t, 5>(),
          output_.packed_accessor64<scalar_t, 5>(),
          indices_.packed_accessor64<int64_t, 5>(),
          randomSamples.packed_accessor64<scalar_t, 3>(),
          poolSizeT, poolSizeH, poolSizeW
        );
      }
    );
    AT_CUDA_CHECK(cudaGetLastError()); 
  }

void fractional_max_pool3d_backward_out_cuda_template(
  Tensor& gradInput,
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef pool_size /* unused */,
  IntArrayRef output_size,
  const Tensor& indices) {
    int64_t dimt = 1;
    int64_t dimh = 2;
    int64_t dimw = 3;

    int64_t outputT = output_size[0];
    int64_t outputH = output_size[1];
    int64_t outputW = output_size[2];

    int64_t ndims = input.ndimension();
    if (ndims == 5) {
      dimt++;
      dimh++;
      dimw++;
    }

    /* sizes */
    int64_t inputT = input.size(dimt);
    int64_t inputH = input.size(dimh);
    int64_t inputW = input.size(dimw);

    TORCH_CHECK(
      outputT == gradOutput.size(dimt),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput time unexpected"
    );
    TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput height unexpected"
    );
    TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool3d_backward_out_cuda_template(): ",
      "gradOutput width unexpected"
    );

    /* resize */
    gradInput.resize_as_(input);
    gradInput.zero_();

    auto gradInput_ = gradInput;
    auto gradOutput_ = gradOutput;
    auto indices_ = indices;

    if(ndims == 4) {
      gradInput_ = gradInput_.reshape({1, gradInput.size(0), inputT,
                                       inputH, inputW});
      gradOutput_ = gradOutput_.reshape({1, gradOutput.size(0), outputT,
                                         outputH, outputW});
      indices_ = indices_.reshape({1, indices.size(0), outputT, outputH,
                                   outputW});
    }

    /* backprop */
    // block is limited to 4 warps
    // grid handles overflow per each plane
    int64_t outputPlaneSize = gradOutput_.size(2) *
      gradOutput_.size(3) * gradOutput_.size(4);
    dim3 grid(
      (outputPlaneSize + 127) / 128, // ceil(outputPlaneSize / 128)
      gradInput_.size(1),
      gradInput_.size(0));
    dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      gradOutput.scalar_type(),
      "fractional_max_pool3d_backward_out_frame",
      [&] {
        fractional_max_pool3d_backward_out_frame<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          gradInput_.packed_accessor64<scalar_t, 5>(),
          gradOutput_.packed_accessor64<scalar_t, 5>(),
          indices_.packed_accessor64<int64_t, 5>()
        );
      }
    );
    AT_CUDA_CHECK(cudaGetLastError()); 
  }

}// namespace

std::tuple<Tensor&, Tensor&> fractional_max_pool3d_out_cuda(
   at::Tensor& output,
   at::Tensor& indices,
   const at::Tensor& input,
   IntArrayRef pool_size,
   IntArrayRef output_size,
   const at::Tensor& randomSamples) {
   fractional_max_pool3d_out_cuda_template(
     output,
     indices,
     input,
     pool_size,
     output_size,
     randomSamples
   );
   return std::tuple<Tensor&, Tensor&>(output, indices);
 }

std::tuple<Tensor, Tensor> fractional_max_pool3d_cuda(
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples) {
    Tensor output = at::empty({0}, input.options());
    Tensor indices = at::empty({0}, input.options().dtype(kLong));
    fractional_max_pool3d_out_cuda_template(
      output,
      indices,
      input,
      pool_size,
      output_size,
      randomSamples
    );
    return std::tuple<Tensor, Tensor>(output, indices);
  }

Tensor& fractional_max_pool3d_backward_out_cuda(
  at::Tensor& gradInput,
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices) {
    fractional_max_pool3d_backward_out_cuda_template(
      gradInput,
      gradOutput_,
      input,
      pool_size,
      output_size,
      indices
    );
    return gradInput;
  }

Tensor fractional_max_pool3d_backward_cuda(
  const at::Tensor& gradOutput,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices) {
    Tensor gradInput = at::empty({0}, input.options());
    fractional_max_pool3d_backward_out_cuda_template(
      gradInput,
      gradOutput,
      input,
      pool_size,
      output_size,
      indices
    );
    return gradInput;
 }

}// native
}// at
