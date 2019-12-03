#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/AccumulateType.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCAtomics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>

#include <ATen/native/cuda/EmbeddingBackwardKernel.cuh>

#include <c10/macros/Macros.h>

namespace at {
namespace native {

namespace {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

// This kernel assumes that all input tensors except `weight` and
// per_sample_weights are contiguous.
template <typename scalar_t>
__global__ void EmbeddingBag_updateOutputKernel(
    int64_t *input, int64_t *offsets, scalar_t *weight, scalar_t *output,
    int64_t *offset2bag, int64_t numIndices, int64_t numBags,
    int64_t featureSize, int64_t weight_stide0, int64_t weight_stride1,
    int mode, int64_t *bag_size, int64_t *max_indices,
    scalar_t* per_sample_weights, int64_t per_sample_weights_stride) {

  // the strategy here is that each bag x feature is handled by a single thread

  using accscalar_t = acc_type<scalar_t, true>;
  int64_t chunksPerBag = THCCeilDiv(featureSize, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < featureSize) {
      int64_t bag = chunk / chunksPerBag;
      scalar_t *weightFeat = weight + featureDim * weight_stride1;
      int64_t begin = offsets[bag];
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
      assert(end >= begin);

      accscalar_t weightFeatSum = 0;
      scalar_t weightFeatMax;

      int64_t bag_size_ = 0;
      int64_t maxWord = -1;
      for (int64_t emb = begin; emb < end; emb++) {
        const int64_t weightRow = input[emb] * weight_stide0;
        scalar_t weightValue = weightFeat[weightRow];

        if (mode == MODE_MAX) {
          if (emb == begin || weightValue > weightFeatMax) {
            weightFeatMax = weightValue;
            maxWord = input[emb];
          }
        } else {
          if (per_sample_weights) {
            accscalar_t scaleWeightBy = static_cast<accscalar_t>(
                per_sample_weights[emb * per_sample_weights_stride]);
            weightFeatSum += scaleWeightBy * static_cast<accscalar_t>(weightValue);
          } else {
            weightFeatSum += static_cast<accscalar_t>(weightValue);
          }
        }

        bag_size_++;
        if (featureDim == 0) {
          offset2bag[emb] = bag;
        }
      }
      if (mode == MODE_MEAN) {
        if (end == begin) {
          bag_size[bag] = 0;
        } else {
          weightFeatSum = weightFeatSum / static_cast<accscalar_t>(bag_size_);
          bag_size[bag] = bag_size_;
        }
      }

      if (mode == MODE_MEAN || mode == MODE_SUM) {
        output[bag * featureSize + featureDim] = static_cast<scalar_t>(weightFeatSum);
      }
      else if (mode == MODE_MAX) {
        if (end == begin) {
          // If bag is empty, set output to 0.
          weightFeatMax = 0;
        }
        max_indices[bag * featureSize + featureDim] = maxWord;
        output[bag * featureSize + featureDim] = weightFeatMax;
      }
    }
  }
}



Tensor embedding_bag_backward_cuda_sum_avg(
                                   const Tensor &grad,
                                   const Tensor &indices,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode,
                                   const Tensor& per_sample_weights) {

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ptrdiff_t numel = indices.numel();

  if (numel == 0) {
    // all empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  int64_t stride = grad_weight.stride(0);

  auto sorted_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  using device_ptr = thrust::device_ptr<int64_t>;

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    sorted_indices.copy_(indices);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential indices
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_indices.data_ptr<int64_t>());
    thrust::copy(policy, count_iter, count_iter + numel, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_indices.data_ptr<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + numel, orig_data,
                        ThrustLTOp<int64_t>());
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Compute an increasing sequence per unique item in sortedIndices:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    auto sorted_data = device_ptr(sorted_indices.data_ptr<int64_t>());
    auto count_data = device_ptr(count.data_ptr<int64_t>());
    thrust::inclusive_scan_by_key(policy, sorted_data, sorted_data + numel,
                                  thrust::make_constant_iterator(1),
                                  count_data);

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
        policy, thrust::make_reverse_iterator(sorted_data + numel),
        thrust::make_reverse_iterator(sorted_data),
        thrust::make_reverse_iterator(count_data + numel),
        thrust::make_reverse_iterator(count_data + numel),
        thrust::equal_to<int64_t>(), thrust::maximum<int64_t>());
  }
  return embedding_backward_cuda_kernel(grad, orig_indices, sorted_indices,
      count, num_weights, /* padding_idx= */ -1, scale_grad_by_freq,
      mode == MODE_MEAN, offset2bag, bag_size, per_sample_weights);
}

template <typename scalar_t>
__global__ void EmbeddingBag_accGradParametersKernel_max(
    int64_t *max_indices, scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t stride, int64_t numBags) {

  using accscalar_t = acc_type<scalar_t, true>;

  int64_t chunksPerBag = THCCeilDiv(stride, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;

      int64_t word_idx = max_indices[bag * stride + featureDim];
      if (word_idx >= 0) {
        // If bag is empty, we have max_indices[idx] set to -1 in forward.
        atomicAdd(&(gradWeight[word_idx * stride + featureDim]),
                gradOutput[bag * stride + featureDim]);
      }
    }
  }
}

Tensor embedding_bag_backward_cuda_max(const Tensor &grad,
                                   const Tensor &max_indices,
                                   int64_t num_weights) {

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#ifdef __HIP_PLATFORM_HCC__
  dim3 block = dim3(64, 4);
#else
  dim3 block = dim3(32, 8);
#endif
  int grid = 1024;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "embedding_bag_backward_cuda_max", [&] {
        EmbeddingBag_accGradParametersKernel_max<
            scalar_t><<<grid, block, 0, stream>>>(
            max_indices.data_ptr<int64_t>(), grad.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(), stride, numBags);
      });

  THCudaCheck(cudaGetLastError());
  return grad_weight;
}
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cuda(const Tensor &weight, const Tensor &indices,
                   const Tensor &offsets, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse,
                   const Tensor& per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag_cuda", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag_cuda", offsets_arg, kLong);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkSameGPU("embedding_bag_cuda", weight_arg, indices_arg);
  checkSameGPU("embedding_bag_cuda", weight_arg, offsets_arg);

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);
  int64_t featureSize = weight.size(1);

  auto bag_size = at::zeros(offsets.sizes(), indices.options());
  auto offset2bag =
      at::zeros({indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto output = at::zeros({offsets.size(0), weight.size(1)}, weight.options());

  Tensor max_indices;

  if (mode == MODE_MAX) {
    max_indices = at::zeros({offsets.size(0), weight.size(1)}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::zeros({0}, indices.options());
  }

#ifdef __HIP_PLATFORM_HCC__
  dim3 block = dim3(64, 4);
#else
  dim3 block = dim3(32, 8);
#endif
  int grid = 1024;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(weight.scalar_type(), "embedding_bag_cuda", [&] {
    EmbeddingBag_updateOutputKernel<scalar_t><<<grid, block, 0, stream>>>(
        indices.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        offset2bag.data_ptr<int64_t>(), numIndices, numBags, featureSize,
        weight.stride(0), weight.stride(1), mode, bag_size.data_ptr<int64_t>(),
        mode == MODE_MAX ? max_indices.data_ptr<int64_t>() : NULL,
        per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() : NULL,
        per_sample_weights.defined() ? per_sample_weights.stride(0) : 0);
  });

  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_cuda(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offsets,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size_,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode,
                                   const Tensor& per_sample_weights) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward in
  // EmbeddingBag.cpp.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_.contiguous();
  auto indices_arg = TensorArg(indices, "indices", 1);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  auto grad_arg = TensorArg(grad, "grad", 1);
  checkSameGPU("embedding_bag_cuda", grad_arg, offsets_arg);
  checkSameGPU("embedding_bag_cuda", grad_arg, indices_arg);


  switch (mode) {
    case MODE_SUM:
    case MODE_MEAN:
      if (mode == MODE_MEAN)
        AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_cuda_sum_avg(grad, indices, offset2bag,
              bag_size_, num_weights, scale_grad_by_freq, mode, per_sample_weights);

    case MODE_MAX:
      AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_cuda_max(grad, max_indices, num_weights);

    default:
      AT_ERROR(
          "Unknown mode for embedding_bag_backward_cuda ", mode);
  }
}

template <typename scalar_t>
__inline__ __device__
static scalar_t warpReduceSum(scalar_t val) {
  for (int offset = C10_WARP_SIZE/2; offset > 0; offset /= 2)
    val += WARP_SHFL_DOWN(val, offset);
  return val;
}

template <typename scalar_t>
__global__ static void _embedding_bag_per_sample_weights_backward_kernel(
    const scalar_t* grad, int64_t grad_stride0, int64_t grad_stride1,
    const scalar_t* weight, int64_t weight_stride0, int64_t weight_stride1,
    const int64_t* indices,  // contiguous
    const int64_t* offset2bag,  // contiguous
    int64_t num_samples,
    int64_t embedding_features,
    scalar_t* output) {
  using accscalar_t = acc_type<scalar_t, true>;
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int warp = idx / C10_WARP_SIZE;
  const int thread_in_warp = idx % C10_WARP_SIZE;
  const int num_warps = blockDim.x * gridDim.x / C10_WARP_SIZE;

  // Each warp is responsible for the accumulation of one sample.
  // This involves doing one dot product between grad[bag_idx] and weight[embedding_idx].
  for (int sample_idx = warp; sample_idx < num_samples; sample_idx += num_warps) {
    accscalar_t result = 0.;
    const int bag_idx = (int)offset2bag[sample_idx];
    const int embedding_idx = (int)indices[sample_idx];
    for (int feature_idx = thread_in_warp; feature_idx < embedding_features;
        feature_idx += C10_WARP_SIZE) {
      result +=
          grad[grad_stride0 * bag_idx + grad_stride1 * feature_idx] *
          weight[weight_stride0 * embedding_idx + weight_stride1 * feature_idx];
    }
    result = warpReduceSum<accscalar_t>(result);
    if (thread_in_warp == 0) {
      output[sample_idx] = result;
    }
  }
}

Tensor _embedding_bag_per_sample_weights_backward_cuda(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  const int threads_per_block = 1024;
  const int warps_per_block = threads_per_block / C10_WARP_SIZE;

  dim3 block(threads_per_block);
  dim3 grid((num_samples + warps_per_block - 1) / warps_per_block);

  auto output = at::empty({num_samples}, grad.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cuda", [&]() {
      _embedding_bag_per_sample_weights_backward_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad.data_ptr<scalar_t>(), grad.stride(0), grad.stride(1),
          weight.data_ptr<scalar_t>(), weight.stride(0), weight.stride(1),
          indices.data_ptr<int64_t>(),
          offset2bag.data_ptr<int64_t>(),
          num_samples,
          embedding_features,
          output.data_ptr<scalar_t>());
    }
  );
  return output;
}

}
}
