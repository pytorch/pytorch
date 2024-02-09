#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/TensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#endif

#include <ATen/cuda/cub.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/EmbeddingBackwardKernel.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/macros/Macros.h>

#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <thrust/iterator/reverse_iterator.h>
#endif

namespace at::native {

#if !CUB_SUPPORTS_SCAN_BY_KEY()
template<typename index_t>
void embedding_dense_backward_cuda_scan(Tensor &sorted_indices, Tensor &count);
#endif

namespace {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

// This kernel assumes that all input tensors except `weight` and
// per_sample_weights are contiguous.
template <typename scalar_t, typename index_t>
__global__ void EmbeddingBag_updateOutputKernel_max(
    const index_t *input, const index_t *offsets, const scalar_t *weight, scalar_t *output,
    index_t *offset2bag, int64_t numIndices, int64_t numBags,
    int64_t featureSize, int64_t weight_stride0, int64_t weight_stride1,
    index_t *bag_size, index_t *max_indices,
    index_t padding_idx, int64_t numRows) {

  // the strategy here is that each bag x feature is handled by a single thread

  int64_t chunksPerBag = ceil_div(featureSize, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < featureSize) {
      int64_t bag = chunk / chunksPerBag;
      const scalar_t *weightFeat = weight + featureDim * weight_stride1;
      int64_t begin = bag == 0 ? 0 : offsets[bag]; // forces first offset to be 0 instead of asserting on it
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
      CUDA_KERNEL_ASSERT(end >= begin);
      scalar_t weightFeatMax = 0;
      int64_t bag_size_ = 0;
      int64_t maxWord = -1;
      for (int64_t emb = begin; emb < end; emb++) {
        bool pad = (input[emb] == padding_idx);
        CUDA_KERNEL_ASSERT(input[emb] < numRows);
        const int64_t weightRow = input[emb] * weight_stride0;
        scalar_t weightValue = weightFeat[weightRow];
        if (bag_size_ == 0 || weightValue > weightFeatMax) {
          weightFeatMax = pad ? weightFeatMax : weightValue;
          maxWord = pad ? maxWord : input[emb];
        }
        bag_size_ += pad ? 0 : 1;

        if (featureDim == 0) {
          offset2bag[emb] = bag;
        }
      }
      bag_size[bag] = bag_size_;
      max_indices[bag * featureSize + featureDim] = maxWord;
      output[bag * featureSize + featureDim] = weightFeatMax;
    }
  }
}

// This kernel assumes that all input tensors except `weight` and
// per_sample_weights are contiguous.
template <typename scalar_t, typename index_t>
__global__ void EmbeddingBag_updateOutputKernel_sum_mean(
    const index_t *input, const index_t *offsets, const scalar_t *weight, scalar_t *output,
    index_t *offset2bag, int64_t numIndices, int64_t numBags,
    int64_t featureSize, int64_t weight_stride0, int64_t weight_stride1,
    int mode, index_t *bag_size,
    const scalar_t* per_sample_weights, int64_t per_sample_weights_stride,
    index_t padding_idx, int64_t numRows) {

  // the strategy here is that each bag x feature is handled by a single thread

  using accscalar_t = acc_type<scalar_t, true>;
  int64_t chunksPerBag = ceil_div(featureSize, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < featureSize) {
      int64_t bag = chunk / chunksPerBag;
      const scalar_t *weightFeat = weight + featureDim * weight_stride1;
      int64_t begin = bag == 0 ? 0 : offsets[bag]; // forces first offset to be 0 instead of asserting on it
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
      CUDA_KERNEL_ASSERT(end >= begin);
      accscalar_t weightFeatSum = 0;
      int64_t bag_size_ = 0;
      for (int64_t emb = begin; emb < end; emb++) {
        bool pad = (input[emb] == padding_idx);
        CUDA_KERNEL_ASSERT(input[emb] < numRows);
        const int64_t weightRow = input[emb] * weight_stride0;
        scalar_t weightValue = weightFeat[weightRow];
        weightValue = pad ? static_cast<scalar_t>(0) : weightValue;
        if (per_sample_weights) {
          accscalar_t scaleWeightBy = static_cast<accscalar_t>(
              per_sample_weights[emb * per_sample_weights_stride]);
          weightFeatSum += scaleWeightBy * static_cast<accscalar_t>(weightValue);
        } else {
          weightFeatSum += static_cast<accscalar_t>(weightValue);
        }
        bag_size_ += pad ? 0 : 1;

        if (featureDim == 0) {
          offset2bag[emb] = bag;
        }
      }
      if (mode == MODE_MEAN) {
        if (bag_size_ != 0) {
          weightFeatSum = weightFeatSum / static_cast<accscalar_t>(bag_size_);
        }
      }
      bag_size[bag] = bag_size_;
      output[bag * featureSize + featureDim] = static_cast<scalar_t>(weightFeatSum);
    }
  }
}

Tensor embedding_bag_backward_cuda_sum_avg(
                                   const Tensor &grad,
                                   const Tensor &indices_,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode,
                                   const Tensor& per_sample_weights,
                                   int64_t padding_idx) {
  auto indices = indices_.contiguous();

  ptrdiff_t num_indices = indices.numel();

  if (num_indices == 0) {
    // all empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  auto sorted_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor count;

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_backward_cuda_sum_avg", [&] () {
    auto range = at::arange(num_indices, indices.options());
    // int64_t nbits = cuda::cub::get_num_bits(num_weights);
    cuda::cub::radix_sort_pairs(
      indices.const_data_ptr<index_t>(), sorted_indices.mutable_data_ptr<index_t>(),
      range.const_data_ptr<index_t>(), orig_indices.mutable_data_ptr<index_t>(),
      num_indices, false/*, 0, nbits*/);
  });

  if (scale_grad_by_freq) {
    count = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_backward_cuda_sum_avg", [&] () {
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      // Compute an increasing sequence per unique item in sortedIndices:
      // sorted: 2 5 5 5 7 7 8 9 9
      //  count: 1 1 2 3 1 2 1 1 2
      auto sorted_data = sorted_indices.const_data_ptr<index_t>();
      auto count_data = count.mutable_data_ptr<index_t>();
      cuda::cub::inclusive_sum_by_key(
        sorted_data,
        at_cuda_detail::cub::ConstantInputIterator<index_t>(1),
        count_data,
        num_indices
      );

      // Take the maximum of each count per unique key in reverse:
      // sorted: 2 5 5 5 7 7 8 9 9
      //  count: 1 3 3 3 2 2 1 2 2
      cuda::cub::inclusive_scan_by_key(
        thrust::make_reverse_iterator(sorted_data + num_indices),
        thrust::make_reverse_iterator(count_data + num_indices),
        thrust::make_reverse_iterator(count_data + num_indices),
        at_cuda_detail::cub::Max(),
        num_indices
      );
    });
#else
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_backward_cuda_sum_avg", [&] () {
      embedding_dense_backward_cuda_scan<index_t>(sorted_indices, count);
    });
#endif
  }
  return embedding_backward_cuda_kernel(grad, orig_indices, sorted_indices,
      count, num_weights, padding_idx, mode == MODE_MEAN, offset2bag,
      bag_size, per_sample_weights);
}

template <typename scalar_t, typename index_t>
__global__ void EmbeddingBag_accGradParametersKernel_max(
    const index_t *max_indices, const scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t stride, int64_t numBags,
    index_t padding_idx, const index_t numel) {

  using accscalar_t = acc_type<scalar_t, true>;

  int64_t chunksPerBag = ceil_div(stride, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;

      index_t word_idx = max_indices[bag * stride + featureDim];
      if (word_idx >= 0 && word_idx != padding_idx) {
        // If bag is empty, we have max_indices[idx] set to -1 in forward.
        fastAtomicAdd(
            gradWeight, static_cast<index_t>(word_idx * stride + featureDim),
            numel, gradOutput[bag * stride + featureDim], true);
      }
    }
  }
}

Tensor embedding_bag_backward_cuda_max(const Tensor &grad,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   int64_t padding_idx) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("embedding_bag_backward_cuda_max");

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#if defined(USE_ROCM)
  dim3 block = dim3(64, 4);
#else
  dim3 block = dim3(32, 8);
#endif
  int grid = 1024;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "embedding_bag_backward_cuda_max", [&] {
        AT_DISPATCH_INDEX_TYPES(max_indices.scalar_type(), "embedding_bag_backward_cuda_max", [&] () {
          EmbeddingBag_accGradParametersKernel_max<
              scalar_t, index_t><<<grid, block, 0, stream>>>(
              max_indices.const_data_ptr<index_t>(), grad.const_data_ptr<scalar_t>(),
              grad_weight.mutable_data_ptr<scalar_t>(), stride, numBags,
              padding_idx, grad_weight.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  });

  return grad_weight;
}
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_forward_only_cuda(const Tensor &weight, const Tensor &indices,
                   const Tensor &offsets, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
                   bool include_last_offset, int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  return _embedding_bag_cuda(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cuda(const Tensor &weight, const Tensor &indices_,
                   const Tensor &offsets_, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt,
                   bool include_last_offset, int64_t padding_idx) {
  TORCH_CHECK(indices_.dim() == 1 || indices_.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices_.dim());
  if (indices_.dim() == 1) {
    TORCH_CHECK(offsets_.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets_.dim());
  }
  TORCH_CHECK(weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_cuda", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_cuda", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_cuda", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkSameGPU("embedding_bag_cuda", weight_arg, indices_arg);
  checkSameGPU("embedding_bag_cuda", weight_arg, offsets_arg);

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);
  if (include_last_offset) {
    // Check https://github.com/pytorch/pytorch/issues/29019
    // We plan to add one more element in offsets, which is equal to the size of
    // indices. Currently for cuda devices, we still use the legacy
    // implementation even this flag is enabled.
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t featureSize = weight.size(1);

  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag =
      at::empty({indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto output = at::empty({numBags, featureSize}, weight.options());

  Tensor max_indices;

  if (mode == MODE_MAX) {
    max_indices = at::empty({numBags, featureSize}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0}, indices.options());
  }

#if defined(USE_ROCM)
  dim3 block = dim3(64, 4);
#else
  dim3 block = dim3(32, 8);
#endif
  int grid = 1024;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "embedding_bag_cuda", [&] {
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cuda", [&] () {
      if (mode == MODE_MAX) {
        EmbeddingBag_updateOutputKernel_max<scalar_t, index_t><<<grid, block, 0, stream>>>(
            indices.const_data_ptr<index_t>(), offsets.const_data_ptr<index_t>(),
            weight.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),
            offset2bag.mutable_data_ptr<index_t>(), numIndices, numBags, featureSize,
            weight.stride(0), weight.stride(1), bag_size.mutable_data_ptr<index_t>(),
            max_indices.mutable_data_ptr<index_t>(),
            padding_idx, weight.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        EmbeddingBag_updateOutputKernel_sum_mean<scalar_t, index_t><<<grid, block, 0, stream>>>(
            indices.const_data_ptr<index_t>(), offsets.const_data_ptr<index_t>(),
            weight.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),
            offset2bag.mutable_data_ptr<index_t>(), numIndices, numBags, featureSize,
            weight.stride(0), weight.stride(1), mode, bag_size.mutable_data_ptr<index_t>(),
            per_sample_weights.defined() ? per_sample_weights.const_data_ptr<scalar_t>() : NULL,
            per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,
            padding_idx, weight.size(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_cuda(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offset2bag,
                                   const Tensor &bag_size_,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   bool scale_grad_by_freq, int64_t mode, const c10::optional<Tensor>& per_sample_weights_opt,
                                   int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward in
  // EmbeddingBag.cpp.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_.contiguous();
  auto indices_arg = TensorArg(indices, "indices", 1);
  auto grad_arg = TensorArg(grad, "grad", 1);
  checkSameGPU("embedding_bag_cuda", grad_arg, indices_arg);


  switch (mode) {
    case MODE_SUM:
    case MODE_MEAN:
      if (mode == MODE_MEAN)
        AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_cuda_sum_avg(grad, indices, offset2bag,
              bag_size_, num_weights, scale_grad_by_freq, mode,
              per_sample_weights, padding_idx);

    case MODE_MAX:
      AT_ASSERT(!per_sample_weights.defined());
      return embedding_bag_backward_cuda_max(grad, max_indices, num_weights,
              padding_idx);

    default:
      AT_ERROR(
          "Unknown mode for embedding_bag_backward_cuda ", mode);
  }
}

template <typename scalar_t, typename index_t>
__global__ static void _embedding_bag_per_sample_weights_backward_kernel(
    const scalar_t* grad, int64_t grad_stride0, int64_t grad_stride1,
    const scalar_t* weight, int64_t weight_stride0, int64_t weight_stride1,
    const index_t* indices,  // contiguous
    const index_t* offset2bag,  // contiguous
    int64_t num_samples,
    int64_t embedding_features,
    scalar_t* output,
    index_t padding_idx) {
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
    if (embedding_idx != padding_idx) {
      for (int feature_idx = thread_in_warp; feature_idx < embedding_features;
          feature_idx += C10_WARP_SIZE) {
        result +=
            grad[grad_stride0 * bag_idx + grad_stride1 * feature_idx] *
            weight[weight_stride0 * embedding_idx + weight_stride1 * feature_idx];
      }
    }
    result = cuda_utils::WarpReduceSum<accscalar_t>(result);
    if (thread_in_warp == 0) {
      output[sample_idx] = result;
    }
  }
}

Tensor _embedding_bag_per_sample_weights_backward_cuda(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  const int threads_per_block = 512;
  const int warps_per_block = threads_per_block / at::cuda::warp_size();

  dim3 block(threads_per_block);
  dim3 grid((num_samples + warps_per_block - 1) / warps_per_block);

  auto output = at::empty({num_samples}, grad.options());

  // Early return when there is no samples in the batch. This saves unnecesary kernel
  // launch, but also prevents cudaGetLastError() to complain about invalid launch args
  if (num_samples == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cuda", [&]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cuda", [&]() {
        _embedding_bag_per_sample_weights_backward_kernel<scalar_t, index_t>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad.const_data_ptr<scalar_t>(), grad.stride(0), grad.stride(1),
            weight.const_data_ptr<scalar_t>(), weight.stride(0), weight.stride(1),
            indices.const_data_ptr<index_t>(),
            offset2bag.const_data_ptr<index_t>(),
            num_samples,
            embedding_features,
            output.mutable_data_ptr<scalar_t>(),
            padding_idx);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
    }
  );
  return output;
}

} // namespace at::native
