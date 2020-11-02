#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/macros/Macros.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>

#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at {
namespace native {

namespace {

inline C10_HOST_DEVICE int32_t round_up(int32_t a, int32_t b) {
  return ((a + b - 1) / b) * b;
}

inline C10_HOST_DEVICE int32_t div_round_up(int32_t a, int32_t b) {
  return ((a + b - 1) / b);
}

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

template <typename scalar_t, int mode, int32_t ILP>
__global__ void EmbeddingBag_updateOutputKernel(
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits>
        per_sample_weights,
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> output,
    PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> offset2bag,
    PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> bag_size,
    PackedTensorAccessor64<int64_t, 2, RestrictPtrTraits> max_indices) {
  const int32_t B = output.size(0);
  const int32_t D = output.size(1);

  // warp per output row.
  const int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  int32_t indices_start =
      offsets[b]; // forces first offset to be 0 instead of asserting on it
  // TODO: this could leverage include_last_offset support.
  int32_t indices_end = (b < B - 1) ? (offsets[b + 1]) : indices.size(0);
  bool weighted = per_sample_weights.size(0) != 0;
  int32_t L = indices_end - indices_start;
  bag_size[b] = L;

  using VecT = memory::aligned_vector<scalar_t, ILP>;
  for (int32_t d = threadIdx.x * ILP; d < round_up(D, C10_WARP_SIZE * ILP);
       d += C10_WARP_SIZE * ILP) {
    acc_type<scalar_t, true> acc[ILP];
    int64_t max_idx[ILP];
    scalar_t max_weight[ILP];
    for (auto ii = 0; ii < ILP; ++ii) {
      acc[ii] = 0.0;
      max_idx[ii] = -1;
      max_weight[ii] = std::numeric_limits<scalar_t>::lowest();
    }
    for (int32_t l_start = 0; l_start < L; l_start += C10_WARP_SIZE) {
      int32_t l = l_start + threadIdx.x;
      const bool valid = l < L;
      int64_t idx = valid ? indices[indices_start + l] : 0;
      acc_type<scalar_t, true> idx_weight = (valid && weighted)
          ? per_sample_weights[indices_start + l]
          : static_cast<scalar_t>(1.0);
      if (valid) {
        offset2bag[indices_start + l] = b;
      }
      for (auto j = 0; j < C10_WARP_SIZE && l_start + j < L; ++j) {
        int64_t idx_j = WARP_SHFL(idx, j);
        acc_type<scalar_t, true> idx_weight_j = WARP_SHFL(idx_weight, j);
        if (d < D) {
          VecT w = *reinterpret_cast<const VecT*>(&weights[idx_j][d]);
          if (mode == MODE_MEAN || mode == MODE_SUM) {
            for (auto ii = 0; ii < ILP; ++ii) {
              acc[ii] += idx_weight_j * w.val[ii];
            }
          } else if (mode == MODE_MAX) {
            for (auto ii = 0; ii < ILP; ++ii) {
              if (w.val[ii] >= max_weight[ii]) {
                max_weight[ii] = w.val[ii];
                max_idx[ii] = idx_j;
              }
            }
          }
        }
      }
    }
    if (d < D) {
      if (mode == MODE_MEAN || mode == MODE_SUM) {
        VecT result;
        for (auto ii = 0; ii < ILP; ++ii) {
          result.val[ii] = (mode == MODE_MEAN && L > 0) ? acc[ii] / L : acc[ii];
        }
        *reinterpret_cast<VecT*>(&output[b][d]) = result;
      } else if (mode == MODE_MAX) {
        VecT result;
        for (auto ii = 0; ii < ILP; ++ii) {
          result.val[ii] = L > 0 ? max_weight[ii] : static_cast<scalar_t>(0.0);
        }
        *reinterpret_cast<VecT*>(&output[b][d]) = result;
        for (auto ii = 0; ii < ILP; ++ii) {
          max_indices[b][d + ii] = max_idx[ii] * D + d + ii;
        }
      }
    }
  }
}

template <typename scalar_t, int32_t ILP>
__global__ void EmbeddingBag_accGradParametersKernel_sum_mean(
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> sorted_indices,
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits>
        sorted_offset2bag,
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits>
        sorted_per_sample_weights,
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> bag_size,
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> grad,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> grad_weight,
    bool scale_grad_by_freq,
    int64_t mode) {
  int32_t B = grad.size(0);
  int32_t D = grad.size(1);
  int32_t id = threadIdx.y + blockIdx.x * blockDim.y;
  if (id >= sorted_indices.size(0)) {
    return;
  }

  const bool segment_start =
      id == 0 || (sorted_indices[id - 1] != sorted_indices[id]);
  if (!segment_start) {
    return;
  }
  const int64_t idx = sorted_indices[id];

  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 0;
  while (true) {
    int segment_continue = 0;
    if (id + SL + threadIdx.x < sorted_indices.size(0)) {
      segment_continue = sorted_indices[id + SL + threadIdx.x] == idx;
    }
    int32_t SL_incr = __popc(__ballot_sync(0xFFFFFFFF, segment_continue));
    SL += SL_incr;
    if (SL_incr != C10_WARP_SIZE) {
      break;
    }
  }
  bool weighted = sorted_per_sample_weights.size(0) != 0;
  // So our warp is responsible for accumulating gradients from
  // sorted_indices[id:id + SL].
  // We can extend this to CTA-per-segment, but not really necessary yet.
  using VecT = memory::aligned_vector<scalar_t, ILP>;

  for (int32_t d = threadIdx.x * ILP; d < round_up(D, ILP * C10_WARP_SIZE);
       d += ILP * C10_WARP_SIZE) {
    acc_type<scalar_t, true> acc[ILP];
#pragma unroll
    for (auto ii = 0; ii < ILP; ++ii) {
      acc[ii] = 0.0;
    }

    for (int32_t sl_start = 0; sl_start < SL; sl_start += C10_WARP_SIZE) {
      int32_t sl = sl_start + threadIdx.x;
      const bool valid = sl < SL;
      int64_t b = valid ? sorted_offset2bag[id + sl] : 0;
      acc_type<scalar_t, true> idx_weight = (valid && weighted)
          ? sorted_per_sample_weights[id + sl]
          : static_cast<scalar_t>(1.0);
      acc_type<scalar_t, true> inv_L = (mode == MODE_MEAN)
          ? (1.0 / static_cast<acc_type<scalar_t, true>>(bag_size[b]))
          : 1.0;
      idx_weight *= inv_L;
      for (auto j = 0; j < C10_WARP_SIZE && sl_start + j < SL; ++j) {
        int64_t b_j = WARP_SHFL(b, j);
        acc_type<scalar_t, true> idx_weight_j = WARP_SHFL(idx_weight, j);
        if (d < D) {
          VecT gw = *reinterpret_cast<const VecT*>(&grad[b_j][d]);

#pragma unroll
          for (auto ii = 0; ii < ILP; ++ii) {
            acc[ii] += idx_weight_j * gw.val[ii];
          }
        }
      }
    }
    if (d < D) {
      VecT result;
#pragma unroll
      for (auto ii = 0; ii < ILP; ++ii) {
        result.val[ii] = scale_grad_by_freq ? acc[ii] / SL : acc[ii];
      }
      *reinterpret_cast<VecT*>(&grad_weight[idx][d]) = result;
    }
  }
}

Tensor embedding_bag_backward_cuda_sum_avg(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights) {
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  if (indices.numel() == 0) {
    // all empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  Tensor sorted_indices;
  Tensor sorted_indices_idx;
  std::tie(sorted_indices, sorted_indices_idx) = indices.sort();
  auto sorted_offset2bag = offset2bag.index_select(0, sorted_indices_idx);
  auto sorted_per_sample_weights = per_sample_weights.defined()
      ? per_sample_weights.index_select(0, sorted_indices_idx)
      : at::empty({0}, grad.options());
  dim3 threads(C10_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE);
  dim3 blocks(div_round_up(
      indices.numel(), CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_backward_cuda_compute_grad_weight",
      [&] {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(
            scalar_t, "embedding_bag_backward_cuda_compute_grad_weight", [&] {

#define X(ILP)                                                                 \
  if (grad.size(1) % ILP == 0 &&                                               \
      memory::can_vectorize_up_to<scalar_t>((char*)grad_weight.data_ptr()) >=  \
          ILP &&                                                               \
      memory::can_vectorize_up_to<scalar_t>((char*)grad.data_ptr()) >= ILP) {  \
    EmbeddingBag_accGradParametersKernel_sum_mean<scalar_t, ILP>               \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(            \
            sorted_indices.packed_accessor64<int64_t, 1, RestrictPtrTraits>(), \
            sorted_offset2bag                                                  \
                .packed_accessor64<int64_t, 1, RestrictPtrTraits>(),           \
            sorted_per_sample_weights                                          \
                .packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),          \
            bag_size.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),       \
            grad.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),          \
            grad_weight.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),   \
            scale_grad_by_freq,                                                \
            mode);                                                             \
    AT_CUDA_CHECK(cudaGetLastError());                                         \
    return;                                                                    \
  }
              X(4);
              X(2);
              X(1);
            });
      });

#undef X
  return grad_weight;
}

Tensor embedding_bag_backward_cuda_max_deterministic(
    const Tensor& grad,
    const Tensor& max_indices,
    int64_t num_weights) {
  Tensor sorted_flat_max_indices;
  Tensor sorted_flat_max_indices_ids;
  auto max_indices_flat = max_indices.flatten();
  auto grad_flat = grad.flatten();

  std::tie(sorted_flat_max_indices, sorted_flat_max_indices_ids) =
      max_indices_flat.sort();
  auto sorted_grads = grad_flat.index_select(0, sorted_flat_max_indices_ids);

  auto reduced_flat_grads =
      at::empty_like(grad_flat, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto reduced_flat_indices =
      at::empty_like(max_indices_flat, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "embedding_bag_backward_cuda_max_deterministic", [&] {
        auto stream = at::cuda::getCurrentCUDAStream();
        auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
        auto policy = thrust::cuda::par(allocator).on(stream);
        auto reduced_end = thrust::reduce_by_key(
            policy,
            sorted_flat_max_indices.data_ptr<int64_t>(),
            sorted_flat_max_indices.data_ptr<int64_t>() +
                sorted_flat_max_indices.numel(),
            sorted_grads.data_ptr<scalar_t>(),
            reduced_flat_indices.data_ptr<int64_t>(),
            reduced_flat_grads.data_ptr<scalar_t>());
        AT_CUDA_CHECK(cudaGetLastError());
        auto reduced_sz = thrust::distance(
            thrust::device_ptr<int64_t>(
                reduced_flat_indices.data_ptr<int64_t>()),
            thrust::device_ptr<int64_t>(reduced_end.first));
        reduced_flat_indices.resize_({reduced_sz});
        reduced_flat_grads.resize_({reduced_sz});
      });

  auto grad_weight_flat =
      at::zeros({num_weights * grad.size(1)}, grad.options());
  auto mask = reduced_flat_indices >= 0;
  grad_weight_flat.scatter_(
      0,
      reduced_flat_indices.masked_select(mask),
      reduced_flat_grads.masked_select(mask));
  return grad_weight_flat.view({num_weights, grad.size(1)});
}

template <typename scalar_t>
__global__ void EmbeddingBag_accGradParametersKernel_max_nondeterministic(
    const PackedTensorAccessor64<int64_t, 2, RestrictPtrTraits> max_indices,
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> gradOutput,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> gradWeight) {
  using accscalar_t = acc_type<scalar_t, true>;
  const int32_t D = max_indices.size(1);
  const int32_t B = max_indices.size(0);
  // single thread per element.
  for (auto id = threadIdx.y + blockDim.y * blockIdx.x; id < B;
       id += gridDim.x * blockDim.y) {
    for (auto d = threadIdx.x; d < D; d += C10_WARP_SIZE) {
      int64_t weight_idx = max_indices[id][d];
      // If bag is empty, we have max_indices[idx] set to -1 in forward.
      if (weight_idx >= 0) {
        int64_t b = weight_idx / D;
        gpuAtomicAdd(&gradWeight[b][d], gradOutput[id][d]);
      }
    }
  }
}

Tensor embedding_bag_backward_cuda_max_nondeterministic(
    const Tensor& grad,
    const Tensor& max_indices,
    int64_t num_weights) {
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  const dim3 threads(C10_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE);
  const dim3 blocks(div_round_up(
      max_indices.size(0), CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(),
      "embedding_bag_backward_cuda_max_nondeterministic",
      [&] {
        EmbeddingBag_accGradParametersKernel_max_nondeterministic<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                max_indices.packed_accessor64<int64_t, 2, RestrictPtrTraits>(),
                grad.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
                grad_weight
                    .packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_weight;
}
} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_cuda(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset) {
  return _embedding_bag_cuda(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cuda(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights,
    bool include_last_offset) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag_cuda", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag_cuda", offsets_arg, kLong);
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
  TORCH_CHECK(featureSize <= std::numeric_limits<int32_t>::max(), "EmbeddingBag: featureSize must be int32")
  TORCH_CHECK(numBags <= std::numeric_limits<int32_t>::max(), "EmbeddingBag: numBags must be int32")

  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag = at::empty(
      {indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto output = at::empty({numBags, featureSize}, weight.options());

  Tensor max_indices;

  if (mode == MODE_MAX) {
    max_indices = at::empty({numBags, featureSize}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0, 0}, indices.options());
  }

  auto _per_sample_weights = per_sample_weights.defined()
      ? per_sample_weights
      : at::empty({0}, weight.options());

  const dim3 threads(C10_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE);
  const dim3 blocks(
      div_round_up(numBags, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_cuda",
      [&] {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "embedding_bag_cuda", [&] {

#define X(mode, ILP)                                                     \
  EmbeddingBag_updateOutputKernel<scalar_t, mode, ILP>                   \
      <<<blocks, threads, 0, stream>>>(                                  \
          indices.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),    \
          offsets.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),    \
          _per_sample_weights                                            \
              .packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),      \
          weight.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),    \
          output.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),    \
          offset2bag.packed_accessor64<int64_t, 1, RestrictPtrTraits>(), \
          bag_size.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),   \
          max_indices.packed_accessor64<int64_t, 2, RestrictPtrTraits>())
          if (mode == MODE_MAX) {
            if (featureSize % 4 == 0 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)weight.data_ptr()) >= 4 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)output.data_ptr()) >= 4) {
              X(MODE_MAX, 4);
            } else {
              X(MODE_MAX, 1);
            }
          }
          if (mode == MODE_MEAN) {
            if (featureSize % 4 == 0 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)weight.data_ptr()) >= 4 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)output.data_ptr()) >= 4) {
              X(MODE_MEAN, 4);
            } else {
              X(MODE_MEAN, 1);
            }
          }
          if (mode == MODE_SUM) {
            if (featureSize % 4 == 0 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)weight.data_ptr()) >= 4 &&
                memory::can_vectorize_up_to<scalar_t>(
                    (char*)output.data_ptr()) >= 4) {
              X(MODE_SUM, 4);
            } else {
              X(MODE_SUM, 1);
            }
          }
        });
      });
#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_cuda(
    const Tensor& grad_,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    const Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward in
  // EmbeddingBag.cpp.
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
      return embedding_bag_backward_cuda_sum_avg(
          grad,
          indices,
          offset2bag,
          bag_size_,
          num_weights,
          scale_grad_by_freq,
          mode,
          per_sample_weights);

    case MODE_MAX:
      AT_ASSERT(!per_sample_weights.defined());
      return globalContext().deterministic()
          ? embedding_bag_backward_cuda_max_deterministic(
                grad, max_indices, num_weights)
          : embedding_bag_backward_cuda_max_nondeterministic(
                grad, max_indices, num_weights);

    default:
      AT_ERROR("Unknown mode for embedding_bag_backward_cuda ", mode);
  }
}

template <typename scalar_t>
__inline__ __device__ static scalar_t warpReduceSum(scalar_t val) {
  for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2)
    val += WARP_SHFL_DOWN(val, offset);
  return val;
}

template <typename scalar_t>
__global__ static void _embedding_bag_per_sample_weights_backward_kernel(
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> grad,
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weight,
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor64<int64_t, 1, RestrictPtrTraits> offset2bag,
    PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> output) {
  const int32_t D = grad.size(1);
  for (int sample_idx = threadIdx.y + blockIdx.x * blockDim.y;
       sample_idx < indices.size(0);
       sample_idx += blockDim.y * gridDim.x) {
    acc_type<scalar_t, true> result = 0.0;
    const int64_t bag_idx = offset2bag[sample_idx];
    const int64_t embedding_idx = indices[sample_idx];
    for (int d = threadIdx.x; d < D; d += C10_WARP_SIZE) {
      result += grad[bag_idx][d] * weight[embedding_idx][d];
    }
    result = warpReduceSum<acc_type<scalar_t, true>>(result);
    if (threadIdx.x == 0) {
      output[sample_idx] = result;
    }
  }
}

Tensor _embedding_bag_per_sample_weights_backward_cuda(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
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

  dim3 block(C10_WARP_SIZE, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE);
  dim3 grid(
      div_round_up(num_samples, CUDA_MAX_THREADS_PER_BLOCK / C10_WARP_SIZE));

  auto output = at::empty({num_samples}, grad.options());

  // Early return when there is no samples in the batch. This saves unnecesary
  // kernel launch, but also prevents cudaGetLastError() to complain about
  // invalid launch args
  if (num_samples == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(),
      "_embedding_bag_per_sample_weights_backward_cuda",
      [&]() {
        _embedding_bag_per_sample_weights_backward_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
                weight.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
                indices.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),
                offset2bag.packed_accessor64<int64_t, 1, RestrictPtrTraits>(),
                output.packed_accessor64<scalar_t, 1, RestrictPtrTraits>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

} // namespace native
} // namespace at
