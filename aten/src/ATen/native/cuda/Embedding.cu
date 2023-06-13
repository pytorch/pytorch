#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>

#include <ATen/cuda/cub.cuh>

#include <ATen/native/cuda/EmbeddingBackwardKernel.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/thread_constants.h>

#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <thrust/iterator/reverse_iterator.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

#if defined(USE_ROCM)
static const int BLOCKDIMY = 16;
#else
static const int BLOCKDIMY = 32;
#endif

template
  <typename scalar_t,
   typename accscalar_t,
   typename index_t>
__global__ void embedding_backward_feature_kernel
  (const index_t* indices,
   const scalar_t* __restrict__ grad,
   scalar_t* __restrict__ grad_weight,
   int n, // OK to pass as int, we don't expect 2 billion+ samples in one shot
   int64_t stride,
   int padding_idx)
{
  extern __shared__ char buf[];
  accscalar_t* smem = (accscalar_t*)buf;
  accscalar_t* my_s = smem + C10_WARP_SIZE*threadIdx.y;
  int* indices_batch = (int*)(buf + sizeof(accscalar_t)*C10_WARP_SIZE*blockDim.y);

  const int s = (int)stride; // OK to make int, we don't expect 2 billion+ embedding row size

  const int f = threadIdx.x + blockIdx.x*blockDim.x; // feature_dim

  for(int batch_start = 0; batch_start < n; batch_start += blockDim.x*blockDim.y)
  {
    // Entire block cooperates to load a batch of 1024 indices to process
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if(batch_start + tid < n)
      indices_batch[tid] = (int)indices[batch_start + tid];

    int batch_end = batch_start + blockDim.x*blockDim.y < n ?
                    batch_start + blockDim.x*blockDim.y : n;

    // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
    for(int chunk_start = batch_start; chunk_start < batch_end; chunk_start += blockDim.y)
    {
      // This does double duty:  it makes sure indices_batch is ready, and it makes sure match-group
      // leaders are done with their accumulates before other warps start loading again.
      __syncthreads();

      int n_this_chunk = (batch_end - chunk_start) < blockDim.y ?
                         (batch_end - chunk_start) : blockDim.y;

      int src_row = chunk_start + threadIdx.y;
      int dst_row = indices_batch[src_row - batch_start]; // This warp's target row in grad_weight

      // All warps load their smem segments with incoming grad data
      if(src_row < n && f < s && dst_row != padding_idx)
        my_s[threadIdx.x] = static_cast<accscalar_t>(grad[src_row*stride + f]);

      __syncthreads();

      // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
      // We need to check if any other warps pulled grad data targeting dst_row.
      // If so, we elect the first warp in each matching group as the leader.
      // Each leader warp serializes the accumulates targeting dst_row in shared memory,
      // then finishes by adding the accumulated buffer to dst_row in grad_weight.
      if(dst_row != padding_idx && src_row < n) // Per-warp exit condition, safe with ballot_sync
      {
        int match_found_this_thread = 0;
        if(threadIdx.x < n_this_chunk)
          match_found_this_thread = (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
#if defined(USE_ROCM)
        unsigned long long int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffsll(matchmask) - 1;
#else
        unsigned int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffs(matchmask) - 1;
#endif

        if(threadIdx.y == first_remaining_peer) // Nominate lowest-indexed warp as the leader
        {
          matchmask ^= (1 << first_remaining_peer);
          while(matchmask)
          {
#if defined(USE_ROCM)
            first_remaining_peer = __ffsll(matchmask) - 1;
#else
            first_remaining_peer = __ffs(matchmask) - 1;
#endif
            my_s[threadIdx.x] += smem[threadIdx.x + C10_WARP_SIZE*first_remaining_peer];
            matchmask ^= (1 << first_remaining_peer);
          }
          if(f < s)
            grad_weight[dst_row*stride + f] += static_cast<scalar_t>(my_s[threadIdx.x]);
        }
      }
    }
  }
}


template <typename scalar_t, typename index_t>
__global__ void embedding_backward_kernel(
  index_t* input, index_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  index_t* count, int64_t numel, int64_t stride, int padding_idx) {

  using accscalar_t = acc_type<scalar_t, true>;
  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])
      && input[idx] != padding_idx) {
    do {
      const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weight_row = ((int) input[idx]) * stride;
      const int grad_row = ((int) indices[idx]) * stride;
      const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;

      accscalar_t gradient[SZ];
      accscalar_t weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * C10_WARP_SIZE;
        if (feature_dim < stride) {
          gradient[ii] = static_cast<accscalar_t>(grad_output[grad_row + feature_dim]);
          weight[ii] = static_cast<accscalar_t>(grad_weight[weight_row + feature_dim]);
        }
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        weight[ii] += gradient[ii] * scale;
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * C10_WARP_SIZE;
        if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

/* Calculate norms of the rows of weight_ptr given by idx_ptr and capture them in norms */
template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void renorm_kernel(
    scalar_t* weights, index_t* indices, accscalar_t max_norm,
    accscalar_t norm_type, int64_t dim,
    int64_t weights_stride0, int64_t weights_stride1,
    const int64_t *num_unique_indices) {
  if (blockIdx.x >= *num_unique_indices) {
    return;
  }

  // Some casting hacks since dynamic shared memory and templates don't work together:
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  int tid = threadIdx.x;
  int base_index = indices[blockIdx.x] * weights_stride0;

  accscalar_t v = 0;
  for (int i = tid; i < dim; i += blockDim.x) {
    auto x = static_cast<accscalar_t>(weights[base_index + i * weights_stride1]);
    if (norm_type == 1) {
      v += std::abs(x);
    } else if (norm_type == 2) {
      v += x * x;
    } else {
      v += std::pow(x, norm_type);
    }
  }

  v = cuda_utils::BlockReduceSum(v, sdata);

  if (tid == 0) {
    sdata[0] = std::pow(v, static_cast<accscalar_t>(1.0 / norm_type));
  }
  __syncthreads();

  // now we renormalize the blocks that need it
  if (sdata[0] > max_norm) {
    auto factor = static_cast<scalar_t>(max_norm / (sdata[0] + 1e-7));
    for (int i = tid; i < dim; i += blockDim.x) {
      weights[base_index + i * weights_stride1] *= factor;
    }
  }
}

} // anonymous namespace

#if !CUB_SUPPORTS_SCAN_BY_KEY()
template<typename index_t>
void embedding_dense_backward_cuda_scan(Tensor &sorted_indices, Tensor &count);
#endif

Tensor embedding_dense_backward_cuda(const Tensor & grad_, const Tensor & indices_,
                               int64_t num_weights, int64_t padding_idx,
                               bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices_, "indices", 1);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});
  checkSameGPU("embedding_backward", grad_arg, indices_arg);

  auto indices = indices_.contiguous();

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_indices <= 3072 && !scale_grad_by_freq) {
    auto indices_contig = indices.contiguous();
    auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());
    int64_t stride = grad_weight.stride(0);
    int warp_size = at::cuda::warp_size();
    dim3 grid(ceil_div(stride, (int64_t)warp_size));
    dim3 block(warp_size, BLOCKDIMY);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad.scalar_type(),
       "embedding_backward",
       [&]
       {
          using accscalar_t = acc_type<scalar_t, true>;
          AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cuda", [&] () {
          embedding_backward_feature_kernel<scalar_t, accscalar_t, index_t>
            <<<grid,
                block,
                sizeof(accscalar_t)*warp_size*BLOCKDIMY + sizeof(int)*warp_size*BLOCKDIMY,
                stream>>>
            (indices_contig.const_data_ptr<index_t>(),
              grad.const_data_ptr<scalar_t>(),
              grad_weight.mutable_data_ptr<scalar_t>(),
              static_cast<int>(num_indices),
              static_cast<int64_t>(stride),
              static_cast<int>(padding_idx));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
       });
    return grad_weight;
  }

  auto sorted_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor count;
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cuda", [&] () {
    auto range = at::arange(num_indices, indices.options());
    int64_t nbits = cuda::cub::get_num_bits(num_weights);
    cuda::cub::radix_sort_pairs(
      indices.const_data_ptr<index_t>(), sorted_indices.mutable_data_ptr<index_t>(),
      range.const_data_ptr<index_t>(), orig_indices.mutable_data_ptr<index_t>(),
      num_indices, false/*, 0, nbits*/);
  });

  if (scale_grad_by_freq) {
    count = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
#if CUB_SUPPORTS_SCAN_BY_KEY()
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cuda", [&] () {
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
        thrust::make_reverse_iterator(static_cast<const index_t*>(count_data) + num_indices),
        thrust::make_reverse_iterator(count_data + num_indices),
        at_cuda_detail::cub::Max(),
        num_indices
      );
    });
#else
    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cuda", [&] () {
      embedding_dense_backward_cuda_scan<index_t>(sorted_indices, count);
    });
#endif
  }

  return embedding_backward_cuda_kernel(grad, orig_indices,
      sorted_indices, count, num_weights, padding_idx);
}

Tensor & embedding_renorm_cuda_(Tensor & self, const Tensor & indices,
                                double max_norm, double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkDim("embedding_renorm_", self_arg, 2);
  checkSameGPU("embedding_renorm", self_arg, indices_arg);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_renorm_cuda_", [&] () {

    auto num_indices = indices.numel();
    auto indices_contig = std::get<0>(indices.sort()).contiguous();
    auto unique_indices = at::empty(indices.numel(), indices.options());
    auto num_unique_indices = at::empty({}, indices.options().dtype(kLong));

    cuda::cub::unique(
      indices_contig.const_data_ptr<index_t>(),
      unique_indices.mutable_data_ptr<index_t>(),
      num_unique_indices.mutable_data_ptr<int64_t>(),
      num_indices
    );

    int warp_size = at::cuda::warp_size();
    TORCH_INTERNAL_ASSERT(num_threads() % warp_size == 0 &&
                  num_threads() <= cuda_utils::kCUDABlockReduceMaxThreads,
                  "BlockReduceSum requires all warps be active");
    const int64_t *num_unique_indices_ptr = num_unique_indices.const_data_ptr<int64_t>();
    dim3 grid = unique_indices.numel();
    dim3 block = num_threads();
    int dim = self.stride(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "embedding_renorm_cuda_", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      renorm_kernel<<<grid, block, (block.x / warp_size) * sizeof(accscalar_t), stream>>>(
        self.mutable_data_ptr<scalar_t>(),
        unique_indices.const_data_ptr<index_t>(),
        static_cast<accscalar_t>(max_norm),
        static_cast<accscalar_t>(norm_type),
        dim, self.stride(0), self.stride(1),
        num_unique_indices_ptr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });
  return self;
}


}  // namespace at::native
