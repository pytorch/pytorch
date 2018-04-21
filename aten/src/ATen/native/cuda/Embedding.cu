#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Error.h"

#include "ATen/cuda/AccumulateType.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>


namespace at { namespace native {

namespace {

static const int WARP_SIZE = 32;

__device__ __forceinline__ bool warp_has_collision(int val) {
  // Compare our value to the values stored in the next 16 lanes,
  // wrapping around at 32. If any pair of values is the same than
  // there is a collision in the warp.
  bool dup = 0;
  const int laneId = threadIdx.x % 32;
  #pragma unroll
  for (int i = 1; i <= 16; i++) {
    dup |= (WARP_SHFL(val, (laneId + i) % 32) == val);
  }
  return __any(dup) != 0;
}

// parallelizes over features
template <typename scalar_t>
__global__ void embedding_backward_feature_kernel(
  int64_t* indices, scalar_t* grad, scalar_t* grad_weight,
  int64_t num_indices, int64_t stride, int padding_idx) {

  const int feature_dim = blockIdx.x * 4 + threadIdx.x / 32;
  if (feature_dim >= stride) {
    return;
  }

  // The strategy here is that each warp handles a single feature
  // dimension.
  // Within that feature dimension, points in the [batch][element]
  // dimension can overlap, and we need to determine if threads want
  // to add to the gradient in a colliding manner.
  // Typically one would use floating-point atomicAdd() to resolve
  // these collisions, but that is non-deterministic if there are
  // collisions. Non-determinism for this code is really bad,
  // especially in RNNs, and is prone to snowballing error.
  // In order to get a deterministic order of execution, we handle
  // non-colliding updates separately from colliding ones. Colliding
  // updates are serialized in their order of execution by using the
  // warp-wide collision detector `warp_has_collision`.
  const int laneId = threadIdx.x % 32;
  for (int64_t i = laneId; i < num_indices; i += WARP_SIZE) {
    const int weight_index = (int)indices[i];
    if (weight_index == padding_idx) {
      continue;
    }

    auto value = grad[i * stride + feature_dim];

    // FIXME: should we accumulate as accreal?
    // Check for collision
    if (warp_has_collision(weight_index)) {
      // Run all lanes sequentially; warp divergence
      for (int i = 0; i < WARP_SIZE; ++i) {
        if (laneId == i) {
          grad_weight[weight_index * stride + feature_dim] += value;
        }
      }
    } else {
      // No collision; warp coherence
      grad_weight[weight_index * stride + feature_dim] += value;
    }
  }
}


template <typename scalar_t>
__global__ void embedding_backward_kernel(
  int64_t* input, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  int64_t* count, int64_t numel, int64_t stride, int padding_idx) {

  using accscalar_t = cuda::acc_type<scalar_t>;
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
        int feature_dim = start_feature + ii * WARP_SIZE;
        if (feature_dim < stride) {
          gradient[ii] = scalar_cast<accscalar_t>(grad_output[grad_row + feature_dim]);
          weight[ii] = scalar_cast<accscalar_t>(grad_weight[weight_row + feature_dim]);
        }
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        weight[ii] += gradient[ii] * scale;
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * WARP_SIZE;
        if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = scalar_cast<scalar_t>(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

/* Calculate norms of the rows of weight_ptr given by idx_ptr and capture them in norms */
template <typename scalar_t, typename accscalar_t>
__global__ void renorm_kernel(
    scalar_t* weights, int64_t* indices, accscalar_t max_norm,
    accscalar_t norm_type, int dim) {

  // Some casting hacks since dynamic shared memory and templates don't work together:
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  int tid = threadIdx.x;
  int base_index = indices[blockIdx.x] * dim;

  accscalar_t v = 0;
  for (int i = tid; i < dim; i += blockDim.x) {
    auto x = scalar_cast<accscalar_t>(weights[base_index + i]);
    if (norm_type == 1) {
      v += std::abs(x);
    } else if (norm_type == 2) {
      v += x * x;
    } else {
      v += std::pow(x, norm_type);
    }
  }

  using Op = ReduceAdd<accscalar_t, accscalar_t>;
  v = reduceBlock<accscalar_t>(sdata, blockDim.x, v, Op(), 0);

  if (tid == 0) {
    sdata[0] = std::pow(v, scalar_cast<accscalar_t>(1.0 / norm_type));
  }
  __syncthreads();

  // now we renormalize the blocks that need it
  if (sdata[0] > max_norm) {
    auto factor = scalar_cast<scalar_t>(max_norm / (sdata[0] + 1e-7));
    for (int i = tid; i < dim; i += blockDim.x) {
      weights[base_index + i] *= factor;
    }
  }
}

} // anonymous namespace

Tensor embedding_backward_cuda(const Tensor & grad_, const Tensor & indices,
                               int64_t num_weights, int64_t padding_idx,
                               bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_backward", indices_arg, kLong);
  checkContiguous("embedding_backward", indices_arg);
  checkSameGPU("embedding_backward", grad_arg, indices_arg);

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  auto grad_weight = at::zeros(grad_.type(), {num_weights, grad_.size(-1)});

  int64_t stride = grad_weight.stride(0);
  cudaStream_t stream = globalContext().getCurrentCUDAStream();

  if (num_indices <= 768 && !scale_grad_by_freq) {
   dim3 grid(THCCeilDiv(stride, (int64_t) 4));
   dim3 block(128);

   AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "embedding_backward", [&] {
     using cuda_scalar_t = cuda::type<scalar_t>;
     embedding_backward_feature_kernel<<<grid, block, 0, stream>>>(
       indices.data<int64_t>(),
       grad.data<cuda_scalar_t>(),
       grad_weight.data<cuda_scalar_t>(),
       num_indices,
       stride,
       padding_idx);
   });

   THCudaCheck(cudaGetLastError());
   return grad_weight;
  }

  auto sorted_indices = indices.type().tensor(indices.sizes());
  auto orig_indices = indices.type().tensor(indices.sizes());
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
    auto orig_data = device_ptr(orig_indices.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data,
                        ThrustLTOp<int64_t>());
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = indices.type().tensor(indices.sizes());

    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Compute an increasing sequence per unique item in sortedIndices:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    auto count_data = device_ptr(count.data<int64_t>());
    thrust::inclusive_scan_by_key(
      policy,
      sorted_data,
      sorted_data + num_indices,
      thrust::make_constant_iterator(1),
      count_data
    );

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
      policy,
      thrust::make_reverse_iterator(sorted_data + num_indices),
      thrust::make_reverse_iterator(sorted_data),
      thrust::make_reverse_iterator(count_data + num_indices),
      thrust::make_reverse_iterator(count_data + num_indices),
      thrust::equal_to<int64_t>(),
      thrust::maximum<int64_t>()
    );
  }

  dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "embedding_backward", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    embedding_backward_kernel<<<grid, block, 0, stream>>>(
      sorted_indices.data<int64_t>(),
      orig_indices.data<int64_t>(),
      grad.data<cuda_scalar_t>(),
      grad_weight.data<cuda_scalar_t>(),
      count.defined() ? count.data<int64_t>() : nullptr,
      num_indices,
      stride,
      padding_idx);
  });
  THCudaCheck(cudaGetLastError());

  return grad_weight;
}

Tensor & embedding_renorm_cuda_(Tensor & self, const Tensor & indices,
                                double max_norm, double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkContiguous("embedding_renorm_", self_arg);
  checkContiguous("embedding_renorm", indices_arg);
  checkDim("embedding_renorm_", self_arg, 2);
  checkSameGPU("embedding_renorm", self_arg, indices_arg);

  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  using device_ptr = thrust::device_ptr<int64_t>;

  auto num_indices = indices.numel();
  auto indices_data = device_ptr(indices.data<int64_t>());

  // FIXME: thrust::unique only removes consecutive elements that are equal.
  // We have race conditions when indices contain duplicates which are not
  // adjacent
  auto unique_indices = indices.type().tensor(indices.numel());
  auto unique_data = device_ptr(unique_indices.data<int64_t>());
  auto end = thrust::unique_copy(policy, indices_data, indices_data + num_indices, unique_data);
  auto num_unique_indices = static_cast<int>(end - unique_data);

  dim3 grid(num_unique_indices);
  dim3 block(128);
  int dim = self.stride(0);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "embedding_backward", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    using accscalar_t = cuda::acc_type<cuda_scalar_t>;
    renorm_kernel<<<grid, block, 128 * sizeof(accscalar_t), stream>>>(
      self.data<cuda_scalar_t>(),
      unique_indices.data<int64_t>(),
      scalar_cast<accscalar_t>(max_norm),
      scalar_cast<accscalar_t>(norm_type),
      dim);
  });
  THCudaCheck(cudaGetLastError());

  return self;
}

}}  // namespace at::native
