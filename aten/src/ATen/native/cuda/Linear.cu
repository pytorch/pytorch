#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorMeta.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

namespace at {
namespace native {

namespace {
// While this might be the a relatively efficient logsumexp-matmul
// implementation withing these that I have seen, this is awfully inefficient
// compared to matrix multiplication and e.g. NVidia cutlass may provide many
// great ideas for improvement
template <typename scalar_t, typename index_t>
__global__ void logmm_kernel(
    // compute res_ij = logsumexp_k lhs_ik + rhs_kj
    // for this compute maxsum_ij = max_k(lhs_ik + rhs_kj)
    // k = reduction dim, using threadIdx.x
    at::cuda::detail::TensorInfo<scalar_t, index_t> res,
    const at::cuda::detail::TensorInfo<scalar_t, index_t> lhs,
    const at::cuda::detail::TensorInfo<scalar_t, index_t> rhs) {
  using accscalar_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

  __shared__ accscalar_t shared_mem[2 * C10_WARP_SIZE];

  index_t j = blockIdx.y;
  index_t i = blockIdx.x;
  index_t b = blockIdx.z;
  int tid = threadIdx.x;
  index_t lhs_batch_offset = 0;
  index_t rhs_batch_offset = 0;
  index_t res_batch_offset = 0;
  for (int d = res.dims - 3; d >= 0; d--) {
    index_t cur_idx = b % res.sizes[d];
    lhs_batch_offset += cur_idx * lhs.strides[d];
    rhs_batch_offset += cur_idx * rhs.strides[d];
    res_batch_offset += cur_idx * res.strides[d];
    b /= res.sizes[d];
  }
  if (j >= rhs.sizes[rhs.dims - 1] || i >= lhs.sizes[lhs.dims - 2] || b > 0) {
    return;
  }
  // reduce within thread
  accscalar_t max = -std::numeric_limits<accscalar_t>::infinity();
  accscalar_t sumexp = 0;

  for (index_t k = threadIdx.x; k < lhs.sizes[lhs.dims - 1]; k += blockDim.x) {
    accscalar_t oldmax = max;
    accscalar_t value = lhs.data
                            [lhs_batch_offset + i * lhs.strides[lhs.dims - 2] +
                             k * lhs.strides[lhs.dims - 1]] +
        rhs.data
            [rhs_batch_offset + k * rhs.strides[rhs.dims - 2] +
             j * rhs.strides[rhs.dims - 1]];

    max = max > value ? max : value;
    if (oldmax == -std::numeric_limits<accscalar_t>::infinity()) {
      // sumexp used to be 0, so the new max is value and we can set 1 here,
      // because we will come back here again
      sumexp = 1;
    } else {
      sumexp *= std::exp(oldmax - max);
      sumexp += std::exp(
          value - max); // if oldmax was not -infinity, max is not either...
    }
  }

  // now we have one value per thread. we'll make it into one value per warp
  // first warpSum to get one value per thread to
  // one value per warp
  for (int k = 0; k < getMSB(C10_WARP_SIZE); ++k) {
    accscalar_t o_max = WARP_SHFL_XOR(max, 1 << k, C10_WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << k, C10_WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= std::exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * std::exp(o_max - max);
    }
  }

  __syncthreads();
  // this writes each warps accumulation into shared memory
  // there are at most C10_WARP_SIZE items left because
  // there are at most C10_WARP_SIZE**2 threads at the beginning
  if (tid % C10_WARP_SIZE == 0) {
    shared_mem[tid / C10_WARP_SIZE * 2] = max;
    shared_mem[tid / C10_WARP_SIZE * 2 + 1] = sumexp;
  }
  __syncthreads();
  if (tid < C10_WARP_SIZE) {
    max =
        (tid < blockDim.x / C10_WARP_SIZE
             ? shared_mem[2 * tid]
             : -std::numeric_limits<accscalar_t>::infinity());
    sumexp = (tid < blockDim.x / C10_WARP_SIZE ? shared_mem[2 * tid + 1] : 0);
  }
  for (int k = 0; k < getMSB(C10_WARP_SIZE); ++k) {
    accscalar_t o_max = WARP_SHFL_XOR(max, 1 << k, C10_WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << k, C10_WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= std::exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * std::exp(o_max - max);
    }
  }

  if (tid == 0) {
    res.data
        [res_batch_offset + i * res.strides[res.dims - 2] +
         j * res.strides[res.dims - 1]] =
        (max > -std::numeric_limits<accscalar_t>::infinity()
             ? std::log(sumexp) + max
             : -std::numeric_limits<accscalar_t>::infinity());
  }
}

template <typename scalar_t>
void log_matmul_cuda_template(
    const Tensor& lhs_,
    const Tensor& rhs_,
    const Tensor& out) {
  TORCH_CHECK(lhs_.is_cuda(), "need cuda tensors");
  TORCH_CHECK(lhs_.device() == rhs_.device(), "need tensors on same GPU");
  TORCH_CHECK(lhs_.dim() >= 2 && rhs_.dim() >= 2, "invalid sizes");
  TORCH_CHECK(lhs_.size(-1) == rhs_.size(-2), "invalid sizes");

  Tensor lhs = lhs_;
  Tensor rhs = rhs_;
  while (lhs.dim() < rhs.dim()) {
    lhs = lhs.unsqueeze(0);
  }
  while (rhs.dim() < lhs.dim()) {
    rhs = rhs.unsqueeze(0);
  }
  std::vector<int64_t> res_size = out.sizes().vec();
  int64_t num_batch = 1;
  for (int d = 0; d < res_size.size() - 2; d++) {
    num_batch *= res_size[d];
  }
  res_size[res_size.size() - 2] = -1;
  res_size[res_size.size() - 1] = -1;
  lhs = lhs.expand(res_size);
  rhs = rhs.expand(res_size);

  using index_t = int32_t;

  auto lhs_info = at::cuda::detail::getTensorInfo<scalar_t, index_t>(lhs);
  auto rhs_info = at::cuda::detail::getTensorInfo<scalar_t, index_t>(rhs);
  auto res_info = at::cuda::detail::getTensorInfo<scalar_t, index_t>(out);

  auto stream = at::cuda::getCurrentCUDAStream();

  int tf = getNumThreads(lhs.size(-1));
  dim3 blocks(lhs.size(-2), rhs.size(-1), num_batch);
  dim3 threads(tf);

  logmm_kernel<<<blocks, threads, 0, stream>>>(res_info, lhs_info, rhs_info);
}
} // namespace

TORCH_IMPL_FUNC(log_matmul_cuda)
(const Tensor& self, const Tensor& other, const Tensor& out) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "log_matmul", [&] {
        log_matmul_cuda_template<scalar_t>(self, other, out);
      });
}

} // namespace native
} // namespace at
