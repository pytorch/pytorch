#include <ATen/ATen.h>
#include <THC/THCTensorMathReduce.cuh>
#include <math.h>

#include <ATen/native/Distance.h>


namespace at { namespace native {

namespace {

static const int forward_threads = 256;

template <typename scalar_t>
static __forceinline__ __device__ scalar_t device_sqrt(scalar_t val);

template <>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template <>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}

template <typename scalar_t>
struct dists {

  static __forceinline__ __device__ scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  // Zero norm
  struct zero {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff != 0.0; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
  };

  // One norm
  struct one {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff); }
  };

  // Special case backward when p is less than two
  struct lt_two {
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1); }
  };

  // Two norm
  struct two {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += diff * diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return device_sqrt<scalar_t>(agg); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : grad * diff / dist; }
  };

  // General p norm
  struct p {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += std::pow(diff, p); }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, static_cast<scalar_t>(1) / p); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1); }
  };

  // Inf norm
  struct inf {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { if (diff > agg) { agg = diff; } }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { if (other > update) { update = other; } }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return grad * sign(diff) * (std::abs(diff) == dist); }
  };

};

template <typename scalar_t, typename F>
__global__ static void pdist_kernel_cuda_impl(scalar_t * result, const scalar_t * self, const int64_t n, const int64_t m, const scalar_t p) {
  const int l = blockIdx.x;
  const int k = blockIdx.y;
  const int stride = blockDim.y;

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<scalar_t>(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

  const scalar_t * const start = self + (l * n + i) * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.y;
  const scalar_t * b = self + (l * n + j) * m + threadIdx.y;
  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }

  // Reduce warps
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    F::agg(agg, WARP_SHFL_DOWN(agg, offset));
  }

  // Reduce block
  // This shared memory is significantly larger than necessary, but the
  // assumption is that it's not a bottleneck, and this is simple
  __shared__ scalar_t shared[forward_threads];
  int lane = threadIdx.y % warpSize;
  int warp_id = threadIdx.y / warpSize;
  if (lane == 0) {
    shared[warp_id] = agg;
  }
  __syncthreads();
  agg = (threadIdx.y < blockDim.y / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    // Only reduce theads with nonzero data
    for (int offset = blockDim.y / warpSize / 2; offset > 0; offset /= 2) {
      F::agg(agg, WARP_SHFL_DOWN(agg, offset));
    }
  }
  if (threadIdx.y == 0) {
    const int64_t combs = n * (n - 1) / 2;
    result[l * combs + k] = F::finish(agg, p);
  }
}

template <typename scalar_t, typename F>
__global__ static void pdist_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * self, const scalar_t * dist, const int64_t n, const int64_t m, const scalar_t p) {
  const int l = blockIdx.x;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride = blockDim.y * gridDim.y;
  const int combs = n * (n - 1) / 2;

  if (k >= combs) {
    return;
  }

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<scalar_t>(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const scalar_t grad_k = grad[l * combs + k];
  const scalar_t dist_k = dist[l * combs + k];

  const scalar_t * const start = self + (l * n + i) * m;
  const scalar_t * const end = start + m;
  const scalar_t * self_i = start + init;
  const scalar_t * self_j = self + (l * n + j) * m + init;
  scalar_t * buff_i = buffer + ((l * (n - 1) + ib) * n + i) * m + init;
  scalar_t * buff_j = buffer + ((l * (n - 1) + jb) * n + j) * m + init;
  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride, buff_j += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
    *buff_j = -res;
  }
}

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, double p) {
  int64_t b = self.size(0);
  int64_t n = self.size(1);
  int64_t m = self.size(2);
  const dim3 grid(b, result.size(1));
  const dim3 block(1, forward_threads);

  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda", [&] {
    if (p == 0.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<grid, block>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (p == 1.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (p == 2.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else if (std::isinf(p)) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    } else {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block>>>(result.data<scalar_t>(), self.data<scalar_t>(), n, m, p);
    }
  });
}

void pdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return;
  }

  const int64_t b = self.size(0);
  const int64_t n = self.size(1);
  const int64_t m = self.size(2);
  const int block_y = 64;
  const int block_z = 4;
  const int grid_y = (m + block_y * 8 - 1) / (block_y * 8);
  const int grid_z = (dist.numel() + block_z - 1) / block_z;
  const dim3 grid(b, grid_y, grid_z);
  const dim3 block(1, block_y, block_z);

  Tensor buffer = at::empty({b, n - 1, n, m}, result.options());
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda_backward", [&] {
    if (p == 1.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), n, m, p);
    } else if (p < 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), n, m, p);
    } else if (p == 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), n, m, p);
    } else if (std::isinf(p)) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), n, m, p);
    } else {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), n, m, p);
    }
  });

  at::sum_out(result, buffer, 1);
}

} // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);

}} // at::native
