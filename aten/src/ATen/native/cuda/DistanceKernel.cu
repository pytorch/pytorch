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
  const int k = blockIdx.x;
  const int stride = blockDim.x;

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<scalar_t>(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

  const scalar_t * const start = self + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = self + j * m + threadIdx.x;
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
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = agg;
  }
  __syncthreads();
  agg = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    // Only reduce theads with nonzero data
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      F::agg(agg, WARP_SHFL_DOWN(agg, offset));
    }
  }
  if (threadIdx.x == 0) {
    result[k] = F::finish(agg, p);
  }
}

template <typename scalar_t, typename F>
__global__ static void pdist_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * self, const scalar_t * dist, int64_t gs, const int64_t n, const int64_t m, const int64_t combs, const scalar_t p) {
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  if (k >= combs) {
    return;
  }

  float n2 = n - .5;
  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<scalar_t>(n2 * n2 - 2 * k - 1)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const scalar_t grad_k = grad[k * gs];
  const scalar_t dist_k = dist[k];

  const scalar_t * const start = self + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * self_i = start + init;
  const scalar_t * self_j = self + j * m + init;
  scalar_t * buff_i = buffer + (ib * n + i) * m + init;
  scalar_t * buff_j = buffer + (jb * n + j) * m + init;
  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride, buff_j += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
    *buff_j = -res;
  }
}

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, double p) {
  const dim3 grid(result.numel());
  const dim3 block(forward_threads);
  int64_t n = self.size(0);
  int64_t m = self.size(1);

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

  const int64_t n = result.size(0);
  int64_t m = self.size(1);
  const int block_x = 64;
  const int block_y = 4;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);
  const int grid_y = (dist.numel() + block_y - 1) / block_y;
  const dim3 grid(grid_x, grid_y);
  const dim3 block(block_x, block_y);

  Tensor buffer = at::empty({n - 1, result.size(0), result.size(1)}, result.options());
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda_backward", [&] {
    if (p == 1.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, dist.numel(), p);
    } else if (p < 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, dist.numel(), p);
    } else if (p == 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, dist.numel(), p);
    } else if (std::isinf(p)) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, dist.numel(), p);
    } else {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block>>>(buffer.data<scalar_t>(), grad.data<scalar_t>(), self.data<scalar_t>(), dist.data<scalar_t>(), grad.stride(0), n, m, dist.numel(), p);
    }
  });

  at::sum_out(result, buffer, 0);
}

} // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);

}} // at::native
