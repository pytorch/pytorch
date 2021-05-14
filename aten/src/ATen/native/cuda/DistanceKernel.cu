#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <THC/THCTensorMathReduce.cuh>
#include <math.h>

#include <ATen/native/Distance.h>

#include <c10/macros/Macros.h>

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
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) {
      return (dist == 0.0 || (diff == 0.0 && p < 1)) ? 0 : (sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1));
    }
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
__device__ static inline scalar_t reduce_agg(scalar_t agg) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    F::agg(agg, WARP_SHFL_DOWN(agg, offset));
  }

  __shared__ scalar_t shared[forward_threads];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = agg;
  }

  __syncthreads();
  agg = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      F::agg(agg, WARP_SHFL_DOWN(agg, offset));
    }
  }
  return agg;
}

template <typename scalar_t, typename F>
__global__ static void pdist_kernel_cuda_impl(scalar_t * result, const scalar_t * self, const int64_t n, const int64_t m, const scalar_t p,
                                              const double n2, const double n2_squared_minus_1) {
  const int64_t k = blockIdx.x;
  const int stride = blockDim.x;

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

  const scalar_t * const start = self + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = self + j * m + threadIdx.x;
  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }

  agg = reduce_agg<scalar_t, F>(agg);
  if (threadIdx.x == 0) {
    result[k] = F::finish(agg, p);
  }
}

template <typename scalar_t, typename F>
__global__ static void cdist_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * x1, const scalar_t * x2, const scalar_t * dist, int64_t gs,
                                                       const scalar_t p, const int64_t r1, const int64_t r2, const int64_t m, const int64_t count, const int64_t r_size, const int64_t l1_size, const int64_t l2_size) {
  const int y = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= count || init >= m) {
    return;
  }
  const int l = y / r_size;
  const int k = y % r_size;
  const int stride = blockDim.x * gridDim.x;
  const int l_size = r_size * m;

  int64_t i = k / r2;
  int64_t j = k % r2;

  const scalar_t grad_k = grad[y];
  const scalar_t dist_k = dist[y];

  const scalar_t * const start = x1 + l * l1_size + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * self_i = start + init;
  const scalar_t * self_j = x2 + l * l2_size + j * m + init;

  scalar_t * buff_i = buffer + l * l_size + (r1 * j + i) * m + init;

  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
  }
}

template <typename scalar_t, typename F>
__global__ static void pdist_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * self, const scalar_t * dist, int64_t gs, const int64_t n, const int64_t m, const int64_t combs, const scalar_t p,
                                                       const double n2, const double n2_squared_minus_1) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride = blockDim.y * gridDim.y;

  if (k >= combs) {
    return;
  }

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * k)));
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

template <typename scalar_t, typename F>
__global__ static void cdist_kernel_cuda_impl(scalar_t * result, const scalar_t * x1, const scalar_t * x2,
    const scalar_t p, const int64_t r1, const int64_t r2, const int64_t m, const int64_t r_size, const int64_t l1_size, const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const scalar_t * const start = x1 + l * l1_size + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = x2 + l * l2_size + j * m + threadIdx.x;

  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }
  agg = reduce_agg<scalar_t, F>(agg);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = F::finish(agg, p);
  }
}

void cdist_kernel_impl(Tensor& result, const Tensor& x1, const Tensor& x2, double p) {
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;
  const dim3 grid(result.numel());
  const dim3 block(forward_threads);

  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cdist_cuda", [&] {
    if (p == 0.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r1, r2, m, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 1.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r1, r2, m, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 2.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r1, r2, m, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (std::isinf(p)) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r1, r2, m, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r1, r2, m, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
}

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, double p) {
  const dim3 grid(result.numel());
  const dim3 block(forward_threads);
  int64_t n = self.size(0);
  int64_t m = self.size(1);
  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_cuda", [&] {
    if (p == 0.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), n, m, p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 1.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), n, m, p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 2.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), n, m, p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (std::isinf(p)) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), n, m, p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), n, m, p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const int block_x = 16;
  // NB: be careful with changing block_y; as it's currently written, grid_y is limited to be 2^16.
  // block_y of 64 gives us max pdist dim1 of 2**24
  const int block_y = 64;
  const int grid_x = (dist.numel() + block_x - 1) / block_x;
  const int grid_y = (m + block_y * 8 - 1) / (block_y * 8);
  const dim3 grid(grid_x, grid_y);
  const dim3 block(block_x, block_y);
  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  Tensor buffer = at::empty({n - 1, result.size(0), result.size(1)}, result.options());
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_cuda_backward", [&] {
    if (p == 1.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(), grad.stride(0), n, m, dist.numel(), p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p < 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(), grad.stride(0), n, m, dist.numel(), p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(), grad.stride(0), n, m, dist.numel(), p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (std::isinf(p)) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(), grad.stride(0), n, m, dist.numel(), p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(), grad.stride(0), n, m, dist.numel(), p, n2, n2_squared_minus_1);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });

  at::sum_out(result, buffer, 0);
}

void cdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0 || x2.numel() == 0) {
    result.fill_(0);
    return;
  }

  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  // Just like we do in the CPU code, assume that result is always batched
  int64_t batch = result.size(0);
  const int block_x = 64;
  const int block_y = 16;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);

  const int64_t count = dist.numel();
  const int64_t grid_temp = (count + block_y - 1) / block_y;

  const int grid_y = (grid_temp - 1) / 65535 + 1;
  const int grid_z = (grid_temp - 1) / grid_y + 1;

  const dim3 grid(grid_x, grid_y, grid_z);
  const dim3 block(block_x, block_y);

  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;
  //current implementation supports only gradient that can be collapsed to 1D. However, to avoid checking this assumption,
  //we call grad.contiguous() before backward, so stride is guaranteed to be 1
  const int64_t gs = 1;

  Tensor buffer = at::empty({batch, r2, r1, m}, result.options());
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist_cuda_backward", [&] {
    if (p == 1.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(),
      gs, p, r1, r2, m, count, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p < 2.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(),
      gs, p, r1, r2, m, count, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (p == 2.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(),
      gs, p, r1, r2, m, count, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (std::isinf(p)) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(),
      gs, p, r1, r2, m, count, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), dist.data_ptr<scalar_t>(),
      gs, p, r1, r2, m, count, r_size, l1_size, l2_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });

  at::sum_out(result, buffer, 1);

}


} // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);
REGISTER_DISPATCH(cdist_stub, &cdist_kernel_impl);
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_impl);

}} // at::native
