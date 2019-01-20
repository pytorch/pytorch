#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <math.h>
#include <THC/THCTensorMathReduce.cuh>

#include <ATen/native/Distance.h>

namespace at {
namespace native {

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
    static __forceinline__ __device__ void inc(
        scalar_t& agg,
        const scalar_t diff,
        const scalar_t p) {
      agg += diff != 0.0;
    }
    static __forceinline__ __device__ scalar_t
    finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static __forceinline__ __device__ void agg(
        scalar_t& update,
        const scalar_t other) {
      update += other;
    }
  };

  // One norm
  struct one {
    static __forceinline__ __device__ void inc(
        scalar_t& agg,
        const scalar_t diff,
        const scalar_t p) {
      agg += diff;
    }
    static __forceinline__ __device__ scalar_t
    finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static __forceinline__ __device__ void agg(
        scalar_t& update,
        const scalar_t other) {
      update += other;
    }
    static __forceinline__ __device__ scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return grad * sign(diff);
    }
  };

  // Special case backward when p is less than two
  struct lt_two {
    static __forceinline__ __device__ scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0 ? 0
                         : sign(diff) * std::pow(std::abs(diff), p - 1) * grad /
              std::pow(dist, p - 1);
    }
  };

  // Two norm
  struct two {
    static __forceinline__ __device__ void inc(
        scalar_t& agg,
        const scalar_t diff,
        const scalar_t p) {
      agg += diff * diff;
    }
    static __forceinline__ __device__ scalar_t
    finish(const scalar_t agg, const scalar_t p) {
      return device_sqrt<scalar_t>(agg);
    }
    static __forceinline__ __device__ void agg(
        scalar_t& update,
        const scalar_t other) {
      update += other;
    }
    static __forceinline__ __device__ scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0 ? 0 : grad * diff / dist;
    }
  };

  // General p norm
  struct p {
    static __forceinline__ __device__ void inc(
        scalar_t& agg,
        const scalar_t diff,
        const scalar_t p) {
      agg += std::pow(diff, p);
    }
    static __forceinline__ __device__ scalar_t
    finish(const scalar_t agg, const scalar_t p) {
      return std::pow(agg, static_cast<scalar_t>(1) / p);
    }
    static __forceinline__ __device__ void agg(
        scalar_t& update,
        const scalar_t other) {
      update += other;
    }
    static __forceinline__ __device__ scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return dist == 0.0 ? 0
                         : diff * std::pow(std::abs(diff), p - 2) * grad /
              std::pow(dist, p - 1);
    }
  };

  // Inf norm
  struct inf {
    static __forceinline__ __device__ void inc(
        scalar_t& agg,
        const scalar_t diff,
        const scalar_t p) {
      if (diff > agg) {
        agg = diff;
      }
    }
    static __forceinline__ __device__ scalar_t
    finish(const scalar_t agg, const scalar_t p) {
      return agg;
    }
    static __forceinline__ __device__ void agg(
        scalar_t& update,
        const scalar_t other) {
      if (other > update) {
        update = other;
      }
    }
    static __forceinline__ __device__ scalar_t backward(
        const scalar_t diff,
        const scalar_t grad,
        const scalar_t dist,
        const scalar_t p) {
      return grad * sign(diff) * (std::abs(diff) == dist);
    }
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
__global__ static void pdist_kernel_cuda_impl(
    scalar_t* result,
    const scalar_t* self,
    const int64_t n,
    const int64_t d,
    const scalar_t p,
    const double n2,
    const double n2_squared_minus_1) {
  const int64_t r_k = blockIdx.x;
  const int64_t b_l = blockIdx.y;
  const int64_t stride = blockDim.x;

  // The -1 accounts for floating point truncation issues
  int64_t n_i = static_cast<int64_t>(
      (n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * r_k)));
  int64_t n_j = r_k - n * n_i + n_i * (n_i + 1) / 2 + n_i + 1;

  const scalar_t* const start = self + (b_l * n + n_i) * d;
  const scalar_t* const end = start + d;
  const scalar_t* a = start + threadIdx.x;
  const scalar_t* b = self + (b_l * n + n_j) * d + threadIdx.x;

  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }

  agg = reduce_agg<scalar_t, F>(agg);
  if (threadIdx.x == 0) {
    const int64_t r = n * (n - 1) / 2;
    result[b_l * r + r_k] = F::finish(agg, p);
  }
}

template <typename scalar_t, typename F>
__global__ static void cdist_backward_kernel_cuda_impl(
    scalar_t* buffer,
    const scalar_t* grad,
    const scalar_t* x1,
    const scalar_t* x2,
    const scalar_t* dist,
    int64_t gs,
    const scalar_t p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t count) {
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  if (k >= count) {
    return;
  }

  int64_t i = k / r2;
  int64_t j = k % r2;

  const scalar_t grad_k = grad[k * gs];
  const scalar_t dist_k = dist[k];

  const scalar_t* const start = x1 + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* self_i = start + init;
  const scalar_t* self_j = x2 + j * m + init;

  scalar_t* buff_i = buffer + (r1 * j + i) * m + init;

  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p);
    *buff_i = res;
  }
}

template <typename scalar_t, typename F>
__global__ static void pdist_backward_kernel_cuda_impl(
    scalar_t* result,
    const scalar_t* grad,
    const scalar_t* self,
    const scalar_t* dist,
    int64_t gs_l,
    int64_t gs_k,
    const int64_t n,
    const int64_t d,
    const int64_t r,
    const scalar_t p,
    const double n2,
    const double n2_squared_minus_1) {
  const int64_t b_l = blockIdx.z;
  const int64_t r_k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t init = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t stride = blockDim.x * gridDim.x;

  if (r_k >= r) {
    return;
  }

  // The -1 accounts for floating point truncation issues
  int64_t n_i = static_cast<int64_t>(
      (n2 - device_sqrt<double>(n2_squared_minus_1 - 2 * r_k)));
  int64_t n_j = r_k - n * n_i + n_i * (n_i + 1) / 2 + n_i + 1;
  int64_t n_ir = n_j - n_i - 1;
  int64_t n_jr = n - 2 - n_i;

  const scalar_t grad_lk = grad[b_l * gs_l + r_k * gs_k];
  const scalar_t dist_lk = dist[b_l * r + r_k];

  const scalar_t* const start = self + (b_l * n + n_i) * d;
  const scalar_t* const end = start + d;
  const scalar_t* self_i = start + init;
  const scalar_t* self_j = self_i + (n_j - n_i) * d;
  scalar_t* result_i = result + ((b_l * (n - 1) + n_ir) * n + n_i) * d + init;
  scalar_t* result_j = result_i + ((n_jr - n_ir) * n + n_j - n_i) * d;
  for (; self_i < end; self_i += stride,
                       self_j += stride,
                       result_i += stride,
                       result_j += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_lk, dist_lk, p);
    *result_i = res;
    *result_j = -res;
  }
}

template <typename scalar_t, typename F>
__global__ static void cdist_kernel_cuda_impl(
    scalar_t* result,
    const scalar_t* x1,
    const scalar_t* x2,
    const scalar_t p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m) {
  const int k = blockIdx.x;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const scalar_t* const start = x1 + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* a = start + threadIdx.x;
  const scalar_t* b = x2 + j * m + threadIdx.x;

  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }
  agg = reduce_agg<scalar_t, F>(agg);
  if (threadIdx.x == 0) {
    result[k] = F::finish(agg, p);
  }
}

void cdist_kernel_impl(
    Tensor& result,
    const Tensor& x1,
    const Tensor& x2,
    double p) {
  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);
  int64_t m = x1.size(-1);
  const dim3 grid(r1 * r2);
  const dim3 block(forward_threads);

  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cdist_cuda", [&] {
    if (p == 0.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<grid, block>>>(
          result.data<scalar_t>(),
          x1.data<scalar_t>(),
          x2.data<scalar_t>(),
          p,
          r1,
          r2,
          m);
    } else if (p == 1.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block>>>(
          result.data<scalar_t>(),
          x1.data<scalar_t>(),
          x2.data<scalar_t>(),
          p,
          r1,
          r2,
          m);
    } else if (p == 2.0) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block>>>(
          result.data<scalar_t>(),
          x1.data<scalar_t>(),
          x2.data<scalar_t>(),
          p,
          r1,
          r2,
          m);
    } else if (std::isinf(p)) {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block>>>(
          result.data<scalar_t>(),
          x1.data<scalar_t>(),
          x2.data<scalar_t>(),
          p,
          r1,
          r2,
          m);
    } else {
      cdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block>>>(
          result.data<scalar_t>(),
          x1.data<scalar_t>(),
          x2.data<scalar_t>(),
          p,
          r1,
          r2,
          m);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());
}

void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, double p) {
  int64_t b = self.size(0);
  int64_t n = self.size(1);
  int64_t d = self.size(2);
  int64_t r = result.size(1);

  AT_CHECK(
      b < (int32_t(1) << 16),
      "The number of batches can't exceed ",
      (int32_t(1) << 16) - 1,
      " but was ",
      b);
  AT_CHECK(
      r < (int64_t(1) << 32),
      "The number of combinations can't exceed ",
      (int64_t(1) << 32) - 1,
      " but was ",
      r);

  const dim3 grid(r, b);
  const dim3 block(forward_threads);

  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do
  // on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_cuda", [&] {
    if (p == 0.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::zero><<<grid, block>>>(
          result.data<scalar_t>(),
          self.data<scalar_t>(),
          n,
          d,
          p,
          n2,
          n2_squared_minus_1);
    } else if (p == 1.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::one><<<grid, block>>>(
          result.data<scalar_t>(),
          self.data<scalar_t>(),
          n,
          d,
          p,
          n2,
          n2_squared_minus_1);
    } else if (p == 2.0) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::two><<<grid, block>>>(
          result.data<scalar_t>(),
          self.data<scalar_t>(),
          n,
          d,
          p,
          n2,
          n2_squared_minus_1);
    } else if (std::isinf(p)) {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf><<<grid, block>>>(
          result.data<scalar_t>(),
          self.data<scalar_t>(),
          n,
          d,
          p,
          n2,
          n2_squared_minus_1);
    } else {
      pdist_kernel_cuda_impl<scalar_t, dists<scalar_t>::p><<<grid, block>>>(
          result.data<scalar_t>(),
          self.data<scalar_t>(),
          n,
          d,
          p,
          n2,
          n2_squared_minus_1);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());
}

void pdist_backward_kernel_impl(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return;
  }
  // Be careful with changing these as they affect the maximum dimensions that
  // backward can run on, however these are currently more permissive than the
  // forward pass
  const int block_x = 16;
  const int block_y = 64;

  const int64_t b = self.size(0);
  const int64_t n = self.size(1);
  const int64_t d = self.size(2);
  const int64_t r = dist.size(1);

  AT_CHECK(
      b < (int32_t(1) << 16),
      "The number of batches can't exceed ",
      (int32_t(1) << 16) - 1,
      " but was ",
      b);
  AT_CHECK(
      r < (int64_t(1) << 32) * block_x,
      "The number of combinations can't exceed ",
      (int64_t(1) << 32) * block_x - 1,
      " but was ",
      r);
  AT_CHECK(
      d < (int32_t(1) << 16) * block_y * 8,
      "The number of dimensions can't exceed ",
      (int32_t(1) << 16) * block_y * 8 - 1,
      " but was ",
      d);

  const int grid_x = (r + block_x - 1) / block_x;
  const int grid_y = (d + block_y * 8 - 1) / (block_y * 8);
  const dim3 grid(grid_x, grid_y, b);
  const dim3 block(block_x, block_y);

  // https://github.com/pytorch/pytorch/issues/15511 demonstrated we need to do
  // some math in fp64 -- this is just minimizing the amount of fp64 math we do
  // on the device.
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  Tensor buffer = at::empty({b, n - 1, n, d}, result.options());
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_cuda_backward", [&] {
    if (p == 1.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              self.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(0),
              grad.stride(1),
              n,
              d,
              r,
              p,
              n2,
              n2_squared_minus_1);
    } else if (p < 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              self.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(0),
              grad.stride(1),
              n,
              d,
              r,
              p,
              n2,
              n2_squared_minus_1);
    } else if (p == 2.0) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              self.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(0),
              grad.stride(1),
              n,
              d,
              r,
              p,
              n2,
              n2_squared_minus_1);
    } else if (std::isinf(p)) {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              self.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(0),
              grad.stride(1),
              n,
              d,
              r,
              p,
              n2,
              n2_squared_minus_1);
    } else {
      pdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              self.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(0),
              grad.stride(1),
              n,
              d,
              r,
              p,
              n2,
              n2_squared_minus_1);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());

  at::sum_out(result, buffer, 1);
}

void cdist_backward_kernel_impl(
    Tensor& result,
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0 || x2.numel() == 0) {
    result.fill_(0);
    return;
  }

  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int block_x = 64;
  const int block_y = 16;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);
  const int grid_y = (dist.numel() + block_y - 1) / block_y;

  const dim3 grid(grid_x, grid_y);
  const dim3 block(block_x, block_y);

  const int64_t count = dist.numel();

  Tensor buffer = at::empty({r2, r1, m}, result.options());
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cdist_cuda_backward", [&] {
    if (p == 1.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::one>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              x1.data<scalar_t>(),
              x2.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(1),
              p,
              r1,
              r2,
              m,
              count);
    } else if (p < 2.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::lt_two>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              x1.data<scalar_t>(),
              x2.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(1),
              p,
              r1,
              r2,
              m,
              count);
    } else if (p == 2.0) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::two>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              x1.data<scalar_t>(),
              x2.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(1),
              p,
              r1,
              r2,
              m,
              count);
    } else if (std::isinf(p)) {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::inf>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              x1.data<scalar_t>(),
              x2.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(1),
              p,
              r1,
              r2,
              m,
              count);
    } else {
      cdist_backward_kernel_cuda_impl<scalar_t, dists<scalar_t>::p>
          <<<grid, block>>>(
              buffer.data<scalar_t>(),
              grad.data<scalar_t>(),
              x1.data<scalar_t>(),
              x2.data<scalar_t>(),
              dist.data<scalar_t>(),
              grad.stride(1),
              p,
              r1,
              r2,
              m,
              count);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());

  at::sum_out(result, buffer, 0);
}

} // anonymous namespace

REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
REGISTER_DISPATCH(pdist_backward_stub, &pdist_backward_kernel_impl);
REGISTER_DISPATCH(cdist_stub, &cdist_kernel_impl);
REGISTER_DISPATCH(cdist_backward_stub, &cdist_backward_kernel_impl);

} // namespace native
} // namespace at
