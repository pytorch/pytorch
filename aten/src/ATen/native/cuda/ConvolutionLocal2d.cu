#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/ConvolutionLocal2d.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv2d_local_backward_native.h>
#include <ATen/ops/_conv2d_local_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {
using at::cuda::detail::CUDA_NUM_THREADS;
using at::cuda::detail::GET_BLOCKS;

// Every kernel below treats one output position (oh, ow) as an independent
// small GEMM against that position's [C_out, K] weight slab, K = C_in*kH*kW.
// The input patch for the position is gathered straight into shared memory,
// so no im2col buffer is ever materialised. Tiles are runtime-sized so the
// block always has kThreads threads; each thread owns kRows outputs.
constexpr int kThreads = 256;
constexpr int kChunk = 32;
constexpr int kRows = 4;
constexpr int kMaxPositions = 16;
constexpr int kSmallBatch = 4;
constexpr size_t kMaxSmem = 48 * 1024;

struct Conv2dLocalDims {
  int batch;
  int in_channels;
  int in_height;
  int in_width;
  int out_channels;
  int out_height;
  int out_width;
  int kernel_height;
  int kernel_width;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;
  int dilation_height;
  int dilation_width;
  int kernel_numel;
  int out_numel_per_channel;
};

Conv2dLocalDims to_dims(const Conv2dLocalParams& p) {
  auto narrow = [](int64_t v) {
    TORCH_CHECK(v <= std::numeric_limits<int>::max(), "conv2d_local: dimension ", v, " is too large for the CUDA kernel");
    return static_cast<int>(v);
  };
  return Conv2dLocalDims{
      narrow(p.batch),
      narrow(p.in_channels),
      narrow(p.in_height),
      narrow(p.in_width),
      narrow(p.out_channels),
      narrow(p.out_height),
      narrow(p.out_width),
      narrow(p.kernel_height),
      narrow(p.kernel_width),
      narrow(p.stride_height),
      narrow(p.stride_width),
      narrow(p.pad_height),
      narrow(p.pad_width),
      narrow(p.dilation_height),
      narrow(p.dilation_width),
      narrow(p.kernel_numel),
      narrow(p.out_numel_per_channel)};
}

// Smallest power of two >= n, clamped to [lo, hi]; the tile width along a dim.
int tile_width(int n, int lo, int hi) {
  int t = lo;
  while (t < n && t < hi) {
    t *= 2;
  }
  return t;
}

// Outputs per thread along the reduced-over dim: enough rows*R to cover it,
// capped at kRows, then halved until the shared-memory tiles fit.
template <typename SmemBytes>
int pick_rows(int extent, int rows, SmemBytes smem_bytes) {
  int r = (extent + rows - 1) / rows;
  r = r >= kRows ? kRows : (r >= 2 ? 2 : 1);
  while (r > 1 && smem_bytes(r) > kMaxSmem) {
    r /= 2;
  }
  TORCH_CHECK(smem_bytes(r) <= kMaxSmem, "conv2d_local: tile does not fit in shared memory");
  return r;
}

// Offset of input element k of the patch at output position (oh, ow) for batch
// row 0, or -1 when it falls in the zero padding. Callers add n * batch_stride.
__device__ __forceinline__ int64_t patch_offset(const Conv2dLocalDims& d, int k, int oh, int ow) {
  const int kw = k % d.kernel_width;
  const int rest = k / d.kernel_width;
  const int kh = rest % d.kernel_height;
  const int ic = rest / d.kernel_height;
  const int ih = oh * d.stride_height - d.pad_height + kh * d.dilation_height;
  const int iw = ow * d.stride_width - d.pad_width + kw * d.dilation_width;
  if (ih < 0 || ih >= d.in_height || iw < 0 || iw >= d.in_width) {
    return -1;
  }
  return (static_cast<int64_t>(ic) * d.in_height + ih) * d.in_width + iw;
}

// out[n, co, l] = bias[co, l] + sum_k patch[n, k] * w[l, co, k]
// Block: P consecutive positions starting at blockIdx.x*P, n-tile blockIdx.y
// (tn = rows*R wide, rows = thread rows per position), co-tile blockIdx.z (tc
// wide). Thread (p, tr, tci) owns n = n0 + tr + r*rows at position l = l0 + p.
// P > 1 only when the batch is too small to fill a block with one position.
template <typename scalar_t, int R>
__global__ void __launch_bounds__(kThreads) conv2d_local_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    Conv2dLocalDims d,
    int tc,
    int P) {
  using acc_t = at::acc_type<scalar_t, true>;
  extern __shared__ char smem_raw[];
  const int rows = kThreads / (tc * P);
  const int tn = rows * R;
  const int a_slot = kChunk * (tn + 1);
  const int b_slot = kChunk * (tc + 1);
  acc_t* As = reinterpret_cast<acc_t*>(smem_raw); // [P][kChunk][tn + 1]
  acc_t* Bs = As + P * a_slot; // [P][kChunk][tc + 1]
  int64_t* offs = reinterpret_cast<int64_t*>(Bs + P * b_slot); // [P][kChunk] patch offsets for n = 0
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;

  const int L = d.out_numel_per_channel;
  const int l0 = blockIdx.x * P;
  const int n0 = blockIdx.y * tn;
  const int co0 = blockIdx.z * tc;
  const int K = d.kernel_numel;

  const int t = threadIdx.x;
  const int p = t / (tc * rows);
  const int ts = t % (tc * rows);
  const int tci = ts % tc;
  const int tr = ts / tc;
  const int l = l0 + p;
  acc_t acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc[r] = acc_t(0);
  }

  for (int k0 = 0; k0 < K; k0 += kChunk) {
    for (int i = t; i < P * kChunk; i += kThreads) {
      const int kk = i % kChunk;
      const int li = l0 + i / kChunk;
      const int k = k0 + kk;
      offs[i] = (k < K && li < L) ? patch_offset(d, k, li / d.out_width, li % d.out_width) : int64_t(-1);
    }
    for (int i = t; i < P * tc * kChunk; i += kThreads) {
      const int kk = i % kChunk;
      const int cc = (i / kChunk) % tc;
      const int pi = i / (kChunk * tc);
      const int k = k0 + kk;
      const int co = co0 + cc;
      const int li = l0 + pi;
      Bs[pi * b_slot + kk * (tc + 1) + cc] = (k < K && co < d.out_channels && li < L)
          ? static_cast<acc_t>(weight[(static_cast<int64_t>(li) * d.out_channels + co) * K + k])
          : acc_t(0);
    }
    __syncthreads();
    for (int i = t; i < P * tn * kChunk; i += kThreads) {
      const int kk = i % kChunk;
      const int nn = (i / kChunk) % tn;
      const int pi = i / (kChunk * tn);
      const int n = n0 + nn;
      const int64_t off = offs[pi * kChunk + kk];
      As[pi * a_slot + kk * (tn + 1) + nn] = (off >= 0 && n < d.batch)
          ? static_cast<acc_t>(input[off + n * batch_stride])
          : acc_t(0);
    }
    __syncthreads();
    const int kmax = min(kChunk, K - k0);
    const acc_t* a_base = As + p * a_slot + tr;
    const acc_t* b_base = Bs + p * b_slot + tci;
    for (int kk = 0; kk < kmax; ++kk) {
      const acc_t b = b_base[kk * (tc + 1)];
      const acc_t* a = a_base + kk * (tn + 1);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r] += a[r * rows] * b;
      }
    }
    __syncthreads();
  }

  const int co = co0 + tci;
  if (co >= d.out_channels || l >= L) {
    return;
  }
  const acc_t b = bias ? static_cast<acc_t>(bias[static_cast<int64_t>(co) * L + l]) : acc_t(0);
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = n0 + tr + r * rows;
    if (n < d.batch) {
      output[(static_cast<int64_t>(n) * d.out_channels + co) * L + l] = static_cast<scalar_t>(acc[r] + b);
    }
  }
}

template <typename acc_t>
__device__ __forceinline__ acc_t warp_sum(acc_t v) {
#pragma unroll
  for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
    v += WARP_SHFL_DOWN(v, offset);
  }
  return v;
}

// Batches of at most kSmallBatch use every weight element only a few times,
// so shared-memory staging is pure overhead. These kernels stream the weight
// slab exactly once with lanes striding over k, warp per (l, co) or per l.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) conv2d_local_forward_small_batch_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    Conv2dLocalDims d) {
  using acc_t = at::acc_type<scalar_t, true>;
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (kThreads / C10_WARP_SIZE) + threadIdx.x / C10_WARP_SIZE;
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int L = d.out_numel_per_channel;
  if (warp >= static_cast<int64_t>(L) * d.out_channels) {
    return;
  }
  const int l = warp / d.out_channels;
  const int co = warp % d.out_channels;
  const int oh = l / d.out_width;
  const int ow = l % d.out_width;
  const int K = d.kernel_numel;
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;
  const scalar_t* wl = weight + (static_cast<int64_t>(l) * d.out_channels + co) * K;
  acc_t acc[kSmallBatch];
#pragma unroll
  for (int n = 0; n < kSmallBatch; ++n) {
    acc[n] = acc_t(0);
  }
  for (int k = lane; k < K; k += C10_WARP_SIZE) {
    const acc_t w = static_cast<acc_t>(wl[k]);
    const int64_t off = patch_offset(d, k, oh, ow);
    if (off >= 0) {
#pragma unroll
      for (int n = 0; n < kSmallBatch; ++n) {
        if (n < d.batch) {
          acc[n] += w * static_cast<acc_t>(input[off + n * batch_stride]);
        }
      }
    }
  }
  const acc_t b = bias ? static_cast<acc_t>(bias[static_cast<int64_t>(co) * L + l]) : acc_t(0);
#pragma unroll
  for (int n = 0; n < kSmallBatch; ++n) {
    const acc_t total = warp_sum(acc[n]);
    if (lane == 0 && n < d.batch) {
      output[(static_cast<int64_t>(n) * d.out_channels + co) * L + l] = static_cast<scalar_t>(total + b);
    }
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) conv2d_local_grad_weight_small_batch_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_weight,
    Conv2dLocalDims d) {
  using acc_t = at::acc_type<scalar_t, true>;
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (kThreads / C10_WARP_SIZE) + threadIdx.x / C10_WARP_SIZE;
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int L = d.out_numel_per_channel;
  if (warp >= static_cast<int64_t>(L) * d.out_channels) {
    return;
  }
  const int l = warp / d.out_channels;
  const int co = warp % d.out_channels;
  const int oh = l / d.out_width;
  const int ow = l % d.out_width;
  const int K = d.kernel_numel;
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;
  acc_t go[kSmallBatch];
#pragma unroll
  for (int n = 0; n < kSmallBatch; ++n) {
    go[n] = n < d.batch ? static_cast<acc_t>(grad_output[(static_cast<int64_t>(n) * d.out_channels + co) * L + l]) : acc_t(0);
  }
  scalar_t* gw = grad_weight + (static_cast<int64_t>(l) * d.out_channels + co) * K;
  for (int k = lane; k < K; k += C10_WARP_SIZE) {
    acc_t acc(0);
    const int64_t off = patch_offset(d, k, oh, ow);
    if (off >= 0) {
#pragma unroll
      for (int n = 0; n < kSmallBatch; ++n) {
        if (n < d.batch) {
          acc += go[n] * static_cast<acc_t>(input[off + n * batch_stride]);
        }
      }
    }
    gw[k] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename acc_t>
__global__ void __launch_bounds__(kThreads) conv2d_local_grad_input_small_batch_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    acc_t* __restrict__ grad_input,
    Conv2dLocalDims d) {
  const int64_t warp = static_cast<int64_t>(blockIdx.x) * (kThreads / C10_WARP_SIZE) + threadIdx.x / C10_WARP_SIZE;
  const int lane = threadIdx.x % C10_WARP_SIZE;
  const int L = d.out_numel_per_channel;
  if (warp >= L) {
    return;
  }
  const int l = warp;
  const int oh = l / d.out_width;
  const int ow = l % d.out_width;
  const int K = d.kernel_numel;
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;
  const scalar_t* wl = weight + static_cast<int64_t>(l) * d.out_channels * K;
  const scalar_t* gol = grad_output + l;
  for (int k = lane; k < K; k += C10_WARP_SIZE) {
    const int64_t off = patch_offset(d, k, oh, ow);
    if (off < 0) {
      continue;
    }
    acc_t acc[kSmallBatch];
#pragma unroll
    for (int n = 0; n < kSmallBatch; ++n) {
      acc[n] = acc_t(0);
    }
    for (int co = 0; co < d.out_channels; ++co) {
      const acc_t w = static_cast<acc_t>(wl[static_cast<int64_t>(co) * K + k]);
#pragma unroll
      for (int n = 0; n < kSmallBatch; ++n) {
        if (n < d.batch) {
          acc[n] += w * static_cast<acc_t>(gol[(static_cast<int64_t>(n) * d.out_channels + co) * L]);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < kSmallBatch; ++n) {
      if (n < d.batch) {
        gpuAtomicAddNoReturn(grad_input + off + n * batch_stride, acc[n]);
      }
    }
  }
}

// grad_w[l, co, k] = sum_n go[n, co, l] * patch[n, k]
// Block: position l, k-tile blockIdx.y (tk = cols*kRows wide), co-tile
// blockIdx.z (tc wide). Thread (tci, kc) owns k = k0 + kc + r*cols.
template <typename scalar_t, int R>
__global__ void __launch_bounds__(kThreads) conv2d_local_grad_weight_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ grad_weight,
    Conv2dLocalDims d,
    int tc) {
  using acc_t = at::acc_type<scalar_t, true>;
  extern __shared__ char smem_raw[];
  const int cols = kThreads / tc;
  const int tk = cols * R;
  acc_t* Gs = reinterpret_cast<acc_t*>(smem_raw); // [kChunk][tc + 1]  (n, co)
  acc_t* As = Gs + kChunk * (tc + 1); // [kChunk][tk + 1]  (n, k)
  int64_t* offs = reinterpret_cast<int64_t*>(As + kChunk * (tk + 1)); // [tk] patch offsets for n = 0
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;

  const int l = blockIdx.x;
  const int oh = l / d.out_width;
  const int ow = l % d.out_width;
  const int k0 = blockIdx.y * tk;
  const int co0 = blockIdx.z * tc;
  const int K = d.kernel_numel;

  const int t = threadIdx.x;
  const int tci = t % tc;
  const int kc = t / tc;
  acc_t acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc[r] = acc_t(0);
  }

  for (int i = t; i < tk; i += kThreads) {
    const int k = k0 + i;
    offs[i] = k < K ? patch_offset(d, k, oh, ow) : int64_t(-1);
  }
  __syncthreads();
  for (int nb = 0; nb < d.batch; nb += kChunk) {
    for (int i = t; i < tc * kChunk; i += kThreads) {
      const int cc = i % tc;
      const int nn = i / tc;
      const int n = nb + nn;
      const int co = co0 + cc;
      Gs[nn * (tc + 1) + cc] = (n < d.batch && co < d.out_channels)
          ? static_cast<acc_t>(grad_output[(static_cast<int64_t>(n) * d.out_channels + co) * d.out_numel_per_channel + l])
          : acc_t(0);
    }
    for (int i = t; i < tk * kChunk; i += kThreads) {
      const int kk = i % tk;
      const int nn = i / tk;
      const int n = nb + nn;
      const int64_t off = offs[kk];
      As[nn * (tk + 1) + kk] = (off >= 0 && n < d.batch)
          ? static_cast<acc_t>(input[off + n * batch_stride])
          : acc_t(0);
    }
    __syncthreads();
    const int nmax = min(kChunk, d.batch - nb);
    for (int nn = 0; nn < nmax; ++nn) {
      const acc_t g = Gs[nn * (tc + 1) + tci];
      const acc_t* a = As + nn * (tk + 1) + kc;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r] += g * a[r * cols];
      }
    }
    __syncthreads();
  }

  const int co = co0 + tci;
  if (co >= d.out_channels) {
    return;
  }
  scalar_t* gw = grad_weight + (static_cast<int64_t>(l) * d.out_channels + co) * K;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int k = k0 + kc + r * cols;
    if (k < K) {
      gw[k] = static_cast<scalar_t>(acc[r]);
    }
  }
}

// grad_in[patch(n, k)] += sum_co go[n, co, l] * w[l, co, k]   (atomic scatter)
// Block: position l, k-tile blockIdx.y (tk wide, one k per thread column),
// n-tile blockIdx.z (tn = rows*kRows wide). Thread (kc, tr) owns k = k0 + kc
// and n = n0 + tr + r*rows. Consecutive threads scatter to consecutive iw.
template <typename scalar_t, typename acc_t, int R>
__global__ void __launch_bounds__(kThreads) conv2d_local_grad_input_scatter_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    acc_t* __restrict__ grad_input,
    Conv2dLocalDims d,
    int tk) {
  extern __shared__ char smem_raw[];
  const int rows = kThreads / tk;
  const int tn = rows * R;
  acc_t* Gs = reinterpret_cast<acc_t*>(smem_raw); // [tn][kChunk + 1]  (n, co)
  acc_t* Ws = Gs + tn * (kChunk + 1); // [kChunk][tk + 1]  (co, k)

  const int l = blockIdx.x;
  const int oh = l / d.out_width;
  const int ow = l % d.out_width;
  const int k0 = blockIdx.y * tk;
  const int n0 = blockIdx.z * tn;
  const int K = d.kernel_numel;
  const scalar_t* wl = weight + static_cast<int64_t>(l) * d.out_channels * K;

  const int t = threadIdx.x;
  const int kc = t % tk;
  const int tr = t / tk;
  acc_t acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc[r] = acc_t(0);
  }

  for (int c0 = 0; c0 < d.out_channels; c0 += kChunk) {
    for (int i = t; i < tn * kChunk; i += kThreads) {
      const int cc = i % kChunk;
      const int nn = i / kChunk;
      const int n = n0 + nn;
      const int co = c0 + cc;
      Gs[nn * (kChunk + 1) + cc] = (n < d.batch && co < d.out_channels)
          ? static_cast<acc_t>(grad_output[(static_cast<int64_t>(n) * d.out_channels + co) * d.out_numel_per_channel + l])
          : acc_t(0);
    }
    for (int i = t; i < tk * kChunk; i += kThreads) {
      const int kk = i % tk;
      const int cc = i / tk;
      const int k = k0 + kk;
      const int co = c0 + cc;
      Ws[cc * (tk + 1) + kk] = (k < K && co < d.out_channels)
          ? static_cast<acc_t>(wl[static_cast<int64_t>(co) * K + k])
          : acc_t(0);
    }
    __syncthreads();
    const int cmax = min(kChunk, d.out_channels - c0);
    for (int cc = 0; cc < cmax; ++cc) {
      const acc_t w = Ws[cc * (tk + 1) + kc];
      const acc_t* g = Gs + tr * (kChunk + 1) + cc;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        acc[r] += g[r * rows * (kChunk + 1)] * w;
      }
    }
    __syncthreads();
  }

  const int k = k0 + kc;
  if (k >= K) {
    return;
  }
  const int64_t off = patch_offset(d, k, oh, ow);
  if (off < 0) {
    return;
  }
  const int64_t batch_stride = static_cast<int64_t>(d.in_channels) * d.in_height * d.in_width;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = n0 + tr + r * rows;
    if (n < d.batch) {
      gpuAtomicAddNoReturn(grad_input + off + n * batch_stride, acc[r]);
    }
  }
}

// Deterministic grad_input: one thread per input element gathering from every
// output it contributed to. Slower (weight reads are uncoalesced) but no atomics.
template <typename scalar_t>
__global__ void C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS) conv2d_local_grad_input_gather_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    int64_t numel,
    Conv2dLocalDims d) {
  using acc_t = at::acc_type<scalar_t, true>;
  const int64_t kernel_numel = d.kernel_numel;
  CUDA_KERNEL_LOOP_TYPE(i, numel, int64_t) {
    const int iw = i % d.in_width;
    int64_t rest = i / d.in_width;
    const int ih = rest % d.in_height;
    rest /= d.in_height;
    const int ic = rest % d.in_channels;
    const int n = rest / d.in_channels;
    const scalar_t* go = grad_output + static_cast<int64_t>(n) * d.out_channels * d.out_numel_per_channel;
    acc_t acc(0);
    for (int kh = 0; kh < d.kernel_height; ++kh) {
      const int oh_num = ih + d.pad_height - kh * d.dilation_height;
      if (oh_num < 0 || oh_num % d.stride_height != 0) {
        continue;
      }
      const int oh = oh_num / d.stride_height;
      if (oh >= d.out_height) {
        continue;
      }
      for (int kw = 0; kw < d.kernel_width; ++kw) {
        const int ow_num = iw + d.pad_width - kw * d.dilation_width;
        if (ow_num < 0 || ow_num % d.stride_width != 0) {
          continue;
        }
        const int ow = ow_num / d.stride_width;
        if (ow >= d.out_width) {
          continue;
        }
        const int l = oh * d.out_width + ow;
        const scalar_t* w = weight + static_cast<int64_t>(l) * d.out_channels * kernel_numel +
            (static_cast<int64_t>(ic) * d.kernel_height + kh) * d.kernel_width + kw;
        for (int oc = 0; oc < d.out_channels; ++oc) {
          acc += static_cast<acc_t>(w[oc * kernel_numel]) *
              static_cast<acc_t>(go[static_cast<int64_t>(oc) * d.out_numel_per_channel + l]);
        }
      }
    }
    grad_input[i] = static_cast<scalar_t>(acc);
  }
}

unsigned int grid_dim(int64_t total, int tile, int64_t limit = 65535) {
  const int64_t blocks = (total + tile - 1) / tile;
  TORCH_CHECK(blocks <= limit, "conv2d_local: tensor too large for the CUDA grid");
  return static_cast<unsigned int>(blocks);
}

} // namespace

Tensor conv2d_local_cuda(
    const Tensor& self,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  const auto p = conv2d_local_shape_check(self, weight, bias, Tensor(), stride, padding, dilation);
  auto input_c = self.contiguous();
  auto weight_c = weight.contiguous();
  auto bias_c = bias.defined() ? bias.contiguous() : bias;
  auto output = at::empty({p.batch, p.out_channels, p.out_height, p.out_width}, self.options());
  const auto d = to_dims(p);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_cuda", [&] {
    if (output.numel() == 0) {
      return;
    }
    using acc_t = at::acc_type<scalar_t, true>;
    if (d.batch <= kSmallBatch) {
      const int64_t warps = static_cast<int64_t>(d.out_numel_per_channel) * d.out_channels;
      const dim3 grid(grid_dim(warps, kThreads / C10_WARP_SIZE, std::numeric_limits<int>::max()));
      conv2d_local_forward_small_batch_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
          input_c.const_data_ptr<scalar_t>(),
          weight_c.const_data_ptr<scalar_t>(),
          bias_c.defined() ? bias_c.const_data_ptr<scalar_t>() : nullptr,
          output.mutable_data_ptr<scalar_t>(),
          d);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    const int tc = tile_width(d.out_channels, 4, 32);
    int rows = kThreads / tc;
    int P = std::min(kMaxPositions, rows / tile_width(d.batch, 1, rows));
    auto smem_bytes = [&](int r) {
      return sizeof(acc_t) * P * kChunk * ((kThreads / (tc * P)) * r + 1 + tc + 1) + sizeof(int64_t) * P * kChunk;
    };
    while (P > 1 && smem_bytes(1) > kMaxSmem) {
      P /= 2;
    }
    rows /= P;
    const int r = pick_rows(d.batch, rows, smem_bytes);
    const int tn = rows * r;
    const dim3 grid(grid_dim(d.out_numel_per_channel, P, std::numeric_limits<int>::max()), grid_dim(d.batch, tn), grid_dim(d.out_channels, tc));
    auto launch = [&](auto kernel) {
      kernel<<<grid, kThreads, smem_bytes(r), stream>>>(
          input_c.const_data_ptr<scalar_t>(),
          weight_c.const_data_ptr<scalar_t>(),
          bias_c.defined() ? bias_c.const_data_ptr<scalar_t>() : nullptr,
          output.mutable_data_ptr<scalar_t>(),
          d,
          tc,
          P);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (r == 4) {
      launch(conv2d_local_forward_kernel<scalar_t, 4>);
    } else if (r == 2) {
      launch(conv2d_local_forward_kernel<scalar_t, 2>);
    } else {
      launch(conv2d_local_forward_kernel<scalar_t, 1>);
    }
  });
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv2d_local_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    std::array<bool, 3> output_mask) {
  const auto p = conv2d_local_shape_check(self, weight, Tensor(), grad_output, stride, padding, dilation);
  const auto d = to_dims(p);
  auto grad_output_c = grad_output.contiguous();
  const auto stream = c10::cuda::getCurrentCUDAStream();
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    auto weight_c = weight.contiguous();
    const bool deterministic = at::globalContext().deterministicAlgorithms();
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_backward_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      if (deterministic) {
        grad_input = at::empty(self.sizes(), self.options());
        if (grad_input.numel() == 0) {
          return;
        }
        conv2d_local_grad_input_gather_kernel<scalar_t><<<GET_BLOCKS(grad_input.numel()), CUDA_NUM_THREADS, 0, stream>>>(
            grad_output_c.const_data_ptr<scalar_t>(),
            weight_c.const_data_ptr<scalar_t>(),
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_input.numel(),
            d);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
      // Atomics accumulate in acc_t; for reduced-precision dtypes that is a
      // float buffer cast back at the end.
      constexpr bool needs_buffer = !std::is_same_v<scalar_t, acc_t>;
      const auto acc_dtype = c10::CppTypeToScalarType<acc_t>::value;
      Tensor acc_buffer = at::zeros(self.sizes(), self.options().dtype(needs_buffer ? acc_dtype : self.scalar_type()));
      if (acc_buffer.numel() > 0 && grad_output_c.numel() > 0 && d.batch <= kSmallBatch) {
        const dim3 grid(grid_dim(d.out_numel_per_channel, kThreads / C10_WARP_SIZE, std::numeric_limits<int>::max()));
        conv2d_local_grad_input_small_batch_kernel<scalar_t, acc_t><<<grid, kThreads, 0, stream>>>(
            grad_output_c.const_data_ptr<scalar_t>(),
            weight_c.const_data_ptr<scalar_t>(),
            acc_buffer.mutable_data_ptr<acc_t>(),
            d);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else if (acc_buffer.numel() > 0 && grad_output_c.numel() > 0) {
        const int tk = tile_width(d.kernel_numel, 4, 32);
        const int rows = kThreads / tk;
        auto smem_bytes = [&](int r) { return sizeof(acc_t) * (rows * r * (kChunk + 1) + kChunk * (tk + 1)); };
        const int r = pick_rows(d.batch, rows, smem_bytes);
        const int tn = rows * r;
        const dim3 grid(d.out_numel_per_channel, grid_dim(d.kernel_numel, tk), grid_dim(d.batch, tn));
        auto launch = [&](auto kernel) {
          kernel<<<grid, kThreads, smem_bytes(r), stream>>>(
              grad_output_c.const_data_ptr<scalar_t>(),
              weight_c.const_data_ptr<scalar_t>(),
              acc_buffer.mutable_data_ptr<acc_t>(),
              d,
              tk);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
        if (r == 4) {
          launch(conv2d_local_grad_input_scatter_kernel<scalar_t, acc_t, 4>);
        } else if (r == 2) {
          launch(conv2d_local_grad_input_scatter_kernel<scalar_t, acc_t, 2>);
        } else {
          launch(conv2d_local_grad_input_scatter_kernel<scalar_t, acc_t, 1>);
        }
      }
      grad_input = needs_buffer ? acc_buffer.to(self.scalar_type()) : acc_buffer;
    });
  }
  if (output_mask[1]) {
    auto input_c = self.contiguous();
    grad_weight = at::empty(weight.sizes(), weight.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_backward_cuda", [&] {
      if (grad_weight.numel() == 0) {
        return;
      }
      using acc_t = at::acc_type<scalar_t, true>;
      if (d.batch <= kSmallBatch) {
        const int64_t warps = static_cast<int64_t>(d.out_numel_per_channel) * d.out_channels;
        const dim3 grid(grid_dim(warps, kThreads / C10_WARP_SIZE, std::numeric_limits<int>::max()));
        conv2d_local_grad_weight_small_batch_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
            grad_output_c.const_data_ptr<scalar_t>(),
            input_c.const_data_ptr<scalar_t>(),
            grad_weight.mutable_data_ptr<scalar_t>(),
            d);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
      }
      const int tc = tile_width(d.out_channels, 4, 32);
      const int cols = kThreads / tc;
      auto smem_bytes = [&](int r) { return sizeof(acc_t) * kChunk * (tc + 1 + cols * r + 1) + sizeof(int64_t) * cols * r; };
      const int r = pick_rows(d.kernel_numel, cols, smem_bytes);
      const int tk = cols * r;
      const dim3 grid(d.out_numel_per_channel, grid_dim(d.kernel_numel, tk), grid_dim(d.out_channels, tc));
      auto launch = [&](auto kernel) {
        kernel<<<grid, kThreads, smem_bytes(r), stream>>>(
            grad_output_c.const_data_ptr<scalar_t>(),
            input_c.const_data_ptr<scalar_t>(),
            grad_weight.mutable_data_ptr<scalar_t>(),
            d,
            tc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      };
      if (r == 4) {
        launch(conv2d_local_grad_weight_kernel<scalar_t, 4>);
      } else if (r == 2) {
        launch(conv2d_local_grad_weight_kernel<scalar_t, 2>);
      } else {
        launch(conv2d_local_grad_weight_kernel<scalar_t, 1>);
      }
  });
  }
  if (output_mask[2]) {
    grad_bias = at::sum(grad_output_c, IntArrayRef{0});
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

} // namespace at::native
