#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Histogram.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at::native {
namespace {

constexpr int histogram_threads = 256;
constexpr int histogram_blocks_per_sm = 4;

template <typename scalar_t, typename index_t, bool linear_bins>
__device__ index_t histogram_bin(
    scalar_t value,
    const scalar_t* edges,
    index_t num_edges) {
  const index_t num_bins = num_edges - 1;
  if (value == edges[num_bins]) {
    return num_bins - 1;
  }

  if constexpr (linear_bins) {
    // Above roughly 2^24 bins, float rounding can require the binary-search fallback.
    // Validate the estimate against the original scalar_t edges.
    const float estimate = static_cast<float>(value - edges[0]) /
        static_cast<float>(edges[num_bins] - edges[0]) * num_bins;
    if (estimate >= 0 && estimate < num_bins) {
      const index_t pos = static_cast<index_t>(estimate);
      if (value >= edges[pos] && value < edges[pos + 1]) {
        return pos;
      }
      if (pos > 0 && value >= edges[pos - 1] && value < edges[pos]) {
        return pos - 1;
      }
      if (pos + 1 < num_bins && value >= edges[pos + 1] &&
          value < edges[pos + 2]) {
        return pos + 1;
      }
    }
    // Fall back for overflow or repeated edges caused by rounding linspace.
  }

  index_t first = 0;
  index_t last = num_edges;
  while (first < last) {
    const index_t mid = c10::cuda::compat::midpoint(first, last);
    if (value < edges[mid]) {
      last = mid;
    } else {
      first = mid + 1;
    }
  }
  return first - 1;
}

template <
    typename scalar_t,
    typename index_t,
    bool linear_bins,
    bool one_dimensional,
    bool weighted,
    bool shared_histogram>
__global__ C10_LAUNCH_BOUNDS_1(histogram_threads) void histogramdd_cuda_kernel(
    const scalar_t* input,
    int64_t num_samples,
    index_t dimensions,
    index_t sample_stride,
    index_t dimension_stride,
    const scalar_t* weight,
    index_t weight_stride,
    const scalar_t* edges,
    const int64_t* edge_offsets,
    scalar_t* histogram,
    index_t num_bins,
    int replicas) {
  extern __shared__ __align__(8) unsigned char shared_memory[];
  scalar_t* accumulation = histogram;
  if constexpr (shared_histogram) {
    auto* shared_hist = reinterpret_cast<scalar_t*>(shared_memory);
    for (index_t bin = threadIdx.x; bin < replicas * num_bins; bin += blockDim.x) {
      shared_hist[bin] = 0;
    }
    accumulation = shared_hist + ((threadIdx.x / C10_WARP_SIZE) & (replicas - 1)) * num_bins;
    __syncthreads();
  }

  const index_t dims = one_dimensional ? 1 : dimensions;
  CUDA_KERNEL_LOOP_TYPE(sample, num_samples, index_t) {
    index_t histogram_index = 0;
    bool in_range = true;
    for (index_t dim = 0; dim < dims; ++dim) {
      const scalar_t value = input[sample * sample_stride + dim * dimension_stride];
      const scalar_t* dim_edges;
      index_t num_edges;
      if constexpr (one_dimensional) {
        dim_edges = edges;
        num_edges = num_bins + 1;
      } else {
        dim_edges = edges + static_cast<index_t>(edge_offsets[dim]);
        num_edges = static_cast<index_t>(edge_offsets[dim + 1] - edge_offsets[dim]);
      }
      // This also excludes NaN samples, matching the CPU implementation.
      if (!(value >= dim_edges[0] && value <= dim_edges[num_edges - 1])) {
        in_range = false;
        break;
      }
      const index_t bin = histogram_bin<scalar_t, index_t, linear_bins>(value, dim_edges, num_edges);
      histogram_index = histogram_index * (num_edges - 1) + bin;
    }
    if (in_range) {
      scalar_t value = weighted ? weight[sample * weight_stride] : scalar_t(1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(USE_ROCM)
      if constexpr (weighted) {
        // Reduce uniform warps without serializing their floating-point atomics.
        const auto peers = __activemask();
        if (peers == 0xffffffffu &&
            __all_sync(peers, histogram_index == WARP_SHFL(histogram_index, 0, C10_WARP_SIZE, peers))) {
          for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
            value += WARP_SHFL_DOWN(value, offset, C10_WARP_SIZE, peers);
          }
          if (threadIdx.x % C10_WARP_SIZE == 0) {
            gpuAtomicAddNoReturn(accumulation + histogram_index, value);
          }
        } else {
          gpuAtomicAddNoReturn(accumulation + histogram_index, value);
        }
      } else {
        // Unit weights need only a peer count, not a floating-point reduction.
        const auto peers = __match_any_sync(__activemask(), histogram_index);
        if ((threadIdx.x % C10_WARP_SIZE) == __ffs(peers) - 1) {
          gpuAtomicAddNoReturn(accumulation + histogram_index, static_cast<scalar_t>(__popc(peers)));
        }
      }
#else
      // ROCm and pre-Volta CUDA use per-sample atomics with shared-memory replicas.
      gpuAtomicAddNoReturn(accumulation + histogram_index, value);
#endif
    }
  }

  if constexpr (shared_histogram) {
    __syncthreads();
    const auto* shared_hist = reinterpret_cast<const scalar_t*>(shared_memory);
    for (index_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
      scalar_t total = 0;
      for (int replica = 0; replica < replicas; ++replica) {
        total += shared_hist[replica * num_bins + bin];
      }
      gpuAtomicAddNoReturn(histogram + bin, total);
    }
  }
}

template <bool linear_bins>
void histogramdd_out_cuda_template(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges) {
  const c10::cuda::CUDAGuard device_guard(self.device());
  globalContext().alertNotDeterministic("histogram_cuda");

  const int64_t dimensions = self.size(-1);
  const int64_t num_samples = std::accumulate(
      self.sizes().begin(), self.sizes().end() - 1, int64_t(1), std::multiplies<int64_t>());
  const Tensor input = self.reshape({num_samples, dimensions});
  const Tensor weights = weight.has_value() ? weight->reshape({num_samples}) : Tensor();
  Tensor histogram = hist.is_contiguous() ? hist : at::empty(hist.sizes(), hist.options());
  histogram.zero_();

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histogram_cuda", [&]() {
    if (dimensions > 0 && num_samples > 0) {
      Tensor edges;
      Tensor edge_offsets;
      Tensor host_offsets; // Keep pinned metadata alive through the kernel launch.
      if (dimensions == 1) {
        edges = bin_edges[0].contiguous();
      } else {
        host_offsets = at::empty(
            {dimensions + 1}, at::TensorOptions().device(kCPU).dtype(kLong).pinned_memory(true));
        int64_t* offsets = host_offsets.mutable_data_ptr<int64_t>();
        offsets[0] = 0;
        for (const auto dim : c10::irange(dimensions)) {
          offsets[dim + 1] = offsets[dim] + bin_edges[dim].numel();
        }
        edges = at::cat(bin_edges);
        edge_offsets = host_offsets.to(self.device(), /*non_blocking=*/true);
      }
      const auto* properties = at::cuda::getCurrentDeviceProperties();
      const int blocks = static_cast<int>(std::min<int64_t>(
          (num_samples - 1) / histogram_threads + 1, properties->multiProcessorCount * histogram_blocks_per_sm));
      const auto stream = at::cuda::getCurrentCUDAStream();
      const auto shared_bytes = histogram.numel() * sizeof(scalar_t);
      const size_t replica_budget = std::min<size_t>(
          properties->sharedMemPerBlock,
          properties->sharedMemPerMultiprocessor / histogram_blocks_per_sm);
      int replicas = 1;
      while (replicas * 2 <= histogram_threads / properties->warpSize &&
             shared_bytes * replicas * 2 <= replica_budget) {
        replicas *= 2;
      }
      const scalar_t* weight_data = weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr;
      const int64_t weight_stride = weights.defined() ? weights.stride(0) : 0;
      const int64_t* offsets = edge_offsets.defined() ? edge_offsets.const_data_ptr<int64_t>() : nullptr;
      const bool use_32bit = canUse32BitIndexMath(input) && canUse32BitIndexMath(histogram) &&
          canUse32BitIndexMath(edges) && (!weights.defined() || canUse32BitIndexMath(weights));
      AT_DISPATCH_INDEX_TYPES(use_32bit ? kInt : kLong, "histogram_cuda_index", [&]() {
        auto launch = [&]<bool one_dimensional, bool weighted>() {
          if (shared_bytes <= properties->sharedMemPerBlock) {
            histogramdd_cuda_kernel<scalar_t, index_t, linear_bins, one_dimensional, weighted, true>
                <<<blocks, histogram_threads, shared_bytes * replicas, stream>>>(
                    input.const_data_ptr<scalar_t>(), num_samples, dimensions, input.stride(0), input.stride(1),
                    weight_data, weight_stride, edges.const_data_ptr<scalar_t>(), offsets,
                    histogram.mutable_data_ptr<scalar_t>(), histogram.numel(), replicas);
          } else {
            histogramdd_cuda_kernel<scalar_t, index_t, linear_bins, one_dimensional, weighted, false>
                <<<blocks, histogram_threads, 0, stream>>>(
                    input.const_data_ptr<scalar_t>(), num_samples, dimensions, input.stride(0), input.stride(1),
                    weight_data, weight_stride, edges.const_data_ptr<scalar_t>(), offsets,
                    histogram.mutable_data_ptr<scalar_t>(), histogram.numel(), 1);
          }
        };
        if (dimensions == 1) {
          if (weights.defined()) {
            launch.template operator()<true, true>();
          } else {
            launch.template operator()<true, false>();
          }
        } else {
          if (weights.defined()) {
            launch.template operator()<false, true>();
          } else {
            launch.template operator()<false, false>();
          }
        }
      });
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });

  if (density) {
    histogram.div_(histogram.sum());
    for (const auto dim : c10::irange(dimensions)) {
      const Tensor bin_lengths = bin_edges[dim].diff();
      std::vector<int64_t> shape(dimensions, 1);
      shape[dim] = bin_lengths.numel();
      histogram.div_(bin_lengths.reshape(shape));
    }
  }
  if (!hist.is_contiguous()) {
    hist.copy_(histogram);
  }
}

void histogramdd_kernel_impl(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges) {
  histogramdd_out_cuda_template<false>(self, weight, density, hist, bin_edges);
}

void histogramdd_linear_kernel_impl(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges,
    bool local_search) {
  // histc uses a separate CUDA kernel; this stub always checks returned edges.
  TORCH_INTERNAL_ASSERT(local_search);
  histogramdd_out_cuda_template<true>(self, weight, density, hist, bin_edges);
}

void histogram_select_outer_bin_edges_impl(
    const Tensor& input,
    const int64_t dimensions,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  Tensor extrema = at::empty({2, dimensions}, input.options());
  Tensor min = extrema.select(0, 0);
  Tensor max = extrema.select(0, 1);
  at::aminmax_out(min, max, input, 0);
  const Tensor extrema_cpu = extrema.to(kCPU);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogram_cuda", [&]() {
    const scalar_t* min_data = extrema_cpu.const_data_ptr<scalar_t>();
    const scalar_t* max_data = min_data + dimensions;
    std::copy(min_data, min_data + dimensions, leftmost_edges.begin());
    std::copy(max_data, max_data + dimensions, rightmost_edges.begin());
  });
}

} // namespace

REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel_impl)
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel_impl)
REGISTER_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_impl)

} // namespace at::native
