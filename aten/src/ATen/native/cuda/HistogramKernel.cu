#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Histogram.h>
#include <c10/cuda/CUDAGuard.h>
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

template <typename scalar_t, bool linear_bins>
__device__ int64_t histogram_bin(
    scalar_t value,
    const scalar_t* edges,
    int64_t num_edges) {
  const int64_t num_bins = num_edges - 1;
  if (value == edges[num_bins]) {
    return num_bins - 1;
  }

  if constexpr (linear_bins) {
    const scalar_t estimate =
        (value - edges[0]) / (edges[num_bins] - edges[0]) * num_bins;
    if (estimate >= 0 && estimate < num_bins) {
      const int64_t pos = static_cast<int64_t>(estimate);
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

  int64_t first = 0;
  int64_t last = num_edges;
  while (first < last) {
    const int64_t mid = first + (last - first) / 2;
    if (value < edges[mid]) {
      last = mid;
    } else {
      first = mid + 1;
    }
  }
  return first - 1;
}

template <typename scalar_t, bool linear_bins, bool shared_histogram>
__global__ void histogramdd_cuda_kernel(
    const scalar_t* input,
    int64_t num_samples,
    int64_t dimensions,
    int64_t sample_stride,
    int64_t dimension_stride,
    const scalar_t* weight,
    int64_t weight_stride,
    const scalar_t* edges,
    const int64_t* edge_offsets,
    scalar_t* histogram,
    int64_t num_bins) {
  extern __shared__ __align__(8) unsigned char shared_memory[];
  scalar_t* accumulation = histogram;
  if constexpr (shared_histogram) {
    accumulation = reinterpret_cast<scalar_t*>(shared_memory);
    for (int64_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
      accumulation[bin] = 0;
    }
    __syncthreads();
  }

  for (int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       sample < num_samples;
       sample += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t histogram_index = 0;
    bool in_range = true;
    for (int64_t dim = 0; dim < dimensions; ++dim) {
      const scalar_t value = input[sample * sample_stride + dim * dimension_stride];
      const scalar_t* dim_edges = edges + edge_offsets[dim];
      const int64_t num_edges = edge_offsets[dim + 1] - edge_offsets[dim];
      // This also excludes NaN samples, matching the CPU implementation.
      if (!(value >= dim_edges[0] && value <= dim_edges[num_edges - 1])) {
        in_range = false;
        break;
      }
      const int64_t bin = histogram_bin<scalar_t, linear_bins>(value, dim_edges, num_edges);
      histogram_index = histogram_index * (num_edges - 1) + bin;
    }
    if (in_range) {
      const scalar_t sample_weight = weight ? weight[sample * weight_stride] : scalar_t(1);
      gpuAtomicAddNoReturn(accumulation + histogram_index, sample_weight);
    }
  }

  if constexpr (shared_histogram) {
    __syncthreads();
    for (int64_t bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
      gpuAtomicAddNoReturn(histogram + bin, accumulation[bin]);
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
  TORCH_CHECK(hist.device() == self.device(), "torch.histogram: hist and input must be on the same device");
  if (weight.has_value()) {
    TORCH_CHECK(weight->device() == self.device(), "torch.histogram: weight and input must be on the same device");
  }
  for (const auto& edges : bin_edges) {
    TORCH_CHECK(edges.device() == self.device(), "torch.histogram: bin edges and input must be on the same device");
  }

  const int64_t dimensions = self.size(-1);
  const int64_t num_samples = std::accumulate(
      self.sizes().begin(), self.sizes().end() - 1, int64_t(1), std::multiplies<int64_t>());
  const Tensor input = self.reshape({num_samples, dimensions});
  const Tensor weights = weight.has_value() ? weight->reshape({num_samples}) : Tensor();
  Tensor histogram = hist.is_contiguous() ? hist : at::empty(hist.sizes(), hist.options());
  histogram.zero_();

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histogram_cuda", [&]() {
    if (dimensions > 0 && num_samples > 0) {
      Tensor host_offsets = at::empty(
          {dimensions + 1}, at::TensorOptions().device(kCPU).dtype(kLong).pinned_memory(true));
      int64_t* offsets = host_offsets.mutable_data_ptr<int64_t>();
      offsets[0] = 0;
      for (const auto dim : c10::irange(dimensions)) {
        offsets[dim + 1] = offsets[dim] + bin_edges[dim].numel();
      }
      const Tensor edges = (dimensions == 1 ? bin_edges[0] : at::cat(bin_edges)).contiguous();
      const Tensor edge_offsets = host_offsets.to(self.device(), /*non_blocking=*/true);
      const auto* properties = at::cuda::getCurrentDeviceProperties();
      const int blocks = static_cast<int>(std::min<int64_t>(
          (num_samples - 1) / histogram_threads + 1, properties->multiProcessorCount * 4));
      const auto stream = at::cuda::getCurrentCUDAStream();
      const auto shared_bytes = histogram.numel() * sizeof(scalar_t);
      const scalar_t* weight_data = weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr;
      const int64_t weight_stride = weights.defined() ? weights.stride(0) : 0;

      if (shared_bytes <= properties->sharedMemPerBlock) {
        histogramdd_cuda_kernel<scalar_t, linear_bins, true>
            <<<blocks, histogram_threads, shared_bytes, stream>>>(
                input.const_data_ptr<scalar_t>(), num_samples, dimensions, input.stride(0), input.stride(1),
                weight_data, weight_stride, edges.const_data_ptr<scalar_t>(), edge_offsets.const_data_ptr<int64_t>(),
                histogram.mutable_data_ptr<scalar_t>(), histogram.numel());
      } else {
        histogramdd_cuda_kernel<scalar_t, linear_bins, false>
            <<<blocks, histogram_threads, 0, stream>>>(
                input.const_data_ptr<scalar_t>(), num_samples, dimensions, input.stride(0), input.stride(1),
                weight_data, weight_stride, edges.const_data_ptr<scalar_t>(), edge_offsets.const_data_ptr<int64_t>(),
                histogram.mutable_data_ptr<scalar_t>(), histogram.numel());
      }
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
    bool /*local_search*/) {
  histogramdd_out_cuda_template<true>(self, weight, density, hist, bin_edges);
}

void histogram_select_outer_bin_edges_impl(
    const Tensor& input,
    const int64_t dimensions,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto [min, max] = at::aminmax(input, 0);
  const Tensor min_cpu = min.to(kCPU);
  const Tensor max_cpu = max.to(kCPU);
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "histogram_cuda", [&]() {
    const scalar_t* min_data = min_cpu.const_data_ptr<scalar_t>();
    const scalar_t* max_data = max_cpu.const_data_ptr<scalar_t>();
    std::copy(min_data, min_data + dimensions, leftmost_edges.begin());
    std::copy(max_data, max_data + dimensions, rightmost_edges.begin());
  });
}

} // namespace

REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel_impl)
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel_impl)
REGISTER_DISPATCH(histogram_select_outer_bin_edges_stub, &histogram_select_outer_bin_edges_impl)

} // namespace at::native
