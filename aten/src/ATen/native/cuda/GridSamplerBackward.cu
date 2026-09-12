#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/GridSampler.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {
namespace {
using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;
constexpr int THREADS = 256;

// NOTE [Deterministic Grid Sampler Input Gradient]
// Each interpolation corner has a fixed record ID. Stable sorting groups these
// records by input destination without changing their order within a segment.
// One logical warp owns each segment/channel tile and uses a fixed reduction
// tree, rather than floating-point atomics. Fixed-size chunks bound temporary
// storage and commit on the caller stream in order; chunk sizes must not depend
// on free memory or scheduling. The caller initializes grad_input to zero.

// Convert a batch-local flattened spatial index using the tensor's own strides.
template <typename scalar_t, int D>
__device__ int64_t spatial_offset(int64_t index, const TensorInfo<scalar_t, int64_t>& tensor,
                                  int first_dim) {
  int64_t offset = 0;
  for (int axis = D - 1; axis >= 0; --axis) {
    offset += (index % tensor.sizes[first_dim + axis]) * tensor.strides[first_dim + axis];
    index /= tensor.sizes[first_dim + axis];
  }
  return offset;
}

// A record is identified by (batch, output spatial index, interpolation corner).
// Invalid neighbors sort after all valid destinations and never reach the output.
template <typename scalar_t, int D>
C10_LAUNCH_BOUNDS_1(THREADS)
__global__ void make_records(
    TensorInfo<const scalar_t, int64_t> input,
    TensorInfo<const scalar_t, int64_t> grid,
    TensorInfo<const scalar_t, int64_t> grad,
    int64_t* keys, int64_t* ids, scalar_t* weights, int64_t* sample_offsets,
    int64_t count, int64_t first_output, int64_t output_spatial, int64_t input_spatial,
    int neighbors, GridSamplerInterpolation mode, GridSamplerPadding padding,
    bool align_corners) {
  const int64_t invalid = input.sizes[0] * input_spatial;
  CUDA_KERNEL_LOOP_TYPE(local_t, count, int64_t) {
    const int64_t t = first_output + local_t;
    const int64_t n = t / output_spatial;
    const int64_t grid_offset = n * grid.strides[0]
        + spatial_offset<const scalar_t, D>(t % output_spatial, grid, 1);
    sample_offsets[local_t] = n * grad.strides[0]
        + spatial_offset<const scalar_t, D>(t % output_spatial, grad, 2);
    scalar_t coord[D];
    int64_t size[D];
    for (int axis = 0; axis < D; ++axis) {
      size[axis] = input.sizes[D + 1 - axis];
      const scalar_t value = grid.data[grid_offset + axis * grid.strides[D + 1]];
      scalar_t unused;
      coord[axis] = mode == GridSamplerInterpolation::Bicubic
          ? at::native::grid_sampler_unnormalize_set_grad(value, size[axis], align_corners, &unused)
          : at::native::grid_sampler_compute_source_index_set_grad(
                value, size[axis], padding, align_corners, &unused);
    }
    scalar_t cubic_x[4], cubic_y[4];
    scalar_t cubic_base_x = 0, cubic_base_y = 0;
    if (mode == GridSamplerInterpolation::Bicubic) {
      cubic_base_x = std::floor(coord[0]);
      cubic_base_y = std::floor(coord[1]);
      at::native::get_cubic_upsampling_coefficients(cubic_x, scalar_t(coord[0] - cubic_base_x));
      at::native::get_cubic_upsampling_coefficients(cubic_y, scalar_t(coord[1] - cubic_base_y));
    }
    for (int q = 0; q < neighbors; ++q) {
      const int64_t r = local_t * neighbors + q;
      int64_t dest[D];
      scalar_t weight = 1;
      scalar_t second_weight = 1;
      if (mode == GridSamplerInterpolation::Bicubic) {
        // Preserve native x-major corner order and (grad * wx) * wy grouping.
        const int x = q / 4, y = q % 4;
        dest[0] = static_cast<int64_t>(at::native::compute_coordinates(
            scalar_t(cubic_base_x - 1 + x), size[0], padding, align_corners));
        dest[1] = static_cast<int64_t>(at::native::compute_coordinates(
            scalar_t(cubic_base_y - 1 + y), size[1], padding, align_corners));
        weight = cubic_x[x];
        second_weight = cubic_y[y];
      } else {
        for (int axis = 0; axis < D; ++axis) {
          if (mode == GridSamplerInterpolation::Nearest) {
            dest[axis] = static_cast<int64_t>(std::nearbyint(coord[axis]));
          } else {
            const int64_t base = static_cast<int64_t>(std::floor(coord[axis]));
            const int upper = (q >> axis) & 1;
            dest[axis] = base + upper;
            const scalar_t factor = upper ? scalar_t(coord[axis] - base)
                                          : scalar_t(base + 1 - coord[axis]);
            weight = weight * factor;
          }
        }
      }
      bool valid = true;
      int64_t key = n * input_spatial;
      int64_t stride = 1;
      for (int axis = 0; axis < D; ++axis) {
        valid &= dest[axis] >= 0 && dest[axis] < size[axis];
        key += dest[axis] * stride;
        stride *= size[axis];
      }
      keys[r] = valid ? key : invalid;
      ids[r] = r;
      const int factors = mode == GridSamplerInterpolation::Bicubic ? 2 : 1;
      weights[r * factors] = weight;
      if (factors == 2) weights[r * 2 + 1] = second_weight;
    }
  }
}

// Evaluate one weighted contribution with the native scalar_t rounding order.
template <typename scalar_t>
__device__ at::opmath_type<scalar_t> record_value(
    TensorInfo<const scalar_t, int64_t> grad, const scalar_t* weights,
    const int64_t* sample_offsets, int64_t r, int64_t c, int neighbor_bits,
    bool bicubic) {
  const scalar_t g = grad.data[sample_offsets[r >> neighbor_bits] + c * grad.strides[1]];
  return bicubic ? scalar_t(scalar_t(g * weights[r * 2]) * weights[r * 2 + 1])
                 : scalar_t(weights[r] * g);
}

// Logical 32-lane groups also work on devices with wider hardware warps.
template <typename acc_t>
__device__ acc_t channel_sum(acc_t value, int channel_tile) {
  for (int delta = 16; delta >= channel_tile; delta /= 2) {
    value += WARP_SHFL_DOWN(value, delta, 32);
  }
  return value;
}

constexpr int PARTIAL_ROWS = 256;

// Only long segments produce partials, so their total number is <= 2*ceil(R/256).
C10_LAUNCH_BOUNDS_1(THREADS)
__global__ void count_long_segments(
    const int64_t* sorted_keys, const int64_t* offsets, const int64_t* num_segments,
    int64_t* counts, int64_t max_segments, int64_t records, int64_t invalid_key) {
  CUDA_KERNEL_LOOP_TYPE(s, max_segments, int64_t) {
    int64_t count = 0;
    if (s < *num_segments && sorted_keys[offsets[s]] != invalid_key) {
      const int64_t end = s + 1 < *num_segments ? offsets[s + 1] : records;
      const int64_t length = end - offsets[s];
      if (length > PARTIAL_ROWS) count = (length + PARTIAL_ROWS - 1) / PARTIAL_ROWS;
    }
    counts[s] = count;
  }
}

// Independent fixed-size ranges expose parallelism even if all samples collide.
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(THREADS)
__global__ void partial_gradients(
    TensorInfo<const scalar_t, int64_t> grad, const int64_t* offsets,
    const int64_t* sorted_ids, const scalar_t* weights, const int64_t* sample_offsets,
    const int64_t* num_segments, const int64_t* counts, const int64_t* prefix,
    at::opmath_type<scalar_t>* partials, int64_t max_segments, int64_t max_partials,
    int64_t records, int neighbor_bits, int channel_bits, bool bicubic) {
  using acc_t = at::opmath_type<scalar_t>;
  const int lane = threadIdx.x % 32;
  const int channel_tile = 1 << channel_bits;
  const int64_t channels = grad.sizes[1];
  const int64_t channel_tiles = (channels + channel_tile - 1) >> channel_bits;
  const int64_t total = prefix[max_segments - 1] + counts[max_segments - 1];
  const int64_t first = int64_t(blockIdx.x) * (THREADS / 32) + threadIdx.x / 32;
  const int64_t step = int64_t(gridDim.x) * (THREADS / 32);
  for (int64_t item = first; item < max_partials * channel_tiles; item += step) {
    const int64_t part = item / channel_tiles;
    if (part >= total) continue;
    int64_t low = 0, high = max_segments;
    while (low < high) {
      const int64_t mid = low + (high - low) / 2;
      if (prefix[mid] <= part) low = mid + 1;
      else high = mid;
    }
    const int64_t segment = low - 1;
    const int64_t c = ((item % channel_tiles) << channel_bits) + (lane & (channel_tile - 1));
    const int64_t begin = offsets[segment] + (part - prefix[segment]) * PARTIAL_ROWS;
    const int64_t segment_end = segment + 1 < *num_segments ? offsets[segment + 1] : records;
    const int64_t end = min(begin + PARTIAL_ROWS, segment_end);
    acc_t acc = 0;
    for (int64_t i = begin + (lane >> channel_bits); i < end; i += 32 >> channel_bits) {
      if (c < channels) acc += record_value(grad, weights, sample_offsets, sorted_ids[i], c, neighbor_bits, bicubic);
    }
    acc = channel_sum(acc, channel_tile);
    if (lane < channel_tile && c < channels) partials[part * channels + c] = acc;
  }
}

// One warp owns the final write for a segment/channel tile. Short segments are
// reduced directly; long ones use fixed-order partials, never floating atomics.
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(THREADS)
__global__ void reduce_segments(
    TensorInfo<const scalar_t, int64_t> grad,
    scalar_t* output, const int64_t* sorted_keys, const int64_t* offsets,
    const int64_t* sorted_ids, const scalar_t* weights, const int64_t* sample_offsets,
    const int64_t* num_segments, int64_t max_segments, int64_t records,
    int64_t input_spatial, int neighbor_bits, int channel_bits, bool bicubic,
    const int64_t* partial_counts, const int64_t* partial_prefix,
    const at::opmath_type<scalar_t>* partials) {
  using acc_t = at::opmath_type<scalar_t>;
  const int lane = threadIdx.x % 32;
  const int channel_tile = 1 << channel_bits;
  const int64_t channels = grad.sizes[1];
  const int64_t channel_tiles = (channels + channel_tile - 1) >> channel_bits;
  const int64_t initial = int64_t(blockIdx.x) * (THREADS / 32) + threadIdx.x / 32;
  const int64_t step = int64_t(gridDim.x) * (THREADS / 32);
  for (int64_t item = initial; item < max_segments * channel_tiles; item += step) {
    const int64_t segment = item / channel_tiles;
    if (segment >= *num_segments) continue;
    const int64_t begin = offsets[segment];
    const int64_t key = sorted_keys[begin];
    if (key == grad.sizes[0] * input_spatial) continue;
    const int64_t c = ((item % channel_tiles) << channel_bits) + (lane & (channel_tile - 1));
    const int64_t end = segment + 1 < *num_segments ? offsets[segment + 1] : records;
    acc_t acc = 0;
    if (partial_counts != nullptr && partial_counts[segment] != 0) {
      const int64_t first_part = partial_prefix[segment];
      const int64_t last_part = first_part + partial_counts[segment];
      for (int64_t p = first_part + (lane >> channel_bits); p < last_part; p += 32 >> channel_bits) {
        if (c < channels) acc += partials[p * channels + c];
      }
    } else {
      for (int64_t i = begin + (lane >> channel_bits); i < end; i += 32 >> channel_bits) {
        if (c < channels) acc += record_value(grad, weights, sample_offsets, sorted_ids[i], c, neighbor_bits, bicubic);
      }
    }
    acc = channel_sum(acc, channel_tile);
    if (lane < channel_tile && c < channels) {
      const int64_t n = key / input_spatial;
      const int64_t index = n * channels * input_spatial + c * input_spatial + key % input_spatial;
      // Chunks commit in stream order; each chunk has one writer per element.
      output[index] = scalar_t(static_cast<acc_t>(output[index]) + acc);
    }
  }
}

template <typename scalar_t, int D>
void launch_grouped(const TensorBase& input, const TensorBase& grid,
                    const TensorBase& grad, const TensorBase& output,
                    GridSamplerInterpolation mode, GridSamplerPadding padding, bool align) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t count = grid.numel() / D;
  if (!count || !input.numel()) return;
  int64_t input_spatial = 1, output_spatial = 1;
  for (int axis = 0; axis < D; ++axis) {
    input_spatial *= input.size(axis + 2);
    output_spatial *= grid.size(axis + 1);
  }
  const int64_t destinations = input.size(0) * input_spatial;
  const bool bicubic = mode == GridSamplerInterpolation::Bicubic;
  const int neighbor_bits = mode == GridSamplerInterpolation::Nearest ? 0 : bicubic ? 4 : D;
  const int neighbors = 1 << neighbor_bits;
  int channel_bits = 0;
  while ((1 << channel_bits) < std::min<int64_t>(input.size(1), 32)) ++channel_bits;
  const int channel_tile = 1 << channel_bits;
  // Bound mapping storage and stay below CUB's INT_MAX record limit. Chunk
  // boundaries depend only on the operator configuration, not free GPU memory.
  constexpr int64_t MAX_RECORDS = 1 << 20;
  const int64_t chunk_outputs = MAX_RECORDS / neighbors;
  for (int64_t first = 0; first < count; first += chunk_outputs) {
    const int64_t chunk = std::min(count - first, chunk_outputs);
    const int64_t records = chunk * neighbors;
    const int64_t max_segments = std::min(records, destinations + 1);
    auto keys = at::empty({records}, input.options().dtype(at::kLong));
    auto sorted_keys = at::empty_like(keys);
    auto ids = at::empty_like(keys);
    auto sorted_ids = at::empty_like(keys);
    auto weights = at::empty({records, bicubic ? 2 : 1}, input.options());
    auto sample_offsets = at::empty({chunk}, input.options().dtype(at::kLong));
    auto num_segments = at::empty({}, input.options().dtype(at::kLong));
    const int blocks = (chunk + THREADS - 1) / THREADS;
    make_records<scalar_t, D><<<blocks, THREADS, 0, stream>>>(
        getTensorInfo<const scalar_t, int64_t>(input), getTensorInfo<const scalar_t, int64_t>(grid),
        getTensorInfo<const scalar_t, int64_t>(grad), keys.mutable_data_ptr<int64_t>(), ids.mutable_data_ptr<int64_t>(),
        weights.mutable_data_ptr<scalar_t>(), sample_offsets.mutable_data_ptr<int64_t>(),
        chunk, first, output_spatial, input_spatial,
        neighbors, mode, padding, align);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::cub::radix_sort_pairs(
        keys.const_data_ptr<int64_t>(), sorted_keys.mutable_data_ptr<int64_t>(),
        ids.const_data_ptr<int64_t>(), sorted_ids.mutable_data_ptr<int64_t>(),
        records, false, 0, at::cuda::cub::get_num_bits(destinations));
    // Release consumed keys before unique_by_key allocates its temporary key output.
    keys = Tensor();
    at::cuda::cub::unique_by_key(
        sorted_keys.const_data_ptr<int64_t>(), cccl_counting_iterator<int64_t>{0},
        ids.mutable_data_ptr<int64_t>(), num_segments.mutable_data_ptr<int64_t>(), records);
    using acc_t = at::opmath_type<scalar_t>;
    Tensor partial_counts, partial_prefix;
    c10::DataPtr partial_storage;
    const int64_t max_partials = 2 * ((records + PARTIAL_ROWS - 1) / PARTIAL_ROWS);
    // Bound payload workspace independently of channel count. Short workloads
    // avoid the extra launches; the direct path remains deterministic.
    constexpr int64_t MAX_PARTIAL_BYTES = 64 * 1024 * 1024;
    if (records >= 1024 &&
        max_partials <= MAX_PARTIAL_BYTES / static_cast<int64_t>(sizeof(acc_t)) / input.size(1)) {
      partial_counts = at::empty({max_segments}, input.options().dtype(at::kLong));
      partial_prefix = at::empty_like(partial_counts);
      count_long_segments<<<(max_segments + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
          sorted_keys.const_data_ptr<int64_t>(), ids.const_data_ptr<int64_t>(),
          num_segments.const_data_ptr<int64_t>(), partial_counts.mutable_data_ptr<int64_t>(),
          max_segments, records, destinations);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      at::cuda::cub::exclusive_sum(partial_counts.const_data_ptr<int64_t>(),
                                  partial_prefix.mutable_data_ptr<int64_t>(), max_segments);
      partial_storage = c10::cuda::CUDACachingAllocator::get()->allocate(max_partials * input.size(1) * sizeof(acc_t));
      const int64_t partial_warps = max_partials * ((input.size(1) + channel_tile - 1) >> channel_bits);
      const int partial_blocks = std::min<int64_t>((partial_warps + THREADS / 32 - 1) / (THREADS / 32), 65535);
      partial_gradients<scalar_t><<<partial_blocks, THREADS, 0, stream>>>(
          getTensorInfo<const scalar_t, int64_t>(grad), ids.const_data_ptr<int64_t>(),
          sorted_ids.const_data_ptr<int64_t>(), weights.const_data_ptr<scalar_t>(),
          sample_offsets.const_data_ptr<int64_t>(), num_segments.const_data_ptr<int64_t>(),
          partial_counts.const_data_ptr<int64_t>(), partial_prefix.const_data_ptr<int64_t>(),
          static_cast<acc_t*>(partial_storage.get()), max_segments, max_partials, records,
          neighbor_bits, channel_bits, bicubic);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int64_t warps = max_segments * ((input.size(1) + channel_tile - 1) >> channel_bits);
    const int reduce_blocks = std::min<int64_t>((warps + THREADS / 32 - 1) / (THREADS / 32), 65535);
    reduce_segments<scalar_t><<<reduce_blocks, THREADS, 0, stream>>>(
        getTensorInfo<const scalar_t, int64_t>(grad), output.mutable_data_ptr<scalar_t>(),
        sorted_keys.const_data_ptr<int64_t>(), ids.const_data_ptr<int64_t>(), sorted_ids.const_data_ptr<int64_t>(),
        weights.const_data_ptr<scalar_t>(), sample_offsets.const_data_ptr<int64_t>(),
        num_segments.const_data_ptr<int64_t>(), max_segments, records, input_spatial,
        neighbor_bits, channel_bits, bicubic,
        partial_counts.defined() ? partial_counts.const_data_ptr<int64_t>() : nullptr,
        partial_prefix.defined() ? partial_prefix.const_data_ptr<int64_t>() : nullptr,
        static_cast<const acc_t*>(partial_storage.get()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace

void launch_grid_sampler_input_backward_kernel(
    const TensorBase& grad_input, const TensorBase& grad_output,
    const TensorBase& input, const TensorBase& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  const auto mode = static_cast<GridSamplerInterpolation>(interpolation_mode);
  const auto padding = static_cast<GridSamplerPadding>(padding_mode);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, input.scalar_type(),
      "grid_sampler_input_backward_cuda", [&] {
        if (input.dim() == 4) {
          launch_grouped<scalar_t, 2>(input, grid, grad_output, grad_input, mode, padding, align_corners);
        } else {
          launch_grouped<scalar_t, 3>(input, grid, grad_output, grad_input, mode, padding, align_corners);
        }
      });
}

} // namespace at::native
