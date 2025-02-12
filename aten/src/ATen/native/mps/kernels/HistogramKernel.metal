#include <metal_stdlib>
using namespace metal;

enum BIN_SELECTION_ALGORITHM {
  LINEAR_INTERPOLATION,
  LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
  BINARY_SEARCH,
};

// Re-implementation of std::upper_bound with some modifications.
template <typename T, typename U>
U upper_bound(constant T* arr, U first, U len, T val) {
  while (len > 0) {
    U half_ = len >> 1;
    U middle = first + half_;

    if (val < arr[middle]) {
      len = half_;
    } else {
      first = middle + 1;
      len -= half_ + 1;
    }
  }
  return first;
}

// The implementation here is mostly taken from the CPU's implementation with
// some modifications. Please see `aten/src/ATen/native/cpu/HistogramKernel.cpp`
// for more details.
template <typename T>
kernel void histogramdd(
    constant T* input_ [[buffer(0)]],
    constant T* weight [[buffer(1)]],
    device T* local_out [[buffer(2)]],
    constant uint* offsets [[buffer(3)]],
    constant size_t& num_dims [[buffer(4)]],
    constant T* bin_seq [[buffer(5)]],
    constant int64_t* num_bin_edges [[buffer(6)]],
    constant T* leftmost_edge [[buffer(7)]],
    constant T* rightmost_edge [[buffer(8)]],
    constant int64_t* local_out_strides [[buffer(9)]],
    constant uint8_t& algorithm [[buffer(10)]],
    constant uint8_t& has_weight [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr T eps = 4e-6;
  bool skip_element = false;
  int64_t hist_index = 0;
  int64_t bin_seq_offset = 0;

  for (size_t dim = 0; dim < num_dims; dim++) {
    T element = input_[offsets[tid * num_dims + dim]];

    // Skips elements which fall outside the specified bins and NaN elements
    // Adding an eps to the edges to eliminate precision issues that cause
    // elements accidentally skipped, this is likely due to the minuscule
    // implementation differences between the CPU and MPS's linspace.
    if (!(element >= (leftmost_edge[dim] - eps) &&
          element <= (rightmost_edge[dim] + eps))) {
      skip_element = true;
      break;
    }
    int64_t pos = -1;

    if (algorithm == BIN_SELECTION_ALGORITHM::BINARY_SEARCH) {
      pos = upper_bound(bin_seq, bin_seq_offset, num_bin_edges[dim], element) -
          bin_seq_offset - 1;
    } else if (
        algorithm == BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION ||
        algorithm ==
            BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
      pos = static_cast<int64_t>(
          (element - leftmost_edge[dim]) * (num_bin_edges[dim] - 1) /
          (rightmost_edge[dim] - leftmost_edge[dim]));
      if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
        int64_t pos_min = max(static_cast<int64_t>(0), pos - 1);
        int64_t pos_max = min(pos + 2, num_bin_edges[dim]);
        pos =
            upper_bound(
                bin_seq, bin_seq_offset + pos_min, pos_max - pos_min, element) -
            bin_seq_offset - 1;
      }
    }

    if (pos == (num_bin_edges[dim] - 1)) {
      pos -= 1;
    }
    hist_index += local_out_strides[dim + 1] * pos;
    bin_seq_offset += num_bin_edges[dim];
  }
  if (!skip_element) {
    // In the unweighted case, the default weight is 1
    local_out[local_out_strides[0] * tid + hist_index] +=
        has_weight ? weight[tid] : 1;
  }
}

#define REGISTER_HISTOGRAMDD_OP(DTYPE)                           \
  template [[host_name("histogramdd_" #DTYPE)]] kernel void      \
  histogramdd<DTYPE>(                                            \
      constant DTYPE * input_ [[buffer(0)]],                     \
      constant DTYPE * weight [[buffer(1)]],                     \
      device DTYPE * local_out [[buffer(2)]],                    \
      constant uint * offsets [[buffer(3)]],                     \
      constant size_t & num_dims [[buffer(4)]],                  \
      constant DTYPE * bin_seq [[buffer(5)]],                    \
      constant int64_t * num_bin_edges [[buffer(6)]],            \
      constant DTYPE * leftmost_edge [[buffer(7)]],              \
      constant DTYPE * rightmost_edge [[buffer(8)]],             \
      constant int64_t * local_out_strides [[buffer(9)]],        \
      constant uint8_t & bin_selection_algorithm [[buffer(10)]], \
      constant uint8_t & has_weight [[buffer(11)]],              \
      uint tid [[thread_position_in_grid]]);

REGISTER_HISTOGRAMDD_OP(float);
REGISTER_HISTOGRAMDD_OP(half);

kernel void kernel_index_offset(
    constant uint* strides [[buffer(0)]],
    device uint* data_offsets [[buffer(1)]],
    constant uint* iter_shape [[buffer(2)]],
    constant uint& num_dimensions [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]]) {
  data_offsets[thread_index] = 0;
  uint32_t idx = thread_index;
  for (uint32_t dim = 0; dim < num_dimensions; dim++) {
    uint32_t reversed_dim = num_dimensions - dim - 1;
    uint32_t remainder = idx % iter_shape[reversed_dim];
    idx /= iter_shape[reversed_dim];

    data_offsets[thread_index] += remainder * strides[reversed_dim];
  }
}
