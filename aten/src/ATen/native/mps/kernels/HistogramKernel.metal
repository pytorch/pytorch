#include <c10/metal/atomic.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

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

template <typename T>
inline long linear_bin(
    T element,
    long num_bins,
    T leftmost_edge,
    T rightmost_edge) {
  return static_cast<long>(
      (element - leftmost_edge) * num_bins / (rightmost_edge - leftmost_edge));
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
    constant int64_t& weight_stride [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr auto eps = T(4e-6);
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
      pos = linear_bin(
          element,
          num_bin_edges[dim] - 1,
          leftmost_edge[dim],
          rightmost_edge[dim]);
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
        (weight_stride >= 0) ? weight[tid * weight_stride] : 1;
  }
}

#define REGISTER_HISTOGRAMDD_OP(DTYPE)                          \
  template [[host_name("histogramdd_" #DTYPE)]] kernel void     \
  histogramdd<DTYPE>(                                           \
      constant DTYPE * input_ [[buffer(0)]],                    \
      constant DTYPE * weight [[buffer(1)]],                    \
      device DTYPE * local_out [[buffer(2)]],                   \
      constant uint * offsets [[buffer(3)]],                    \
      constant size_t& num_dims [[buffer(4)]],                  \
      constant DTYPE* bin_seq [[buffer(5)]],                    \
      constant int64_t* num_bin_edges [[buffer(6)]],            \
      constant DTYPE* leftmost_edge [[buffer(7)]],              \
      constant DTYPE* rightmost_edge [[buffer(8)]],             \
      constant int64_t* local_out_strides [[buffer(9)]],        \
      constant uint8_t& bin_selection_algorithm [[buffer(10)]], \
      constant int64_t& weight_stride [[buffer(11)]],           \
      uint tid [[thread_position_in_grid]]);

REGISTER_HISTOGRAMDD_OP(float);
REGISTER_HISTOGRAMDD_OP(half);
REGISTER_HISTOGRAMDD_OP(bfloat);
REGISTER_HISTOGRAMDD_OP(int);
REGISTER_HISTOGRAMDD_OP(long);
REGISTER_HISTOGRAMDD_OP(short);
REGISTER_HISTOGRAMDD_OP(char);
REGISTER_HISTOGRAMDD_OP(uchar);

template <typename T>
inline long histc_bin(
    T element,
    long num_bins,
    T leftmost_edge,
    T rightmost_edge) {
  if (!(element >= leftmost_edge && element <= rightmost_edge)) {
    return -1;
  }
  long pos = linear_bin(element, num_bins, leftmost_edge, rightmost_edge);
  return metal::clamp(pos, 0L, num_bins - 1);
}

template <typename T, bool dense>
kernel void histc_atomic_global(
    constant T* input [[buffer(0)]],
    device AtomicType_t<long>* counts [[buffer(1)]],
    constant uint* offsets [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    constant long& num_bins [[buffer(4)]],
    constant T* bin_edges [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= num_elements) {
    return;
  }
  T element = input[dense ? tid : offsets[tid]];
  long bin = histc_bin(element, num_bins, bin_edges[0], bin_edges[num_bins]);
  if (bin >= 0) {
    AtomicType<long>::atomic_add(counts, bin, 1);
  }
}

template <typename T, bool dense>
kernel void histc_atomic_threadgroup(
    constant T* input [[buffer(0)]],
    device AtomicType_t<long>* counts [[buffer(1)]],
    constant uint* offsets [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    constant long& num_bins [[buffer(4)]],
    constant T* bin_edges [[buffer(5)]],
    constant uint& total_threads [[buffer(6)]],
    threadgroup atomic_uint* local_counts [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  for (uint bin = local_tid; bin < num_bins; bin += threads_per_threadgroup) {
    atomic_store_explicit(&local_counts[bin], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The host limits num_elements to UINT32_MAX, so a local bin count fits in
  // uint.
  for (uint index = tid; index < num_elements; index += total_threads) {
    T element = input[dense ? index : offsets[index]];
    long bin = histc_bin(element, num_bins, bin_edges[0], bin_edges[num_bins]);
    if (bin >= 0) {
      atomic_fetch_add_explicit(&local_counts[bin], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint bin = local_tid; bin < num_bins; bin += threads_per_threadgroup) {
    uint value = atomic_load_explicit(&local_counts[bin], memory_order_relaxed);
    if (value != 0) {
      AtomicType<long>::atomic_add(counts, bin, static_cast<long>(value));
    }
  }
}

#define REGISTER_HISTC_ATOMIC_OP(DTYPE)                                        \
  template [[host_name("histc_atomic_global_dense_" #DTYPE)]] kernel void      \
  histc_atomic_global<DTYPE, true>(                                            \
      constant DTYPE*,                                                         \
      device AtomicType_t<long>*,                                              \
      constant uint*,                                                          \
      constant uint&,                                                          \
      constant long&,                                                          \
      constant DTYPE*,                                                         \
      uint);                                                                   \
  template [[host_name("histc_atomic_global_strided_" #DTYPE)]] kernel void    \
  histc_atomic_global<DTYPE, false>(                                           \
      constant DTYPE*,                                                         \
      device AtomicType_t<long>*,                                              \
      constant uint*,                                                          \
      constant uint&,                                                          \
      constant long&,                                                          \
      constant DTYPE*,                                                         \
      uint);                                                                   \
  template [[host_name("histc_atomic_threadgroup_dense_" #DTYPE)]] kernel void \
  histc_atomic_threadgroup<DTYPE, true>(                                       \
      constant DTYPE*,                                                         \
      device AtomicType_t<long>*,                                              \
      constant uint*,                                                          \
      constant uint&,                                                          \
      constant long&,                                                          \
      constant DTYPE*,                                                         \
      constant uint&,                                                          \
      threadgroup atomic_uint*,                                                \
      uint,                                                                    \
      uint,                                                                    \
      uint);                                                                   \
  template                                                                     \
      [[host_name("histc_atomic_threadgroup_strided_" #DTYPE)]] kernel void    \
      histc_atomic_threadgroup<DTYPE, false>(                                  \
          constant DTYPE*,                                                     \
          device AtomicType_t<long>*,                                          \
          constant uint*,                                                      \
          constant uint&,                                                      \
          constant long&,                                                      \
          constant DTYPE*,                                                     \
          constant uint&,                                                      \
          threadgroup atomic_uint*,                                            \
          uint,                                                                \
          uint,                                                                \
          uint)

REGISTER_HISTC_ATOMIC_OP(float);
REGISTER_HISTC_ATOMIC_OP(half);
REGISTER_HISTC_ATOMIC_OP(bfloat);
REGISTER_HISTC_ATOMIC_OP(int);
REGISTER_HISTC_ATOMIC_OP(long);
REGISTER_HISTC_ATOMIC_OP(short);
REGISTER_HISTC_ATOMIC_OP(char);
REGISTER_HISTC_ATOMIC_OP(uchar);

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
