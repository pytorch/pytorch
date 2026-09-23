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
// Flat index into the histogram for element `tid`, or -1 when the element falls
// outside the outer edges or is NaN. `hist_strides` is indexed by dimension.
template <typename T>
inline long histogramdd_index(
    constant T* input_,
    constant int64_t* input_strides,
    size_t num_dims,
    constant T* bin_seq,
    constant int64_t* num_bin_edges,
    constant int64_t* hist_strides,
    uint8_t algorithm,
    uint tid) {
  long hist_index = 0;
  long bin_seq_offset = 0;

  for (size_t dim = 0; dim < num_dims; dim++) {
    T element = input_[tid * input_strides[0] + dim * input_strides[1]];
    const T leftmost_edge = bin_seq[bin_seq_offset];
    const T rightmost_edge = bin_seq[bin_seq_offset + num_bin_edges[dim] - 1];

    // Skips elements which fall outside the specified bins and NaN elements.
    // The CPU kernel compares against the edges exactly; widening this test
    // counts elements that both CPU and numpy drop, and lets the linear paths
    // derive an index outside the histogram.
    if (!(element >= leftmost_edge && element <= rightmost_edge)) {
      return -1;
    }

    long pos = -1;
    if (algorithm == BIN_SELECTION_ALGORITHM::BINARY_SEARCH) {
      pos = upper_bound(bin_seq, bin_seq_offset, num_bin_edges[dim], element) -
          bin_seq_offset - 1;
    } else if (
        algorithm == BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION ||
        algorithm ==
            BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
      pos = linear_bin(
          element, num_bin_edges[dim] - 1, leftmost_edge, rightmost_edge);
      if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
        long pos_min = max(0L, pos - 1);
        long pos_max = min(pos + 2, num_bin_edges[dim]);
        pos =
            upper_bound(
                bin_seq, bin_seq_offset + pos_min, pos_max - pos_min, element) -
            bin_seq_offset - 1;
      }
    }

    // The last bin is closed, so an element on the rightmost edge belongs to
    // the final bin. The linear paths can also land outside for very narrow
    // ranges, so clamp rather than only folding the top edge down.
    pos = metal::clamp(pos, 0L, num_bin_edges[dim] - 2);
    hist_index += hist_strides[dim] * pos;
    bin_seq_offset += num_bin_edges[dim];
  }
  return hist_index;
}

template <typename T>
kernel void histogramdd(
    constant T* input_ [[buffer(0)]],
    constant T* weight [[buffer(1)]],
    device T* local_out [[buffer(2)]],
    constant int64_t* input_strides [[buffer(3)]],
    constant size_t& num_dims [[buffer(4)]],
    constant T* bin_seq [[buffer(5)]],
    constant int64_t* num_bin_edges [[buffer(6)]],
    constant int64_t* local_out_strides [[buffer(7)]],
    constant uint8_t& algorithm [[buffer(8)]],
    constant int64_t& weight_stride [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  const long hist_index = histogramdd_index(
      input_,
      input_strides,
      num_dims,
      bin_seq,
      num_bin_edges,
      local_out_strides + 1,
      algorithm,
      tid);
  if (hist_index >= 0) {
    // In the unweighted case, the default weight is 1
    local_out[local_out_strides[0] * tid + hist_index] +=
        (weight_stride >= 0) ? weight[tid * weight_stride] : 1;
  }
}

// Unweighted counts are integers, so every thread can accumulate into one
// shared histogram with 32-bit atomics instead of owning a [numel, bins] slice.
template <typename T>
kernel void histogramdd_atomic(
    constant T* input_ [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant int64_t* input_strides [[buffer(2)]],
    constant size_t& num_dims [[buffer(3)]],
    constant T* bin_seq [[buffer(4)]],
    constant int64_t* num_bin_edges [[buffer(5)]],
    constant int64_t* hist_strides [[buffer(6)]],
    constant uint8_t& algorithm [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const long hist_index = histogramdd_index(
      input_,
      input_strides,
      num_dims,
      bin_seq,
      num_bin_edges,
      hist_strides,
      algorithm,
      tid);
  if (hist_index >= 0) {
    atomic_fetch_add_explicit(&counts[hist_index], 1, memory_order_relaxed);
  }
}

#define REGISTER_HISTOGRAMDD_OP(DTYPE)                         \
  template [[host_name("histogramdd_" #DTYPE)]] kernel void    \
  histogramdd<DTYPE>(                                          \
      constant DTYPE * input_ [[buffer(0)]],                   \
      constant DTYPE * weight [[buffer(1)]],                   \
      device DTYPE * local_out [[buffer(2)]],                  \
      constant int64_t* input_strides [[buffer(3)]],           \
      constant size_t& num_dims [[buffer(4)]],                 \
      constant DTYPE* bin_seq [[buffer(5)]],                   \
      constant int64_t* num_bin_edges [[buffer(6)]],           \
      constant int64_t* local_out_strides [[buffer(7)]],       \
      constant uint8_t& bin_selection_algorithm [[buffer(8)]], \
      constant int64_t& weight_stride [[buffer(9)]],           \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_HISTOGRAMDD_ATOMIC_OP(DTYPE)                      \
  template [[host_name("histogramdd_atomic_" #DTYPE)]] kernel void \
  histogramdd_atomic<DTYPE>(                                       \
      constant DTYPE * input_ [[buffer(0)]],                       \
      device atomic_uint * counts [[buffer(1)]],                   \
      constant int64_t* input_strides [[buffer(2)]],               \
      constant size_t& num_dims [[buffer(3)]],                     \
      constant DTYPE* bin_seq [[buffer(4)]],                       \
      constant int64_t* num_bin_edges [[buffer(5)]],               \
      constant int64_t* hist_strides [[buffer(6)]],                \
      constant uint8_t& bin_selection_algorithm [[buffer(7)]],     \
      uint tid [[thread_position_in_grid]]);

REGISTER_HISTOGRAMDD_ATOMIC_OP(float);
REGISTER_HISTOGRAMDD_ATOMIC_OP(half);
REGISTER_HISTOGRAMDD_ATOMIC_OP(bfloat);

REGISTER_HISTOGRAMDD_OP(float);
REGISTER_HISTOGRAMDD_OP(half);
REGISTER_HISTOGRAMDD_OP(bfloat);

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

// The host caps num_elements at UINT32_MAX, so no bin count can overflow uint.
// counts is written here and converted to the output dtype by the caller.
template <typename T>
kernel void histc_atomic_global(
    constant T* input [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant long& input_stride [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    constant long& num_bins [[buffer(4)]],
    constant T* bin_edges [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= num_elements) {
    return;
  }
  T element = input[tid * input_stride];
  long bin = histc_bin(element, num_bins, bin_edges[0], bin_edges[num_bins]);
  if (bin >= 0) {
    atomic_fetch_add_explicit(&counts[bin], 1, memory_order_relaxed);
  }
}

template <typename T>
kernel void histc_atomic_threadgroup(
    constant T* input [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant long& input_stride [[buffer(2)]],
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
    T element = input[index * input_stride];
    long bin = histc_bin(element, num_bins, bin_edges[0], bin_edges[num_bins]);
    if (bin >= 0) {
      atomic_fetch_add_explicit(&local_counts[bin], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint bin = local_tid; bin < num_bins; bin += threads_per_threadgroup) {
    uint value = atomic_load_explicit(&local_counts[bin], memory_order_relaxed);
    if (value != 0) {
      atomic_fetch_add_explicit(&counts[bin], value, memory_order_relaxed);
    }
  }
}

#define REGISTER_HISTC_ATOMIC_OP(DTYPE)                                  \
  template [[host_name("histc_atomic_global_" #DTYPE)]] kernel void      \
  histc_atomic_global<DTYPE>(                                            \
      constant DTYPE*,                                                   \
      device atomic_uint*,                                               \
      constant long&,                                                    \
      constant uint&,                                                    \
      constant long&,                                                    \
      constant DTYPE*,                                                   \
      uint);                                                             \
  template [[host_name("histc_atomic_threadgroup_" #DTYPE)]] kernel void \
  histc_atomic_threadgroup<DTYPE>(                                       \
      constant DTYPE*,                                                   \
      device atomic_uint*,                                               \
      constant long&,                                                    \
      constant uint&,                                                    \
      constant long&,                                                    \
      constant DTYPE*,                                                   \
      constant uint&,                                                    \
      threadgroup atomic_uint*,                                          \
      uint,                                                              \
      uint,                                                              \
      uint)

REGISTER_HISTC_ATOMIC_OP(float);
REGISTER_HISTC_ATOMIC_OP(half);
REGISTER_HISTC_ATOMIC_OP(bfloat);
REGISTER_HISTC_ATOMIC_OP(int);
REGISTER_HISTC_ATOMIC_OP(long);
REGISTER_HISTC_ATOMIC_OP(short);
REGISTER_HISTC_ATOMIC_OP(char);
REGISTER_HISTC_ATOMIC_OP(uchar);
