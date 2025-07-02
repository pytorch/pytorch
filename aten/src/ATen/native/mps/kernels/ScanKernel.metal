#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include <c10/metal/common.h>
#include <c10/metal/utils.h>

using c10::metal::accum_t;

#if __METAL_VERSION__ < 310
template <typename T, typename acc_t = accum_t<T>>
struct CumMinOp {
  static acc_t apply(acc_t a, acc_t b) {
    return metal::min(a, b);
  }
  static acc_t identity() {
    return static_cast<acc_t>(
        metal::is_floating_point_v<T> ? metal::numeric_limits<T>::infinity()
                                      : metal::numeric_limits<T>::max());
  }
};

template <typename T, typename acc_t = accum_t<T>>
struct CumMaxOp {
  static acc_t apply(acc_t a, acc_t b) {
    return metal::max(a, b);
  }
  static acc_t identity() {
    return static_cast<acc_t>(
        metal::is_floating_point_v<T> ? -metal::numeric_limits<T>::infinity()
                                      : metal::numeric_limits<T>::lowest());
  }
};

// Inclusive scan along innermost dimension for contiguous tensors
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_contiguous_innermost_dim(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= num_rows)
    return;

  const uint offset = row * row_size;

  acc_t accumulator = Op::identity();

  for (uint col = 0; col < row_size; col++) {
    T val = input[offset + col];
    acc_t accum_val = static_cast<acc_t>(val);
    accumulator = Op::apply(accumulator, accum_val);
    output[offset + col] = static_cast<T>(accumulator);
  }
}

// Inclusive scan along outer dimension for contiguous tensors
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_contiguous_outer_dim(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& num_orows [[buffer(2)]],
    constant uint& num_irows [[buffer(3)]],
    constant uint& row_size [[buffer(4)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint orow = thread_index / num_irows;
  const uint irow = thread_index % num_irows;

  if (orow >= num_orows)
    return;

  acc_t accumulator = Op::identity();

  const uint idx_base = orow * row_size * num_irows + irow;
  for (uint col = 0, idx = idx_base; col < row_size; col++, idx += num_irows) {
    T val = input[idx];
    acc_t accum_val = static_cast<acc_t>(val);
    accumulator = Op::apply(accumulator, accum_val);
    output[idx] = static_cast<T>(accumulator);
  }
}

// Inclusive scan with indices along innermost dimension for contiguous tensors
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_with_indices_contiguous_innermost_dim(
    constant T* input [[buffer(0)]],
    device T* values [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& row_size [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= num_rows)
    return;

  const uint offset = row * row_size;

  acc_t accumulator = Op::identity();
  int64_t best_idx = 0;

  for (uint col = 0; col < row_size; col++) {
    T val = input[offset + col];
    acc_t accum_val = static_cast<acc_t>(val);
    if (col == 0 || Op::apply(accum_val, accumulator) == accum_val) {
      accumulator = accum_val;
      best_idx = col;
    }
    values[offset + col] = static_cast<T>(accumulator);
    indices[offset + col] = best_idx;
  }
}

// Inclusive scan with indices along outer dimension for contiguous tensors
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_with_indices_contiguous_outer_dim(
    constant T* input [[buffer(0)]],
    device T* values [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant uint& num_orows [[buffer(3)]],
    constant uint& num_irows [[buffer(4)]],
    constant uint& row_size [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint orow = thread_index / num_irows;
  const uint irow = thread_index % num_irows;

  if (orow >= num_orows)
    return;

  acc_t accumulator = Op::identity();
  int64_t best_idx = 0;

  const uint idx_base = orow * row_size * num_irows + irow;
  for (uint col = 0, idx = idx_base; col < row_size; col++, idx += num_irows) {
    T val = input[idx];
    acc_t accum_val = static_cast<acc_t>(val);
    if (col == 0 || Op::apply(accum_val, accumulator) == accum_val) {
      accumulator = accum_val;
      best_idx = col;
    }
    values[idx] = static_cast<T>(accumulator);
    indices[idx] = best_idx;
  }
}

// Shared utility functions for strided kernels
inline long calculate_non_scan_elements(
    constant long* sizes,
    uint ndim,
    uint scan_dim) {
  long total = 1;
  for (uint i = 0; i < ndim; ++i) {
    if (i != scan_dim) {
      total *= sizes[i];
    }
  }
  return total;
}

inline void thread_index_to_coordinates(
    uint index,
    int pos[c10::metal::max_ndim],
    constant long* sizes,
    uint ndim,
    uint scan_dim) {
  long remaining_index = index;
  for (uint i = 0; i < ndim; ++i) {
    if (i != scan_dim) {
      pos[i] = remaining_index % sizes[i];
      remaining_index /= sizes[i];
    } else {
      pos[i] = 0;
    }
  }
}

inline long calculate_base_offset(
    int pos[c10::metal::max_ndim],
    constant long* strides,
    uint ndim,
    uint scan_dim) {
  long offset = 0;
  for (uint i = 0; i < ndim; ++i) {
    if (i != scan_dim) {
      offset += pos[i] * strides[i];
    }
  }
  return offset;
}

// Generic strided scan kernel
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_strided(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* input_strides [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& scan_dim [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long total_non_scan_elements =
      calculate_non_scan_elements(sizes, ndim, scan_dim);
  if (thread_index >= total_non_scan_elements) {
    return;
  }

  int pos[c10::metal::max_ndim];
  thread_index_to_coordinates(thread_index, pos, sizes, ndim, scan_dim);

  const long input_base_offset =
      calculate_base_offset(pos, input_strides, ndim, scan_dim);
  const long output_base_offset =
      calculate_base_offset(pos, output_strides, ndim, scan_dim);

  acc_t accumulator = Op::identity();
  const long scan_size = sizes[scan_dim];
  const long input_scan_stride = input_strides[scan_dim];
  const long output_scan_stride = output_strides[scan_dim];

  for (long scan_idx = 0; scan_idx < scan_size; scan_idx++) {
    const long input_offset = input_base_offset + scan_idx * input_scan_stride;
    const long output_offset =
        output_base_offset + scan_idx * output_scan_stride;

    T val = input[input_offset];
    acc_t accum_val = static_cast<acc_t>(val);
    accumulator = Op::apply(accumulator, accum_val);
    output[output_offset] = static_cast<T>(accumulator);
  }
}

// Generic strided scan with indices kernel
template <typename T, typename Op, typename acc_t = accum_t<T>>
kernel void scan_with_indices_strided(
    constant T* input [[buffer(0)]],
    device T* values [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* input_strides [[buffer(4)]],
    constant long* values_strides [[buffer(5)]],
    constant long* indices_strides [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    constant uint& scan_dim [[buffer(8)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long total_non_scan_elements =
      calculate_non_scan_elements(sizes, ndim, scan_dim);
  if (thread_index >= total_non_scan_elements) {
    return;
  }

  int pos[c10::metal::max_ndim];
  thread_index_to_coordinates(thread_index, pos, sizes, ndim, scan_dim);

  const long input_base_offset =
      calculate_base_offset(pos, input_strides, ndim, scan_dim);
  const long values_base_offset =
      calculate_base_offset(pos, values_strides, ndim, scan_dim);
  const long indices_base_offset =
      calculate_base_offset(pos, indices_strides, ndim, scan_dim);

  acc_t accumulator = Op::identity();
  int64_t best_idx = 0;
  const long scan_size = sizes[scan_dim];
  const long input_scan_stride = input_strides[scan_dim];
  const long values_scan_stride = values_strides[scan_dim];
  const long indices_scan_stride = indices_strides[scan_dim];

  for (long scan_idx = 0; scan_idx < scan_size; scan_idx++) {
    const long input_offset = input_base_offset + scan_idx * input_scan_stride;
    const long values_offset =
        values_base_offset + scan_idx * values_scan_stride;
    const long indices_offset =
        indices_base_offset + scan_idx * indices_scan_stride;

    T val = input[input_offset];
    acc_t accum_val = static_cast<acc_t>(val);
    if (scan_idx == 0 || Op::apply(accum_val, accumulator) == accum_val) {
      accumulator = accum_val;
      best_idx = scan_idx;
    }
    values[values_offset] = static_cast<T>(accumulator);
    indices[indices_offset] = best_idx;
  }
}

#define REGISTER_SCAN_OP(OP_NAME, OP_CLASS, DTYPE)                             \
  template [[host_name(#OP_NAME "_contiguous_innermost_" #DTYPE)]] kernel void \
  scan_contiguous_innermost_dim<DTYPE, OP_CLASS<DTYPE>>(                       \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * output [[buffer(1)]],                                     \
      constant uint & num_rows [[buffer(2)]],                                  \
      constant uint & row_size [[buffer(3)]],                                  \
      uint row [[thread_position_in_grid]]);                                   \
                                                                               \
  template [[host_name(#OP_NAME "_contiguous_outer_" #DTYPE)]] kernel void     \
  scan_contiguous_outer_dim<DTYPE, OP_CLASS<DTYPE>>(                           \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * output [[buffer(1)]],                                     \
      constant uint & num_orows [[buffer(2)]],                                 \
      constant uint & num_irows [[buffer(3)]],                                 \
      constant uint & row_size [[buffer(4)]],                                  \
      uint thread_index [[thread_position_in_grid]]);                          \
                                                                               \
  template [[host_name(#OP_NAME "_strided_" #DTYPE)]] kernel void              \
  scan_strided<DTYPE, OP_CLASS<DTYPE>>(                                        \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * output [[buffer(1)]],                                     \
      constant long* sizes [[buffer(2)]],                                      \
      constant long* input_strides [[buffer(3)]],                              \
      constant long* output_strides [[buffer(4)]],                             \
      constant uint& ndim [[buffer(5)]],                                       \
      constant uint& scan_dim [[buffer(6)]],                                   \
      uint thread_index [[thread_position_in_grid]]);

#define REGISTER_SCAN_WITH_INDICES_OP(OP_NAME, OP_CLASS, DTYPE)                \
  template [[host_name(#OP_NAME "_contiguous_innermost_" #DTYPE)]] kernel void \
  scan_with_indices_contiguous_innermost_dim<DTYPE, OP_CLASS<DTYPE>>(          \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * values [[buffer(1)]],                                     \
      device int64_t* indices [[buffer(2)]],                                   \
      constant uint& num_rows [[buffer(3)]],                                   \
      constant uint& row_size [[buffer(4)]],                                   \
      uint row [[thread_position_in_grid]]);                                   \
                                                                               \
  template [[host_name(#OP_NAME "_contiguous_outer_" #DTYPE)]] kernel void     \
  scan_with_indices_contiguous_outer_dim<DTYPE, OP_CLASS<DTYPE>>(              \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * values [[buffer(1)]],                                     \
      device int64_t* indices [[buffer(2)]],                                   \
      constant uint& num_orows [[buffer(3)]],                                  \
      constant uint& num_irows [[buffer(4)]],                                  \
      constant uint& row_size [[buffer(5)]],                                   \
      uint thread_index [[thread_position_in_grid]]);                          \
                                                                               \
  template [[host_name(#OP_NAME "_strided_" #DTYPE)]] kernel void              \
  scan_with_indices_strided<DTYPE, OP_CLASS<DTYPE>>(                           \
      constant DTYPE * input [[buffer(0)]],                                    \
      device DTYPE * values [[buffer(1)]],                                     \
      device int64_t* indices [[buffer(2)]],                                   \
      constant long* sizes [[buffer(3)]],                                      \
      constant long* input_strides [[buffer(4)]],                              \
      constant long* values_strides [[buffer(5)]],                             \
      constant long* indices_strides [[buffer(6)]],                            \
      constant uint& ndim [[buffer(7)]],                                       \
      constant uint& scan_dim [[buffer(8)]],                                   \
      uint thread_index [[thread_position_in_grid]]);

// Scan operations with indices
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, float);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, half);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, long);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, int);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, short);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, char);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, uchar);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bool);

REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, float);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, half);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, long);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, int);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, short);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, char);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, uchar);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bool);

#else // __METAL_VERSION__ >= 310

// The reminder of this file contains cummin and cummax implementations adapted
// from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scan.h
//
// The original MLX kernels have been modified to integrate with PyTorch's MPS
// backend. Most notably:
//  - Keeping track and returning indices, which MLX kernels don't do.
//  - Perform computations on half/bfloat tensors at higher precision (float)
//  via c10::metal::accum_t
//
// Original work is licensed under MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

inline uint64_t simd_shuffle_and_fill_up(
    uint64_t data,
    uint64_t filling,
    uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline int64_t simd_shuffle_and_fill_up(
    int64_t data,
    int64_t filling,
    uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

#define DEFINE_SIMD_SCAN()                                               \
  template <typename U, metal::enable_if_t<sizeof(U) < 8, bool> = true>  \
  U simd_scan(U val) {                                                   \
    return simd_scan_impl(val);                                          \
  }                                                                      \
                                                                         \
  template <typename U, metal::enable_if_t<sizeof(U) == 8, bool> = true> \
  U simd_scan(U val) {                                                   \
    for (int i = 1; i <= 16; i *= 2) {                                   \
      val = operator()(val, simd_shuffle_and_fill_up(val, init, i));     \
    }                                                                    \
    return val;                                                          \
  }

#define DEFINE_SIMD_EXCLUSIVE_SCAN()                                     \
  template <typename U, metal::enable_if_t<sizeof(U) < 8, bool> = true>  \
  U simd_exclusive_scan(U val) {                                         \
    return simd_exclusive_scan_impl(val);                                \
  }                                                                      \
                                                                         \
  template <typename U, metal::enable_if_t<sizeof(U) == 8, bool> = true> \
  U simd_exclusive_scan(U val) {                                         \
    val = simd_scan(val);                                                \
    return simd_shuffle_and_fill_up(val, init, 1);                       \
  }

// Pair structure to hold value and index for cummin/cummax operations
template <typename T, typename acc_t = accum_t<T>>
struct ValueIndexPair {
  acc_t value;
  int64_t index;
};

// Helper function to create ValueIndexPair
template <typename T, typename acc_t = accum_t<T>>
inline ValueIndexPair<T, acc_t> make_pair(acc_t v, int64_t i) {
  ValueIndexPair<T, acc_t> result;
  result.value = v;
  result.index = i;
  return result;
}

// Helper function for shuffling pairs in SIMD operations
template <typename T, typename acc_t = accum_t<T>>
inline ValueIndexPair<T, acc_t> simd_shuffle_pair(
    ValueIndexPair<T, acc_t> data,
    uint16_t lane) {
  return make_pair<T, acc_t>(
      simd_shuffle(data.value, lane), simd_shuffle(data.index, lane));
}

template <typename T, typename acc_t = accum_t<T>>
struct CumMinOp {
  using pair_t = ValueIndexPair<T, acc_t>;

  static constexpr constant acc_t init_val = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::max());

  static pair_t get_init() {
    return make_pair<T, acc_t>(init_val, 0);
  }

  pair_t operator()(pair_t a, pair_t b) {
    if (::metal::isnan(static_cast<float>(a.value)) &&
        ::metal::isnan(static_cast<float>(b.value))) {
      return (a.index >= b.index) ? a : b;
    } else if (::metal::isnan(static_cast<float>(a.value))) {
      return a;
    } else if (::metal::isnan(static_cast<float>(b.value))) {
      return b;
    } else if (a.value < b.value) {
      return a;
    } else if (a.value > b.value) {
      return b;
    } else {
      return (a.index >= b.index) ? a : b;
    }
  }

  // For SIMD operations, we need to handle pairs differently
  pair_t simd_scan(pair_t val) {
    // For pairs, we need to implement scan manually since SIMD doesn't support
    // pairs directly
    pair_t init_val = get_init();
    for (int i = 1; i <= 16; i *= 2) {
      pair_t shuffled = make_pair<T, acc_t>(
          simd_shuffle_and_fill_up(val.value, init_val.value, i),
          simd_shuffle_and_fill_up(val.index, init_val.index, i));
      val = operator()(val, shuffled);
    }
    return val;
  }

  pair_t simd_exclusive_scan(pair_t val) {
    val = simd_scan(val);
    pair_t init_val = get_init();
    return simd_shuffle_and_fill_up_pair(val, init_val, 1);
  }

 private:
  pair_t simd_shuffle_pair(pair_t data, uint16_t delta) {
    pair_t init_val = get_init();
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, init_val.value, delta),
        simd_shuffle_and_fill_up(data.index, init_val.index, delta));
  }

  pair_t simd_shuffle_and_fill_up_pair(
      pair_t data,
      pair_t filling,
      uint16_t delta) {
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, filling.value, delta),
        simd_shuffle_and_fill_up(data.index, filling.index, delta));
  }
};

template <typename T, typename acc_t = accum_t<T>>
struct CumMaxOp {
  using pair_t = ValueIndexPair<T, acc_t>;

  static constexpr constant acc_t init_val = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? -metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::lowest());

  static pair_t get_init() {
    return make_pair<T, acc_t>(init_val, 0);
  }

  pair_t operator()(pair_t a, pair_t b) {
    if (::metal::isnan(static_cast<float>(a.value)) &&
        ::metal::isnan(static_cast<float>(b.value))) {
      return (a.index >= b.index) ? a : b;
    } else if (::metal::isnan(static_cast<float>(a.value))) {
      return a;
    } else if (::metal::isnan(static_cast<float>(b.value))) {
      return b;
    } else if (a.value > b.value) {
      return a;
    } else if (a.value < b.value) {
      return b;
    } else {
      return (a.index >= b.index) ? a : b;
    }
  }

  // For SIMD operations, we need to handle pairs differently
  pair_t simd_scan(pair_t val) {
    // For pairs, we need to implement scan manually since SIMD doesn't support
    // pairs directly
    pair_t init_val = get_init();
    for (int i = 1; i <= 16; i *= 2) {
      pair_t shuffled = make_pair<T, acc_t>(
          simd_shuffle_and_fill_up(val.value, init_val.value, i),
          simd_shuffle_and_fill_up(val.index, init_val.index, i));
      val = operator()(val, shuffled);
    }
    return val;
  }

  pair_t simd_exclusive_scan(pair_t val) {
    val = simd_scan(val);
    pair_t init_val = get_init();
    return simd_shuffle_and_fill_up_pair(val, init_val, 1);
  }

 private:
  pair_t simd_shuffle_pair(pair_t data, uint16_t delta) {
    pair_t init_val = get_init();
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, init_val.value, delta),
        simd_shuffle_and_fill_up(data.index, init_val.index, delta));
  }

  pair_t simd_shuffle_and_fill_up_pair(
      pair_t data,
      pair_t filling,
      uint16_t delta) {
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, filling.value, delta),
        simd_shuffle_and_fill_up(data.index, filling.index, delta));
  }
};

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void load_unsafe(acc_t values[N_READS], const device T* input) {
  for (int i = 0; i < N_READS; i++) {
    values[i] = static_cast<acc_t>(input[i]);
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void load_safe(
    acc_t values[N_READS],
    const device T* input,
    int start,
    int total,
    acc_t init) {
  for (int i = 0; i < N_READS; i++) {
    values[i] = (start + i < total) ? static_cast<acc_t>(input[i]) : init;
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void write_unsafe(acc_t values[N_READS], device T* out) {
  for (int i = 0; i < N_READS; i++) {
    out[i] = static_cast<T>(values[i]);
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void write_safe(
    acc_t values[N_READS],
    device T* out,
    int start,
    int total) {
  for (int i = 0; i < N_READS; i++) {
    if (start + i < total) {
      out[i] = static_cast<T>(values[i]);
    }
  }
}

// Utility function for ceiling division
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_with_indices_innermost_dim(
    const device T* in [[buffer(0)]],
    device T* out_values [[buffer(1)]],
    device int64_t* out_indices [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  Op op;
  using pair_t = typename Op::pair_t;

  // Position the pointers
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out_values += offset;
  out_indices += offset;

  // Compute the number of simd_groups
  uint simd_groups = lsize.x / simd_size;

  // Allocate memory
  pair_t prefix = op.get_init();
  pair_t values[N_READS];
  threadgroup pair_t simdgroup_sums[32];

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Compute the block offset
    uint offset_idx = r * lsize.x * N_READS + lid.x * N_READS;

    // Read the values as pairs
    for (int i = 0; i < N_READS; i++) {
      if ((offset_idx + i) < axis_size) {
        values[i] = make_pair<T, acc_t>(
            static_cast<acc_t>(in[offset_idx + i]), offset_idx + i);
      } else {
        values[i] = op.get_init();
      }
    }

    // Compute an inclusive scan per thread
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums
    pair_t prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);

    // Write simdgroup_sums to SM
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      pair_t prev_simdgroup =
          op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
      simdgroup_sums[simd_lane_id] = prev_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute the output
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }

    // Write the values
    for (int i = 0; i < N_READS; i++) {
      if ((offset_idx + i) < axis_size) {
        out_values[offset_idx + i] = static_cast<T>(values[i].value);
        out_indices[offset_idx + i] = values[i].index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Share the prefix
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
  }
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_with_indices_outer_dim(
    const device T* in [[buffer(0)]],
    device T* out_values [[buffer(1)]],
    device int64_t* out_indices [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    const constant size_t& stride [[buffer(4)]],
    const constant size_t& stride_blocks [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int simd_size = 32;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(T);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;
  using pair_t = typename Op::pair_t;

  threadgroup pair_t read_buffer[BM * BN_pad];
  pair_t values[n_scans];
  pair_t prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = op.get_init();
  }

  // Compute offsets
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out_values += offset + global_index_x + read_offset_x;
  out_indices += offset + global_index_x + read_offset_x;
  threadgroup pair_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup pair_t* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;

    // Read into shared memory as pairs
    for (int i = 0; i < N_READS; i++) {
      if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
        // For cummin/cummax, the index should represent the position along the
        // scan axis
        read_into[i] = make_pair<T, acc_t>(
            static_cast<acc_t>(in[index_y * stride + i]), index_y);
      } else {
        read_into[i] = op.get_init();
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read strided into registers
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Perform the scan
    for (int i = 0; i < n_scans; i++) {
      values[i] = op.simd_scan(values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = make_pair<T, acc_t>(
          simd_shuffle(values[i].value, simd_size - 1),
          simd_shuffle(values[i].index, simd_size - 1));
    }

    // Write to shared memory
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to device memory
    for (int i = 0; i < N_READS; i++) {
      if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
        out_values[index_y * stride + i] = static_cast<T>(read_into[i].value);
        out_indices[index_y * stride + i] = read_into[i].index;
      }
    }
  }
}

#define REGISTER_SCAN_WITH_INDICES_OP(OP_NAME, OP_CLASS, DTYPE, NREADS) \
  template [[host_name(#OP_NAME "_innermost_" #DTYPE)]] [[kernel]] void \
  scan_with_indices_innermost_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(      \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out_values [[buffer(1)]],                           \
      device int64_t* out_indices [[buffer(2)]],                        \
      const constant size_t& axis_size [[buffer(3)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);           \
                                                                        \
  template [[host_name(#OP_NAME "_outer_" #DTYPE)]] [[kernel]] void     \
  scan_with_indices_outer_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(          \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out_values [[buffer(1)]],                           \
      device int64_t* out_indices [[buffer(2)]],                        \
      const constant size_t& axis_size [[buffer(3)]],                   \
      const constant size_t& stride [[buffer(4)]],                      \
      const constant size_t& stride_blocks [[buffer(5)]],               \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

// Scan with indices operations for cummin/cummax
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, float, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, half, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bfloat, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, long, 2);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, int, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, short, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, char, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, uchar, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bool, 4);

REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, float, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, half, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bfloat, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, long, 2);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, int, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, short, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, char, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, uchar, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bool, 4);

#endif
