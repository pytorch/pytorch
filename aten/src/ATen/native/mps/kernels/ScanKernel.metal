#include <metal_stdlib>
using namespace metal;

#include <c10/metal/common.h>
#include <c10/metal/utils.h>

using c10::metal::accum_t;

template <typename T, typename acc_t = accum_t<T>>
struct CumSumOp {
  static acc_t apply(acc_t a, acc_t b) {
    return a + b;
  }
  static acc_t identity() {
    return acc_t(0);
  }
};

template <typename T, typename acc_t = accum_t<T>>
struct CumProdOp {
  static acc_t apply(acc_t a, acc_t b) {
    return a * b;
  }
  static acc_t identity() {
    return acc_t(1);
  }
};

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

#if __METAL_VERSION__ >= 310
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bfloat);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bfloat);
#endif
