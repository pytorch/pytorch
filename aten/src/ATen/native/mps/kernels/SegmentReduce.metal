#include <metal_stdlib>
using namespace metal;

// Must match the integer values of at::native::ReductionType.
enum class ReductionType : int32_t {
  MAX = 0,
  MEAN = 1,
  MIN = 2,
  SUM = 3,
  PROD = 4,
};

template <typename T>
inline T get_initial_value(ReductionType reduction_type) {
  switch (reduction_type) {
    case ReductionType::SUM:
      return T(0);
    case ReductionType::MEAN:
      return T(0);
    case ReductionType::MAX:
      return T(-INFINITY);
    case ReductionType::MIN:
      return T(INFINITY);
    case ReductionType::PROD:
      return T(1);
    default:
      return T(0);
  }
}

template <typename T>
kernel void segment_reduce_forward(
    device T* output_data [[buffer(0)]],
    constant T* values_data [[buffer(1)]],
    constant int* lengths_cumsum_data [[buffer(2)]],
    constant int* lengths_data [[buffer(3)]],
    constant long& segment_count [[buffer(4)]],
    constant long& outer_offset [[buffer(5)]],
    constant long& inner_offset [[buffer(6)]],
    constant long& data_size_axis [[buffer(7)]],
    constant ReductionType& reduction_type [[buffer(8)]],
    constant T& initial_value [[buffer(9)]],
    constant bool& is_initial_set [[buffer(10)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= (uint)(outer_offset * segment_count * inner_offset)) {
    return;
  }

  uint lane_id = idx % inner_offset;
  uint row_id = idx / inner_offset;
  uint dim_idx = row_id % segment_count;
  uint outer_idx = row_id / segment_count;

  uint offset_idx = outer_idx * (segment_count + 1) + dim_idx;
  uint offset_start = lengths_cumsum_data[offset_idx];
  uint offset_end = lengths_cumsum_data[offset_idx + 1];

  T accumulator =
      is_initial_set ? initial_value : get_initial_value<T>(reduction_type);

  for (uint i = offset_start; i < offset_end; i++) {
    uint data_index =
        outer_idx * data_size_axis * inner_offset + i * inner_offset + lane_id;
    T val = values_data[data_index];
    switch (reduction_type) {
      case ReductionType::SUM:
      case ReductionType::MEAN:
        accumulator += val;
        break;
      case ReductionType::MAX:
        accumulator = (isnan(val) || val > accumulator) ? val : accumulator;
        break;
      case ReductionType::MIN:
        accumulator = (isnan(val) || val < accumulator) ? val : accumulator;
        break;
      case ReductionType::PROD:
        accumulator *= val;
        break;
      default:
        break;
    }
  }

  if (reduction_type == ReductionType::MEAN) {
    uint seg_length = offset_end - offset_start;
    if (seg_length > 0 || !is_initial_set) {
      accumulator /= T(seg_length);
    }
  }

  uint output_idx = outer_idx * segment_count * inner_offset +
      dim_idx * inner_offset + lane_id;
  output_data[output_idx] = accumulator;
}

template <typename T>
kernel void segment_reduce_backward(
    device T* grad_input_data [[buffer(0)]],
    constant T* grad_output_data [[buffer(1)]],
    constant T* output_data [[buffer(2)]],
    constant T* values_data [[buffer(3)]],
    constant int* lengths_cumsum_data [[buffer(4)]],
    constant int* lengths_data [[buffer(5)]],
    constant long& segment_count [[buffer(6)]],
    constant long& outer_offset [[buffer(7)]],
    constant long& inner_offset [[buffer(8)]],
    constant long& data_size_axis [[buffer(9)]],
    constant ReductionType& reduction_type [[buffer(10)]],
    constant T& initial_prod_value [[buffer(11)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= (uint)(outer_offset * segment_count * inner_offset)) {
    return;
  }

  uint lane_id = idx % inner_offset;
  uint row_id = idx / inner_offset;
  uint dim_idx = row_id % segment_count;
  uint outer_idx = row_id / segment_count;

  uint offset_idx = outer_idx * (segment_count + 1) + dim_idx;
  uint offset_start = lengths_cumsum_data[offset_idx];
  uint offset_end = lengths_cumsum_data[offset_idx + 1];

  uint output_idx = outer_idx * segment_count * inner_offset +
      dim_idx * inner_offset + lane_id;
  T grad_val = grad_output_data[output_idx];
  T output_val = output_data[output_idx];

  uint data_base_idx = outer_idx * data_size_axis * inner_offset + lane_id;

  if (reduction_type == ReductionType::SUM ||
      reduction_type == ReductionType::MEAN) {
    if (reduction_type == ReductionType::MEAN && offset_end > offset_start) {
      grad_val /= T(offset_end - offset_start);
    }

    for (uint i = offset_start; i < offset_end; i++) {
      uint data_index = data_base_idx + i * inner_offset;
      grad_input_data[data_index] = grad_val;
    }
  } else if (
      reduction_type == ReductionType::MIN ||
      reduction_type == ReductionType::MAX) {
    uint tied_count = 0;
    for (uint i = offset_start; i < offset_end; i++) {
      uint data_index = data_base_idx + i * inner_offset;
      T val = values_data[data_index];
      if (isnan(val) || val == output_val) {
        tied_count++;
      }
    }

    if (tied_count > 0) {
      T grad_per_tied = grad_val / T(tied_count);
      for (uint i = offset_start; i < offset_end; i++) {
        uint data_index = data_base_idx + i * inner_offset;
        T val = values_data[data_index];
        if (isnan(val) || val == output_val) {
          grad_input_data[data_index] = grad_per_tied;
        }
      }
    }
  } else if (reduction_type == ReductionType::PROD) {
    for (uint i = offset_start; i < offset_end; i++) {
      uint data_index = data_base_idx + i * inner_offset;
      T val = values_data[data_index];
      if (isnan(val) || val == T(0)) {
        T exclusive_prod = initial_prod_value;
        for (uint k = offset_start; k < offset_end; k++) {
          if (k != i) {
            uint k_data_index = data_base_idx + k * inner_offset;
            exclusive_prod *= values_data[k_data_index];
          }
        }
        grad_input_data[data_index] = grad_val * exclusive_prod;
      } else {
        grad_input_data[data_index] = grad_val * output_val / val;
      }
    }
  }
}

#define REGISTER_SEGMENT_REDUCE(T)                     \
  template [[host_name("segment_reduce_forward_" #T)]] \
  kernel void segment_reduce_forward<T>(               \
      device T*,                                       \
      constant T*,                                     \
      constant int*,                                   \
      constant int*,                                   \
      constant long&,                                  \
      constant long&,                                  \
      constant long&,                                  \
      constant long&,                                  \
      constant ReductionType&,                         \
      constant T&,                                     \
      constant bool&,                                  \
      uint);

#define REGISTER_SEGMENT_REDUCE_BACKWARD(T)             \
  template [[host_name("segment_reduce_backward_" #T)]] \
  kernel void segment_reduce_backward<T>(               \
      device T*,                                        \
      constant T*,                                      \
      constant T*,                                      \
      constant T*,                                      \
      constant int*,                                    \
      constant int*,                                    \
      constant long&,                                   \
      constant long&,                                   \
      constant long&,                                   \
      constant long&,                                   \
      constant ReductionType&,                          \
      constant T&,                                      \
      uint);

REGISTER_SEGMENT_REDUCE(float)
REGISTER_SEGMENT_REDUCE(half)
REGISTER_SEGMENT_REDUCE(bfloat)

REGISTER_SEGMENT_REDUCE_BACKWARD(float)
REGISTER_SEGMENT_REDUCE_BACKWARD(half)
REGISTER_SEGMENT_REDUCE_BACKWARD(bfloat)
