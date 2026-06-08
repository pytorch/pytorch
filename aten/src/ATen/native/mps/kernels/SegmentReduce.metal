#include <metal_stdlib>
#include <c10/metal/reduction_utils.h>
#include <ATen/native/mps/kernels/SegmentReduce.h>

using namespace metal;
using at::native::ReductionType;
using c10::metal::get_initial_value;

template <typename T, typename index_t, typename lengths_t, ReductionType reduction>
kernel void segment_reduce_forward(
    device T* output_data [[buffer(0)]],
    constant T* values_data [[buffer(1)]],
    constant index_t* lengths_cumsum_data [[buffer(2)]],
    constant lengths_t* lengths_data [[buffer(3)]],
    constant SegmentReduceParams& params [[buffer(4)]],
    constant T& initial_value [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  if ((index_t)idx >=
      (index_t)params.outer_offset * (index_t)params.segment_count * (index_t)params.inner_offset) {
    return;
  }

  index_t lane_id = (index_t)idx % (index_t)params.inner_offset;
  index_t row_id = (index_t)idx / (index_t)params.inner_offset;
  index_t dim_idx = row_id % (index_t)params.segment_count;
  index_t outer_idx = row_id / (index_t)params.segment_count;

  index_t offset_idx = outer_idx * (index_t)(params.segment_count + 1) + dim_idx;
  index_t offset_start = lengths_cumsum_data[offset_idx];
  index_t offset_end = lengths_cumsum_data[offset_idx + 1];

  T accumulator =
      params.is_initial_set ? initial_value : get_initial_value<T>(reduction);

  for (index_t i = offset_start; i < offset_end; i++) {
    index_t data_index =
        outer_idx * (index_t)params.data_size_axis * (index_t)params.inner_offset +
        i * (index_t)params.inner_offset + lane_id;

    T val = values_data[data_index];
    if (reduction == ReductionType::SUM || reduction == ReductionType::MEAN) {
      accumulator += val;
    } else if (reduction == ReductionType::MAX) {
      accumulator = (isnan(val) || val > accumulator) ? val : accumulator;
    } else if (reduction == ReductionType::MIN) {
      accumulator = (isnan(val) || val < accumulator) ? val : accumulator;
    } else {
      accumulator *= val;
    }
  }

  if (reduction == ReductionType::MEAN) {
    index_t seg_length = offset_end - offset_start;
    if (seg_length > 0 || !params.is_initial_set) {
      accumulator /= T(seg_length);
    }
  }

  index_t output_idx =
      outer_idx * (index_t)params.segment_count * (index_t)params.inner_offset +
      dim_idx * (index_t)params.inner_offset + lane_id;
  output_data[output_idx] = accumulator;
}

template <typename T, typename index_t, typename lengths_t, ReductionType reduction>
kernel void segment_reduce_backward(
    device T* grad_input_data [[buffer(0)]],
    constant T* grad_output_data [[buffer(1)]],
    constant T* output_data [[buffer(2)]],
    constant T* values_data [[buffer(3)]],
    constant index_t* lengths_cumsum_data [[buffer(4)]],
    constant lengths_t* lengths_data [[buffer(5)]],
    constant SegmentReduceParams& params [[buffer(6)]],
    constant T& initial_prod_value [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
  if ((index_t)idx >=
      (index_t)params.outer_offset * (index_t)params.segment_count * (index_t)params.inner_offset) {
    return;
  }

  index_t lane_id = (index_t)idx % (index_t)params.inner_offset;
  index_t row_id = (index_t)idx / (index_t)params.inner_offset;
  index_t dim_idx = row_id % (index_t)params.segment_count;
  index_t outer_idx = row_id / (index_t)params.segment_count;

  index_t offset_idx = outer_idx * (index_t)(params.segment_count + 1) + dim_idx;
  index_t offset_start = lengths_cumsum_data[offset_idx];
  index_t offset_end = lengths_cumsum_data[offset_idx + 1];

  index_t output_idx =
      outer_idx * (index_t)params.segment_count * (index_t)params.inner_offset +
      dim_idx * (index_t)params.inner_offset + lane_id;
  T grad_val = grad_output_data[output_idx];
  T output_val = output_data[output_idx];

  index_t data_base_idx =
      outer_idx * (index_t)params.data_size_axis * (index_t)params.inner_offset + lane_id;

  if (reduction == ReductionType::SUM ||
      reduction == ReductionType::MEAN) {
    if (reduction == ReductionType::MEAN && offset_end > offset_start) {
      grad_val /= T(offset_end - offset_start);
    }

    for (index_t i = offset_start; i < offset_end; i++) {
      index_t data_index = data_base_idx + i * (index_t)params.inner_offset;
      grad_input_data[data_index] = grad_val;
    }
  } else if (
      reduction == ReductionType::MIN ||
      reduction == ReductionType::MAX) {
    index_t tied_count = 0;
    for (index_t i = offset_start; i < offset_end; i++) {
      index_t data_index = data_base_idx + i * (index_t)params.inner_offset;
      T val = values_data[data_index];
      if (isnan(val) || val == output_val) {
        tied_count++;
      }
    }

    if (tied_count > 0) {
      T grad_per_tied = grad_val / T(tied_count);
      for (index_t i = offset_start; i < offset_end; i++) {
        index_t data_index = data_base_idx + i * (index_t)params.inner_offset;
        T val = values_data[data_index];
        if (isnan(val) || val == output_val) {
          grad_input_data[data_index] = grad_per_tied;
        }
      }
    }
  } else if (reduction == ReductionType::PROD) {
    for (index_t i = offset_start; i < offset_end; i++) {
      index_t data_index = data_base_idx + i * (index_t)params.inner_offset;
      T val = values_data[data_index];
      if (isnan(val) || val == T(0)) {
        T exclusive_prod = initial_prod_value;
        for (index_t k = offset_start; k < offset_end; k++) {
          if (k != i) {
            index_t k_data_index = data_base_idx + k * (index_t)params.inner_offset;
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

#define REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, REDUCTION)                              \
  template [[host_name("segment_reduce_forward_" #T "_" #IDX_T "_" #LENGTHS_T "_" #REDUCTION)]] \
  kernel void segment_reduce_forward<T, IDX_T, LENGTHS_T, ReductionType::REDUCTION>(         \
      device T*,                                                                   \
      constant T*,                                                                 \
      constant IDX_T*,                                                             \
      constant LENGTHS_T*,                                                             \
      constant SegmentReduceParams&,                                               \
      constant T&,                                                                 \
      uint);

#define REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, REDUCTION)                              \
  template [[host_name("segment_reduce_backward_" #T "_" #IDX_T "_" #LENGTHS_T "_" #REDUCTION)]]          \
  kernel void segment_reduce_backward<T, IDX_T, LENGTHS_T, ReductionType::REDUCTION>(                 \
      device T*,                                                                            \
      constant T*,                                                                          \
      constant T*,                                                                          \
      constant T*,                                                                          \
      constant IDX_T*,                                                                      \
      constant LENGTHS_T*,                                                                      \
      constant SegmentReduceParams&,                                                        \
      constant T&,                                                                          \
      uint);

#define REGISTER_SEGMENT_REDUCE_ALL_REDUCTIONS(T, IDX_T, LENGTHS_T)       \
  REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, MAX)                       \
  REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, MEAN)                      \
  REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, MIN)                       \
  REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, SUM)                       \
  REGISTER_SEGMENT_REDUCE(T, IDX_T, LENGTHS_T, PROD)

#define REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_REDUCTIONS(T, IDX_T, LENGTHS_T) \
  REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, MAX)                 \
  REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, MEAN)                \
  REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, MIN)                 \
  REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, SUM)                 \
  REGISTER_SEGMENT_REDUCE_BACKWARD(T, IDX_T, LENGTHS_T, PROD)

#define REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(T, LENGTHS_T)                      \
  REGISTER_SEGMENT_REDUCE_ALL_REDUCTIONS(T, int, LENGTHS_T)                      \
  REGISTER_SEGMENT_REDUCE_ALL_REDUCTIONS(T, long, LENGTHS_T)

#define REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(T, LENGTHS_T)                  \
  REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_REDUCTIONS(T, int, LENGTHS_T)                  \
  REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_REDUCTIONS(T, long, LENGTHS_T)

REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(float, int)
REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(float, long)
REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(half, int)
REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(half, long)
REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(bfloat, int)
REGISTER_SEGMENT_REDUCE_ALL_IDX_TYPES(bfloat, long)

REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(float, int)
REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(float, long)
REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(half, int)
REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(half, long)
REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(bfloat, int)
REGISTER_SEGMENT_REDUCE_BACKWARD_ALL_IDX_TYPES(bfloat, long)
