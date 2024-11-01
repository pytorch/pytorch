#include <metal_stdlib>
using namespace metal;

// The bucketization kernels are mostly copied-n-pasted from bucketization.cu.

template <typename input_t>
int64_t lower_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val,
    constant int64_t* data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t lower_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t upper_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val,
    constant int64_t* data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t upper_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t, typename output_t>
kernel void searchsorted_sorter(
    constant input_t* data_in [[buffer(0)]],
    constant input_t* data_bd [[buffer(1)]],
    device output_t* data_out [[buffer(2)]],
    constant int64_t& idim_in [[buffer(3)]],
    constant int64_t& idim_bd [[buffer(4)]],
    constant int64_t& numel_in [[buffer(5)]],
    constant int64_t& right [[buffer(6)]],
    constant int64_t& is_1d_boundaries [[buffer(7)]],
    constant int64_t* data_sort [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]) {
  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in; tid += tptg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right
        ? lower_bound<input_t>(
              data_bd, start_bd, end_bd, data_in[tid], data_sort) -
            start_bd
        : upper_bound<input_t>(
              data_bd, start_bd, end_bd, data_in[tid], data_sort) -
            start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

template <typename input_t, typename output_t>
kernel void searchsorted(
    constant input_t* data_in [[buffer(0)]],
    constant input_t* data_bd [[buffer(1)]],
    device output_t* data_out [[buffer(2)]],
    constant int64_t& idim_in [[buffer(3)]],
    constant int64_t& idim_bd [[buffer(4)]],
    constant int64_t& numel_in [[buffer(5)]],
    constant int64_t& right [[buffer(6)]],
    constant int64_t& is_1d_boundaries [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]]) {
  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in; tid += tptg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right
        ? lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) -
            start_bd
        : upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) -
            start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

#define REGISTER_SEARCHSORTED_OP(INPUT_T, OUTPUT_T)                          \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T                \
                       "_sorter")]] kernel void                              \
  searchsorted_sorter<INPUT_T, OUTPUT_T>(                                    \
      constant INPUT_T * data_in [[buffer(0)]],                              \
      constant INPUT_T * data_bd [[buffer(1)]],                              \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant int64_t & idim_in [[buffer(3)]],                              \
      constant int64_t & idim_bd [[buffer(4)]],                              \
      constant int64_t & numel_in [[buffer(5)]],                             \
      constant int64_t & right [[buffer(6)]],                                \
      constant int64_t & is_1d_boundaries [[buffer(7)]],                     \
      constant int64_t * data_sort [[buffer(8)]],                            \
      uint2 tgid [[threadgroup_position_in_grid]],                           \
      uint2 tid2 [[thread_position_in_threadgroup]],                         \
      uint2 tptg [[threads_per_threadgroup]]);                               \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T)]] kernel void \
  searchsorted<INPUT_T, OUTPUT_T>(                                           \
      constant INPUT_T * data_in [[buffer(0)]],                              \
      constant INPUT_T * data_bd [[buffer(1)]],                              \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant int64_t & idim_in [[buffer(3)]],                              \
      constant int64_t & idim_bd [[buffer(4)]],                              \
      constant int64_t & numel_in [[buffer(5)]],                             \
      constant int64_t & right [[buffer(6)]],                                \
      constant int64_t & is_1d_boundaries [[buffer(7)]],                     \
      uint2 tgid [[threadgroup_position_in_grid]],                           \
      uint2 tid2 [[thread_position_in_threadgroup]],                         \
      uint2 tptg [[threads_per_threadgroup]]);

REGISTER_SEARCHSORTED_OP(float, int);
REGISTER_SEARCHSORTED_OP(float, long);
REGISTER_SEARCHSORTED_OP(half, int);
REGISTER_SEARCHSORTED_OP(half, long);
#if __METAL_VERSION__ >= 310
REGISTER_SEARCHSORTED_OP(bfloat, int);
REGISTER_SEARCHSORTED_OP(bfloat, long);
#endif
REGISTER_SEARCHSORTED_OP(char, int);
REGISTER_SEARCHSORTED_OP(char, long);
REGISTER_SEARCHSORTED_OP(uchar, int);
REGISTER_SEARCHSORTED_OP(uchar, long);
REGISTER_SEARCHSORTED_OP(short, int);
REGISTER_SEARCHSORTED_OP(short, long);
REGISTER_SEARCHSORTED_OP(int, int);
REGISTER_SEARCHSORTED_OP(int, long);
REGISTER_SEARCHSORTED_OP(long, int);
REGISTER_SEARCHSORTED_OP(long, long);
