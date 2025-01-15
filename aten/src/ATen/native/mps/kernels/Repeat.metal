template <typename T>
kernel void repeat_interleave(
    constant T* repeat_ptr [[buffer(0)]],
    constant int64_t* cumsum_ptr [[buffer(1)]],
    device T* result_ptr [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  int64_t end = cumsum_ptr[tid];
  T repeat = repeat_ptr[tid];
  int64_t start = end - repeat;
  for (uint j = start; j < end; j++) {
    result_ptr[j] = tid;
  }
}

template [[host_name("repeat_interleave_int32_t")]] kernel void
repeat_interleave<int32_t>(
    constant int32_t*,
    constant int64_t*,
    device int32_t*,
    uint);

template [[host_name("repeat_interleave_int64_t")]] kernel void
repeat_interleave<int64_t>(
    constant int64_t*,
    constant int64_t*,
    device int64_t*,
    uint);
