template <typename T>
kernel void repeat_2d(
    device const T* input,
    device T* output,
    constant int4& sizes_strides,
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const auto src = int2(pos) % sizes_strides.xy;
  output[pos.y * grid.x + pos.x] =
      input[src.x * sizes_strides.z + src.y * sizes_strides.w];
}

#define REGISTER_REPEAT(T, WIDTH)                                       \
  template [[host_name("repeat_2d_" #WIDTH)]] kernel void repeat_2d<T>( \
      device const T*, device T*, constant int4&, uint2, uint2)

REGISTER_REPEAT(uchar, 1);
REGISTER_REPEAT(ushort, 2);
REGISTER_REPEAT(uint, 4);
REGISTER_REPEAT(ulong, 8);

template <typename T>
kernel void repeat_interleave(
    device const T* repeat_ptr [[buffer(0)]],
    device const int64_t* cumsum_ptr [[buffer(1)]],
    device T* result_ptr [[buffer(2)]],
    constant int64_t& repeat_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int64_t end = cumsum_ptr[tid];
  T repeat = repeat_ptr[tid * repeat_stride];
  int64_t start = end - repeat;
  for (uint j = start; j < end; j++) {
    result_ptr[j] = tid;
  }
}

template [[host_name("repeat_interleave_int32_t")]] kernel void
repeat_interleave<int32_t>(
    device const int32_t*,
    device const int64_t*,
    device int32_t*,
    constant int64_t&,
    uint);

template [[host_name("repeat_interleave_int64_t")]] kernel void
repeat_interleave<int64_t>(
    device const int64_t*,
    device const int64_t*,
    device int64_t*,
    constant int64_t&,
    uint);
