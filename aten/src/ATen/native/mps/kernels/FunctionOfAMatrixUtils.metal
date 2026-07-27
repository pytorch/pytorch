#include <c10/metal/indexing.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T, typename C, typename IdxT>
kernel void compute_linear_combination(
    device void* out_ptr [[buffer(0)]],
    constant void* in_ptr [[buffer(1)]],
    constant void* coeff_ptr [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* out_strides [[buffer(4)]],
    constant long* in_strides [[buffer(5)]],
    constant long* coeff_strides [[buffer(6)]],
    constant vec<IdxT, 4>& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const auto ndim = uint(params.w);
  IdxT pos[max_ndim];
  pos_from_thread_index(IdxT(tid), pos, sizes, ndim);
  const auto out_off = offset_from_coord(pos, out_strides, ndim);
  const auto in_off = offset_from_coord(pos, in_strides, ndim);
  const auto coeff_off = offset_from_coord(pos, coeff_strides, ndim);
  T acc = T(0);
  for (IdxT i = 0; i < params.z; ++i) {
    const auto a =
        val_at_offs<T>(in_ptr, in_off + i * params.x * IdxT(sizeof(T)));
    const auto b =
        val_at_offs<C>(coeff_ptr, coeff_off + i * params.y * IdxT(sizeof(C)));
    acc += a * b;
  }
  ref_at_offs<T>(out_ptr, out_off) += acc;
}

#define INSTANTIATE_CLC(T, C, IDX, SUF)                                   \
  template[[host_name("compute_linear_combination_" #T SUF)]] kernel void \
  compute_linear_combination<T, C, IDX>(                                  \
      device void* out_ptr [[buffer(0)]],                                 \
      constant void* in_ptr [[buffer(1)]],                                \
      constant void* coeff_ptr [[buffer(2)]],                             \
      constant long* sizes [[buffer(3)]],                                 \
      constant long* out_strides [[buffer(4)]],                           \
      constant long* in_strides [[buffer(5)]],                            \
      constant long* coeff_strides [[buffer(6)]],                         \
      constant vec<IDX, 4>& params [[buffer(7)]],                         \
      uint tid [[thread_position_in_grid]]);

#define INSTANTIATE_CLC_ALL(T, C)    \
  INSTANTIATE_CLC(T, C, int, "_u32") \
  INSTANTIATE_CLC(T, C, long, "_u64")

INSTANTIATE_CLC_ALL(float, float);
INSTANTIATE_CLC_ALL(half, half);
INSTANTIATE_CLC_ALL(bfloat, bfloat);
INSTANTIATE_CLC_ALL(float2, float);
