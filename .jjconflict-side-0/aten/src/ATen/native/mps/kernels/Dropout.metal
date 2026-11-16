#include <c10/metal/random.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;

// Fused forward dropout: one Philox-4x32-10 round per thread produces 4
// uniforms; each is thresholded against `p_comp = 1 - p` to derive a mask
// element, and the corresponding output is written as `mask ? input * scale :
// 0`. This collapses what used to be two MPS launches (bernoulli + mask*scale)
// into a single bandwidth-bound pass. `cast_to` converts the scalar `0` and
// the scaled product into the destination dtype while doing the right thing
// for complex T (i.e. zero is `T(0, 0)`).
template <typename T>
kernel void dropout_fwd(
    device T* output [[buffer(0)]],
    device bool* mask [[buffer(1)]],
    device const T* input [[buffer(2)]],
    constant float2& params [[buffer(3)]],
    constant long2& seed_base_offset [[buffer(4)]],
    constant uint& numel [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float p_comp = params.x;
  float scale = params.y;
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    float u = c10::metal::detail::uint32_to_uniform_float(raw[i]);
    bool m = u < p_comp;
    mask[base + i] = m;
    output[base + i] = c10::metal::cast_to<T>(m ? input[base + i] * scale : 0);
  }
}

// Backward: grad_input = mask ? grad * scale : 0. Single streaming pass.
template <typename T>
kernel void dropout_bwd(
    device T* grad_input [[buffer(0)]],
    device const T* grad_output [[buffer(1)]],
    device const bool* mask [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  grad_input[tid] =
      c10::metal::cast_to<T>(mask[tid] ? grad_output[tid] * scale : 0);
}

#define REGISTER_DROPOUT(DTYPE)                             \
  template [[host_name("dropout_fwd_" #DTYPE)]] kernel void \
  dropout_fwd<DTYPE>(                                       \
      device DTYPE*,                                        \
      device bool*,                                         \
      device const DTYPE*,                                  \
      constant float2&,                                     \
      constant long2&,                                      \
      constant uint&,                                       \
      uint);                                                \
  template [[host_name("dropout_bwd_" #DTYPE)]] kernel void \
  dropout_bwd<DTYPE>(                                       \
      device DTYPE*,                                        \
      device const DTYPE*,                                  \
      device const bool*,                                   \
      constant float&,                                      \
      uint)

REGISTER_DROPOUT(float);
REGISTER_DROPOUT(half);
REGISTER_DROPOUT(bfloat);
REGISTER_DROPOUT(float2);
REGISTER_DROPOUT(half2);
