#include <ATen/native/mps/kernels/Convolution.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;

// Direct NCHW conv fallback for filter dims >= 256 (MPSGraph miscomputes); each
// thread does R consecutive ow outputs. I is int when all numels fit in int32.
template <typename T, typename I, int R>
kernel void conv2d(
    constant T* input [[buffer(0)]],
    constant T* weight [[buffer(1)]],
    constant T* bias [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant Conv2DParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  using acc_t = ::c10::metal::accum_t<T>;

  const uint owBlocks = (uint(p.outW) + R - 1) / R;
  uint rem = tid;
  const int ow0 = int(rem % owBlocks) * R;
  rem /= owBlocks;
  const int oh = int(rem % uint(p.outH));
  rem /= uint(p.outH);
  const int co = int(rem % uint(p.C_out));
  const int n = int(rem / uint(p.C_out));

  const int cin_base = (co / p.C_out_per_group) * p.C_in_per_group;
  const int ih0 = oh * p.sH - p.padH;
  const int iw_base = ow0 * p.sW - p.padW;
  // Full unit-stride block: loads need no per-lane bounds checks and merge.
  const bool full = (ow0 + R <= p.outW) && (p.sW == 1);

  const acc_t b = p.has_bias ? static_cast<acc_t>(bias[co]) : acc_t(0);
  acc_t acc[R];
  for (int r = 0; r < R; ++r) {
    acc[r] = b;
  }

  const I in_n_off = static_cast<I>(n) * p.C_in * p.H * p.W;
  const I w_co_off = static_cast<I>(co) * p.C_in_per_group * p.kH * p.kW;
  for (int ci = 0; ci < p.C_in_per_group; ++ci) {
    const I in_c_off = in_n_off + static_cast<I>(cin_base + ci) * p.H * p.W;
    const I w_c_off = w_co_off + static_cast<I>(ci) * p.kH * p.kW;
    for (int kh = 0; kh < p.kH; ++kh) {
      const int ih = ih0 + kh * p.dH;
      if (ih < 0 || ih >= p.H) {
        continue;
      }
      const I in_row = in_c_off + static_cast<I>(ih) * p.W;
      const I w_row = w_c_off + static_cast<I>(kh) * p.kW;
      for (int kw = 0; kw < p.kW; ++kw) {
        const int iw = iw_base + kw * p.dW;
        const acc_t wv = static_cast<acc_t>(weight[w_row + kw]);
        if (full && iw >= 0 && iw + R <= p.W) {
          for (int r = 0; r < R; ++r) {
            acc[r] += static_cast<acc_t>(input[in_row + iw + r]) * wv;
          }
        } else {
          for (int r = 0; r < R; ++r) {
            const int iwr = iw + r * p.sW;
            if (ow0 + r < p.outW && iwr >= 0 && iwr < p.W) {
              acc[r] += static_cast<acc_t>(input[in_row + iwr]) * wv;
            }
          }
        }
      }
    }
  }
  const I out_row = ((static_cast<I>(n) * p.C_out + co) * p.outH + oh) * p.outW;
  for (int r = 0; r < R; ++r) {
    if (ow0 + r < p.outW) {
      output[out_row + ow0 + r] = static_cast<T>(acc[r]);
    }
  }
}

#define REGISTER_CONV2D_OP(DTYPE, ITYPE, INAME, R)                       \
  template [[host_name("conv2d_r" #R "_" INAME "_" #DTYPE)]] kernel void \
  conv2d<DTYPE, ITYPE, R>(                                               \
      constant DTYPE * input [[buffer(0)]],                              \
      constant DTYPE * weight [[buffer(1)]],                             \
      constant DTYPE * bias [[buffer(2)]],                               \
      device DTYPE * output [[buffer(3)]],                               \
      constant Conv2DParams & p [[buffer(4)]],                           \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_CONV2D_DTYPE(DTYPE)         \
  REGISTER_CONV2D_OP(DTYPE, int, "i32", 1);  \
  REGISTER_CONV2D_OP(DTYPE, int, "i32", 4);  \
  REGISTER_CONV2D_OP(DTYPE, long, "i64", 1); \
  REGISTER_CONV2D_OP(DTYPE, long, "i64", 4);

REGISTER_CONV2D_DTYPE(float);
REGISTER_CONV2D_DTYPE(half);
REGISTER_CONV2D_DTYPE(bfloat);
