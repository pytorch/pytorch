#include <metal_stdlib>

using namespace metal;

// Threads don't interact, so this only caps the threadgroup. Measured up to 7%
// faster than the default of 1024.
constant constexpr uint kCellMaxThreadsPerThreadgroup = 128;

template <typename T>
[[max_total_threads_per_threadgroup(kCellMaxThreadsPerThreadgroup)]]
kernel void lstm_cell_forward(
    device T* gates,
    device const T* recurrent,
    device const T* cx,
    device T* cy,
    device T* hy,
    device const T* bias,
    constant uint& hidden_size,
    uint tid [[thread_position_in_grid]]) {
  const ulong h = hidden_size, col = tid % h, offset = (tid / h) * 4 * h + col;
  float4 g;
  for (uint k = 0; k < 4; ++k) {
    g[k] = float(gates[offset + k * h]) + float(recurrent[offset + k * h]) +
        float(bias[col + k * h]);
  }
  const float cell = precise::tanh(g.z);
  g = 1.0f / (1.0f + precise::exp(-g));
  g.z = cell;
  const float c = g.y * float(cx[tid]) + g.x * g.z;
  cy[tid] = T(c);
  hy[tid] = T(g.w * precise::tanh(c));
  for (uint k = 0; k < 4; ++k) {
    gates[offset + k * h] = T(g[k]);
  }
}

template <typename T>
kernel void lstm_cell_backward(
    device const T* gates,
    device const T* cx,
    device const T* cy,
    device const T* dy,
    device const T* dh,
    device T* dc,
    device T* dg,
    constant uint& hidden_size,
    uint tid [[thread_position_in_grid]]) {
  const ulong h = hidden_size, offset = (tid / h) * 4 * h + tid % h;
  float4 g;
  for (uint k = 0; k < 4; ++k) {
    g[k] = float(gates[offset + k * h]);
  }
  const float grad_h = float(dy[tid]) + float(dh[tid]);
  const float c = precise::tanh(float(cy[tid]));
  const float grad_c = float(dc[tid]) + grad_h * g.w * (1.0f - c * c);
  dc[tid] = T(grad_c * g.y);
  const float4 grad =
      float4(grad_c * g.z, grad_c * float(cx[tid]), grad_c * g.x, grad_h * c) *
      float4(
          g.x * (1.0f - g.x),
          g.y * (1.0f - g.y),
          1.0f - g.z * g.z,
          g.w * (1.0f - g.w));
  for (uint k = 0; k < 4; ++k) {
    dg[offset + k * h] = T(grad[k]);
  }
}

#define REGISTER_LSTM(T)                                       \
  template [[host_name("lstm_cell_forward_" #T)]] kernel void  \
  lstm_cell_forward<T>(                                        \
      device T*,                                               \
      device const T*,                                         \
      device const T*,                                         \
      device T*,                                               \
      device T*,                                               \
      device const T*,                                         \
      constant uint&,                                          \
      uint);                                                   \
  template [[host_name("lstm_cell_backward_" #T)]] kernel void \
  lstm_cell_backward<T>(                                       \
      device const T*,                                         \
      device const T*,                                         \
      device const T*,                                         \
      device const T*,                                         \
      device const T*,                                         \
      device T*,                                               \
      device T*,                                               \
      constant uint&,                                          \
      uint);

REGISTER_LSTM(float);
REGISTER_LSTM(half);
REGISTER_LSTM(bfloat);
