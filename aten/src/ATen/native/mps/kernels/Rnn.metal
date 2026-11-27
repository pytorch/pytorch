#include <ATen/native/mps/kernels/Rnn.h>
#include <metal_stdlib>

using namespace metal;

template <typename T, typename idx_t>
[[max_total_threads_per_threadgroup(kCellMaxThreadsPerThreadgroup)]]
kernel void lstm_cell_forward(
    device T* gates,
    device const T* recurrent,
    device const T* cx,
    device T* cy,
    device T* hy,
    device const T* bias,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kLstmGates * h + col;
  float4 g;
  for (uint k = 0; k < kLstmGates; ++k) {
    g[k] = float(gates[offset + k * h]) + float(recurrent[offset + k * h]) +
        float(bias[col + k * h]);
  }
  const float cell = precise::tanh(g.z);
  g = 1.0f / (1.0f + precise::exp(-g));
  g.z = cell;
  const float c = g.y * float(cx[idx]) + g.x * g.z;
  cy[idx] = T(c);
  hy[idx] = T(g.w * precise::tanh(c));
  for (uint k = 0; k < kLstmGates; ++k) {
    gates[offset + k * h] = T(g[k]);
  }
}

template <typename T, typename idx_t>
kernel void lstm_cell_backward(
    device const T* gates,
    device const T* cx,
    device const T* cy,
    device const T* dy,
    device const T* dh,
    device T* dc,
    device T* dg,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kLstmGates * h + col;
  float4 g;
  for (uint k = 0; k < kLstmGates; ++k) {
    g[k] = float(gates[offset + k * h]);
  }
  const float grad_h = float(dy[idx]) + float(dh[idx]);
  const float c = precise::tanh(float(cy[idx]));
  const float grad_c = float(dc[idx]) + grad_h * g.w * (1.0f - c * c);
  dc[idx] = T(grad_c * g.y);
  const float4 grad =
      float4(grad_c * g.z, grad_c * float(cx[idx]), grad_c * g.x, grad_h * c) *
      float4(
          g.x * (1.0f - g.x),
          g.y * (1.0f - g.y),
          1.0f - g.z * g.z,
          g.w * (1.0f - g.w));
  for (uint k = 0; k < kLstmGates; ++k) {
    dg[offset + k * h] = T(grad[k]);
  }
}

#define REGISTER_LSTM(T)                                             \
  template [[host_name("lstm_cell_forward_" #T "_u32")]] [[kernel]]  \
  decltype(lstm_cell_forward<T, uint>) lstm_cell_forward<T, uint>;   \
  template [[host_name("lstm_cell_forward_" #T "_u64")]] [[kernel]]  \
  decltype(lstm_cell_forward<T, ulong>) lstm_cell_forward<T, ulong>; \
  template [[host_name("lstm_cell_backward_" #T "_u32")]] [[kernel]] \
  decltype(lstm_cell_backward<T, uint>) lstm_cell_backward<T, uint>; \
  template [[host_name("lstm_cell_backward_" #T "_u64")]] [[kernel]] \
  decltype(lstm_cell_backward<T, ulong>) lstm_cell_backward<T, ulong>

REGISTER_LSTM(float);
REGISTER_LSTM(half);
REGISTER_LSTM(bfloat);
