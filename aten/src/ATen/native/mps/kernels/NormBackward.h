#pragma once
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

template <typename T>
struct NormBackwardCoeffs {
  opmath_t<T> scale, shift, slope;
};

template <typename T>
struct NormBackwardRow {
  threadgroup opmath_t<T>* tg_a;
  threadgroup opmath_t<T>* tg_b;
  opmath_t<T> sum_grad, sum_prod;
  uint size, index, tid, lsize;
};

template <typename T, typename Op, int reads = c10::metal::ILP_PER_THREAD>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_single_row(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant uint& row_size [[buffer(5)]],
    device typename Op::extra_t* extra [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x;
  const uint base = tid * reads;
  const ulong row_offset = ulong(row) * row_size + base;
  device const T* dy = grad_out + row_offset;
  device const T* x = input + row_offset;
  device T* dx = grad_input + row_offset;
  const int count = min(reads, max(0, int(row_size) - int(base)));
  const Op op{opmath(stat_a[row]), opmath(stat_b[row]), extra};
  opmath grad[reads], dir[reads];
  opmath sum_grad = 0, sum_prod = 0;
#pragma unroll
  for (int i = 0; i < reads; i++) {
    if (i < count) {
      grad[i] = op.grad(opmath(dy[i]), base + i);
      dir[i] = op.dir(opmath(x[i]));
      sum_grad += grad[i];
      sum_prod += grad[i] * dir[i];
    }
  }
  threadgroup opmath tg_a[simdgroup_size], tg_b[simdgroup_size];
  const NormBackwardRow<T> row_state{
      tg_a, tg_b, sum_grad, sum_prod, row_size, row, tid, lsize};
  const auto c = op.coeffs(row_state);
#pragma unroll
  for (int i = 0; i < reads; i++) {
    if (i < count) {
      dx[i] = static_cast<T>(c.scale * (grad[i] - c.shift - dir[i] * c.slope));
    }
  }
}

template <typename T, typename Op, int reads = c10::metal::ILP_PER_THREAD>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_looped(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant uint& row_size [[buffer(5)]],
    device typename Op::extra_t* extra [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x;
  const ulong row_offset = ulong(row) * row_size;
  device const T* dy = grad_out + row_offset;
  device const T* x = input + row_offset;
  device T* dx = grad_input + row_offset;
  const Op op{opmath(stat_a[row]), opmath(stat_b[row]), extra};
  opmath sum_grad = 0, sum_prod = 0;
  for (uint r = tid * reads; r < row_size; r += lsize * reads) {
#pragma unroll
    for (int i = 0; i < reads; i++) {
      const uint j = r + i;
      if (j < row_size) {
        const opmath grad = op.grad(opmath(dy[j]), j);
        sum_grad += grad;
        sum_prod += grad * op.dir(opmath(x[j]));
      }
    }
  }
  threadgroup opmath tg_a[simdgroup_size], tg_b[simdgroup_size];
  const NormBackwardRow<T> row_state{
      tg_a, tg_b, sum_grad, sum_prod, row_size, row, tid, lsize};
  const auto c = op.coeffs(row_state);
  for (uint r = tid * reads; r < row_size; r += lsize * reads) {
#pragma unroll
    for (int i = 0; i < reads; i++) {
      const uint j = r + i;
      if (j < row_size) {
        const opmath grad = op.grad(opmath(dy[j]), j);
        const opmath dir = op.dir(opmath(x[j]));
        dx[j] = static_cast<T>(c.scale * (grad - c.shift - dir * c.slope));
      }
    }
  }
}

template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_column_partials(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device typename Op::acc_t* part [[buffer(2)]],
    constant uint2& rows_cols [[buffer(3)]],
    device const T* stat_a [[buffer(4)]],
    device const T* stat_b [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 ntg [[threadgroups_per_grid]],
    uint2 pos [[thread_position_in_threadgroup]]) {
  using opmath = opmath_t<T>;
  using acc_t = typename Op::acc_t;
  const uint rows = rows_cols.x, cols = rows_cols.y;
  const uint col = tg.x * simdgroup_size + pos.x;
  const uint row0 = tg.y * simdgroup_size + pos.y;
  const uint row_step = ntg.y * simdgroup_size;
  acc_t acc = 0;
  if (col < cols) {
    ulong idx = ulong(row0) * cols + col;
    const ulong idx_step = ulong(row_step) * cols;
    for (uint i = row0; i < rows; i += row_step, idx += idx_step) {
      acc += Op::column_term(
          opmath(grad_out[idx]), opmath(input[idx]), stat_a, stat_b, i);
    }
  }
  threadgroup acc_t buf[simdgroup_size][simdgroup_size];
  buf[pos.y][pos.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const acc_t total = c10::metal::simd_sum(buf[pos.x][pos.y]);
  const uint out_col = tg.x * simdgroup_size + pos.y;
  if (pos.x == 0 && out_col < cols) {
    part[tg.y * cols + out_col] = total;
  }
}
