#pragma once
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

// Backward of a per-row normalization: grad_input = scale * (grad - shift -
// dir * slope), where Op supplies grad/dir per element and turns the row sums
// of grad and grad * dir into the coefficients in Op::finish.
template <typename T>
struct NormBackwardCoeffs {
  opmath_t<T> scale, shift, slope;
};

// The row kernels below take the same buffers, so the host binds them once and
// launches one or two passes picked by select_norm_row_config. dims holds
// {rows, row_size, lanes per row, chunk size}.

// Rows of at most simdgroup_size * ILP_PER_THREAD elements: each row is owned
// by `lanes` (a power of two) threads of a simdgroup and reduced with shuffles,
// so one threadgroup covers many rows.
template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_packed(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant uint4& dims [[buffer(6)]],
    device typename Op::extra_t* extra [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  constexpr int reads = ILP_PER_THREAD;
  const uint rows = dims.x, row_size = dims.y, lanes = dims.z;
  // Not via a flat thread index, which wraps at 2**32 threads
  const uint row = tg.y * (tsize.x / lanes) + pos.x / lanes;
  const uint lane = pos.x % lanes, base = lane * reads;
  const bool active = row < rows;
  const uint stat_row = min(row, rows - 1);
  const ulong offset = ulong(row) * row_size + base;
  const int count = active ? min(reads, max(0, int(row_size) - int(base))) : 0;
  const Op op{opmath(stat_a[stat_row]), opmath(stat_b[stat_row]), extra};
  opmath grad[reads], dir[reads];
  opmath sum_grad = 0, sum_prod = 0;
#pragma unroll
  for (int i = 0; i < reads; i++) {
    if (i < count) {
      grad[i] = op.grad(opmath(grad_out[offset + i]), base + i);
      dir[i] = op.dir(opmath(input[offset + i]));
      sum_grad += grad[i];
      sum_prod += grad[i] * dir[i];
    }
  }
  for (ushort delta = lanes / 2; delta > 0; delta /= 2) {
    sum_grad += simd_shuffle_xor(sum_grad, delta);
    sum_prod += simd_shuffle_xor(sum_prod, delta);
  }
  const auto c =
      op.finish(sum_grad, sum_prod, row_size, row, active && lane == 0);
#pragma unroll
  for (int i = 0; i < reads; i++) {
    if (i < count) {
      grad_input[offset + i] =
          static_cast<T>(c.scale * (grad[i] - c.shift - dir[i] * c.slope));
    }
  }
}

// One threadgroup per row, each thread holding ILP_PER_THREAD elements of it.
template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_single_row(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant uint4& dims [[buffer(6)]],
    device typename Op::extra_t* extra [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  constexpr int reads = ILP_PER_THREAD;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x, row_size = dims.y;
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
  const auto sums =
      threadgroup_sum2(tg_a, tg_b, sum_grad, sum_prod, tid, lsize);
  const auto c = op.finish(sums[0], sums[1], row_size, row, tid == 0);
#pragma unroll
  for (int i = 0; i < reads; i++) {
    if (i < count) {
      dx[i] = static_cast<T>(c.scale * (grad[i] - c.shift - dir[i] * c.slope));
    }
  }
}

// One threadgroup per row too long for registers, read once for the sums and
// once more for grad_input.
template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_looped(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant uint4& dims [[buffer(6)]],
    device typename Op::extra_t* extra [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  constexpr int reads = ILP_PER_THREAD;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x, row_size = dims.y;
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
  const auto sums =
      threadgroup_sum2(tg_a, tg_b, sum_grad, sum_prod, tid, lsize);
  const auto c = op.finish(sums[0], sums[1], row_size, row, tid == 0);
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

// Rows too few to fill the GPU are split into chunks: threadgroup (chunk, row)
// reduces its slice into part[row][chunk], then norm_backward_apply adds up the
// row's partials and writes grad_input over the same slice.
template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_row_partials(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device float2* part [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    device typename Op::extra_t* extra [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 ntg [[threadgroups_per_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  constexpr int reads = ILP_PER_THREAD;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x;
  const uint row_size = dims.y, start = tg.x * dims.w;
  const uint len = min(dims.w, row_size - start);
  const ulong offset = ulong(row) * row_size + start;
  const Op op{opmath(stat_a[row]), opmath(stat_b[row]), extra};
  opmath sum_grad = 0, sum_prod = 0;
  for (uint r = tid * reads; r < len; r += lsize * reads) {
#pragma unroll
    for (int i = 0; i < reads; i++) {
      if (r + i < len) {
        const opmath grad =
            op.grad(opmath(grad_out[offset + r + i]), start + r + i);
        sum_grad += grad;
        sum_prod += grad * op.dir(opmath(input[offset + r + i]));
      }
    }
  }
  threadgroup opmath tg_a[simdgroup_size], tg_b[simdgroup_size];
  const auto sums =
      threadgroup_sum2(tg_a, tg_b, sum_grad, sum_prod, tid, lsize);
  if (tid == 0) {
    part[ulong(row) * ntg.x + tg.x] = float2(sums[0], sums[1]);
  }
}

template <typename T, typename Op>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_apply(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    device const float2* part [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    device typename Op::extra_t* extra [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 ntg [[threadgroups_per_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  constexpr int reads = ILP_PER_THREAD;
  const uint row = tg.y, tid = pos.x, lsize = tsize.x;
  threadgroup float2 total;
  if (tid < simdgroup_size) {
    float2 acc = 0;
    for (uint c = tid; c < ntg.x; c += simdgroup_size) {
      acc += part[ulong(row) * ntg.x + c];
    }
    acc = simd_sum(acc);
    if (tid == 0) {
      total = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row_size = dims.y, start = tg.x * dims.w;
  const uint len = min(dims.w, row_size - start);
  const ulong offset = ulong(row) * row_size + start;
  const Op op{opmath(stat_a[row]), opmath(stat_b[row]), extra};
  const auto c =
      op.finish(total.x, total.y, row_size, row, tg.x == 0 && tid == 0);
  for (uint r = tid * reads; r < len; r += lsize * reads) {
#pragma unroll
    for (int i = 0; i < reads; i++) {
      if (r + i < len) {
        const opmath grad =
            op.grad(opmath(grad_out[offset + r + i]), start + r + i);
        const opmath dir = op.dir(opmath(input[offset + r + i]));
        grad_input[offset + r + i] =
            static_cast<T>(c.scale * (grad - c.shift - dir * c.slope));
      }
    }
  }
}

template <typename A>
void store_column(float acc, device A* out, device A*, uint col) {
  out[col] = static_cast<A>(acc);
}

template <typename A>
void store_column(float2 acc, device A* out_a, device A* out_b, uint col) {
  if (out_a != nullptr) {
    out_a[col] = static_cast<A>(acc.x);
  }
  if (out_b != nullptr) {
    out_b[col] = static_cast<A>(acc.y);
  }
}

// Column sums of Op::column_term over a tsize.x (columns) by tsize.y (rows, a
// power of two) tile. Partial sums go to part[split][col], or with a single
// split and no part buffer straight to the outputs.
template <typename T, typename Op, typename A>
[[max_total_threads_per_threadgroup(c10::metal::max_threadgroup_size)]]
kernel void norm_backward_column_tiled(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* stat_a [[buffer(2)]],
    device const T* stat_b [[buffer(3)]],
    device typename Op::acc_t* part [[buffer(4)]],
    device A* out_a [[buffer(5)]],
    device A* out_b [[buffer(6)]],
    constant uint2& rows_cols [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 ntg [[threadgroups_per_grid]],
    uint2 pos [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  using opmath = opmath_t<T>;
  using acc_t = typename Op::acc_t;
  const uint rows = rows_cols.x, cols = rows_cols.y;
  const uint col = tg.x * tsize.x + pos.x;
  const uint row_step = ntg.y * tsize.y;
  acc_t acc = 0;
  if (col < cols) {
    // Count iterations so a 32-bit r cannot wrap; a 64-bit counter is ~20%
    // slower on cache-resident shapes
    uint r = tg.y * tsize.y + pos.y;
    const uint steps = r < rows ? (rows - r - 1) / row_step + 1 : 0;
    const ulong idx_step = ulong(row_step) * cols;
    ulong idx = ulong(r) * cols + col;
    for (uint i = 0; i < steps; ++i, r += row_step, idx += idx_step) {
      acc += Op::column_term(
          opmath(grad_out[idx]), opmath(input[idx]), stat_a, stat_b, r);
    }
  }
  threadgroup acc_t buf[c10::metal::max_threadgroup_size];
  const uint lid = pos.y * tsize.x + pos.x;
  buf[lid] = acc;
  for (uint s = tsize.y / 2; s > 0; s /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pos.y < s) {
      acc += buf[lid + s * tsize.x];
      buf[lid] = acc;
    }
  }
  if (pos.y != 0 || col >= cols) {
    return;
  }
  if (part != nullptr) {
    part[tg.y * cols + col] = acc;
  } else {
    store_column(acc, out_a, out_b, col);
  }
}

#define REGISTER_NORM_BACKWARD_ROWS(PREFIX, T, OP)                         \
  template [[host_name(PREFIX "_packed_" #T)]] [[kernel]]                  \
  decltype(norm_backward_packed<T, OP<T>>) norm_backward_packed<T, OP<T>>; \
  template [[host_name(PREFIX "_single_row_" #T)]] [[kernel]]              \
  decltype(norm_backward_single_row<T, OP<T>>)                             \
      norm_backward_single_row<T, OP<T>>;                                  \
  template [[host_name(PREFIX "_looped_" #T)]] [[kernel]]                  \
  decltype(norm_backward_looped<T, OP<T>>) norm_backward_looped<T, OP<T>>; \
  template [[host_name(PREFIX "_row_partials_" #T)]] [[kernel]]            \
  decltype(norm_backward_row_partials<T, OP<T>>)                           \
      norm_backward_row_partials<T, OP<T>>;                                \
  template [[host_name(PREFIX "_apply_" #T)]] [[kernel]]                   \
  decltype(norm_backward_apply<T, OP<T>>) norm_backward_apply<T, OP<T>>
