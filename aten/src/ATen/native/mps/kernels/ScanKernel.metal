#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include <c10/metal/common.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>

using namespace c10::metal;

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    return c10::metal::logaddexp(x, y);
  }
};

C10_METAL_CONSTEXPR auto simd_size = c10::metal::simdgroup_size;

// The reminder of this file contains cummin and cummax implementations adapted
// from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scan.h
//
// The original MLX kernels have been modified to integrate with PyTorch's MPS
// backend. Most notably:
//  - Keeping track and returning indices, which MLX kernels don't do.
//  - Perform computations on half/bfloat tensors at higher precision (float)
//  via c10::metal::accum_t
//
// Original work is licensed under MIT License:
// https://github.com/ml-explore/mlx/blob/main/LICENSE

inline int64_t simd_shuffle_and_fill_up(
    int64_t data,
    int64_t filling,
    uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

// Inclusive (and derived exclusive) SIMD scan for scalar acc_t ops, expressed
// through the op's operator(). Shared by LogCumSumExp/CumProd/CumSum.
template <typename Op, typename acc_t>
inline acc_t simd_scan(Op op, acc_t x) {
  for (int i = 1; i <= 16; i *= 2) {
    x = op(x, simd_shuffle_and_fill_up(x, Op::init, i));
  }
  return x;
}

template <typename Op, typename acc_t>
inline acc_t simd_exclusive_scan(Op op, acc_t x) {
  return simd_shuffle_and_fill_up(simd_scan(op, x), Op::init, 1);
}

template <typename T, typename acc_t = accum_t<T>>
struct LogCumSumExpOp {
  static constexpr constant acc_t init = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? -metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::lowest());

  acc_t operator()(acc_t a, acc_t b) {
    return LogAddExp{}(a, b);
  }
};

template <typename T, ::metal::enable_if_t<!is_complex_v<T>, bool> = true>
constexpr T cum_op_init(int v) {
  return static_cast<T>(v);
}
template <typename T, ::metal::enable_if_t<is_complex_v<T>, bool> = true>
constexpr T cum_op_init(int v) {
  return T(static_cast<float>(v), 0.0f);
}

template <typename T, typename acc_t = accum_t<T>>
struct CumProdOp {
  static constexpr constant acc_t init = cum_op_init<acc_t>(1);

  acc_t operator()(acc_t a, acc_t b) {
    return c10::metal::mul(a, b);
  }
};

template <typename T, typename acc_t = accum_t<T>>
struct CumSumOp {
  static constexpr constant acc_t init = cum_op_init<acc_t>(0);

  acc_t operator()(acc_t a, acc_t b) {
    return a + b;
  }
};

// Pair structure to hold value and index for cummin/cummax operations
template <typename T, typename acc_t = accum_t<T>>
struct ValueIndexPair {
  acc_t value;
  int64_t index;
};

// Helper function to create ValueIndexPair
template <typename T, typename acc_t = accum_t<T>>
inline ValueIndexPair<T, acc_t> make_pair(acc_t v, int64_t i) {
  ValueIndexPair<T, acc_t> result;
  result.value = v;
  result.index = i;
  return result;
}

template <typename T, typename acc_t>
inline ValueIndexPair<T, acc_t> simd_shuffle_and_fill_up(
    ValueIndexPair<T, acc_t> data,
    ValueIndexPair<T, acc_t> filling,
    uint16_t delta) {
  return make_pair<T, acc_t>(
      simd_shuffle_and_fill_up(data.value, filling.value, delta),
      simd_shuffle_and_fill_up(data.index, filling.index, delta));
}

// Value/index SIMD scan shared by cummin and cummax; the op supplies get_init()
// and operator(). Mirrors the scalar overloads above (which the pair shuffle
// overload lets these reuse).
template <typename Op, typename T, typename acc_t>
inline ValueIndexPair<T, acc_t> simd_scan(Op op, ValueIndexPair<T, acc_t> x) {
  ValueIndexPair<T, acc_t> init = op.get_init();
  for (int i = 1; i <= 16; i *= 2) {
    x = op(x, simd_shuffle_and_fill_up(x, init, i));
  }
  return x;
}

template <typename Op, typename T, typename acc_t>
inline ValueIndexPair<T, acc_t> simd_exclusive_scan(
    Op op,
    ValueIndexPair<T, acc_t> x) {
  return simd_shuffle_and_fill_up(simd_scan(op, x), op.get_init(), 1);
}

template <typename T, typename acc_t = accum_t<T>>
struct CumMinOp {
  using pair_t = ValueIndexPair<T, acc_t>;

  static constexpr constant acc_t init_val = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::max());

  static pair_t get_init() {
    return make_pair<T, acc_t>(init_val, 0);
  }

  pair_t operator()(pair_t a, pair_t b) {
    if (::metal::isnan(static_cast<float>(a.value)) &&
        ::metal::isnan(static_cast<float>(b.value))) {
      return (a.index >= b.index) ? a : b;
    } else if (::metal::isnan(static_cast<float>(a.value))) {
      return a;
    } else if (::metal::isnan(static_cast<float>(b.value))) {
      return b;
    } else if (a.value < b.value) {
      return a;
    } else if (a.value > b.value) {
      return b;
    } else {
      return (a.index >= b.index) ? a : b;
    }
  }
};

template <typename T, typename acc_t = accum_t<T>>
struct CumMaxOp {
  using pair_t = ValueIndexPair<T, acc_t>;

  static constexpr constant acc_t init_val = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? -metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::lowest());

  static pair_t get_init() {
    return make_pair<T, acc_t>(init_val, 0);
  }

  pair_t operator()(pair_t a, pair_t b) {
    if (::metal::isnan(static_cast<float>(a.value)) &&
        ::metal::isnan(static_cast<float>(b.value))) {
      return (a.index >= b.index) ? a : b;
    } else if (::metal::isnan(static_cast<float>(a.value))) {
      return a;
    } else if (::metal::isnan(static_cast<float>(b.value))) {
      return b;
    } else if (a.value > b.value) {
      return a;
    } else if (a.value < b.value) {
      return b;
    } else {
      return (a.index >= b.index) ? a : b;
    }
  }
};

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void load_unsafe(acc_t values[N_READS], const device T* input) {
  for (int i = 0; i < N_READS; i++) {
    values[i] = static_cast<acc_t>(input[i]);
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void load_safe(
    acc_t values[N_READS],
    const device T* input,
    int start,
    int total,
    acc_t init) {
  for (int i = 0; i < N_READS; i++) {
    values[i] = (start + i < total) ? static_cast<acc_t>(input[i]) : init;
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void write_unsafe(acc_t values[N_READS], device T* out) {
  for (int i = 0; i < N_READS; i++) {
    out[i] = static_cast<T>(values[i]);
  }
}

template <typename T, int N_READS, typename acc_t = accum_t<T>>
inline void write_safe(
    acc_t values[N_READS],
    device T* out,
    int start,
    int total) {
  for (int i = 0; i < N_READS; i++) {
    if (start + i < total) {
      out[i] = static_cast<T>(values[i]);
    }
  }
}

// Utility function for ceiling division
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// Reduce one acc_t per thread to the total, broadcast to all threads.
// `smem` holds >= simd_size acc_t; Op must define simd_scan and init.
template <typename Op, typename acc_t>
inline acc_t threadgroup_reduce(
    acc_t val,
    threadgroup acc_t* smem,
    uint lsize_x,
    uint simd_lane_id,
    uint simd_group_id) {
  Op op;
  uint simd_groups = lsize_x / simd_size;
  acc_t simd_total = simd_shuffle(simd_scan(op, val), simd_size - 1);
  if (simd_lane_id == 0) {
    smem[simd_group_id] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc_t v = (simd_lane_id < simd_groups) ? smem[simd_lane_id] : Op::init;
    acc_t total = simd_shuffle(simd_scan(op, v), simd_size - 1);
    if (simd_lane_id == 0) {
      smem[0] = total;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  acc_t result = smem[0];
  // Let callers safely reuse `smem` after the reduction.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
inline acc_t block_reduce_impl(
    const device T* in,
    uint len,
    threadgroup acc_t* smem,
    uint lid_x,
    uint lsize_x,
    uint simd_lane_id,
    uint simd_group_id) {
  Op op;
  acc_t acc = Op::init;
  acc_t values[N_READS];
  for (uint r = 0; r < ceildiv(len, N_READS * lsize_x); r++) {
    uint offset = r * lsize_x * N_READS + lid_x * N_READS;
    if ((offset + N_READS) < len) {
      load_unsafe<T, N_READS>(values, in + offset);
    } else {
      load_safe<T, N_READS>(values, in + offset, offset, len, Op::init);
    }
    for (int i = 0; i < N_READS; i++) {
      acc = op(acc, values[i]);
    }
  }
  return threadgroup_reduce<Op>(
      acc, smem, lsize_x, simd_lane_id, simd_group_id);
}

// Threadgroup inclusive scan of `len` elements seeded with `init_prefix`
// (carry-in from preceding blocks). `simdgroup_sums` holds >= simd_size acc_t.
// IN is the read type, T the write type (differ only for int32->int64 fusion).
template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
inline void block_scan_impl(
    const device IN* in,
    device T* out,
    uint len,
    acc_t init_prefix,
    threadgroup acc_t* simdgroup_sums,
    uint lid_x,
    uint lsize_x,
    uint simd_lane_id,
    uint simd_group_id) {
  Op op;
  uint simd_groups = lsize_x / simd_size;
  acc_t prefix = init_prefix;
  acc_t values[N_READS];
  for (uint r = 0; r < ceildiv(len, N_READS * lsize_x); r++) {
    uint offset = r * lsize_x * N_READS + lid_x * N_READS;
    if ((offset + N_READS) < len) {
      load_unsafe<IN, N_READS>(values, in + offset);
    } else {
      load_safe<IN, N_READS>(values, in + offset, offset, len, Op::init);
    }
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }
    acc_t prev_thread = simd_exclusive_scan(op, values[N_READS - 1]);
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      simdgroup_sums[simd_lane_id] =
          simd_exclusive_scan(op, simdgroup_sums[simd_lane_id]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }
    if ((offset + N_READS) < len) {
      write_unsafe<T, N_READS>(values, out + offset);
    } else {
      write_safe<T, N_READS>(values, out + offset, offset, len);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
    // Ensure every thread has read the carry before the next iteration
    // overwrites simdgroup_sums; otherwise long (multi-iteration) scans race.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Inclusive scan along innermost dimension for contiguous tensors
template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_innermost_dim(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // One threadgroup scans one row [gid.y, gid.z]; see block_scan_impl.
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  threadgroup acc_t simdgroup_sums[simd_size];
  block_scan_impl<T, Op, N_READS, IN>(
      in + offset,
      out + offset,
      uint(axis_size),
      Op::init,
      simdgroup_sums,
      lid.x,
      lsize.x,
      simd_lane_id,
      simd_group_id);
}

// Inclusive scan along outer dimension for contiguous tensors
template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_outer_dim(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(T);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup acc_t read_buffer[BM * BN_pad];
  acc_t values[n_scans];
  acc_t prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = Op::init;
  }

  // Compute offsets
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  threadgroup acc_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup acc_t* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;

    // Read into shared memory with type conversion
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = static_cast<acc_t>(in[index_y * stride + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          read_into[i] = static_cast<acc_t>(in[index_y * stride + i]);
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read strided into registers
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Perform the scan
    for (int i = 0; i < n_scans; i++) {
      values[i] = simd_scan(op, values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }

    // Write to shared memory
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to device memory with type conversion
    if (check_index_y < axis_size && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        out[index_y * stride + i] = static_cast<T>(read_into[i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
          out[index_y * stride + i] = static_cast<T>(read_into[i]);
        }
      }
    }
  }
}

// Fused transpose-scan: read storage [axis_size, n_cols] strided, scan along
// axis, write contiguous [n_cols, axis_size]; both accesses coalesce, no copy.
template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_innermost_transposed(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& n_cols [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = BN + 1; // coprime with 32 banks -> conflict-free
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup acc_t read_buffer[BM * BN_pad];
  acc_t values[n_scans];
  acc_t prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = Op::init;
  }

  size_t col_block = gid.y + gsize.y * size_t(gid.z);
  if (col_block >= stride_blocks) {
    return;
  }
  size_t col_base = col_block * BN;
  uint col_limit = uint(min(size_t(BN), n_cols - col_base));

  uint read_offset_y = (lid.x * N_READS) / BN; // axis within chunk
  uint read_offset_x = (lid.x * N_READS) % BN; // column within block
  const device IN* in_p = in + col_base + read_offset_x;
  threadgroup acc_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup acc_t* read_from =
      read_buffer + simd_lane_id * BN_pad + simd_group_id * n_scans;

  for (uint j = 0; j < axis_size; j += BM) {
    uint index_y = j + read_offset_y;
    // Coalesced strided read: consecutive threads read consecutive columns.
    if (index_y < axis_size && (read_offset_x + N_READS) <= col_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = static_cast<acc_t>(in_p[size_t(index_y) * n_cols + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = (index_y < axis_size && (read_offset_x + i) < col_limit)
            ? static_cast<acc_t>(in_p[size_t(index_y) * n_cols + i])
            : Op::init;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane = axis position within the chunk; each simdgroup owns n_scans
    // columns.
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < n_scans; i++) {
      values[i] = simd_scan(op, values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }

    // Transposed coalesced write straight from registers: consecutive lanes
    // (axis positions) hit consecutive contiguous-output addresses.
    uint axis_pos = j + simd_lane_id;
    if (axis_pos < axis_size) {
      for (int i = 0; i < n_scans; i++) {
        uint col = uint(col_base) + simd_group_id * n_scans + i;
        if (col < n_cols) {
          out[size_t(col) * axis_size + axis_pos] = static_cast<T>(values[i]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_with_indices_innermost_dim(
    const device T* in [[buffer(0)]],
    device T* out_values [[buffer(1)]],
    device int64_t* out_indices [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  using pair_t = typename Op::pair_t;

  // Position the pointers
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out_values += offset;
  out_indices += offset;

  // Compute the number of simd_groups
  uint simd_groups = lsize.x / simd_size;

  // Allocate memory
  pair_t prefix = op.get_init();
  pair_t values[N_READS];
  threadgroup pair_t simdgroup_sums[32];

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Compute the block offset
    uint offset_idx = r * lsize.x * N_READS + lid.x * N_READS;

    // Read the values as pairs
    for (int i = 0; i < N_READS; i++) {
      if ((offset_idx + i) < axis_size) {
        values[i] = make_pair<T, acc_t>(
            static_cast<acc_t>(in[offset_idx + i]), offset_idx + i);
      } else {
        values[i] = op.get_init();
      }
    }

    // Compute an inclusive scan per thread
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums
    pair_t prev_thread = simd_exclusive_scan(op, values[N_READS - 1]);

    // Write simdgroup_sums to SM
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      pair_t prev_simdgroup =
          simd_exclusive_scan(op, simdgroup_sums[simd_lane_id]);
      simdgroup_sums[simd_lane_id] = prev_simdgroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute the output
    for (int i = 0; i < N_READS; i++) {
      values[i] = op(values[i], prefix);
      values[i] = op(values[i], simdgroup_sums[simd_group_id]);
      values[i] = op(values[i], prev_thread);
    }

    // Write the values
    for (int i = 0; i < N_READS; i++) {
      if ((offset_idx + i) < axis_size) {
        out_values[offset_idx + i] = static_cast<T>(values[i].value);
        out_indices[offset_idx + i] = values[i].index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Share the prefix
    if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
      simdgroup_sums[0] = values[N_READS - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = simdgroup_sums[0];
  }
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_with_indices_outer_dim(
    const device T* in [[buffer(0)]],
    device T* out_values [[buffer(1)]],
    device int64_t* out_indices [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    const constant size_t& stride [[buffer(4)]],
    const constant size_t& stride_blocks [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BN_pad = 32 + 16 / sizeof(T);
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;
  using pair_t = typename Op::pair_t;

  threadgroup pair_t read_buffer[BM * BN_pad];
  pair_t values[n_scans];
  pair_t prefix[n_scans];
  for (int i = 0; i < n_scans; i++) {
    prefix[i] = op.get_init();
  }

  // Compute offsets
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t offset = full_gid / stride_blocks * axis_size * stride;
  size_t global_index_x = full_gid % stride_blocks * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = stride - global_index_x;
  in += offset + global_index_x + read_offset_x;
  out_values += offset + global_index_x + read_offset_x;
  out_indices += offset + global_index_x + read_offset_x;
  threadgroup pair_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup pair_t* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  for (uint j = 0; j < axis_size; j += BM) {
    // Calculate the indices for the current thread
    uint index_y = j + read_offset_y;
    uint check_index_y = index_y;

    // Read into shared memory as pairs
    for (int i = 0; i < N_READS; i++) {
      if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
        // For cummin/cummax, the index should represent the position along the
        // scan axis
        read_into[i] = make_pair<T, acc_t>(
            static_cast<acc_t>(in[index_y * stride + i]), index_y);
      } else {
        read_into[i] = op.get_init();
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read strided into registers
    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Perform the scan
    for (int i = 0; i < n_scans; i++) {
      values[i] = simd_scan(op, values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = make_pair<T, acc_t>(
          simd_shuffle(values[i].value, simd_size - 1),
          simd_shuffle(values[i].index, simd_size - 1));
    }

    // Write to shared memory
    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to device memory
    for (int i = 0; i < N_READS; i++) {
      if (check_index_y < axis_size && (read_offset_x + i) < stride_limit) {
        out_values[index_y * stride + i] = static_cast<T>(read_into[i].value);
        out_indices[index_y * stride + i] = read_into[i].index;
      }
    }
  }
}

// Three-pass scan (block_reduce -> scan_block_sums -> block_carry) over a
// [n_scans, axis_size] tensor; splits a long axis across threadgroups.
template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_block_reduce(
    const device IN* in [[buffer(0)]],
    device acc_t* block_sums [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& block_size [[buffer(3)]],
    const constant size_t& num_blocks [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup acc_t smem[simd_size];
  size_t block_start = size_t(gid.x) * block_size;
  uint len = uint(min(size_t(block_size), axis_size - block_start));
  acc_t total = block_reduce_impl<IN, Op, N_READS, acc_t>(
      in + size_t(gid.y) * axis_size + block_start,
      len,
      smem,
      lid.x,
      lsize.x,
      simd_lane_id,
      simd_group_id);
  if (lid.x == 0) {
    block_sums[size_t(gid.y) * num_blocks + gid.x] = total;
  }
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_block_sums(
    device acc_t* block_sums [[buffer(0)]],
    const constant size_t& num_blocks [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup acc_t smem[simd_size];
  device acc_t* p = block_sums + size_t(gid.y) * num_blocks;
  block_scan_impl<acc_t, Op, N_READS, acc_t>(
      p,
      p,
      uint(num_blocks),
      Op::init,
      smem,
      lid.x,
      lsize.x,
      simd_lane_id,
      simd_group_id);
}

template <
    typename T,
    typename Op,
    int N_READS,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_block_carry(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device acc_t* block_sums [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    const constant size_t& block_size [[buffer(4)]],
    const constant size_t& num_blocks [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup acc_t simdgroup_sums[simd_size];
  Op op;
  // Carry-in = sum of preceding blocks' totals; block_sums is tiny/cached, so
  // summing inline beats a separate block-scan dispatch.
  const device acc_t* row_sums = block_sums + size_t(gid.y) * num_blocks;
  acc_t partial = Op::init;
  for (uint b = lid.x; b < gid.x; b += lsize.x) {
    partial = op(partial, row_sums[b]);
  }
  acc_t carry = threadgroup_reduce<Op>(
      partial, simdgroup_sums, lsize.x, simd_lane_id, simd_group_id);

  size_t block_start = size_t(gid.x) * block_size;
  uint len = uint(min(size_t(block_size), axis_size - block_start));
  size_t base = size_t(gid.y) * axis_size + block_start;
  block_scan_impl<T, Op, N_READS, IN>(
      in + base,
      out + base,
      len,
      carry,
      simdgroup_sums,
      lid.x,
      lsize.x,
      simd_lane_id,
      simd_group_id);
}

// Single-pass decoupled look-back (Merrill & Garland), 1 read + 1 write.
// Dynamic tile-id claim gives forward progress; metal3.1 -> two 32-bit sentinel
// words.
constant constexpr uint kScanEmpty = 0xFFFFFFFFu;

inline uint scan_encode(float value) {
  uint bits = as_type<uint>(value);
  return bits == kScanEmpty ? 0x7FC00000u
                            : bits; // canonical NaN, never the sentinel
}

// Orders the look-back's cross-threadgroup carry handoff (producer release
// before publish, consumer acquire after observe); relaxed atomics alone don't.
// seq_cst fences exist only on Metal 3.2+, so on macOS 14 this is a no-op and
// the caller routes to the multi-block kernels instead.
inline void scan_lookback_fence() {
#if defined(__HAVE_ATOMIC_FENCE__)
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst);
#endif
}

// Walk predecessor slots back from `look` (stepping by `stride`, stopping after
// `first`), folding each into the carry; a resolved inclusive prefix short-
// circuits. Returns the exclusive prefix for the current tile.
template <typename Op, typename acc_t>
inline acc_t scan_lookback(
    device atomic_uint* aggregates,
    device atomic_uint* inclusive,
    uint look,
    uint first,
    uint stride) {
  Op op;
  acc_t carry = Op::init;
  while (true) {
    uint iw = atomic_load_explicit(&inclusive[look], memory_order_relaxed);
    if (iw != kScanEmpty) {
      scan_lookback_fence(); // acquire the producer's writes before using them
      return op(as_type<acc_t>(iw), carry);
    }
    uint aw = atomic_load_explicit(&aggregates[look], memory_order_relaxed);
    if (aw != kScanEmpty) {
      scan_lookback_fence(); // acquire
      carry = op(as_type<acc_t>(aw), carry);
      if (look == first) {
        return carry;
      }
      look -= stride;
    }
  }
}

template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_contig_decoupled(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    device atomic_uint* tile_counter [[buffer(2)]],
    device atomic_uint* aggregates [[buffer(3)]],
    device atomic_uint* inclusive [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    const constant uint& num_tiles [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup uint tg_tile_id;
  threadgroup acc_t simdgroup_sums[simd_size];
  threadgroup acc_t tg_total;
  threadgroup acc_t tg_carry;

  // 1) Dynamically claim a tile id (the forward-progress guarantee).
  if (lid == 0) {
    tg_tile_id =
        atomic_fetch_add_explicit(tile_counter, 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint tile_id = tg_tile_id;
  const uint tile_in_row = tile_id % num_tiles;
  const uint row = tile_id / num_tiles;

  const uint tile = lsize * N_READS;
  const uint block_start = tile_in_row * tile;
  const uint len = min(tile, axis_size - block_start);
  const size_t base = size_t(row) * axis_size + block_start;

  // 2) Local inclusive scan of the tile, kept in registers (carry applied
  // later).
  acc_t values[N_READS];
  const uint offset = lid * N_READS;
  if (offset + N_READS <= len) {
    load_unsafe<T, N_READS>(values, in + base + offset);
  } else {
    load_safe<T, N_READS>(values, in + base + offset, offset, len, Op::init);
  }
  for (int i = 1; i < N_READS; i++) {
    values[i] = op(values[i], values[i - 1]);
  }
  const uint simd_groups = lsize / simd_size;
  acc_t prev_thread = simd_exclusive_scan(op, values[N_READS - 1]);
  if (simd_lane_id == simd_size - 1) {
    simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc_t v =
        (simd_lane_id < simd_groups) ? simdgroup_sums[simd_lane_id] : Op::init;
    simdgroup_sums[simd_lane_id] = simd_exclusive_scan(op, v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const acc_t simd_prefix = simdgroup_sums[simd_group_id];
  for (int i = 0; i < N_READS; i++) {
    values[i] = op(op(values[i], simd_prefix), prev_thread);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
    tg_total = values[N_READS - 1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const acc_t block_total = tg_total;

  // 3-5) Publish aggregate, look back for the exclusive prefix, publish
  // inclusive. A single thread owns the cross-threadgroup protocol; the rest
  // wait below.
  if (lid == 0) {
    acc_t carry = Op::init;
    if (tile_in_row == 0) {
      // Tile 0's aggregate is already its inclusive prefix.
      scan_lookback_fence(); // release before publishing the slot
      atomic_store_explicit(
          &inclusive[tile_id], scan_encode(block_total), memory_order_relaxed);
    } else {
      scan_lookback_fence(); // release
      atomic_store_explicit(
          &aggregates[tile_id], scan_encode(block_total), memory_order_relaxed);
      carry = scan_lookback<Op, acc_t>(
          aggregates, inclusive, tile_id - 1, tile_id - tile_in_row, 1);
      scan_lookback_fence(); // release
      atomic_store_explicit(
          &inclusive[tile_id],
          scan_encode(op(carry, block_total)),
          memory_order_relaxed);
    }
    tg_carry = carry;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const acc_t carry = tg_carry;

  // 6) Seed each element with the exclusive prefix and write out.
  for (int i = 0; i < N_READS; i++) {
    values[i] = op(carry, values[i]);
  }
  if (offset + N_READS <= len) {
    write_unsafe<T, N_READS>(values, out + base + offset);
  } else {
    write_safe<T, N_READS>(values, out + base + offset, offset, len);
  }
}

// Componentwise inclusive scan of one tile into registers (no carry); tile
// total per component goes to tg_total. Scalar per-component simd scans (not
// the float4-batched path) keep int64 accumulators exact past 2^24.
template <typename Op, int N_READS, int VEC, typename acc_t, typename IN>
inline void scan_vec_tile_inclusive(
    const device IN* in,
    size_t base,
    uint len,
    uint soff,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup acc_t* simdgroup_sums, // simd_size * VEC
    threadgroup acc_t* tg_total, // VEC
    thread acc_t* vals) { // N_READS * VEC
  Op op;
  const uint simd_groups = lsize / simd_size;
  for (int i = 0; i < N_READS; i++) {
    const uint s = soff + i;
    for (int c = 0; c < VEC; c++) {
      vals[i * VEC + c] = (s < len)
          ? static_cast<acc_t>(in[base + size_t(s) * VEC + c])
          : Op::init;
    }
  }
  for (int i = 1; i < N_READS; i++) {
    for (int c = 0; c < VEC; c++) {
      vals[i * VEC + c] = op(vals[i * VEC + c], vals[(i - 1) * VEC + c]);
    }
  }
  acc_t prev_thread[VEC];
  for (int c = 0; c < VEC; c++) {
    const acc_t last = vals[(N_READS - 1) * VEC + c];
    prev_thread[c] = simd_exclusive_scan(op, last);
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id * VEC + c] = op(prev_thread[c], last);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    for (int c = 0; c < VEC; c++) {
      acc_t v = (simd_lane_id < simd_groups)
          ? simdgroup_sums[simd_lane_id * VEC + c]
          : Op::init;
      simdgroup_sums[simd_lane_id * VEC + c] = simd_exclusive_scan(op, v);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N_READS; i++) {
    for (int c = 0; c < VEC; c++) {
      vals[i * VEC + c] =
          op(op(vals[i * VEC + c], simdgroup_sums[simd_group_id * VEC + c]),
             prev_thread[c]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == simd_groups - 1 && simd_lane_id == simd_size - 1) {
    for (int c = 0; c < VEC; c++) {
      tg_total[c] = vals[(N_READS - 1) * VEC + c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// VEC-wide decoupled look-back. A tile's VEC columns publish behind one
// `status` word (0 empty, 1 aggregate, 2 inclusive ready), so the carry handoff
// needs one device fence per step regardless of column count. Walks
// predecessors of `look` down to `first`, folding into carry[0..VEC); an
// inclusive prefix short-circuits.
template <typename Op, typename acc_t, int VEC>
inline void scan_lookback_vec(
    device atomic_uint* status,
    device atomic_uint* aggregates,
    device atomic_uint* inclusive,
    uint look,
    uint first,
    thread acc_t* carry) {
  Op op;
  while (true) {
    uint st;
    do {
      st = atomic_load_explicit(&status[look], memory_order_relaxed);
    } while (st == 0u);
    scan_lookback_fence(); // acquire the producer's value writes
    device atomic_uint* slot = (st == 2u) ? inclusive : aggregates;
    for (int c = 0; c < VEC; c++) {
      uint w =
          atomic_load_explicit(&slot[look * VEC + c], memory_order_relaxed);
      carry[c] = op(as_type<acc_t>(w), carry[c]);
    }
    if (st == 2u || look == first) {
      return;
    }
    look -= 1;
  }
}

// Single-pass decoupled look-back for an outer scan with inner stride VEC, over
// [n_orows, axis_size, VEC] contiguous, one threadgroup per (orow, axis-tile).
// 1 read + 1 write, no global reduce/carry barrier. Float-accumulate only.
template <
    typename T,
    typename Op,
    int N_READS,
    int VEC,
    typename acc_t = accum_t<T>>
kernel void scan_strided_decoupled(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    device atomic_uint* tile_counter [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    device atomic_uint* aggregates [[buffer(4)]],
    device atomic_uint* inclusive [[buffer(5)]],
    const constant uint& axis_size [[buffer(6)]],
    const constant uint& num_tiles [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup uint tg_tile_id;
  threadgroup acc_t simdgroup_sums[simd_size * VEC];
  threadgroup acc_t tg_total[VEC];
  threadgroup acc_t tg_carry[VEC];

  if (lid == 0) {
    tg_tile_id =
        atomic_fetch_add_explicit(tile_counter, 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint tile_id = tg_tile_id;
  const uint tile_in_row = tile_id % num_tiles;
  const uint row = tile_id / num_tiles;

  const uint tile = lsize * N_READS;
  const uint block_start = tile_in_row * tile;
  const uint len = min(tile, axis_size - block_start);
  const size_t base = (size_t(row) * axis_size + block_start) * VEC;

  acc_t vals[N_READS * VEC];
  scan_vec_tile_inclusive<Op, N_READS, VEC, acc_t, T>(
      in,
      base,
      len,
      lid * N_READS,
      lsize,
      simd_lane_id,
      simd_group_id,
      simdgroup_sums,
      tg_total,
      vals);

  if (lid == 0) {
    acc_t carry[VEC];
    for (int c = 0; c < VEC; c++) {
      carry[c] = Op::init;
    }
    if (tile_in_row == 0) {
      for (int c = 0; c < VEC; c++) {
        atomic_store_explicit(
            &inclusive[tile_id * VEC + c],
            scan_encode(tg_total[c]),
            memory_order_relaxed);
      }
      scan_lookback_fence(); // release values before publishing the flag
      atomic_store_explicit(&status[tile_id], 2u, memory_order_relaxed);
    } else {
      for (int c = 0; c < VEC; c++) {
        atomic_store_explicit(
            &aggregates[tile_id * VEC + c],
            scan_encode(tg_total[c]),
            memory_order_relaxed);
      }
      scan_lookback_fence(); // release
      atomic_store_explicit(&status[tile_id], 1u, memory_order_relaxed);
      scan_lookback_vec<Op, acc_t, VEC>(
          status,
          aggregates,
          inclusive,
          tile_id - 1,
          tile_id - tile_in_row,
          carry);
      for (int c = 0; c < VEC; c++) {
        atomic_store_explicit(
            &inclusive[tile_id * VEC + c],
            scan_encode(op(carry[c], tg_total[c])),
            memory_order_relaxed);
      }
      scan_lookback_fence(); // release
      atomic_store_explicit(&status[tile_id], 2u, memory_order_relaxed);
    }
    for (int c = 0; c < VEC; c++) {
      tg_carry[c] = carry[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint soff = lid * N_READS;
  for (int i = 0; i < N_READS; i++) {
    const uint s = soff + i;
    if (s < len) {
      for (int c = 0; c < VEC; c++) {
        out[base + size_t(s) * VEC + c] =
            static_cast<T>(op(tg_carry[c], vals[i * VEC + c]));
      }
    }
  }
}

// Int small-stride outer scan over [n_orows, axis, VEC], componentwise: int has
// no decoupled look-back (no 64-bit atomic on metal3.1), so a 2-pass
// multi-block (block_reduce sums each block; block_carry re-scans seeded with
// prior blocks).
template <
    typename T,
    typename Op,
    int N_READS,
    int VEC,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_vec_block_reduce(
    const device IN* in [[buffer(0)]],
    device acc_t* block_sums [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& block_size [[buffer(3)]],
    const constant size_t& num_blocks [[buffer(4)]],
    const constant size_t& n_orows [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup acc_t smem[simd_size];
  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t block_id = full_gid % num_blocks;
  size_t orow = full_gid / num_blocks;
  if (orow >= n_orows) {
    return;
  }
  size_t block_start = block_id * block_size;
  if (block_start >= axis_size) {
    if (lid.x == 0) {
      for (int c = 0; c < VEC; c++) {
        block_sums[(orow * num_blocks + block_id) * VEC + c] = Op::init;
      }
    }
    return;
  }
  uint len = uint(min(block_size, axis_size - block_start));
  size_t base = (orow * axis_size + block_start) * VEC;

  acc_t acc[VEC];
  for (int c = 0; c < VEC; c++) {
    acc[c] = Op::init;
  }
  for (uint r = 0; r < ceildiv(len, N_READS * lsize.x); r++) {
    for (int i = 0; i < N_READS; i++) {
      uint s = r * lsize.x * N_READS + lid.x * N_READS + i;
      if (s < len) {
        for (int c = 0; c < VEC; c++) {
          acc[c] =
              op(acc[c], static_cast<acc_t>(in[base + size_t(s) * VEC + c]));
        }
      }
    }
  }
  for (int c = 0; c < VEC; c++) {
    acc_t total = threadgroup_reduce<Op>(
        acc[c], smem, lsize.x, simd_lane_id, simd_group_id);
    if (lid.x == 0) {
      block_sums[(orow * num_blocks + block_id) * VEC + c] = total;
    }
  }
}

template <
    typename T,
    typename Op,
    int N_READS,
    int VEC,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_vec_block_carry(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device acc_t* block_sums [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    const constant size_t& block_size [[buffer(4)]],
    const constant size_t& num_blocks [[buffer(5)]],
    const constant size_t& n_orows [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup acc_t simdgroup_sums[simd_size * VEC];
  threadgroup acc_t tg_total[VEC];
  threadgroup acc_t tg_carry[VEC];

  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t block_id = full_gid % num_blocks;
  size_t orow = full_gid / num_blocks;
  if (orow >= n_orows) {
    return;
  }
  size_t block_start = block_id * block_size;
  if (block_start >= axis_size) {
    return;
  }
  uint len = uint(min(block_size, axis_size - block_start));
  size_t base = (orow * axis_size + block_start) * VEC;

  // Carry-in = sum of all preceding blocks' per-component totals (parallel over
  // threads, then threadgroup-reduced). block_sums is small and cached.
  acc_t partial[VEC];
  for (int c = 0; c < VEC; c++) {
    partial[c] = Op::init;
  }
  for (uint b = lid.x; b < block_id; b += lsize.x) {
    for (int c = 0; c < VEC; c++) {
      partial[c] =
          op(partial[c], block_sums[(orow * num_blocks + b) * VEC + c]);
    }
  }
  for (int c = 0; c < VEC; c++) {
    acc_t carry = threadgroup_reduce<Op>(
        partial[c], simdgroup_sums, lsize.x, simd_lane_id, simd_group_id);
    if (lid.x == 0) {
      tg_carry[c] = carry;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  acc_t prefix[VEC];
  for (int c = 0; c < VEC; c++) {
    prefix[c] = tg_carry[c];
  }
  const uint tile = lsize.x * N_READS;
  const uint soff = lid.x * N_READS;
  for (uint t0 = 0; t0 < len; t0 += tile) {
    uint tlen = min(tile, len - t0);
    acc_t vals[N_READS * VEC];
    scan_vec_tile_inclusive<Op, N_READS, VEC, acc_t, IN>(
        in,
        base + size_t(t0) * VEC,
        tlen,
        soff,
        lsize.x,
        simd_lane_id,
        simd_group_id,
        simdgroup_sums,
        tg_total,
        vals);
    for (int i = 0; i < N_READS; i++) {
      const uint s = soff + i;
      if (s < tlen) {
        for (int c = 0; c < VEC; c++) {
          out[base + size_t(t0 + s) * VEC + c] =
              static_cast<T>(op(prefix[c], vals[i * VEC + c]));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < VEC; c++) {
      prefix[c] = op(prefix[c], tg_total[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Three-pass outer-axis scan (no transpose): BM x BN strided smem tiling, axis
// split into blocks; totals feed scan_block_sums (pass 2) unchanged.
template <
    typename T,
    typename Op,
    int N_READS,
    int BN,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_strided_block_reduce(
    const device IN* in [[buffer(0)]],
    device acc_t* block_sums [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    const constant size_t& stride_blocks [[buffer(4)]],
    const constant size_t& block_size [[buffer(5)]],
    const constant size_t& num_blocks [[buffer(6)]],
    const constant size_t& n_orows [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 32;
  // Odd padding (coprime with the 32 shared-memory banks) keeps the
  // column-strided scan read (stride BN_pad) bank-conflict free.
  constexpr int BN_pad = BN + 1;
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup acc_t read_buffer[BM * BN_pad];
  acc_t values[n_scans];
  acc_t partial[n_scans];
  for (int i = 0; i < n_scans; i++) {
    partial[i] = Op::init;
  }

  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t block_id = full_gid % num_blocks;
  size_t rest = full_gid / num_blocks;
  size_t irow_tile = rest % stride_blocks;
  size_t orow = rest / stride_blocks;
  if (orow >= n_orows) {
    return;
  }

  size_t offset = orow * axis_size * stride;
  size_t global_index_x = irow_tile * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;

  uint stride_limit = uint(stride - global_index_x);
  in += offset + global_index_x + read_offset_x;
  threadgroup acc_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup acc_t* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  uint axis_start = uint(block_id * block_size);
  uint axis_end = uint(min(axis_start + block_size, axis_size));

  for (uint j = axis_start; j < axis_end; j += BM) {
    uint index_y = j + read_offset_y;
    if (index_y < axis_end && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = static_cast<acc_t>(in[size_t(index_y) * stride + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (index_y < axis_end && (read_offset_x + i) < stride_limit) {
          read_into[i] = static_cast<acc_t>(in[size_t(index_y) * stride + i]);
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < n_scans; i++) {
      acc_t s = simd_scan(op, values[i]);
      partial[i] = op(partial[i], simd_shuffle(s, simd_size - 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // One lane per simdgroup writes the block totals for its columns.
  if (simd_lane_id == 0) {
    for (int i = 0; i < n_scans; i++) {
      size_t irow = global_index_x + scan_offset_x + i;
      if (irow < stride) {
        block_sums[(orow * stride + irow) * num_blocks + block_id] = partial[i];
      }
    }
  }
}

template <
    typename T,
    typename Op,
    int N_READS,
    int BN,
    typename IN = T,
    typename acc_t = accum_t<T>>
kernel void scan_strided_block_carry(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device acc_t* block_sums [[buffer(2)]],
    const constant size_t& axis_size [[buffer(3)]],
    const constant size_t& stride [[buffer(4)]],
    const constant size_t& stride_blocks [[buffer(5)]],
    const constant size_t& block_size [[buffer(6)]],
    const constant size_t& num_blocks [[buffer(7)]],
    const constant size_t& n_orows [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 32;
  // Odd padding (coprime with the 32 shared-memory banks) keeps the
  // column-strided scan read (stride BN_pad) bank-conflict free.
  constexpr int BN_pad = BN + 1;
  constexpr int n_simds = BN / N_READS;
  constexpr int n_scans = BN / n_simds;
  Op op;

  threadgroup acc_t read_buffer[BM * BN_pad];
  acc_t values[n_scans];
  acc_t prefix[n_scans];

  size_t full_gid = gid.y + gsize.y * size_t(gid.z);
  size_t block_id = full_gid % num_blocks;
  size_t rest = full_gid / num_blocks;
  size_t irow_tile = rest % stride_blocks;
  size_t orow = rest / stride_blocks;
  if (orow >= n_orows) {
    return;
  }

  size_t offset = orow * axis_size * stride;
  size_t global_index_x = irow_tile * BN;
  uint read_offset_y = (lid.x * N_READS) / BN;
  uint read_offset_x = (lid.x * N_READS) % BN;
  uint scan_offset_y = simd_lane_id;
  uint scan_offset_x = simd_group_id * n_scans;
  uint stride_limit = uint(stride - global_index_x);

  // Seed each column's running prefix with its carry-in (inclusive sum of all
  // preceding blocks for that scan).
  for (int i = 0; i < n_scans; i++) {
    size_t irow = global_index_x + scan_offset_x + i;
    if (block_id > 0 && irow < stride) {
      prefix[i] =
          block_sums[(orow * stride + irow) * num_blocks + (block_id - 1)];
    } else {
      prefix[i] = Op::init;
    }
  }

  in += offset + global_index_x + read_offset_x;
  out += offset + global_index_x + read_offset_x;
  threadgroup acc_t* read_into =
      read_buffer + read_offset_y * BN_pad + read_offset_x;
  threadgroup acc_t* read_from =
      read_buffer + scan_offset_y * BN_pad + scan_offset_x;

  uint axis_start = uint(block_id * block_size);
  uint axis_end = uint(min(axis_start + block_size, axis_size));

  for (uint j = axis_start; j < axis_end; j += BM) {
    uint index_y = j + read_offset_y;
    if (index_y < axis_end && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        read_into[i] = static_cast<acc_t>(in[size_t(index_y) * stride + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (index_y < axis_end && (read_offset_x + i) < stride_limit) {
          read_into[i] = static_cast<acc_t>(in[size_t(index_y) * stride + i]);
        } else {
          read_into[i] = Op::init;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < n_scans; i++) {
      values[i] = read_from[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < n_scans; i++) {
      values[i] = simd_scan(op, values[i]);
      values[i] = op(values[i], prefix[i]);
      prefix[i] = simd_shuffle(values[i], simd_size - 1);
    }

    for (int i = 0; i < n_scans; i++) {
      read_from[i] = values[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (index_y < axis_end && (read_offset_x + N_READS) < stride_limit) {
      for (int i = 0; i < N_READS; i++) {
        out[size_t(index_y) * stride + i] = static_cast<T>(read_into[i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (index_y < axis_end && (read_offset_x + i) < stride_limit) {
          out[size_t(index_y) * stride + i] = static_cast<T>(read_into[i]);
        }
      }
    }
  }
}

// Segmented scan for a tiny innermost axis with very many rows: pack many rows
// per threadgroup (one-threadgroup-per-row would launch far too many).
template <typename T, typename Op, typename IN = T, typename acc_t = accum_t<T>>
kernel void scan_tiny_innermost(
    const device IN* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant uint& axis_size [[buffer(2)]],
    const constant uint& n_scans [[buffer(3)]],
    const constant uint& rows_per_tg [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]) {
  constexpr uint TILE = 2048;
  threadgroup acc_t buf[TILE];
  Op op;

  uint elems = rows_per_tg * axis_size; // <= TILE
  size_t base = size_t(tgid.x) * elems;
  size_t total = size_t(n_scans) * axis_size;

  for (uint i = lid.x; i < elems; i += tg_size.x) {
    buf[i] = (base + i < total) ? static_cast<acc_t>(in[base + i]) : Op::init;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = lid.x; row < rows_per_tg; row += tg_size.x) {
    if (base + size_t(row) * axis_size >= total) {
      break;
    }
    uint off = row * axis_size;
    acc_t acc = buf[off];
    for (uint j = 1; j < axis_size; j++) {
      acc = op(acc, buf[off + j]);
      buf[off + j] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lid.x; i < elems; i += tg_size.x) {
    if (base + i < total) {
      out[base + i] = static_cast<T>(buf[i]);
    }
  }
}

#define REGISTER_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS)              \
  template [[host_name(#OP_NAME "_innermost_" #DTYPE)]] [[kernel]] void \
  scan_innermost_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(                   \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);           \
                                                                        \
  template [[host_name(#OP_NAME "_outer_" #DTYPE)]] [[kernel]] void     \
  scan_outer_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(                       \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      const constant size_t& stride [[buffer(3)]],                      \
      const constant size_t& stride_blocks [[buffer(4)]],               \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_SCAN_WITH_INDICES_OP(OP_NAME, OP_CLASS, DTYPE, NREADS) \
  template [[host_name(#OP_NAME "_innermost_" #DTYPE)]] [[kernel]] void \
  scan_with_indices_innermost_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(      \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out_values [[buffer(1)]],                           \
      device int64_t* out_indices [[buffer(2)]],                        \
      const constant size_t& axis_size [[buffer(3)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);           \
                                                                        \
  template [[host_name(#OP_NAME "_outer_" #DTYPE)]] [[kernel]] void     \
  scan_with_indices_outer_dim<DTYPE, OP_CLASS<DTYPE>, NREADS>(          \
      const device DTYPE* in [[buffer(0)]],                             \
      device DTYPE* out_values [[buffer(1)]],                           \
      device int64_t* out_indices [[buffer(2)]],                        \
      const constant size_t& axis_size [[buffer(3)]],                   \
      const constant size_t& stride [[buffer(4)]],                      \
      const constant size_t& stride_blocks [[buffer(5)]],               \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS)         \
  template [[host_name(#OP_NAME "_block_reduce_" #DTYPE)]] [[kernel]] void    \
  scan_block_reduce<DTYPE, OP_CLASS<DTYPE>, NREADS>(                          \
      const device DTYPE* in [[buffer(0)]],                                   \
      device accum_t<DTYPE>* block_sums [[buffer(1)]],                        \
      const constant size_t& axis_size [[buffer(2)]],                         \
      const constant size_t& block_size [[buffer(3)]],                        \
      const constant size_t& num_blocks [[buffer(4)]],                        \
      uint3 gid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 lsize [[threads_per_threadgroup]],                                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);                 \
                                                                              \
  template [[host_name(#OP_NAME "_scan_block_sums_" #DTYPE)]] [[kernel]] void \
  scan_block_sums<DTYPE, OP_CLASS<DTYPE>, NREADS>(                            \
      device accum_t<DTYPE> * block_sums [[buffer(0)]],                       \
      const constant size_t& num_blocks [[buffer(1)]],                        \
      uint3 gid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 lsize [[threads_per_threadgroup]],                                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);                 \
                                                                              \
  template [[host_name(#OP_NAME "_block_carry_" #DTYPE)]] [[kernel]] void     \
  scan_block_carry<DTYPE, OP_CLASS<DTYPE>, NREADS>(                           \
      const device DTYPE* in [[buffer(0)]],                                   \
      device DTYPE* out [[buffer(1)]],                                        \
      const device accum_t<DTYPE>* block_sums [[buffer(2)]],                  \
      const constant size_t& axis_size [[buffer(3)]],                         \
      const constant size_t& block_size [[buffer(4)]],                        \
      const constant size_t& num_blocks [[buffer(5)]],                        \
      uint3 gid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 lsize [[threads_per_threadgroup]],                                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_DECOUPLED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS)           \
  template [[host_name(#OP_NAME "_contig_decoupled_" #DTYPE)]] [[kernel]] void \
  scan_contig_decoupled<DTYPE, OP_CLASS<DTYPE>, NREADS>(                       \
      const device DTYPE* in [[buffer(0)]],                                    \
      device DTYPE* out [[buffer(1)]],                                         \
      device atomic_uint* tile_counter [[buffer(2)]],                          \
      device atomic_uint* aggregates [[buffer(3)]],                            \
      device atomic_uint* inclusive [[buffer(4)]],                             \
      const constant uint& axis_size [[buffer(5)]],                            \
      const constant uint& num_tiles [[buffer(6)]],                            \
      uint lid [[thread_position_in_threadgroup]],                             \
      uint lsize [[threads_per_threadgroup]],                                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

// Single-pass strided decoupled look-back, one (NREADS, VEC) per registration.
#define REGISTER_STRIDED_DECOUPLED_SCAN_OP(                            \
    OP_NAME, OP_CLASS, DTYPE, NREADS, VEC)                             \
  template [[host_name(#OP_NAME "_strided_decoupled_" #VEC "_" #NREADS \
                                "_" #DTYPE)]] [[kernel]] void          \
  scan_strided_decoupled<DTYPE, OP_CLASS<DTYPE>, NREADS, VEC>(         \
      const device DTYPE* in [[buffer(0)]],                            \
      device DTYPE* out [[buffer(1)]],                                 \
      device atomic_uint* tile_counter [[buffer(2)]],                  \
      device atomic_uint* status [[buffer(3)]],                        \
      device atomic_uint* aggregates [[buffer(4)]],                    \
      device atomic_uint* inclusive [[buffer(5)]],                     \
      const constant uint& axis_size [[buffer(6)]],                    \
      const constant uint& num_tiles [[buffer(7)]],                    \
      uint lid [[thread_position_in_threadgroup]],                     \
      uint lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

// Integer small-stride (VEC in {2,3,4}) two-pass vectorized multi-block scan.
#define REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, VEC) \
  template [[host_name(#OP_NAME "_vec_block_reduce_" #VEC                      \
                                "_" #DTYPE)]] [[kernel]] void                  \
  scan_vec_block_reduce<DTYPE, OP_CLASS<DTYPE>, NREADS, VEC>(                  \
      const device DTYPE* in [[buffer(0)]],                                    \
      device accum_t<DTYPE>* block_sums [[buffer(1)]],                         \
      const constant size_t& axis_size [[buffer(2)]],                          \
      const constant size_t& block_size [[buffer(3)]],                         \
      const constant size_t& num_blocks [[buffer(4)]],                         \
      const constant size_t& n_orows [[buffer(5)]],                            \
      uint3 gid [[threadgroup_position_in_grid]],                              \
      uint3 gsize [[threadgroups_per_grid]],                                   \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint3 lsize [[threads_per_threadgroup]],                                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);                  \
                                                                               \
  template [[host_name(#OP_NAME "_vec_block_carry_" #VEC                       \
                                "_" #DTYPE)]] [[kernel]] void                  \
  scan_vec_block_carry<DTYPE, OP_CLASS<DTYPE>, NREADS, VEC>(                   \
      const device DTYPE* in [[buffer(0)]],                                    \
      device DTYPE* out [[buffer(1)]],                                         \
      const device accum_t<DTYPE>* block_sums [[buffer(2)]],                   \
      const constant size_t& axis_size [[buffer(3)]],                          \
      const constant size_t& block_size [[buffer(4)]],                         \
      const constant size_t& num_blocks [[buffer(5)]],                         \
      const constant size_t& n_orows [[buffer(6)]],                            \
      uint3 gid [[threadgroup_position_in_grid]],                              \
      uint3 gsize [[threadgroups_per_grid]],                                   \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint3 lsize [[threads_per_threadgroup]],                                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(                        \
    OP_NAME, OP_CLASS, DTYPE, NREADS)                                   \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 2); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 3); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 4); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 8)

// Fused int32 -> int64 vectorized multi-block scan.
#define REGISTER_VEC_MULTIBLOCK_SCAN_OP_PROMOTED(                              \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, VEC)                       \
  template [[host_name(#OP_NAME "_vec_block_reduce_" #VEC "_" #IN_DTYPE        \
                                "_" #OUT_DTYPE)]] [[kernel]] void              \
  scan_vec_block_reduce<                                                       \
      OUT_DTYPE,                                                               \
      OP_CLASS<OUT_DTYPE>,                                                     \
      NREADS,                                                                  \
      VEC,                                                                     \
      IN_DTYPE>(                                                               \
      const device IN_DTYPE* in [[buffer(0)]],                                 \
      device accum_t<OUT_DTYPE>* block_sums [[buffer(1)]],                     \
      const constant size_t& axis_size [[buffer(2)]],                          \
      const constant size_t& block_size [[buffer(3)]],                         \
      const constant size_t& num_blocks [[buffer(4)]],                         \
      const constant size_t& n_orows [[buffer(5)]],                            \
      uint3 gid [[threadgroup_position_in_grid]],                              \
      uint3 gsize [[threadgroups_per_grid]],                                   \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint3 lsize [[threads_per_threadgroup]],                                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);                  \
                                                                               \
  template [[host_name(#OP_NAME "_vec_block_carry_" #VEC "_" #IN_DTYPE         \
                                "_" #OUT_DTYPE)]] [[kernel]] void              \
  scan_vec_block_carry<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, VEC, IN_DTYPE>( \
      const device IN_DTYPE* in [[buffer(0)]],                                 \
      device OUT_DTYPE* out [[buffer(1)]],                                     \
      const device accum_t<OUT_DTYPE>* block_sums [[buffer(2)]],               \
      const constant size_t& axis_size [[buffer(3)]],                          \
      const constant size_t& block_size [[buffer(4)]],                         \
      const constant size_t& num_blocks [[buffer(5)]],                         \
      const constant size_t& n_orows [[buffer(6)]],                            \
      uint3 gid [[threadgroup_position_in_grid]],                              \
      uint3 gsize [[threadgroups_per_grid]],                                   \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint3 lsize [[threads_per_threadgroup]],                                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC_PROMOTED( \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS)       \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP_PROMOTED(               \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 2); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP_PROMOTED(               \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 3); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP_PROMOTED(               \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 4); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP_PROMOTED(               \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 8)

#define REGISTER_TRANSPOSED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS)         \
  template [[host_name(#OP_NAME "_innermost_transposed_" #DTYPE)]] [[kernel]] \
  void scan_innermost_transposed<DTYPE, OP_CLASS<DTYPE>, NREADS>(             \
      const device DTYPE* in [[buffer(0)]],                                   \
      device DTYPE* out [[buffer(1)]],                                        \
      const constant size_t& axis_size [[buffer(2)]],                         \
      const constant size_t& n_cols [[buffer(3)]],                            \
      const constant size_t& stride_blocks [[buffer(4)]],                     \
      uint3 gid [[threadgroup_position_in_grid]],                             \
      uint3 gsize [[threadgroups_per_grid]],                                  \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_TRANSPOSED_SCAN_OP_PROMOTED(                                  \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS)                            \
  template [[host_name(#OP_NAME "_innermost_transposed_" #IN_DTYPE             \
                                "_" #OUT_DTYPE)]] [[kernel]] void              \
  scan_innermost_transposed<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, IN_DTYPE>( \
      const device IN_DTYPE* in [[buffer(0)]],                                 \
      device OUT_DTYPE* out [[buffer(1)]],                                     \
      const constant size_t& axis_size [[buffer(2)]],                          \
      const constant size_t& n_cols [[buffer(3)]],                             \
      const constant size_t& stride_blocks [[buffer(4)]],                      \
      uint3 gid [[threadgroup_position_in_grid]],                              \
      uint3 gsize [[threadgroups_per_grid]],                                   \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

// Strided multi-block kernels are templated on tile width BN (host fits it to
// n_irows); signatures are BN-independent.
#define REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, BN) \
  template [[host_name(#OP_NAME "_strided_block_reduce_" #BN           \
                                "_" #DTYPE)]] [[kernel]] void          \
  scan_strided_block_reduce<DTYPE, OP_CLASS<DTYPE>, NREADS, BN>(       \
      const device DTYPE* in [[buffer(0)]],                            \
      device accum_t<DTYPE>* block_sums [[buffer(1)]],                 \
      const constant size_t& axis_size [[buffer(2)]],                  \
      const constant size_t& stride [[buffer(3)]],                     \
      const constant size_t& stride_blocks [[buffer(4)]],              \
      const constant size_t& block_size [[buffer(5)]],                 \
      const constant size_t& num_blocks [[buffer(6)]],                 \
      const constant size_t& n_orows [[buffer(7)]],                    \
      uint3 gid [[threadgroup_position_in_grid]],                      \
      uint3 gsize [[threadgroups_per_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);          \
                                                                       \
  template [[host_name(#OP_NAME "_strided_block_carry_" #BN            \
                                "_" #DTYPE)]] [[kernel]] void          \
  scan_strided_block_carry<DTYPE, OP_CLASS<DTYPE>, NREADS, BN>(        \
      const device DTYPE* in [[buffer(0)]],                            \
      device DTYPE* out [[buffer(1)]],                                 \
      const device accum_t<DTYPE>* block_sums [[buffer(2)]],           \
      const constant size_t& axis_size [[buffer(3)]],                  \
      const constant size_t& stride [[buffer(4)]],                     \
      const constant size_t& stride_blocks [[buffer(5)]],              \
      const constant size_t& block_size [[buffer(6)]],                 \
      const constant size_t& num_blocks [[buffer(7)]],                 \
      const constant size_t& n_orows [[buffer(8)]],                    \
      uint3 gid [[threadgroup_position_in_grid]],                      \
      uint3 gsize [[threadgroups_per_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_STRIDED_SCAN_OP_ALL_BN(OP_NAME, OP_CLASS, DTYPE, NREADS) \
  REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 8);          \
  REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 16);         \
  REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 32)

#define REGISTER_TINY_SCAN_OP(OP_NAME, OP_CLASS, DTYPE)                      \
  template [[host_name(#OP_NAME "_tiny_innermost_" #DTYPE)]] [[kernel]] void \
  scan_tiny_innermost<DTYPE, OP_CLASS<DTYPE>>(                               \
      const device DTYPE* in [[buffer(0)]],                                  \
      device DTYPE* out [[buffer(1)]],                                       \
      const constant uint& axis_size [[buffer(2)]],                          \
      const constant uint& n_scans [[buffer(3)]],                            \
      const constant uint& rows_per_tg [[buffer(4)]],                        \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint3 tg_size [[threads_per_threadgroup]])

// Fused integer-widening scans: read IN, accumulate/write OUT (int32 -> int64).
// Named "{in}_{out}" so the host scans the narrow input directly (no upcast).
#define REGISTER_SCAN_OP_PROMOTED(                                      \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS)                     \
  template [[host_name(#OP_NAME "_innermost_" #IN_DTYPE                 \
                                "_" #OUT_DTYPE)]] [[kernel]] void       \
  scan_innermost_dim<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, IN_DTYPE>( \
      const device IN_DTYPE* in [[buffer(0)]],                          \
      device OUT_DTYPE* out [[buffer(1)]],                              \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);           \
                                                                        \
  template [[host_name(#OP_NAME "_outer_" #IN_DTYPE                     \
                                "_" #OUT_DTYPE)]] [[kernel]] void       \
  scan_outer_dim<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, IN_DTYPE>(     \
      const device IN_DTYPE* in [[buffer(0)]],                          \
      device OUT_DTYPE* out [[buffer(1)]],                              \
      const constant size_t& axis_size [[buffer(2)]],                   \
      const constant size_t& stride [[buffer(3)]],                      \
      const constant size_t& stride_blocks [[buffer(4)]],               \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_MULTIBLOCK_SCAN_OP_PROMOTED(                          \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS)                    \
  template [[host_name(#OP_NAME "_block_reduce_" #IN_DTYPE             \
                                "_" #OUT_DTYPE)]] [[kernel]] void      \
  scan_block_reduce<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, IN_DTYPE>( \
      const device IN_DTYPE* in [[buffer(0)]],                         \
      device accum_t<OUT_DTYPE>* block_sums [[buffer(1)]],             \
      const constant size_t& axis_size [[buffer(2)]],                  \
      const constant size_t& block_size [[buffer(3)]],                 \
      const constant size_t& num_blocks [[buffer(4)]],                 \
      uint3 gid [[threadgroup_position_in_grid]],                      \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint3 lsize [[threads_per_threadgroup]],                         \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);          \
                                                                       \
  template [[host_name(#OP_NAME "_block_carry_" #IN_DTYPE              \
                                "_" #OUT_DTYPE)]] [[kernel]] void      \
  scan_block_carry<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, NREADS, IN_DTYPE>(  \
      const device IN_DTYPE* in [[buffer(0)]],                         \
      device OUT_DTYPE* out [[buffer(1)]],                             \
      const device accum_t<OUT_DTYPE>* block_sums [[buffer(2)]],       \
      const constant size_t& axis_size [[buffer(3)]],                  \
      const constant size_t& block_size [[buffer(4)]],                 \
      const constant size_t& num_blocks [[buffer(5)]],                 \
      uint3 gid [[threadgroup_position_in_grid]],                      \
      uint3 lid [[thread_position_in_threadgroup]],                    \
      uint3 lsize [[threads_per_threadgroup]],                         \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_STRIDED_SCAN_OP_PROMOTED(                                 \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, BN)                    \
  template [[host_name(#OP_NAME "_strided_block_reduce_" #BN "_" #IN_DTYPE \
                                "_" #OUT_DTYPE)]] [[kernel]] void          \
  scan_strided_block_reduce<                                               \
      OUT_DTYPE,                                                           \
      OP_CLASS<OUT_DTYPE>,                                                 \
      NREADS,                                                              \
      BN,                                                                  \
      IN_DTYPE>(                                                           \
      const device IN_DTYPE* in [[buffer(0)]],                             \
      device accum_t<OUT_DTYPE>* block_sums [[buffer(1)]],                 \
      const constant size_t& axis_size [[buffer(2)]],                      \
      const constant size_t& stride [[buffer(3)]],                         \
      const constant size_t& stride_blocks [[buffer(4)]],                  \
      const constant size_t& block_size [[buffer(5)]],                     \
      const constant size_t& num_blocks [[buffer(6)]],                     \
      const constant size_t& n_orows [[buffer(7)]],                        \
      uint3 gid [[threadgroup_position_in_grid]],                          \
      uint3 gsize [[threadgroups_per_grid]],                               \
      uint3 lid [[thread_position_in_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);              \
                                                                           \
  template [[host_name(#OP_NAME "_strided_block_carry_" #BN "_" #IN_DTYPE  \
                                "_" #OUT_DTYPE)]] [[kernel]] void          \
  scan_strided_block_carry<                                                \
      OUT_DTYPE,                                                           \
      OP_CLASS<OUT_DTYPE>,                                                 \
      NREADS,                                                              \
      BN,                                                                  \
      IN_DTYPE>(                                                           \
      const device IN_DTYPE* in [[buffer(0)]],                             \
      device OUT_DTYPE* out [[buffer(1)]],                                 \
      const device accum_t<OUT_DTYPE>* block_sums [[buffer(2)]],           \
      const constant size_t& axis_size [[buffer(3)]],                      \
      const constant size_t& stride [[buffer(4)]],                         \
      const constant size_t& stride_blocks [[buffer(5)]],                  \
      const constant size_t& block_size [[buffer(6)]],                     \
      const constant size_t& num_blocks [[buffer(7)]],                     \
      const constant size_t& n_orows [[buffer(8)]],                        \
      uint3 gid [[threadgroup_position_in_grid]],                          \
      uint3 gsize [[threadgroups_per_grid]],                               \
      uint3 lid [[thread_position_in_threadgroup]],                        \
      uint simd_lane_id [[thread_index_in_simdgroup]],                     \
      uint simd_group_id [[simdgroup_index_in_threadgroup]])

#define REGISTER_STRIDED_SCAN_OP_ALL_BN_PROMOTED(          \
    OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS)        \
  REGISTER_STRIDED_SCAN_OP_PROMOTED(                       \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 8);  \
  REGISTER_STRIDED_SCAN_OP_PROMOTED(                       \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 16); \
  REGISTER_STRIDED_SCAN_OP_PROMOTED(                       \
      OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE, NREADS, 32)

#define REGISTER_TINY_SCAN_OP_PROMOTED(OP_NAME, OP_CLASS, IN_DTYPE, OUT_DTYPE) \
  template [[host_name(#OP_NAME "_tiny_innermost_" #IN_DTYPE                   \
                                "_" #OUT_DTYPE)]] [[kernel]] void              \
  scan_tiny_innermost<OUT_DTYPE, OP_CLASS<OUT_DTYPE>, IN_DTYPE>(               \
      const device IN_DTYPE* in [[buffer(0)]],                                 \
      device OUT_DTYPE* out [[buffer(1)]],                                     \
      const constant uint& axis_size [[buffer(2)]],                            \
      const constant uint& n_scans [[buffer(3)]],                              \
      const constant uint& rows_per_tg [[buffer(4)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                            \
      uint3 tg_size [[threads_per_threadgroup]])

// Simple scan operations
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, float, 4);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, half, 4);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, bfloat, 4);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, float2, 2);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, half2, 4);

REGISTER_SCAN_OP(cumprod, CumProdOp, float2, 2);
REGISTER_SCAN_OP(cumprod, CumProdOp, half2, 4);

// Real-typed cumsum/cumprod fallback for scan axes outside the rightmost 4
// dims, where MPSGraph's MPSNDArrayScan asserts (issue #184844).
REGISTER_SCAN_OP(cumsum, CumSumOp, float, 4);
REGISTER_SCAN_OP(cumsum, CumSumOp, half, 4);
REGISTER_SCAN_OP(cumsum, CumSumOp, bfloat, 4);
REGISTER_SCAN_OP(cumsum, CumSumOp, int, 4);
REGISTER_SCAN_OP(cumsum, CumSumOp, long, 2);

REGISTER_SCAN_OP(cumprod, CumProdOp, float, 4);
REGISTER_SCAN_OP(cumprod, CumProdOp, half, 4);
REGISTER_SCAN_OP(cumprod, CumProdOp, bfloat, 4);
REGISTER_SCAN_OP(cumprod, CumProdOp, int, 4);
REGISTER_SCAN_OP(cumprod, CumProdOp, long, 2);

// Multi-block variants for long scan axes with low scan-count parallelism.
REGISTER_MULTIBLOCK_SCAN_OP(cumsum, CumSumOp, float, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumsum, CumSumOp, half, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumsum, CumSumOp, bfloat, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumsum, CumSumOp, int, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumsum, CumSumOp, long, 2);

REGISTER_MULTIBLOCK_SCAN_OP(cumprod, CumProdOp, float, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumprod, CumProdOp, half, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumprod, CumProdOp, bfloat, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumprod, CumProdOp, int, 4);
REGISTER_MULTIBLOCK_SCAN_OP(cumprod, CumProdOp, long, 2);

// Single-pass decoupled look-back variants (float-accumulate dtypes only).
REGISTER_DECOUPLED_SCAN_OP(cumsum, CumSumOp, float, 16);
REGISTER_DECOUPLED_SCAN_OP(cumsum, CumSumOp, half, 16);
REGISTER_DECOUPLED_SCAN_OP(cumsum, CumSumOp, bfloat, 16);
REGISTER_DECOUPLED_SCAN_OP(cumprod, CumProdOp, float, 16);
REGISTER_DECOUPLED_SCAN_OP(cumprod, CumProdOp, half, 16);
REGISTER_DECOUPLED_SCAN_OP(cumprod, CumProdOp, bfloat, 16);

// Narrow-stride outer-scan look-back, inner stride VEC in {2,4,8,16}. NREADS x
// VEC ~ the contig path's register footprint; host VEC->NREADS map must match.
#define REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(OP_NAME, OP_CLASS, DTYPE) \
  REGISTER_STRIDED_DECOUPLED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, 8, 2);        \
  REGISTER_STRIDED_DECOUPLED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, 4, 4);        \
  REGISTER_STRIDED_DECOUPLED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, 4, 8);        \
  REGISTER_STRIDED_DECOUPLED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, 2, 16)
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumsum, CumSumOp, float);
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumsum, CumSumOp, half);
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumsum, CumSumOp, bfloat);
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumprod, CumProdOp, float);
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumprod, CumProdOp, half);
REGISTER_STRIDED_DECOUPLED_SCAN_OP_ALL_VEC(cumprod, CumProdOp, bfloat);

// Strided multi-block variants, one tile width (BN) per registration. The host
// picks the smallest BN >= n_irows (capped at 32) for good tile utilization.
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumsum, CumSumOp, float, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumsum, CumSumOp, half, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumsum, CumSumOp, bfloat, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumsum, CumSumOp, int, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumsum, CumSumOp, long, 2);

REGISTER_STRIDED_SCAN_OP_ALL_BN(cumprod, CumProdOp, float, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumprod, CumProdOp, half, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumprod, CumProdOp, bfloat, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumprod, CumProdOp, int, 4);
REGISTER_STRIDED_SCAN_OP_ALL_BN(cumprod, CumProdOp, long, 2);

// Small-stride (VEC in {2,3,4,8}) vectorized multi-block scan: deterministic,
// reads exactly VEC, avoiding the strided kernel's min BN=8 tile waste.
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumsum, CumSumOp, long, 2);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumprod, CumProdOp, long, 2);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumsum, CumSumOp, float, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumsum, CumSumOp, half, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumsum, CumSumOp, bfloat, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumprod, CumProdOp, float, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumprod, CumProdOp, half, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC(cumprod, CumProdOp, bfloat, 4);

// Odd narrow vec widths {5,6,7}, float only: read exactly VEC instead of idling
// 8-n_irows lanes in a BN=8 strided tile. Int/long stay on the strided kernel.
#define REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(                         \
    OP_NAME, OP_CLASS, DTYPE, NREADS)                                   \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 5); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 6); \
  REGISTER_VEC_MULTIBLOCK_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 7)
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumsum, CumSumOp, float, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumsum, CumSumOp, half, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumsum, CumSumOp, bfloat, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumprod, CumProdOp, float, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumprod, CumProdOp, half, 4);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_NARROW(cumprod, CumProdOp, bfloat, 4);

// Tight strided tile widths {12,24}, float-accumulate dtypes only: fits n_irows
// in (8,12] and (16,24] without the loose BN=16/32 lane waste (n_irows=12 ->
// BN=16 idles 4 of 16 lanes, regressing bf16). Int/long use the {8,16,32} set.
#define REGISTER_STRIDED_SCAN_OP_TIGHT(OP_NAME, OP_CLASS, DTYPE, NREADS) \
  REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 12);        \
  REGISTER_STRIDED_SCAN_OP(OP_NAME, OP_CLASS, DTYPE, NREADS, 24)
REGISTER_STRIDED_SCAN_OP_TIGHT(cumsum, CumSumOp, float, 4);
REGISTER_STRIDED_SCAN_OP_TIGHT(cumsum, CumSumOp, half, 4);
REGISTER_STRIDED_SCAN_OP_TIGHT(cumsum, CumSumOp, bfloat, 4);
REGISTER_STRIDED_SCAN_OP_TIGHT(cumprod, CumProdOp, float, 4);
REGISTER_STRIDED_SCAN_OP_TIGHT(cumprod, CumProdOp, half, 4);
REGISTER_STRIDED_SCAN_OP_TIGHT(cumprod, CumProdOp, bfloat, 4);

// Tiny-innermost-axis variants (many short scans).
REGISTER_TINY_SCAN_OP(cumsum, CumSumOp, float);
REGISTER_TINY_SCAN_OP(cumsum, CumSumOp, half);
REGISTER_TINY_SCAN_OP(cumsum, CumSumOp, bfloat);
REGISTER_TINY_SCAN_OP(cumsum, CumSumOp, int);
REGISTER_TINY_SCAN_OP(cumsum, CumSumOp, long);

REGISTER_TINY_SCAN_OP(cumprod, CumProdOp, float);
REGISTER_TINY_SCAN_OP(cumprod, CumProdOp, half);
REGISTER_TINY_SCAN_OP(cumprod, CumProdOp, bfloat);
REGISTER_TINY_SCAN_OP(cumprod, CumProdOp, int);
REGISTER_TINY_SCAN_OP(cumprod, CumProdOp, long);

// Fused int32 -> int64. NREADS matches int64 (same int64-accumulator
// footprint).
REGISTER_SCAN_OP_PROMOTED(cumsum, CumSumOp, int, long, 2);
REGISTER_SCAN_OP_PROMOTED(cumprod, CumProdOp, int, long, 2);
REGISTER_MULTIBLOCK_SCAN_OP_PROMOTED(cumsum, CumSumOp, int, long, 2);
REGISTER_MULTIBLOCK_SCAN_OP_PROMOTED(cumprod, CumProdOp, int, long, 2);
REGISTER_STRIDED_SCAN_OP_ALL_BN_PROMOTED(cumsum, CumSumOp, int, long, 2);
REGISTER_STRIDED_SCAN_OP_ALL_BN_PROMOTED(cumprod, CumProdOp, int, long, 2);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC_PROMOTED(
    cumsum,
    CumSumOp,
    int,
    long,
    2);
REGISTER_VEC_MULTIBLOCK_SCAN_OP_ALL_VEC_PROMOTED(
    cumprod,
    CumProdOp,
    int,
    long,
    2);

// Fused transpose-scan (dense non-contiguous input, innermost logical axis).
REGISTER_TRANSPOSED_SCAN_OP(cumsum, CumSumOp, float, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumsum, CumSumOp, half, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumsum, CumSumOp, bfloat, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumsum, CumSumOp, long, 2);
REGISTER_TRANSPOSED_SCAN_OP(cumprod, CumProdOp, float, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumprod, CumProdOp, half, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumprod, CumProdOp, bfloat, 4);
REGISTER_TRANSPOSED_SCAN_OP(cumprod, CumProdOp, long, 2);
REGISTER_TRANSPOSED_SCAN_OP_PROMOTED(cumsum, CumSumOp, int, long, 2);
REGISTER_TRANSPOSED_SCAN_OP_PROMOTED(cumprod, CumProdOp, int, long, 2);
REGISTER_TINY_SCAN_OP_PROMOTED(cumsum, CumSumOp, int, long);
REGISTER_TINY_SCAN_OP_PROMOTED(cumprod, CumProdOp, int, long);

// Scan with indices operations for cummin/cummax
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, float, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, half, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bfloat, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, long, 2);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, int, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, short, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, char, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, uchar, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummin, CumMinOp, bool, 4);

REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, float, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, half, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bfloat, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, long, 2);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, int, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, short, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, char, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, uchar, 4);
REGISTER_SCAN_WITH_INDICES_OP(cummax, CumMaxOp, bool, 4);
