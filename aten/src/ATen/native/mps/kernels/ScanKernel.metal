#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include <c10/metal/common.h>
#include <c10/metal/utils.h>

using c10::metal::accum_t;

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    // Reference:
    // https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
    T min_val = c10::metal::min(x, y);
    T max_val = c10::metal::max(x, y);

    if (min_val != max_val || metal::isfinite(min_val)) {
      // nan will be propagated here
      return c10::metal::log1p(metal::exp(min_val - max_val)) + max_val;
    } else {
      // special case to correctly handle infinite cases
      return x;
    }
  };
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

inline uint64_t simd_shuffle_and_fill_up(
    uint64_t data,
    uint64_t filling,
    uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

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

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

#define DEFINE_SIMD_SCAN()                                               \
  template <typename U, metal::enable_if_t<sizeof(U) < 8, bool> = true>  \
  U simd_scan(U val) {                                                   \
    return simd_scan_impl(val);                                          \
  }                                                                      \
                                                                         \
  template <typename U, metal::enable_if_t<sizeof(U) == 8, bool> = true> \
  U simd_scan(U val) {                                                   \
    for (int i = 1; i <= 16; i *= 2) {                                   \
      val = operator()(val, simd_shuffle_and_fill_up(val, init, i));     \
    }                                                                    \
    return val;                                                          \
  }

#define DEFINE_SIMD_EXCLUSIVE_SCAN()                                     \
  template <typename U, metal::enable_if_t<sizeof(U) < 8, bool> = true>  \
  U simd_exclusive_scan(U val) {                                         \
    return simd_exclusive_scan_impl(val);                                \
  }                                                                      \
                                                                         \
  template <typename U, metal::enable_if_t<sizeof(U) == 8, bool> = true> \
  U simd_exclusive_scan(U val) {                                         \
    val = simd_scan(val);                                                \
    return simd_shuffle_and_fill_up(val, init, 1);                       \
  }

template <typename T, typename acc_t = accum_t<T>>
struct LogCumSumExpOp {
  static constexpr constant acc_t init = static_cast<acc_t>(
      metal::is_floating_point_v<T> ? -metal::numeric_limits<T>::infinity()
                                    : metal::numeric_limits<T>::lowest());

  acc_t operator()(acc_t a, acc_t b) {
    return LogAddExp{}(a, b);
  }

  acc_t simd_scan(acc_t x) {
    for (int i = 1; i <= 16; i *= 2) {
      acc_t other = simd_shuffle_and_fill_up(x, init, i);
      x = LogAddExp{}(x, other);
    }
    return x;
  }

  acc_t simd_exclusive_scan(acc_t x) {
    x = simd_scan(x);
    return simd_shuffle_and_fill_up(x, init, 1);
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

// Helper function for shuffling pairs in SIMD operations
template <typename T, typename acc_t = accum_t<T>>
inline ValueIndexPair<T, acc_t> simd_shuffle_pair(
    ValueIndexPair<T, acc_t> data,
    uint16_t lane) {
  return make_pair<T, acc_t>(
      simd_shuffle(data.value, lane), simd_shuffle(data.index, lane));
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

  // For SIMD operations, we need to handle pairs differently
  pair_t simd_scan(pair_t val) {
    // For pairs, we need to implement scan manually since SIMD doesn't support
    // pairs directly
    pair_t init_val = get_init();
    for (int i = 1; i <= 16; i *= 2) {
      pair_t shuffled = make_pair<T, acc_t>(
          simd_shuffle_and_fill_up(val.value, init_val.value, i),
          simd_shuffle_and_fill_up(val.index, init_val.index, i));
      val = operator()(val, shuffled);
    }
    return val;
  }

  pair_t simd_exclusive_scan(pair_t val) {
    val = simd_scan(val);
    pair_t init_val = get_init();
    return simd_shuffle_and_fill_up_pair(val, init_val, 1);
  }

 private:
  pair_t simd_shuffle_pair(pair_t data, uint16_t delta) {
    pair_t init_val = get_init();
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, init_val.value, delta),
        simd_shuffle_and_fill_up(data.index, init_val.index, delta));
  }

  pair_t simd_shuffle_and_fill_up_pair(
      pair_t data,
      pair_t filling,
      uint16_t delta) {
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, filling.value, delta),
        simd_shuffle_and_fill_up(data.index, filling.index, delta));
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

  // For SIMD operations, we need to handle pairs differently
  pair_t simd_scan(pair_t val) {
    // For pairs, we need to implement scan manually since SIMD doesn't support
    // pairs directly
    pair_t init_val = get_init();
    for (int i = 1; i <= 16; i *= 2) {
      pair_t shuffled = make_pair<T, acc_t>(
          simd_shuffle_and_fill_up(val.value, init_val.value, i),
          simd_shuffle_and_fill_up(val.index, init_val.index, i));
      val = operator()(val, shuffled);
    }
    return val;
  }

  pair_t simd_exclusive_scan(pair_t val) {
    val = simd_scan(val);
    pair_t init_val = get_init();
    return simd_shuffle_and_fill_up_pair(val, init_val, 1);
  }

 private:
  pair_t simd_shuffle_pair(pair_t data, uint16_t delta) {
    pair_t init_val = get_init();
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, init_val.value, delta),
        simd_shuffle_and_fill_up(data.index, init_val.index, delta));
  }

  pair_t simd_shuffle_and_fill_up_pair(
      pair_t data,
      pair_t filling,
      uint16_t delta) {
    return make_pair<T, acc_t>(
        simd_shuffle_and_fill_up(data.value, filling.value, delta),
        simd_shuffle_and_fill_up(data.index, filling.index, delta));
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

// Inclusive scan along innermost dimension for contiguous tensors
template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_innermost_dim(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  // Position the pointers
  size_t offset = (gid.y + gsize.y * size_t(gid.z)) * axis_size;
  in += offset;
  out += offset;

  // Compute the number of simd_groups
  uint simd_groups = lsize.x / simd_size;

  // Allocate memory
  acc_t prefix = Op::init;
  acc_t values[N_READS];
  threadgroup acc_t simdgroup_sums[32];

  // Loop over the reduced axis in blocks of size ceildiv(axis_size,
  // N_READS*lsize)
  //    Read block
  //    Compute inclusive scan of the block
  //      Compute inclusive scan per thread
  //      Compute exclusive scan of thread sums in simdgroup
  //      Write simdgroup sums in SM
  //      Compute exclusive scan of simdgroup sums
  //      Compute the output by scanning prefix, prev_simdgroup, prev_thread,
  //      value
  //    Write block

  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Compute the block offset
    uint offset = r * lsize.x * N_READS + lid.x * N_READS;

    // Read the values
    if ((offset + N_READS) < axis_size) {
      load_unsafe<T, N_READS>(values, in + offset);
    } else {
      load_safe<T, N_READS>(values, in + offset, offset, axis_size, Op::init);
    }

    // Compute an inclusive scan per thread
    for (int i = 1; i < N_READS; i++) {
      values[i] = op(values[i], values[i - 1]);
    }

    // Compute exclusive scan of thread sums
    acc_t prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);

    // Write simdgroup_sums to SM
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      acc_t prev_simdgroup =
          op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
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
    if ((offset + N_READS) < axis_size) {
      write_unsafe<T, N_READS>(values, out + offset);
    } else {
      write_safe<T, N_READS>(values, out + offset, offset, axis_size);
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

// Inclusive scan along outer dimension for contiguous tensors
template <typename T, typename Op, int N_READS, typename acc_t = accum_t<T>>
kernel void scan_outer_dim(
    const device T* in [[buffer(0)]],
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
      values[i] = op.simd_scan(values[i]);
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
    pair_t prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);

    // Write simdgroup_sums to SM
    if (simd_lane_id == simd_size - 1) {
      simdgroup_sums[simd_group_id] = op(prev_thread, values[N_READS - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exclusive scan of simdgroup_sums
    if (simd_group_id == 0) {
      pair_t prev_simdgroup =
          op.simd_exclusive_scan(simdgroup_sums[simd_lane_id]);
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
      values[i] = op.simd_scan(values[i]);
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

// Simple scan operations
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, float, 4);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, half, 4);
REGISTER_SCAN_OP(logcumsumexp, LogCumSumExpOp, bfloat, 4);

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
