// LSD radix sort kernels for MPS (Path 3 in Sort.mm).
//
// One full pass over an RBITS-bit digit = three kernels:
//   radix_count   : per-block digit histograms (rows x digits x n_blocks).
//   radix_scan    : exclusive scan over the histogram to get global offsets.
//                   May be skipped if radix_scatter is launched with FUSED_SCAN
//                   (small-problem case - the scatter does the scan inline).
//   radix_scatter : each block sorts its own elements by the current digit
//                   using a per-bit split, then scatters to global offsets.
// radix_count_scan fuses the count+scan into a single per-row dispatch when
// n_blocks is small enough to fit a flat histogram in tgmem.
//
// Host runs (bits_per_key / RBITS) passes, ping-ponging buffers each pass.
// The `_final` scatter variant writes long indices for the last pass.
//
// Radix is naturally stable: the per-bit split preserves order within zeros
// and ones, and the global prefix-sum scatter preserves cross-block order.
// to_radix_key() maps a key to a uint whose natural order matches sort order
// (handles float sign bits, signed-int two's-complement, and descending).
#pragma once

// Unsigned bit-container same size as T; default uint covers 4-byte types.
template <typename T>
struct radix_bits {
  using type = uint;
};
template <>
struct radix_bits<half> {
  using type = ushort;
};
template <>
struct radix_bits<bfloat> {
  using type = ushort;
};
template <>
struct radix_bits<short> {
  using type = ushort;
};
template <>
struct radix_bits<char> {
  using type = uchar;
};
template <>
struct radix_bits<uchar> {
  using type = uchar;
};
template <>
struct radix_bits<bool> {
  using type = uchar;
};

// Map key to uint where uint-order matches sort-order, handles floats,
// signed/unsigned ints and descending.
template <typename T>
inline ::metal::enable_if_t<::metal::is_floating_point_v<T>, uint> to_radix_key(
    T val,
    bool desc) {
  using U = typename radix_bits<T>::type;
  constexpr uint nbits = sizeof(T) * 8;
  constexpr uint key_mask = (nbits == 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u);
  constexpr uint sign_bit = 1u << (nbits - 1);
  // val != val is NaN check; metal::isnan is buggy at large TGs on M2.
  if (val != val)
    return desc ? 0u : key_mask;
  uint bits = uint(as_type<U>(val));
  uint is_neg = bits >> (nbits - 1);
  uint mask = (uint(0) - is_neg) | sign_bit;
  uint result = (bits ^ mask) & key_mask;
  if (desc)
    result = (~result) & key_mask;
  return result;
}

template <typename T>
inline ::metal::enable_if_t<
    !::metal::is_floating_point_v<T> && ::metal::is_signed_v<T>,
    uint>
to_radix_key(T val, bool desc) {
  using U = typename radix_bits<T>::type;
  constexpr U sign_bit = U(1) << (sizeof(T) * 8 - 1);
  U result = as_type<U>(val) ^ sign_bit;
  if (desc)
    result = U(~result);
  return uint(result);
}

template <typename T>
inline ::metal::enable_if_t<
    !::metal::is_floating_point_v<T> && !::metal::is_signed_v<T>,
    uint>
to_radix_key(T val, bool desc) {
  using U = typename radix_bits<T>::type;
  U result = U(val);
  if (desc)
    result = U(~result);
  return uint(result);
}

template <typename T, short RTPTG, short EPT, short RBITS>
kernel void radix_count(
    const device T* input [[buffer(0)]],
    device uint* histograms [[buffer(1)]],
    constant int3& dims [[buffer(2)]],
    constant bool& desc [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  const int sort_size = dims.x;
  const int n_blocks = dims.y;
  const int shift = dims.z;
  constexpr int RSIZE = 1 << RBITS;
  constexpr uint RMASK = RSIZE - 1;
  constexpr int ELEMS_PER_TG = RTPTG * EPT;
  constexpr int SIMD_W = 32;
  uint lid = lid3.x;
  uint lane = lid & (SIMD_W - 1);
  int row = tid.y;
  int block_idx = tid.x;
  int block_start = block_idx * ELEMS_PER_TG;
  int items_this_block = min(ELEMS_PER_TG, sort_size - block_start);
  const device T* rk = input + row * sort_size + block_start;

  threadgroup atomic_uint local_hist[RSIZE];
  for (uint d = lid; d < uint(RSIZE); d += RTPTG)
    atomic_store_explicit(&local_hist[d], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (RBITS <= 4) {
    uint my_counts[RSIZE];
#pragma unroll
    for (uint d = 0; d < uint(RSIZE); ++d)
      my_counts[d] = 0;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      int pos = i * RTPTG + int(lid);
      if (pos < items_this_block) {
        uint key = to_radix_key(rk[pos], desc);
        my_counts[(key >> shift) & RMASK]++;
      }
    }
#pragma unroll
    for (uint d = 0; d < uint(RSIZE); ++d) {
      uint s = simd_sum(my_counts[d]);
      if (lane == 0 && s > 0)
        atomic_fetch_add_explicit(&local_hist[d], s, memory_order_relaxed);
    }
  } else {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      int pos = i * RTPTG + int(lid);
      if (pos < items_this_block) {
        uint key = to_radix_key(rk[pos], desc);
        uint d = (key >> shift) & RMASK;
        atomic_fetch_add_explicit(&local_hist[d], 1u, memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint d = lid; d < uint(RSIZE); d += RTPTG) {
    uint count = atomic_load_explicit(&local_hist[d], memory_order_relaxed);
    histograms[row * RSIZE * n_blocks + d * n_blocks + block_idx] = count;
  }
}

// Fused count+scan: one TG per row, replaces radix_count+radix_scan when
// n_blocks <= MAX_BLOCKS (gated host-side)
template <typename T, ushort RTPTG, ushort EPT, ushort RBITS, ushort MAX_BLOCKS>
kernel void radix_count_scan(
    const device T* input [[buffer(0)]],
    device uint* histograms [[buffer(1)]],
    constant int3& dims [[buffer(2)]],
    constant bool& desc [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  const int sort_size = dims.x;
  const int n_blocks = dims.y;
  const int shift = dims.z;
  constexpr int RSIZE = 1 << RBITS;
  constexpr uint RMASK = RSIZE - 1;
  constexpr int ELEMS_PER_TG = RTPTG * EPT;
  constexpr int SIMD_W = 32;
  constexpr int MAX_SIMD_GROUPS = RTPTG / SIMD_W;
  constexpr int MAX_ENTRIES = int(RSIZE) * int(MAX_BLOCKS);

  uint lid = lid3.x;
  uint simd_id = lid / SIMD_W;
  uint lane = lid & (SIMD_W - 1);
  int row = tid.y;
  const int n_entries = RSIZE * n_blocks;

  // Flat histogram, layout hists_flat[d * n_blocks + b] to match global layout.
  threadgroup atomic_uint hists_flat[MAX_ENTRIES];
  for (uint i = lid; i < uint(n_entries); i += RTPTG)
    atomic_store_explicit(&hists_flat[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int b = 0; b < n_blocks; b++) {
    int block_start = b * ELEMS_PER_TG;
    int items_this_block = min(ELEMS_PER_TG, sort_size - block_start);
    const device T* rk = input + row * sort_size + block_start;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      int pos = i * RTPTG + int(lid);
      if (pos < items_this_block) {
        uint key = to_radix_key(rk[pos], desc);
        uint d = (key >> shift) & RMASK;
        atomic_fetch_add_explicit(
            &hists_flat[d * n_blocks + b], 1u, memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Exclusive prefix sum across n_entries slots (same pattern as radix_scan).
  uint chunk = (uint(n_entries) + RTPTG - 1) / RTPTG;
  uint my_start = min(lid * chunk, uint(n_entries));
  uint my_end = min(my_start + chunk, uint(n_entries));

  uint local_sum = 0;
  for (uint i = my_start; i < my_end; ++i)
    local_sum += atomic_load_explicit(&hists_flat[i], memory_order_relaxed);

  threadgroup uint simd_totals[MAX_SIMD_GROUPS];
  uint my_prefix = simd_prefix_exclusive_sum(local_sum);
  uint simd_total = simd_sum(local_sum);
  if (lane == 0)
    simd_totals[simd_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0 && lane < uint(MAX_SIMD_GROUPS)) {
    uint t = simd_totals[lane];
    simd_totals[lane] = simd_prefix_exclusive_sum(t);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint my_base = simd_totals[simd_id] + my_prefix;

  uint running = my_base;
  for (uint i = my_start; i < my_end; ++i) {
    uint v = atomic_load_explicit(&hists_flat[i], memory_order_relaxed);
    atomic_store_explicit(&hists_flat[i], running, memory_order_relaxed);
    running += v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lid; i < uint(n_entries); i += RTPTG) {
    histograms[row * n_entries + i] =
        atomic_load_explicit(&hists_flat[i], memory_order_relaxed);
  }
}

constant constexpr int SCAN_TPTG = 1024;

[[host_name("radix_scan")]]
kernel void radix_scan(
    device uint* histograms [[buffer(0)]],
    constant int& n_entries [[buffer(1)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  constexpr int SIMD_W = 32;
  constexpr int MAX_SIMD_GROUPS = SCAN_TPTG / SIMD_W;
  int row = tid.y;
  device uint* rh = histograms + row * n_entries;
  uint lid = lid3.x;
  uint simd_id = lid / SIMD_W;
  uint lane = lid % SIMD_W;

  uint chunk = (uint(n_entries) + SCAN_TPTG - 1) / SCAN_TPTG;
  uint my_start = min(lid * chunk, uint(n_entries));
  uint my_end = min(my_start + chunk, uint(n_entries));

  uint local_sum = 0;
  for (uint i = my_start; i < my_end; ++i)
    local_sum += rh[i];

  threadgroup uint simd_totals[MAX_SIMD_GROUPS];
  uint my_prefix = simd_prefix_exclusive_sum(local_sum);
  uint simd_total = simd_sum(local_sum);
  if (lane == 0)
    simd_totals[simd_id] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0 && lane < uint(MAX_SIMD_GROUPS)) {
    uint t = simd_totals[lane];
    simd_totals[lane] = simd_prefix_exclusive_sum(t);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint my_base = simd_totals[simd_id] + my_prefix;

  uint running = my_base;
  for (uint i = my_start; i < my_end; ++i) {
    uint v = rh[i];
    rh[i] = running;
    running += v;
  }
}

// Must match host kMaxFusedBlocks in Sort.mm.
constexpr constant int kMaxFusedBlocks = 4;
template <
    typename T,
    typename InIdxT,
    typename OutIdxT,
    short RTPTG,
    short EPT,
    short RBITS,
    bool FUSED_SCAN = false>
kernel void radix_scatter(
    const device T* keys_in [[buffer(0)]],
    const device InIdxT* vals_in [[buffer(1)]],
    device T* keys_out [[buffer(2)]],
    device OutIdxT* vals_out [[buffer(3)]],
    const device uint* offsets [[buffer(4)]],
    constant int3& dims [[buffer(5)]],
    constant bool2& flags [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]]) {
  const int sort_size = dims.x;
  const int n_blocks = dims.y;
  const int shift = dims.z;
  const bool desc = flags.x;
  const bool first_pass = flags.y;
  constexpr int RSIZE = 1 << RBITS;
  constexpr uint RMASK = RSIZE - 1;
  constexpr int ELEMS_PER_TG = RTPTG * EPT;
  constexpr int SIMD_W = 32;
  constexpr int MAX_SIMD_GROUPS = RTPTG / SIMD_W;
  constexpr int FUSED_BUF_SIZE = FUSED_SCAN ? RSIZE * kMaxFusedBlocks : 1;
  uint lid = lid3.x;
  uint simd_id = lid / SIMD_W;
  uint lane = lid & (SIMD_W - 1);
  int row = tid.y;
  int block_idx = tid.x;
  int block_start = block_idx * ELEMS_PER_TG;
  int items_this_block = min(ELEMS_PER_TG, sort_size - block_start);

  threadgroup T stage_keys[ELEMS_PER_TG];
  threadgroup InIdxT stage_idxs[ELEMS_PER_TG];
  threadgroup uint simd_sum_buf[MAX_SIMD_GROUPS];
  threadgroup uint tg_total_zeros;
  threadgroup uint block_offsets[RSIZE];
  threadgroup uint digit_start[RSIZE];

  threadgroup uint fused_buf[FUSED_BUF_SIZE];

#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    int pos = i * RTPTG + int(lid);
    if (pos < items_this_block) {
      stage_keys[pos] = keys_in[row * sort_size + block_start + pos];
      stage_idxs[pos] = first_pass
          ? InIdxT(block_start + pos)
          : vals_in[row * sort_size + block_start + pos];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  T local_keys[EPT];
  InIdxT local_idxs[EPT];
  uint my_active_start = min(uint(lid) * uint(EPT), uint(items_this_block));
  uint my_active_end = min((uint(lid) + 1) * uint(EPT), uint(items_this_block));
  uint my_active_count = my_active_end - my_active_start;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    if (uint(i) < my_active_count) {
      local_keys[i] = stage_keys[my_active_start + i];
      local_idxs[i] = stage_idxs[my_active_start + i];
    } else {
      local_keys[i] = sort_init<T>(desc);
      local_idxs[i] = InIdxT(0);
    }
  }

  for (int bit = 0; bit < RBITS; ++bit) {
    bool bits[EPT];
    uint my_zeros = 0;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      if (uint(i) < my_active_count) {
        uint d = (to_radix_key(local_keys[i], desc) >> shift) & RMASK;
        bits[i] = (d >> bit) & 1;
        if (!bits[i])
          ++my_zeros;
      } else {
        bits[i] = false;
      }
    }

    uint simd_prefix_val = simd_prefix_exclusive_sum(my_zeros);
    uint simd_total = simd_sum(my_zeros);
    if (lane == 0)
      simd_sum_buf[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
      uint t = (lane < uint(MAX_SIMD_GROUPS)) ? simd_sum_buf[lane] : 0u;
      uint p = simd_prefix_exclusive_sum(t);
      if (lane < uint(MAX_SIMD_GROUPS))
        simd_sum_buf[lane] = p;
      if (lane == uint(MAX_SIMD_GROUPS - 1))
        tg_total_zeros = p + t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_zero_prefix = simd_sum_buf[simd_id] + simd_prefix_val;
    uint total_zeros = tg_total_zeros;
    uint my_one_prefix = my_active_start - my_zero_prefix;

    uint next_zero = my_zero_prefix;
    uint next_one = total_zeros + my_one_prefix;

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      if (uint(i) < my_active_count) {
        uint new_pos = bits[i] ? next_one++ : next_zero++;
        stage_keys[new_pos] = local_keys[i];
        stage_idxs[new_pos] = local_idxs[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (bit < RBITS - 1) {
#pragma unroll
      for (int i = 0; i < EPT; ++i) {
        if (uint(i) < my_active_count) {
          local_keys[i] = stage_keys[my_active_start + i];
          local_idxs[i] = stage_idxs[my_active_start + i];
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    uint pos = my_active_start + uint(i);
    if (uint(i) < my_active_count) {
      uint d_here = (to_radix_key(stage_keys[pos], desc) >> shift) & RMASK;
      uint d_prev;
      if (pos == 0u) {
        d_prev = 0xFFFFFFFFu;
      } else {
        d_prev = (to_radix_key(stage_keys[pos - 1u], desc) >> shift) & RMASK;
      }
      if (d_here != d_prev) {
        digit_start[d_here] = pos;
      }
    }
  }
  if (FUSED_SCAN) {
    int n_entries = RSIZE * n_blocks;
    for (int i = int(lid); i < n_entries; i += RTPTG) {
      fused_buf[i] = offsets[row * n_entries + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk = (uint(n_entries) + RTPTG - 1) / RTPTG;
    uint my_start = min(lid * chunk, uint(n_entries));
    uint my_end = min(my_start + chunk, uint(n_entries));

    uint local_sum = 0;
    for (uint i = my_start; i < my_end; ++i)
      local_sum += fused_buf[i];

    uint my_prefix = simd_prefix_exclusive_sum(local_sum);
    uint simd_total = simd_sum(local_sum);
    if (lane == 0)
      simd_sum_buf[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane < uint(MAX_SIMD_GROUPS)) {
      uint t = simd_sum_buf[lane];
      simd_sum_buf[lane] = simd_prefix_exclusive_sum(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint my_base = simd_sum_buf[simd_id] + my_prefix;
    uint running = my_base;
    for (uint i = my_start; i < my_end; ++i) {
      uint v = fused_buf[i];
      fused_buf[i] = running;
      running += v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = lid; d < uint(RSIZE); d += RTPTG)
      block_offsets[d] = fused_buf[d * n_blocks + block_idx];
  } else {
    for (uint d = lid; d < uint(RSIZE); d += RTPTG)
      block_offsets[d] =
          offsets[row * RSIZE * n_blocks + d * n_blocks + block_idx];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    int pos = i * RTPTG + int(lid);
    if (pos < items_this_block) {
      T k = stage_keys[pos];
      uint idx = stage_idxs[pos];
      uint d = (to_radix_key(k, desc) >> shift) & RMASK;
      uint local_rank = uint(pos) - digit_start[d];
      uint global_pos = block_offsets[d] + local_rank;
      keys_out[row * sort_size + global_pos] = k;
      vals_out[row * sort_size + global_pos] = OutIdxT(idx);
    }
  }
}

#define INSTANTIATE_RADIX(T, RTPTG, EPT, RBITS)                                \
  template [[host_name("radix_count_" #T "_" #RBITS "bit")]]                   \
  kernel void radix_count<T, RTPTG, EPT, RBITS>(                               \
      const device T*,                                                         \
      device uint*,                                                            \
      constant int3&,                                                          \
      constant bool&,                                                          \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_" #T "_" #RBITS "bit")]]                 \
  kernel void radix_scatter<T, uint, uint, RTPTG, EPT, RBITS, false>(          \
      const device T*,                                                         \
      const device uint*,                                                      \
      device T*,                                                               \
      device uint*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_final_" #T "_" #RBITS "bit")]]           \
  kernel void radix_scatter<T, uint, long, RTPTG, EPT, RBITS, false>(          \
      const device T*,                                                         \
      const device uint*,                                                      \
      device T*,                                                               \
      device long*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_fused_" #T "_" #RBITS "bit")]]           \
  kernel void radix_scatter<T, uint, uint, RTPTG, EPT, RBITS, true>(           \
      const device T*,                                                         \
      const device uint*,                                                      \
      device T*,                                                               \
      device uint*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_fused_final_" #T "_" #RBITS "bit")]]     \
  kernel void radix_scatter<T, uint, long, RTPTG, EPT, RBITS, true>(           \
      const device T*,                                                         \
      const device uint*,                                                      \
      device T*,                                                               \
      device long*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
                                                                               \
  template [[host_name("radix_scatter_" #T "_" #RBITS "bit_u16")]]             \
  kernel void radix_scatter<T, ushort, ushort, RTPTG, EPT, RBITS, false>(      \
      const device T*,                                                         \
      const device ushort*,                                                    \
      device T*,                                                               \
      device ushort*,                                                          \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_final_" #T "_" #RBITS "bit_u16")]]       \
  kernel void radix_scatter<T, ushort, long, RTPTG, EPT, RBITS, false>(        \
      const device T*,                                                         \
      const device ushort*,                                                    \
      device T*,                                                               \
      device long*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_fused_" #T "_" #RBITS "bit_u16")]]       \
  kernel void radix_scatter<T, ushort, ushort, RTPTG, EPT, RBITS, true>(       \
      const device T*,                                                         \
      const device ushort*,                                                    \
      device T*,                                                               \
      device ushort*,                                                          \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
  template [[host_name("radix_scatter_fused_final_" #T "_" #RBITS "bit_u16")]] \
  kernel void radix_scatter<T, ushort, long, RTPTG, EPT, RBITS, true>(         \
      const device T*,                                                         \
      const device ushort*,                                                    \
      device T*,                                                               \
      device long*,                                                            \
      const device uint*,                                                      \
      constant int3&,                                                          \
      constant bool2&,                                                         \
      uint3,                                                                   \
      uint3);                                                                  \
                                                                               \
  template [[host_name("radix_count_scan_" #T "_" #RBITS "bit_mb4")]]          \
  kernel void radix_count_scan<T, RTPTG, EPT, RBITS, 4>(                       \
      const device T*,                                                         \
      device uint*,                                                            \
      constant int3&,                                                          \
      constant bool&,                                                          \
      uint3,                                                                   \
      uint3);

INSTANTIATE_RADIX(char, 512, 8, 4);
INSTANTIATE_RADIX(uchar, 512, 8, 4);
INSTANTIATE_RADIX(bool, 512, 8, 4);
INSTANTIATE_RADIX(half, 1024, 4, 8);
INSTANTIATE_RADIX(bfloat, 1024, 4, 8);
INSTANTIATE_RADIX(short, 1024, 4, 8);
INSTANTIATE_RADIX(ushort, 1024, 4, 8);
INSTANTIATE_RADIX(float, 512, 4, 8);
INSTANTIATE_RADIX(int, 512, 4, 8);
INSTANTIATE_RADIX(uint, 512, 4, 8);

#define INSTANTIATE_RADIX_TPTG512(T)                                          \
  template [[host_name("radix_count_" #T "_8bit_tptg512")]]                   \
  kernel void radix_count<T, 512, 4, 8>(                                      \
      const device T*,                                                        \
      device uint*,                                                           \
      constant int3&,                                                         \
      constant bool&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_" #T "_8bit_tptg512")]]                 \
  kernel void radix_scatter<T, uint, uint, 512, 4, 8, false>(                 \
      const device T*,                                                        \
      const device uint*,                                                     \
      device T*,                                                              \
      device uint*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_final_" #T "_8bit_tptg512")]]           \
  kernel void radix_scatter<T, uint, long, 512, 4, 8, false>(                 \
      const device T*,                                                        \
      const device uint*,                                                     \
      device T*,                                                              \
      device long*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_fused_" #T "_8bit_tptg512")]]           \
  kernel void radix_scatter<T, uint, uint, 512, 4, 8, true>(                  \
      const device T*,                                                        \
      const device uint*,                                                     \
      device T*,                                                              \
      device uint*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_fused_final_" #T "_8bit_tptg512")]]     \
  kernel void radix_scatter<T, uint, long, 512, 4, 8, true>(                  \
      const device T*,                                                        \
      const device uint*,                                                     \
      device T*,                                                              \
      device long*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
                                                                              \
  template [[host_name("radix_scatter_" #T "_8bit_tptg512_u16")]]             \
  kernel void radix_scatter<T, ushort, ushort, 512, 4, 8, false>(             \
      const device T*,                                                        \
      const device ushort*,                                                   \
      device T*,                                                              \
      device ushort*,                                                         \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_final_" #T "_8bit_tptg512_u16")]]       \
  kernel void radix_scatter<T, ushort, long, 512, 4, 8, false>(               \
      const device T*,                                                        \
      const device ushort*,                                                   \
      device T*,                                                              \
      device long*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_fused_" #T "_8bit_tptg512_u16")]]       \
  kernel void radix_scatter<T, ushort, ushort, 512, 4, 8, true>(              \
      const device T*,                                                        \
      const device ushort*,                                                   \
      device T*,                                                              \
      device ushort*,                                                         \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("radix_scatter_fused_final_" #T "_8bit_tptg512_u16")]] \
  kernel void radix_scatter<T, ushort, long, 512, 4, 8, true>(                \
      const device T*,                                                        \
      const device ushort*,                                                   \
      device T*,                                                              \
      device long*,                                                           \
      const device uint*,                                                     \
      constant int3&,                                                         \
      constant bool2&,                                                        \
      uint3,                                                                  \
      uint3);

INSTANTIATE_RADIX_TPTG512(half);
INSTANTIATE_RADIX_TPTG512(bfloat);
INSTANTIATE_RADIX_TPTG512(short);
INSTANTIATE_RADIX_TPTG512(ushort);
