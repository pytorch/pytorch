
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Sort kernels for MPS. This file contains single-block sort path: one
// threadgroup per row, used when the segment fits in threadgroup memory.
// Multi-block merge and radix paths are planned for follow-up PRs.
//
// File layout:
//   1. Shared comparators & padding  (sort_compare, sort_init)
//   2. Merge primitives              (merge_partition, merge_step)
//   3. SIMD bitonic building blocks  (sort_shuffle_xor, bitonic_substage,
//                                     simd_bitonic_sort4)
//   4. Block merge sort              (block_merge_sort)
//   5. Single-block sort kernel      (sort_block)
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Shared comparators & padding
// -----------------------------------------------------------------------------

template <typename T>
inline bool sort_compare(T a, T b, bool desc) {
  return desc ? c10::metal::less(b, a) : c10::metal::less(a, b);
}

template <
    bool STABLE,
    typename T,
    typename IdxT,
    ::metal::enable_if_t<STABLE, bool> = true>
inline bool bitonic_lt(
    T vi,
    IdxT ii,
    T vp,
    IdxT ip,
    bool /*i_am_low*/,
    bool desc) {
  if (sort_compare(vi, vp, desc))
    return true;
  if (sort_compare(vp, vi, desc))
    return false;
  return ii < ip;
}

template <
    bool STABLE,
    typename T,
    typename IdxT,
    ::metal::enable_if_t<!STABLE, bool> = true>
inline bool bitonic_lt(
    T vi,
    IdxT /*ii*/,
    T vp,
    IdxT /*ip*/,
    bool i_am_low,
    bool desc) {
  return sort_compare(vi, vp, desc) ||
      (!sort_compare(vp, vi, desc) && i_am_low);
}

// Padding value for out-of-range slots. Chosen to sort to the end of the
// output (largest for asc, smallest for desc) so padding never lands among
// real data. Floats use NaN on asc because c10::metal::less puts NaNs last.
template <
    typename T,
    ::metal::enable_if_t<::metal::is_same_v<T, bool>, bool> = true>
inline T sort_init(bool desc) {
  return !desc;
}

template <
    typename T,
    ::metal::enable_if_t<
        !::metal::is_same_v<T, bool> && ::metal::is_floating_point_v<T>,
        bool> = true>
inline T sort_init(bool desc) {
  return desc ? T(-INFINITY) : T(NAN);
}

template <
    typename T,
    ::metal::enable_if_t<
        !::metal::is_same_v<T, bool> && !::metal::is_floating_point_v<T>,
        bool> = true>
inline T sort_init(bool desc) {
  return desc ? metal::numeric_limits<T>::lowest()
              : metal::numeric_limits<T>::max();
}

// -----------------------------------------------------------------------------
// 2. Merge primitives
//
// merge_partition finds the split point on the merge-path diagonal for two
// sorted runs; merge_step sequentially merges TN elements from that split.
// -----------------------------------------------------------------------------

template <typename T>
inline int merge_partition(
    const threadgroup T* A,
    const threadgroup T* B,
    int a_sz,
    int b_sz,
    int diag,
    bool desc) {
  int lo = max(0, diag - b_sz), hi = min(diag, a_sz);
  while (lo < hi) {
    int m = lo + (hi - lo) / 2;
    if (sort_compare(B[diag - 1 - m], A[m], desc)) {
      hi = m;
    } else {
      lo = m + 1;
    }
  }
  return hi;
}

template <typename T, typename IdxT, short N>
inline void merge_step(
    const threadgroup T* A,
    const threadgroup T* B,
    const threadgroup IdxT* Ai,
    const threadgroup IdxT* Bi,
    int a_sz,
    int b_sz,
    thread T (&v)[N],
    thread IdxT (&idx)[N],
    bool desc) {
  T init = sort_init<T>(desc);
  int a = 0, b = 0;
  for (int i = 0; i < N; ++i) {
    T va = (a < a_sz) ? A[a] : init;
    T vb = (b < b_sz) ? B[b] : init;
    bool tb = (b < b_sz) && (a >= a_sz || sort_compare(vb, va, desc));
    v[i] = tb ? vb : va;
    idx[i] = tb ? Bi[b] : ((a < a_sz) ? Ai[a] : IdxT(0));
    b += int(tb);
    a += int(!tb);
  }
}

// -----------------------------------------------------------------------------
// 3. SIMD bitonic building blocks
//
// simd_bitonic_sort4 sorts TN=4 elements per lane across a full SIMD group
// (128 elements) as the first stage of block_merge_sort. The substages within
// a lane use register swaps. substages across lanes use simd_shuffle_xor.
// -----------------------------------------------------------------------------

template <
    typename T,
    ::metal::enable_if_t<::metal::is_same_v<T, bool>, bool> = true>
inline T sort_shuffle_xor(T v, ushort delta) {
  return bool(simd_shuffle_xor(uint(v), delta));
}

template <
    typename T,
    ::metal::enable_if_t<sizeof(T) == 1 && !::metal::is_same_v<T, bool>, bool> =
        true>
inline T sort_shuffle_xor(T v, ushort delta) {
  uchar u = as_type<uchar>(v);
  return as_type<T>(uchar(simd_shuffle_xor(uint(u), delta)));
}

template <typename T, ::metal::enable_if_t<sizeof(T) == 2, bool> = true>
inline T sort_shuffle_xor(T v, ushort delta) {
  ushort u = as_type<ushort>(v);
  return as_type<T>(ushort(simd_shuffle_xor(uint(u), delta)));
}

template <typename T, ::metal::enable_if_t<sizeof(T) == 4, bool> = true>
inline T sort_shuffle_xor(T v, ushort delta) {
  return simd_shuffle_xor(v, delta);
}

template <typename T, ::metal::enable_if_t<sizeof(T) == 8, bool> = true>
inline T sort_shuffle_xor(T v, ushort delta) {
  ulong u = as_type<ulong>(v);
  uint lo = simd_shuffle_xor(uint(u), delta);
  uint hi = simd_shuffle_xor(uint(u >> 32), delta);
  return as_type<T>(ulong(lo) | (ulong(hi) << 32));
}

template <
    typename T,
    typename IdxT,
    short TN,
    int K,
    int OFFSET,
    bool STABLE,
    ::metal::enable_if_t<(OFFSET < TN), bool> = true>
inline void bitonic_substage(
    thread T (&v)[TN],
    thread IdxT (&idx)[TN],
    uint lane,
    bool desc) {
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    short pi = i ^ OFFSET;
    if (pi > i) {
      int global_p = int(lane) * TN + i;
      bool ascending = (global_p & K) == 0;
      T vi = v[i], vp = v[pi];
      IdxT ii = idx[i], ip = idx[pi];
      bool vi_first =
          bitonic_lt<STABLE>(vi, ii, vp, ip, /*i_am_low=*/false, desc);
      bool do_swap = ascending ? !vi_first : vi_first;
      v[i] = do_swap ? vp : vi;
      v[pi] = do_swap ? vi : vp;
      idx[i] = do_swap ? ip : ii;
      idx[pi] = do_swap ? ii : ip;
    }
  }
}

template <
    typename T,
    typename IdxT,
    short TN,
    int K,
    int OFFSET,
    bool STABLE,
    ::metal::enable_if_t<(OFFSET >= TN), bool> = true>
inline void bitonic_substage(
    thread T (&v)[TN],
    thread IdxT (&idx)[TN],
    uint lane,
    bool desc) {
  constexpr ushort LANE_OFFSET = OFFSET / TN;
  bool i_am_low = (lane & uint(LANE_OFFSET)) == 0;
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    T vi = v[i];
    IdxT ii = idx[i];
    T vp = sort_shuffle_xor(vi, LANE_OFFSET);
    IdxT ip = sort_shuffle_xor(ii, LANE_OFFSET);
    int global_p = int(lane) * TN + i;
    bool ascending = (global_p & K) == 0;
    // STABLE=false uses i_am_low so the two lanes agree on which element
    // each keeps (without it, equal values make both lanes grab the same
    // one and we get duplicate indices).
    // STABLE=true ignores i_am_low and uses the lower original index - both
    // lanes still agree because their (ii vs ip) views are mirror images.
    bool vi_first = bitonic_lt<STABLE>(vi, ii, vp, ip, i_am_low, desc);
    bool should_take = vi_first != (ascending == i_am_low);
    v[i] = should_take ? vp : vi;
    idx[i] = should_take ? ip : ii;
  }
}

template <typename T, typename IdxT, bool STABLE>
inline void simd_bitonic_sort4(
    thread T (&v)[4],
    thread IdxT (&idx)[4],
    uint lane,
    bool desc) {
  bitonic_substage<T, IdxT, 4, 2, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 4, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 4, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 8, 4, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 8, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 8, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 16, 8, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 4, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 16, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 32, 16, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 8, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 4, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 32, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 64, 32, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 16, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 8, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 4, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 64, 1, STABLE>(v, idx, lane, desc);

  bitonic_substage<T, IdxT, 4, 128, 64, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 32, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 16, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 8, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 4, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 2, STABLE>(v, idx, lane, desc);
  bitonic_substage<T, IdxT, 4, 128, 1, STABLE>(v, idx, lane, desc);
}

// -----------------------------------------------------------------------------
// 4. Block merge sort
//
// Sorts ELEMS_PER_TG = TPTG*TN elements held in threadgroup memory (tv/ti).
// First stage is a SIMD bitonic sort per 128-element SIMD group; subsequent
// stages double the merged run size up to TPTG.
// -----------------------------------------------------------------------------

template <typename T, typename IdxT, short TPTG, short TN, bool STABLE>
inline void block_merge_sort(
    threadgroup T* tv,
    threadgroup IdxT* ti,
    uint lid,
    bool desc) {
  static_assert(
      TN == 4 && TPTG >= 32, "block_merge_sort requires TN==4 and TPTG>=32");
  int base = lid * TN;
  thread T lv[TN];
  thread IdxT li[TN];
  for (int i = 0; i < TN; ++i) {
    lv[i] = tv[base + i];
    li[i] = ti[base + i];
  }

  simd_bitonic_sort4<T, IdxT, STABLE>(lv, li, lid & 31u, desc);

  if (TPTG >= 64) {
    constexpr int mt_first = 64;
    for (int i = 0; i < TN; ++i) {
      tv[base + i] = lv[i];
      ti[base + i] = li[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int grp = lid / mt_first, lane = lid % mt_first;
    int sz = TN * mt_first, st = sz * grp;
    int hsz = sz / 2;
    int diag = TN * lane;
    int p = merge_partition(tv + st, tv + st + hsz, hsz, hsz, diag, desc);
    merge_step<T, IdxT, TN>(
        tv + st + p,
        tv + st + hsz + diag - p,
        ti + st + p,
        ti + st + hsz + diag - p,
        hsz - p,
        hsz - diag + p,
        lv,
        li,
        desc);
  }
  for (int mt = 128; mt <= TPTG; mt *= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < TN; ++i) {
      tv[base + i] = lv[i];
      ti[base + i] = li[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int grp = lid / mt, lane = lid % mt;
    int sz = TN * mt, st = sz * grp;
    int hsz = sz / 2;
    int diag = TN * lane;
    int p = merge_partition(tv + st, tv + st + hsz, hsz, hsz, diag, desc);
    merge_step<T, IdxT, TN>(
        tv + st + p,
        tv + st + hsz + diag - p,
        ti + st + p,
        ti + st + hsz + diag - p,
        hsz - p,
        hsz - diag + p,
        lv,
        li,
        desc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < TN; ++i) {
    tv[base + i] = lv[i];
    ti[base + i] = li[i];
  }
}

// =============================================================================
// 5. Single-block sort kernel
//
// One threadgroup per row. The whole segment is loaded into threadgroup
// memory, sorted in place with block_merge_sort and written out. Selected
// by the host when size <= TPTG*TN for the largest available TPTG.
// =============================================================================

template <typename T, short TPTG, short TN, bool STABLE>
kernel void sort_block(
    const device T* inp [[buffer(0)]],
    device T* out_vals [[buffer(1)]],
    device long* out_idx [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant long2& strides [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const long stride_sort = strides.x;
  const long stride_seg = strides.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  threadgroup T tgv[ELEMS_PER_TG];
  threadgroup uint tgi[ELEMS_PER_TG];

  T init = sort_init<T>(desc);
  long base_in = tid.y * stride_seg;
  long base_out = long(tid.y) * long(size);
  for (int i = lid.x; i < ELEMS_PER_TG; i += TPTG) {
    tgv[i] = i < size ? inp[base_in + i * stride_sort] : init;
    tgi[i] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort<T, uint, TPTG, TN, STABLE>(tgv, tgi, lid.x, desc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = lid.x; i < size; i += TPTG) {
    out_vals[base_out + i] = tgv[i];
    out_idx[base_out + i] = long(tgi[i]);
  }
}

// TODO: reuse DEFAULT_ILP from c10/metal/common.h for TN
#define INSTANTIATE_SORT_VARIANT(T, TPTG, TN, STABLE, SUFFIX)              \
  template[[host_name("sort_block_" #T "_tptg" #TPTG SUFFIX)]] kernel void \
  sort_block<T, TPTG, TN, STABLE>(                                         \
      const device T*,                                                     \
      device T*,                                                           \
      device long*,                                                        \
      constant int&,                                                       \
      constant long2&,                                                     \
      constant bool&,                                                      \
      uint3,                                                               \
      uint3);

#define INSTANTIATE_SORT(T, TPTG, TN)              \
  INSTANTIATE_SORT_VARIANT(T, TPTG, TN, false, "") \
  INSTANTIATE_SORT_VARIANT(T, TPTG, TN, true, "_stable")

#define INSTANTIATE_ALL_TPTG(T) \
  INSTANTIATE_SORT(T, 32, 4)    \
  INSTANTIATE_SORT(T, 64, 4)    \
  INSTANTIATE_SORT(T, 128, 4)   \
  INSTANTIATE_SORT(T, 256, 4)   \
  INSTANTIATE_SORT(T, 512, 4)

#define INSTANTIATE_ALL_TPTG_1024(T) \
  INSTANTIATE_ALL_TPTG(T)            \
  INSTANTIATE_SORT(T, 1024, 4)

INSTANTIATE_ALL_TPTG_1024(float);
INSTANTIATE_ALL_TPTG_1024(half);
INSTANTIATE_ALL_TPTG_1024(bfloat);
INSTANTIATE_ALL_TPTG_1024(int);
INSTANTIATE_ALL_TPTG(long);
INSTANTIATE_ALL_TPTG_1024(short);
INSTANTIATE_ALL_TPTG_1024(char);
INSTANTIATE_ALL_TPTG_1024(uchar);
INSTANTIATE_ALL_TPTG_1024(bool);
INSTANTIATE_ALL_TPTG_1024(ushort);
INSTANTIATE_ALL_TPTG_1024(uint);
INSTANTIATE_ALL_TPTG(ulong);
