// topk / kthvalue kernels: reuse the sort load/sort/scatter bodies
// (SortMerge.h, SortRadix.h) and only change the writeback to emit sorted ranks
// [offset, offset+count) as out[row*count + (rank-offset)]. topk = {0, k};
// kthvalue = the count==1 case {k-1, 1}.
#pragma once
#include <ATen/native/mps/kernels/SortMerge.h>
#include <ATen/native/mps/kernels/SortRadix.h>

template <typename T, short TPTG, short TN>
kernel void sort_block_topk(
    const device T* inp [[buffer(0)]],
    device T* out_vals [[buffer(1)]],
    device long* out_idx [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant long2& strides [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    constant int2& sel [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr int ELEMS_PER_TG = TPTG * TN;
  threadgroup T tgv[ELEMS_PER_TG];
  threadgroup uint tgi[ELEMS_PER_TG];
  sort_block_body<T, TPTG, TN, /*STABLE=*/false>(
      inp, tgv, tgi, size, strides.x, strides.y, desc, tid.y, lid.x);
  const int offset = sel.x;
  const int count = sel.y;
  long base_out = long(tid.y) * long(count);
  for (int j = int(lid.x); j < count; j += TPTG) {
    out_vals[base_out + j] = tgv[offset + j];
    out_idx[base_out + j] = long(tgi[offset + j]);
  }
}

template <typename T, typename InIdxT, short TPTG, short TN>
kernel void mb_merge_final_topk(
    const device T* vi [[buffer(0)]],
    const device InIdxT* ii [[buffer(1)]],
    device T* vo [[buffer(2)]],
    device long* io [[buffer(3)]],
    constant int3& dims [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    constant int2& sel [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const int size = dims.x;
  const int merge_tiles = dims.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  vi += tid.y * size;
  ii += tid.y * size;

  threadgroup T tgv[ELEMS_PER_TG];
  threadgroup InIdxT tgi[ELEMS_PER_TG];
  mb_merge_body<T, InIdxT, TPTG, TN>(
      vi, ii, tgv, tgi, size, merge_tiles, tid.x, desc, lid.x);

  // tgv[local] holds global sorted rank base+local,
  // emit those in [offset, offset+count).
  const int offset = sel.x;
  const int count = sel.y;
  const int base = int(tid.x) * ELEMS_PER_TG;
  long base_out = long(tid.y) * long(count);
  for (int local = int(lid.x); local < ELEMS_PER_TG; local += TPTG) {
    int g = base + local;
    if (g >= offset && g < offset + count && g < size) {
      vo[base_out + (g - offset)] = tgv[local];
      io[base_out + (g - offset)] = long(tgi[local]);
    }
  }
}

// Float keyed merge: sort to_radix_key(value) as an integer key (branch-free vs
// the NaN-aware float compare). desc is baked into the key, so only load/store
// transform.
template <typename T, typename KeyT, typename IdxT, short TPTG, short TN>
kernel void mb_sort_block_fkey(
    const device T* inp [[buffer(0)]],
    device KeyT* dv [[buffer(1)]],
    device IdxT* di [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant long2& strides [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const long stride_sort = strides.x;
  const long stride_seg = strides.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  long seg = tid.y * stride_seg;
  int blk = tid.x * ELEMS_PER_TG;
  threadgroup KeyT tgv[ELEMS_PER_TG];
  threadgroup IdxT tgi[ELEMS_PER_TG];
  for (int i = lid.x; i < ELEMS_PER_TG; i += TPTG) {
    int g = blk + i;
    tgv[i] = g < size ? KeyT(to_radix_key(inp[seg + g * stride_sort], desc))
                      : sort_init<KeyT>(false);
    tgi[i] = g < size ? IdxT(g) : ~IdxT(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort<KeyT, IdxT, TPTG, TN, false>(tgv, tgi, lid.x, false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int row = tid.y * size;
  for (int i = lid.x; i < ELEMS_PER_TG; i += TPTG) {
    int g = blk + i;
    if (g < size) {
      dv[row + g] = tgv[i];
      di[row + g] = tgi[i];
    }
  }
}

template <typename T, typename KeyT, typename InIdxT, short TPTG, short TN>
kernel void mb_merge_final_fkey(
    const device KeyT* vi [[buffer(0)]],
    const device InIdxT* ii [[buffer(1)]],
    device T* vo [[buffer(2)]],
    device long* io [[buffer(3)]],
    constant int3& dims [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const int size = dims.x;
  const int merge_tiles = dims.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  vi += tid.y * size;
  ii += tid.y * size;
  vo += tid.y * size;
  io += tid.y * size;
  threadgroup KeyT tgv[ELEMS_PER_TG];
  threadgroup InIdxT tgi[ELEMS_PER_TG];
  mb_merge_body<KeyT, InIdxT, TPTG, TN>(
      vi, ii, tgv, tgi, size, merge_tiles, tid.x, false, lid.x);
  int base = tid.x * ELEMS_PER_TG;
  for (int i = lid.x; i < ELEMS_PER_TG; i += TPTG) {
    int g = base + i;
    if (g < size) {
      vo[g] = from_radix_key<T>(tgv[i], desc);
      io[g] = long(tgi[i]);
    }
  }
}

template <typename T, typename KeyT, typename InIdxT, short TPTG, short TN>
kernel void mb_merge_final_topk_fkey(
    const device KeyT* vi [[buffer(0)]],
    const device InIdxT* ii [[buffer(1)]],
    device T* vo [[buffer(2)]],
    device long* io [[buffer(3)]],
    constant int3& dims [[buffer(4)]],
    constant bool& desc [[buffer(5)]],
    constant int2& sel [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const int size = dims.x;
  const int merge_tiles = dims.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  vi += tid.y * size;
  ii += tid.y * size;
  threadgroup KeyT tgv[ELEMS_PER_TG];
  threadgroup InIdxT tgi[ELEMS_PER_TG];
  mb_merge_body<KeyT, InIdxT, TPTG, TN>(
      vi, ii, tgv, tgi, size, merge_tiles, tid.x, false, lid.x);
  const int offset = sel.x;
  const int count = sel.y;
  const int base = int(tid.x) * ELEMS_PER_TG;
  long base_out = long(tid.y) * long(count);
  for (int local = int(lid.x); local < ELEMS_PER_TG; local += TPTG) {
    int g = base + local;
    if (g >= offset && g < offset + count && g < size) {
      vo[base_out + (g - offset)] = from_radix_key<T>(tgv[local], desc);
      io[base_out + (g - offset)] = long(tgi[local]);
    }
  }
}

template <
    typename T,
    typename InIdxT,
    short RTPTG,
    short EPT,
    short RBITS,
    bool FUSED_SCAN = false>
kernel void radix_scatter_topk(
    const device T* keys_in [[buffer(0)]],
    const device InIdxT* vals_in [[buffer(1)]],
    device T* keys_out [[buffer(2)]],
    device long* vals_out [[buffer(3)]],
    const device uint* offsets [[buffer(4)]],
    constant int3& dims [[buffer(5)]],
    constant bool2& flags [[buffer(6)]],
    constant int2& sel [[buffer(7)]],
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

  radix_scatter_body<T, InIdxT, RTPTG, EPT, RBITS, FUSED_SCAN>(
      keys_in,
      vals_in,
      offsets,
      stage_keys,
      stage_idxs,
      simd_sum_buf,
      &tg_total_zeros,
      block_offsets,
      digit_start,
      fused_buf,
      sort_size,
      n_blocks,
      shift,
      desc,
      first_pass,
      row,
      block_idx,
      items_this_block,
      lid,
      simd_id,
      lane);

  const int offset = sel.x;
  const int count = sel.y;
  long base_out = long(row) * long(count);
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    int pos = i * RTPTG + int(lid);
    if (pos < items_this_block) {
      T k = stage_keys[pos];
      uint idx = stage_idxs[pos];
      uint d = (to_radix_key(k, desc) >> shift) & RMASK;
      uint local_rank = uint(pos) - digit_start[d];
      uint global_pos = block_offsets[d] + local_rank;
      if (global_pos >= uint(offset) && global_pos < uint(offset + count)) {
        keys_out[base_out + long(global_pos) - long(offset)] = k;
        vals_out[base_out + long(global_pos) - long(offset)] = long(idx);
      }
    }
  }
}

#define INSTANTIATE_TOPK_SORT_BLOCK(T, TPTG, TN)                          \
  template [[host_name("sort_block_topk_" #T "_tptg" #TPTG)]] kernel void \
  sort_block_topk<T, TPTG, TN>(                                           \
      const device T*,                                                    \
      device T*,                                                          \
      device long*,                                                       \
      constant int&,                                                      \
      constant long2&,                                                    \
      constant bool&,                                                     \
      constant int2&,                                                     \
      uint3,                                                              \
      uint3);

#define INSTANTIATE_TOPK_MB_MERGE_FINAL(T, TPTG, TN)                     \
  template [[host_name("mb_merge_final_topk_" #T "_tptg" #TPTG)]]        \
  kernel void mb_merge_final_topk<T, uint, TPTG, TN>(                    \
      const device T*,                                                   \
      const device uint*,                                                \
      device T*,                                                         \
      device long*,                                                      \
      constant int3&,                                                    \
      constant bool&,                                                    \
      constant int2&,                                                    \
      uint3,                                                             \
      uint3);                                                            \
  template [[host_name("mb_merge_final_topk_" #T "_tptg" #TPTG "_u16")]] \
  kernel void mb_merge_final_topk<T, ushort, TPTG, TN>(                  \
      const device T*,                                                   \
      const device ushort*,                                              \
      device T*,                                                         \
      device long*,                                                      \
      constant int3&,                                                    \
      constant bool&,                                                    \
      constant int2&,                                                    \
      uint3,                                                             \
      uint3);

#define INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, TPTG, TN) \
  INSTANTIATE_TOPK_SORT_BLOCK(T, TPTG, TN)            \
  INSTANTIATE_TOPK_MB_MERGE_FINAL(T, TPTG, TN)

#define INSTANTIATE_TOPK_ALL_TPTG(T)          \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 32, 4)  \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 64, 4)  \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 128, 4) \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 256, 4) \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 512, 4)

#define INSTANTIATE_TOPK_ALL_TPTG_1024(T) \
  INSTANTIATE_TOPK_ALL_TPTG(T)            \
  INSTANTIATE_TOPK_BLOCK_AND_MERGE(T, 1024, 4)

INSTANTIATE_TOPK_ALL_TPTG_1024(float);
INSTANTIATE_TOPK_ALL_TPTG_1024(half);
INSTANTIATE_TOPK_ALL_TPTG_1024(bfloat);
INSTANTIATE_TOPK_ALL_TPTG_1024(int);
INSTANTIATE_TOPK_ALL_TPTG(long);
INSTANTIATE_TOPK_ALL_TPTG_1024(short);
INSTANTIATE_TOPK_ALL_TPTG_1024(char);
INSTANTIATE_TOPK_ALL_TPTG_1024(uchar);
INSTANTIATE_TOPK_ALL_TPTG_1024(bool);
INSTANTIATE_TOPK_ALL_TPTG_1024(ushort);
INSTANTIATE_TOPK_ALL_TPTG_1024(uint);
INSTANTIATE_TOPK_ALL_TPTG(ulong);

#define _SCATTER_TOPK_SIG(T, InIdxT)                                       \
  const device T*, const device InIdxT*, device T*, device long*,          \
      const device uint*, constant int3&, constant bool2&, constant int2&, \
      uint3, uint3

#define INSTANTIATE_TOPK_RADIX_SCATTER(T, InIdxT, RTPTG, EPT, RBITS, NAME) \
  template [[host_name("radix_scatter_final_topk_" NAME)]]                 \
  kernel void radix_scatter_topk<T, InIdxT, RTPTG, EPT, RBITS, false>(     \
      _SCATTER_TOPK_SIG(T, InIdxT));                                       \
  template [[host_name("radix_scatter_fused_final_topk_" NAME)]]           \
  kernel void radix_scatter_topk<T, InIdxT, RTPTG, EPT, RBITS, true>(      \
      _SCATTER_TOPK_SIG(T, InIdxT));

#define INSTANTIATE_TOPK_RADIX(T, RTPTG, EPT, RBITS)   \
  INSTANTIATE_TOPK_RADIX_SCATTER(                      \
      T, uint, RTPTG, EPT, RBITS, #T "_" #RBITS "bit") \
  INSTANTIATE_TOPK_RADIX_SCATTER(                      \
      T, ushort, RTPTG, EPT, RBITS, #T "_" #RBITS "bit_u16")

INSTANTIATE_TOPK_RADIX(char, 512, 8, 4);
INSTANTIATE_TOPK_RADIX(uchar, 512, 8, 4);
INSTANTIATE_TOPK_RADIX(bool, 512, 8, 4);
INSTANTIATE_TOPK_RADIX(half, 1024, 4, 8);
INSTANTIATE_TOPK_RADIX(bfloat, 1024, 4, 8);
INSTANTIATE_TOPK_RADIX(short, 1024, 4, 8);
INSTANTIATE_TOPK_RADIX(ushort, 1024, 4, 8);
INSTANTIATE_TOPK_RADIX(float, 512, 4, 8);
INSTANTIATE_TOPK_RADIX(int, 512, 4, 8);
INSTANTIATE_TOPK_RADIX(uint, 512, 4, 8);

#define INSTANTIATE_TOPK_RADIX_TPTG512(T)                                \
  INSTANTIATE_TOPK_RADIX_SCATTER(T, uint, 512, 4, 8, #T "_8bit_tptg512") \
  INSTANTIATE_TOPK_RADIX_SCATTER(T, ushort, 512, 4, 8, #T "_8bit_tptg512_u16")

INSTANTIATE_TOPK_RADIX_TPTG512(half);
INSTANTIATE_TOPK_RADIX_TPTG512(bfloat);
INSTANTIATE_TOPK_RADIX_TPTG512(short);
INSTANTIATE_TOPK_RADIX_TPTG512(ushort);

// Keyed float merge: instantiated only for large-segment TPTGs ({512, 1024}),
// where the host-side keyed gate (sort_size >= 8192) lands; 256 is a safety
// margin.
#define INSTANTIATE_FKEY(T, KeyT, TPTG, TN)                                   \
  template [[host_name("mb_sort_block_fkey_" #T "_tptg" #TPTG)]]              \
  kernel void mb_sort_block_fkey<T, KeyT, uint, TPTG, TN>(                    \
      const device T*,                                                        \
      device KeyT*,                                                           \
      device uint*,                                                           \
      constant int&,                                                          \
      constant long2&,                                                        \
      constant bool&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("mb_sort_block_fkey_" #T "_tptg" #TPTG "_u16")]]       \
  kernel void mb_sort_block_fkey<T, KeyT, ushort, TPTG, TN>(                  \
      const device T*,                                                        \
      device KeyT*,                                                           \
      device ushort*,                                                         \
      constant int&,                                                          \
      constant long2&,                                                        \
      constant bool&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("mb_merge_final_fkey_" #T "_tptg" #TPTG)]]             \
  kernel void mb_merge_final_fkey<T, KeyT, uint, TPTG, TN>(                   \
      const device KeyT*,                                                     \
      const device uint*,                                                     \
      device T*,                                                              \
      device long*,                                                           \
      constant int3&,                                                         \
      constant bool&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("mb_merge_final_fkey_" #T "_tptg" #TPTG "_u16")]]      \
  kernel void mb_merge_final_fkey<T, KeyT, ushort, TPTG, TN>(                 \
      const device KeyT*,                                                     \
      const device ushort*,                                                   \
      device T*,                                                              \
      device long*,                                                           \
      constant int3&,                                                         \
      constant bool&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("mb_merge_final_topk_fkey_" #T "_tptg" #TPTG)]]        \
  kernel void mb_merge_final_topk_fkey<T, KeyT, uint, TPTG, TN>(              \
      const device KeyT*,                                                     \
      const device uint*,                                                     \
      device T*,                                                              \
      device long*,                                                           \
      constant int3&,                                                         \
      constant bool&,                                                         \
      constant int2&,                                                         \
      uint3,                                                                  \
      uint3);                                                                 \
  template [[host_name("mb_merge_final_topk_fkey_" #T "_tptg" #TPTG "_u16")]] \
  kernel void mb_merge_final_topk_fkey<T, KeyT, ushort, TPTG, TN>(            \
      const device KeyT*,                                                     \
      const device ushort*,                                                   \
      device T*,                                                              \
      device long*,                                                           \
      constant int3&,                                                         \
      constant bool&,                                                         \
      constant int2&,                                                         \
      uint3,                                                                  \
      uint3);

#define INSTANTIATE_FKEY_ALL(T, KeyT) \
  INSTANTIATE_FKEY(T, KeyT, 256, 4)   \
  INSTANTIATE_FKEY(T, KeyT, 512, 4)   \
  INSTANTIATE_FKEY(T, KeyT, 1024, 4)

INSTANTIATE_FKEY_ALL(float, uint);
INSTANTIATE_FKEY_ALL(half, ushort);
INSTANTIATE_FKEY_ALL(bfloat, ushort);
