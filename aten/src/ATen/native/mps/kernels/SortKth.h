// Kth value metal kernels, most of the code here is reusing the load/sort/scatter bodies
// from the sort kernels (SortMerge.h and SortRadix.h) and only customize the writeback
// Instead of emitting all sort_size elements per row, they emit one element per row at
// sorted rank `kth`. Output is [n_rows], i.e. one scalar per row
#pragma once
#include <ATen/native/mps/kernels/SortMerge.h>
#include <ATen/native/mps/kernels/SortRadix.h>


template <typename T, short TPTG, short TN>
kernel void sort_block_kth(const device T* inp [[buffer(0)]],
                           device T* out_vals [[buffer(1)]],
                           device long* out_idx [[buffer(2)]],
                           constant int& size [[buffer(3)]],
                           constant long2& strides [[buffer(4)]],
                           constant bool& desc [[buffer(5)]],
                           constant int& kth [[buffer(6)]],
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr int ELEMS_PER_TG = TPTG * TN;
  threadgroup T tgv[ELEMS_PER_TG];
  threadgroup uint tgi[ELEMS_PER_TG];
  sort_block_body<T, TPTG, TN, /*STABLE=*/false>(inp, tgv, tgi, size, strides.x, strides.y, desc, tid.y, lid.x);
  if (lid.x == 0) {
    out_vals[tid.y] = tgv[kth];
    out_idx[tid.y] = long(tgi[kth]);
  }
}

template <typename T, typename InIdxT, short TPTG, short TN>
kernel void mb_merge_final_kth(const device T* vi [[buffer(0)]],
                               const device InIdxT* ii [[buffer(1)]],
                               device T* vo [[buffer(2)]],
                               device long* io [[buffer(3)]],
                               constant int3& dims [[buffer(4)]],
                               constant bool& desc [[buffer(5)]],
                               constant int& kth [[buffer(6)]],
                               uint3 tid [[threadgroup_position_in_grid]],
                               uint3 lid [[thread_position_in_threadgroup]]) {
  const int size = dims.x;
  const int merge_tiles = dims.y;
  constexpr int ELEMS_PER_TG = TPTG * TN;
  vi += tid.y * size;
  ii += tid.y * size;

  threadgroup T tgv[ELEMS_PER_TG];
  threadgroup InIdxT tgi[ELEMS_PER_TG];
  mb_merge_body<T, InIdxT, TPTG, TN>(vi, ii, tgv, tgi, size, merge_tiles, tid.x, desc, lid.x);

  int base = tid.x * ELEMS_PER_TG;
  if (kth >= base && kth < base + ELEMS_PER_TG && kth < size) {
    if (lid.x == 0) {
      int local = kth - base;
      vo[tid.y] = tgv[local];
      io[tid.y] = long(tgi[local]);
    }
  }
}

template <typename T, typename InIdxT, short RTPTG, short EPT, short RBITS, bool FUSED_SCAN = false>
kernel void radix_scatter_kth(const device T* keys_in [[buffer(0)]],
                              const device InIdxT* vals_in [[buffer(1)]],
                              device T* keys_out [[buffer(2)]],
                              device long* vals_out [[buffer(3)]],
                              const device uint* offsets [[buffer(4)]],
                              constant int3& dims [[buffer(5)]],
                              constant bool2& flags [[buffer(6)]],
                              constant int& kth [[buffer(7)]],
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

  radix_scatter_body<T, InIdxT, RTPTG, EPT, RBITS, FUSED_SCAN>(keys_in,
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

#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    int pos = i * RTPTG + int(lid);
    if (pos < items_this_block) {
      T k = stage_keys[pos];
      uint idx = stage_idxs[pos];
      uint d = (to_radix_key(k, desc) >> shift) & RMASK;
      uint local_rank = uint(pos) - digit_start[d];
      uint global_pos = block_offsets[d] + local_rank;
      if (global_pos == uint(kth)) {
        keys_out[row] = k;
        vals_out[row] = long(idx);
      }
    }
  }
}

#define INSTANTIATE_KTH_SORT_BLOCK(T, TPTG, TN)                                                                  \
  template [[host_name("sort_block_kth_" #T "_tptg" #TPTG)]] kernel void sort_block_kth<T, TPTG, TN>(            \
      const device T*,                                                                                           \
      device T*,                                                                                                 \
      device long*,                                                                                              \
      constant int&,                                                                                             \
      constant long2&,                                                                                           \
      constant bool&,                                                                                            \
      constant int&,                                                                                             \
      uint3,                                                                                                     \
      uint3);

#define INSTANTIATE_KTH_MB_MERGE_FINAL(T, TPTG, TN)                                                              \
  template [[host_name("mb_merge_final_kth_" #T "_tptg" #TPTG)]]                                                 \
  kernel void mb_merge_final_kth<T, uint, TPTG, TN>(const device T*,                                             \
                                                    const device uint*,                                          \
                                                    device T*,                                                   \
                                                    device long*,                                                \
                                                    constant int3&,                                              \
                                                    constant bool&,                                              \
                                                    constant int&,                                               \
                                                    uint3,                                                       \
                                                    uint3);                                                      \
  template [[host_name("mb_merge_final_kth_" #T "_tptg" #TPTG "_u16")]]                                          \
  kernel void mb_merge_final_kth<T, ushort, TPTG, TN>(const device T*,                                           \
                                                      const device ushort*,                                      \
                                                      device T*,                                                 \
                                                      device long*,                                              \
                                                      constant int3&,                                            \
                                                      constant bool&,                                            \
                                                      constant int&,                                             \
                                                      uint3,                                                     \
                                                      uint3);

#define INSTANTIATE_KTH_BLOCK_AND_MERGE(T, TPTG, TN) \
  INSTANTIATE_KTH_SORT_BLOCK(T, TPTG, TN)            \
  INSTANTIATE_KTH_MB_MERGE_FINAL(T, TPTG, TN)

#define INSTANTIATE_KTH_ALL_TPTG(T)            \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 32, 4)    \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 64, 4)    \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 128, 4)   \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 256, 4)   \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 512, 4)

#define INSTANTIATE_KTH_ALL_TPTG_1024(T) \
  INSTANTIATE_KTH_ALL_TPTG(T)            \
  INSTANTIATE_KTH_BLOCK_AND_MERGE(T, 1024, 4)

INSTANTIATE_KTH_ALL_TPTG_1024(float);
INSTANTIATE_KTH_ALL_TPTG_1024(half);
INSTANTIATE_KTH_ALL_TPTG_1024(bfloat);
INSTANTIATE_KTH_ALL_TPTG_1024(int);
INSTANTIATE_KTH_ALL_TPTG(long);
INSTANTIATE_KTH_ALL_TPTG_1024(short);
INSTANTIATE_KTH_ALL_TPTG_1024(char);
INSTANTIATE_KTH_ALL_TPTG_1024(uchar);
INSTANTIATE_KTH_ALL_TPTG_1024(bool);
INSTANTIATE_KTH_ALL_TPTG_1024(ushort);
INSTANTIATE_KTH_ALL_TPTG_1024(uint);
INSTANTIATE_KTH_ALL_TPTG(ulong);

#define _SCATTER_KTH_SIG(T, InIdxT)                                                            \
  const device T*, const device InIdxT*, device T*, device long*, const device uint*,          \
      constant int3&, constant bool2&, constant int&, uint3, uint3

#define INSTANTIATE_KTH_RADIX_SCATTER(T, InIdxT, RTPTG, EPT, RBITS, NAME)                                       \
  template [[host_name("radix_scatter_final_kth_" NAME)]]                                                       \
  kernel void radix_scatter_kth<T, InIdxT, RTPTG, EPT, RBITS, false>(_SCATTER_KTH_SIG(T, InIdxT));              \
  template [[host_name("radix_scatter_fused_final_kth_" NAME)]]                                                 \
  kernel void radix_scatter_kth<T, InIdxT, RTPTG, EPT, RBITS, true>(_SCATTER_KTH_SIG(T, InIdxT));

#define INSTANTIATE_KTH_RADIX(T, RTPTG, EPT, RBITS)                              \
  INSTANTIATE_KTH_RADIX_SCATTER(T, uint, RTPTG, EPT, RBITS, #T "_" #RBITS "bit") \
  INSTANTIATE_KTH_RADIX_SCATTER(T, ushort, RTPTG, EPT, RBITS, #T "_" #RBITS "bit_u16")

INSTANTIATE_KTH_RADIX(char, 512, 8, 4);
INSTANTIATE_KTH_RADIX(uchar, 512, 8, 4);
INSTANTIATE_KTH_RADIX(bool, 512, 8, 4);
INSTANTIATE_KTH_RADIX(half, 1024, 4, 8);
INSTANTIATE_KTH_RADIX(bfloat, 1024, 4, 8);
INSTANTIATE_KTH_RADIX(short, 1024, 4, 8);
INSTANTIATE_KTH_RADIX(ushort, 1024, 4, 8);
INSTANTIATE_KTH_RADIX(float, 512, 4, 8);
INSTANTIATE_KTH_RADIX(int, 512, 4, 8);
INSTANTIATE_KTH_RADIX(uint, 512, 4, 8);

#define INSTANTIATE_KTH_RADIX_TPTG512(T)                                \
  INSTANTIATE_KTH_RADIX_SCATTER(T, uint, 512, 4, 8, #T "_8bit_tptg512") \
  INSTANTIATE_KTH_RADIX_SCATTER(T, ushort, 512, 4, 8, #T "_8bit_tptg512_u16")

INSTANTIATE_KTH_RADIX_TPTG512(half);
INSTANTIATE_KTH_RADIX_TPTG512(bfloat);
INSTANTIATE_KTH_RADIX_TPTG512(short);
INSTANTIATE_KTH_RADIX_TPTG512(ushort);
