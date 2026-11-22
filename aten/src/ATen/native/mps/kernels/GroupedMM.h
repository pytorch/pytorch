#pragma once

#include <ATen/native/mps/kernels/Matmul.h>
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t kGroupedMMMediumTileCols = 128;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsMax = 32;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsSimdgroups = 16;
C10_METAL_CONSTEXPR uint32_t kGroupedMMFewRowsLaneChunk = 16;

template <typename idx_t>
struct GroupedMMParams {
  MatmulParams<idx_t> mm;
  uint32_t groups;
  idx_t batch_stride;
};

#ifdef __METAL__
inline uint32_t grouped_mm_offset(int32_t offset, uint32_t limit) {
  if (offset <= 0) {
    return 0;
  }
  return min(static_cast<uint32_t>(offset), limit);
}

// Index of the group whose jagged extent holds idx, or groups if none does
inline uint32_t grouped_mm_find_group(
    device const int32_t* offsets,
    uint32_t groups,
    uint32_t idx) {
  uint32_t group = 0;
  while (group < groups) {
    const uint32_t mid = group + (groups - group) / 2;
    if (offsets[mid] <= static_cast<int32_t>(idx)) {
      group = mid + 1;
    } else {
      groups = mid;
    }
  }
  return group;
}

// Maps a flat tile index along the jagged dimension (rows or columns, capped
// at limit) to its (group, first index, valid count) triple by walking the
// jagged group extents; padding tiles leave the zero-initialized outputs
// unchanged.
template <int TILE>
inline void grouped_mm_group_tile(
    device const int32_t* offsets,
    uint32_t groups,
    uint32_t limit,
    uint32_t tile,
    thread uint32_t& group,
    thread uint32_t& tile_first,
    thread uint32_t& tile_count) {
  uint32_t start = 0;
  uint32_t tile_start = 0;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint32_t end = grouped_mm_offset(offsets[g], limit);
    const uint32_t count = end > start ? end - start : 0;
    const uint32_t tiles =
        c10::metal::ceil_div(count, static_cast<uint32_t>(TILE));
    if (tile < tile_start + tiles) {
      group = g;
      tile_first = start + (tile - tile_start) * TILE;
      tile_count = min(static_cast<uint32_t>(TILE), end - tile_first);
      return;
    }
    tile_start += tiles;
    start = max(start, end);
  }
}

// One thread searches the offsets, then the tile is broadcast through
// threadgroup memory to the whole threadgroup.
template <int TILE>
inline bool grouped_mm_group_tile_broadcast(
    device const int32_t* offsets,
    uint32_t groups,
    uint32_t limit,
    uint32_t tile,
    uint32_t tid,
    threadgroup uint32_t* tile_info,
    thread uint32_t& group,
    thread uint32_t& tile_first,
    thread uint32_t& tile_count) {
  if (tid == 0) {
    uint32_t g = 0;
    uint32_t first = 0;
    uint32_t count = 0;
    grouped_mm_group_tile<TILE>(offsets, groups, limit, tile, g, first, count);
    tile_info[0] = g;
    tile_info[1] = first;
    tile_info[2] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  group = tile_info[0];
  tile_first = tile_info[1];
  tile_count = tile_info[2];
  return tile_count != 0;
}
#endif
