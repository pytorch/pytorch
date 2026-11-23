#pragma once
#include <c10/metal/common.h>

// Byte geometry for one dispatch of the inner_contiguous_scatter kernel: a
// CONTIGUOUS input copied into regularly-strided blocks of the output -- each
// slice_bytes run is placed at off_bytes + k * out_stride_bytes. Backs the
// contiguous cat fast path and (in future) same-dtype scatter into strided
// views. idx_type_t is the byte index width: uint32_t unless an addressed
// extent can exceed 4GB.
template <typename idx_type_t = uint32_t>
struct StridedBlockParams {
  idx_type_t chunk_base;
  idx_type_t slice_bytes;
  idx_type_t out_stride_bytes;
  idx_type_t off_bytes;
  uint32_t nbytes;
};
