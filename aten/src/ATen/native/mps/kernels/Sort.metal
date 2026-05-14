// Sort kernels for MPS. The host (Sort.mm) picks one of three paths based on
// the segment size and type:
//   Path 1 - single-block: one threadgroup per row, used when the segment
//            fits in threadgroup memory.
//   Path 2 - multi-block:  segment split into ELEMS_PER_TG-sized blocks,
//            each block sorted independently, then log2(n_blocks) passes
//            of pairwise merges. Used when sort_size exceeds what one
//            threadgroup can hold, or when single-block would dispatch
//            too few TGs to keep the GPU busy.
//   Path 3 - radix sort:   classic LSD radix over RBITS-bit digits.
//            radix_count -> radix_scan -> radix_scatter, repeated per
//            digit (with optional fused count+scan or fused scan+scatter
//            when n_blocks is small). Selected for radix-friendly types
//            (elem_size <= 4) when the dispatch count beats merge.
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

#include <ATen/native/mps/kernels/SortMerge.h>
#include <ATen/native/mps/kernels/SortRadix.h>
