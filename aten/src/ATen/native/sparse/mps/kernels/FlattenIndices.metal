#include <metal_stdlib>
using namespace metal;


kernel void flatten_indices_kernel(
    device const long* indices        [[ buffer(0) ]],
    device const long* row_muls       [[ buffer(1) ]],
    device long*       flat_indices   [[ buffer(2) ]],
    constant uint&     sparse_dim     [[ buffer(3) ]],
    constant uint&     nnz            [[ buffer(4) ]],
    constant long&     idx_stride0    [[ buffer(5) ]],
    constant long&     idx_stride1    [[ buffer(6) ]],
    uint               gid            [[ thread_position_in_grid ]]) {

  if (gid >= nnz) return;

  long flat = 0;
  for (uint d = 0; d < sparse_dim; ++d) {
    long off = (long)d * idx_stride0 + (long)gid * idx_stride1;
    long v = indices[off];
    flat += v * row_muls[d];
  }
  flat_indices[gid] = flat;
}