#include <metal_stdlib>
using namespace metal;


kernel void flatten_indices_kernel(
    device const long* indices        [[ buffer(0) ]],
    device const long* row_muls       [[ buffer(1) ]],
    device long*       flat_indices   [[ buffer(2) ]],
    constant uint&     sparse_dim     [[ buffer(3) ]],
    constant long2&    idx_strides    [[ buffer(4) ]],
    uint               gid            [[ thread_position_in_grid ]]) {
  long flat = 0;
  for (uint d = 0; d < sparse_dim; ++d) {
    long off = (long)d * idx_strides.x + (long)gid * idx_strides.y;
    long v = indices[off];
    flat += v * row_muls[d];
  }
  flat_indices[gid] = flat;
}