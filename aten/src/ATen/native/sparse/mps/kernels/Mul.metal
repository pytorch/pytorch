#include <metal_stdlib>
#include <c10/metal/indexing.h>
using namespace metal;


template <typename T>
kernel void dense_sparse_mul_kernel(
    device const T* dense         [[buffer(0)]],
    device const T* values        [[buffer(1)]],
    device T* out_values          [[buffer(2)]],
    device const long* indices    [[buffer(3)]],
    device const long* sizes      [[buffer(4)]],
    constant uint& nnz            [[buffer(5)]],
    constant uint& ndim_i          [[buffer(6)]],
    constant uint& view_cols      [[buffer(7)]],
    uint3 gid                     [[thread_position_in_grid]])
{
  uint col = gid.x;
  uint i = gid.z;

  long key = 0;
  for (uint d = 0; d < ndim_i; ++d) {
    long idx_d = indices[(ulong)d * (ulong)nnz + (ulong)i];
    const auto sz_d  = sizes[d];
    key = key * sz_d + idx_d;
  }

  ulong dense_idx = (ulong)key * (ulong)view_cols + (ulong)col;
  ulong val_idx = (ulong)i * (ulong)view_cols + (ulong)col;

  const auto a = static_cast<float>(values[val_idx]);
  const auto b = static_cast<float>(dense[dense_idx]);
  out_values[val_idx] = static_cast<T>(a * b);
}

kernel void intersect_binary_search(
    device const long*  keysA        [[buffer(0)]],
    device const long*  keysB        [[buffer(1)]],
    device long*        outA_idx     [[buffer(2)]],
    device long*        outB_idx     [[buffer(3)]],
    device atomic_uint* counter      [[buffer(4)]],
    constant uint&      lenB         [[buffer(5)]],
    constant bool&      A_is_lhs     [[buffer(6)]],
    uint3               tid_in_grid  [[thread_position_in_grid]])
{
  uint gid = tid_in_grid.x;

  long key = keysA[gid];

  // lower_bound in B
  uint lo = 0;
  uint hi = lenB;
  while (lo < hi) {
    uint mid = (lo + hi) >> 1;
    long v = keysB[mid];
    if (v < key) lo = mid + 1;
    else         hi = mid;
  }

  if (lo < lenB && keysB[lo] == key) {
    uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    if (A_is_lhs) {
      outA_idx[pos] = (long)gid;
      outB_idx[pos] = (long)lo;
    } else {
      outA_idx[pos] = (long)lo;
      outB_idx[pos] = (long)gid;
    }
  }
}


template <typename T>
kernel void fused_gather_mul_kernel(
    device const T*    lhs_vals      [[buffer(0)]],
    device const T*    rhs_vals      [[buffer(1)]],
    device const long* lhs_sel       [[buffer(2)]],
    device const long* rhs_sel       [[buffer(3)]],
    device const long* lhs_indices   [[buffer(4)]],
    device long*       out_indices   [[buffer(5)]],
    device T*          out_vals      [[buffer(6)]],
    constant uint&     nDimI         [[buffer(7)]],
    constant uint&     L             [[buffer(8)]],
    constant uint&     M             [[buffer(9)]],
    constant uint&     view_cols     [[buffer(10)]],
    uint3              gid           [[thread_position_in_grid]])
{
  const uint col = gid.x;
  const uint k = gid.z;

  const long iL = lhs_sel[k];
  const long iR = rhs_sel[k];

  if (col < view_cols) {
    const ulong offL = (ulong)iL * (ulong)view_cols + (ulong)col;
    const ulong offR = (ulong)iR * (ulong)view_cols + (ulong)col;
    const ulong offO = (ulong)k  * (ulong)view_cols + (ulong)col;

    const float a = (float)lhs_vals[offL];
    const float b = (float)rhs_vals[offR];
    out_vals[offO] = (T)(a * b);
  }

  // One thread per match copies the indices column
  if (col == 0) {
    const ulong uL = (ulong)L;
    const ulong uM = (ulong)M;
    const ulong src_col = (ulong)iL; // gather from lhs
    for (uint d = 0; d < nDimI; ++d) {
      const long v = lhs_indices[(ulong)d * uL + src_col];
      out_indices[(ulong)d * uM + (ulong)k] = v;
    }
  }
}

kernel void gather_int64_columns(
    device const long* in         [[buffer(0)]],
    device const long* sel        [[buffer(1)]],
    device long*       out        [[buffer(2)]],
    constant uint&     nDimI      [[buffer(3)]],
    constant uint&     L          [[buffer(4)]],
    constant uint&     M          [[buffer(5)]],
    uint               g          [[thread_position_in_grid]])
{
  uint d32 = g / M;
  uint k32 = g - d32 * M;

  ulong d = (ulong)d32;
  ulong k = (ulong)k32;

  long src_col = sel[k];

  long v = in[d * (ulong)L + (ulong)src_col];
  out[d * (ulong)M + k] = v;
}

template <typename T>
kernel void pairwise_gather_mul_kernel(
    device const T* lhs_vals      [[buffer(0)]],
    device const T* rhs_vals      [[buffer(1)]],
    device const long* lhs_sel    [[buffer(2)]],
    device const long* rhs_sel    [[buffer(3)]],
    device T* out_vals            [[buffer(4)]],
    constant uint& M              [[buffer(5)]],
    constant uint& view_cols      [[buffer(6)]],
    uint3 gid                     [[thread_position_in_grid]])
{
  uint col = gid.x;
  uint k = gid.z;

  long iL = lhs_sel[k];
  long iR = rhs_sel[k];

  ulong offL = (ulong)iL * (ulong)view_cols + (ulong)col;
  ulong offR = (ulong)iR * (ulong)view_cols + (ulong)col;
  ulong offO = (ulong)k * (ulong)view_cols + (ulong)col;

  float a = (float)lhs_vals[offL];
  float b = (float)rhs_vals[offR];
  out_vals[offO] = (T)(a * b);
}

#define INSTANTIATE_DENSE_SPARSE_MUL(DTYPE)                          \
  template [[host_name("dense_sparse_mul_kernel_" #DTYPE)]] kernel void \
  dense_sparse_mul_kernel<DTYPE>(                                    \
      device const DTYPE* dense         [[buffer(0)]],               \
      device const DTYPE* values        [[buffer(1)]],               \
      device DTYPE* out_values          [[buffer(2)]],               \
      device const long* indices        [[buffer(3)]],               \
      device const long* sizes          [[buffer(4)]],               \
      constant uint& nnz                [[buffer(5)]],               \
      constant uint& nDimI              [[buffer(6)]],               \
      constant uint& view_cols          [[buffer(7)]],               \
      uint3 gid                         [[thread_position_in_grid]]);

INSTANTIATE_DENSE_SPARSE_MUL(float);
INSTANTIATE_DENSE_SPARSE_MUL(half);
INSTANTIATE_DENSE_SPARSE_MUL(bfloat);

#define INSTANTIATE_FUSED_GATHER_MUL(DTYPE)                                   \
  template [[host_name("fused_gather_mul_kernel_" #DTYPE)]] kernel void       \
  fused_gather_mul_kernel<DTYPE>(                                             \
      device const DTYPE* lhs_vals      [[buffer(0)]],                        \
      device const DTYPE* rhs_vals      [[buffer(1)]],                        \
      device const long*  lhs_sel       [[buffer(2)]],                        \
      device const long*  rhs_sel       [[buffer(3)]],                        \
      device const long*  lhs_indices   [[buffer(4)]],                        \
      device long*        out_indices   [[buffer(5)]],                        \
      device DTYPE*       out_vals      [[buffer(6)]],                        \
      constant uint&      nDimI         [[buffer(7)]],                        \
      constant uint&      L             [[buffer(8)]],                        \
      constant uint&      M             [[buffer(9)]],                        \
      constant uint&      view_cols     [[buffer(10)]],                       \
      uint3               gid           [[thread_position_in_grid]]);

INSTANTIATE_FUSED_GATHER_MUL(float);
INSTANTIATE_FUSED_GATHER_MUL(half);
INSTANTIATE_FUSED_GATHER_MUL(bfloat);