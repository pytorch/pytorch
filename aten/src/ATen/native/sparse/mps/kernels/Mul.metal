#include <c10/metal/indexing.h>
using namespace metal;

inline uint lower_bound_i64(device const long* arr, uint lo, uint hi, long key) {
  uint l = lo, r = hi;
  while (l < r) {
    uint m = (l + r) >> 1;
    long v = arr[m];
    if (v < key) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

inline uint upper_bound_i64(device const long* arr, uint lo, uint hi, long key) {
  uint l = lo, r = hi;
  while (l < r) {
    uint m = (l + r) >> 1;
    long v = arr[m];
    if (v <= key) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

kernel void build_row_ptr_from_sorted_rows_by_batch(
    device const long* rows        [[buffer(0)]],
    device const long* batch_ptr   [[buffer(1)]],
    device long*       row_ptr     [[buffer(2)]],
    constant uint2&    dims        [[buffer(3)]],
    uint3              tid         [[thread_position_in_grid]])
{
  const uint I = dims.x;
  const uint B = dims.y;

  const uint i = tid.x;
  const uint b = tid.y;

  if (b >= B || i > I) return;

  const uint base = (uint)batch_ptr[b];
  const uint lim  = (uint)batch_ptr[b + 1];

  const ulong out_base = (ulong)b * (ulong)(I + 1);

  if (i == I) {
    row_ptr[out_base + (ulong)I] = (long)lim;
  } else {
    const long key = (long)i;
    const uint pos = lower_bound_i64(rows, base, lim, key);
    row_ptr[out_base + (ulong)i] = (long)pos;
  }
}

kernel void spmm_bmm_coo_rows_grouped(
    device const long*   rows      [[buffer(0)]],
    device const long*   cols      [[buffer(1)]],
    device const float*  vals      [[buffer(2)]],
    device const float*  dense     [[buffer(3)]],
    device float*        out       [[buffer(4)]],
    device const long*   row_ptr   [[buffer(5)]],
    constant uint4&      dims      [[buffer(6)]],
    uint3                tid       [[thread_position_in_grid]],
    uint3                ltid      [[thread_position_in_threadgroup]],
    uint3                tptg      [[threads_per_threadgroup]])
{
  const uint B = dims.x;
  const uint I = dims.y;
  const uint J = dims.z;
  const uint K = dims.w;

  const uint b = tid.z;
  const uint i = tid.y;
  const uint lane = ltid.x;
  const uint tgW  = tptg.x;

  const ulong rp_base = (ulong)b * (ulong)(I + 1);
  const uint start = (uint)row_ptr[rp_base + (ulong)i];
  const uint end   = (uint)row_ptr[rp_base + (ulong)i + 1];

  for (uint k = lane; k < K; k += tgW) {
    float acc = 0.0f;
    for (uint p = start; p < end; ++p) {
      const uint c = (uint)cols[p];
      const float v = vals[p];
      const uint d_off = ((b * J) + c) * K + k;
      acc = fma(v, dense[d_off], acc);
    }
    const uint y_off = ((b * I) + i) * K + k;
    out[y_off] = acc;
  }
}

template <typename T>
kernel void dense_sparse_mul_kernel(
    device const T* dense         [[buffer(0)]],
    device const T* values        [[buffer(1)]],
    device T* out_values          [[buffer(2)]],
    device const long* indices    [[buffer(3)]],
    device const long* sizes      [[buffer(4)]],
    constant uint3& sparse_params [[buffer(5)]],
    uint3 gid                     [[thread_position_in_grid]])
{
  uint col = gid.x;
  uint i = gid.z;
  uint nnz = sparse_params.x;
  uint ndim_i = sparse_params.y;
  uint view_cols = sparse_params.z;

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
    constant uint2&    dims_input    [[buffer(7)]],
    constant uint2&    dims_output   [[buffer(8)]],
    uint3              gid           [[thread_position_in_grid]])
{
  const uint col = gid.x;
  const uint k = gid.z;
  const uint n_dim_i = dims_input.x;
  const uint L = dims_input.y;
  const uint M = dims_output.x;
  const uint view_cols = dims_output.y;

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
    for (uint d = 0; d < n_dim_i; ++d) {
      const long v = lhs_indices[(ulong)d * uL + src_col];
      out_indices[(ulong)d * uM + (ulong)k] = v;
    }
  }
}


kernel void build_batch_ptr_from_sorted_batches(
    device const long* batches       [[buffer(0)]],
    device long*       batch_ptr     [[buffer(1)]],
    constant uint&     nnz           [[buffer(2)]],
    constant uint&     B             [[buffer(3)]],
    uint3              tid           [[thread_position_in_grid]])
{
  uint b = tid.x;

  if (b == B) {
    batch_ptr[b] = (long)nnz;
    return;
  }

  uint lo = 0;
  uint hi = nnz;
  long key = (long)b;
  while (lo < hi) {
    uint mid = (lo + hi) >> 1;
    long v = batches[mid];
    if (v < key) lo = mid + 1;
    else         hi = mid;
  }
  batch_ptr[b] = (long)lo;
}

kernel void spmm_bmm_coo(
    device const long*   rows         [[buffer(0)]],
    device const long*   cols         [[buffer(1)]],
    device const float*  vals         [[buffer(2)]],
    device const float*  dense        [[buffer(3)]],
    device float*        out          [[buffer(4)]],
    device const long*   batch_ptr    [[buffer(5)]],
    constant uint4&      dims         [[buffer(6)]],
    constant float&      alpha        [[buffer(7)]],
    constant float&      beta         [[buffer(8)]],
    uint3                tid          [[thread_position_in_grid]])
{
  const uint K = dims.w;
  const uint J = dims.z;
  const uint I = dims.y;
  const uint B = dims.x;

  const uint k = tid.x;
  const uint b = tid.y;
  const uint i = tid.z;

  const uint base = (uint)batch_ptr[b];
  const uint lim = (uint)batch_ptr[b + 1];

  // find row segment [start, end) for row i within this batch
  const uint start = lower_bound_i64(rows, base, lim, (long)i);
  const uint end = upper_bound_i64(rows, base, lim, (long)i);

  float acc = 0.0f;
  for (uint p = start; p < end; ++p) {
    const uint c = (uint)cols[p];
    const float v = vals[p];
    const uint dense_off = ((b * J) + c) * K + k;
    acc += v * dense[dense_off];
  }

  const uint out_off = ((b * I) + i) * K + k;
  float y = 0.0f;
  if (beta != 0.0f) {
    y = out[out_off] * beta;
  }
  out[out_off] = y + alpha * acc;
}



kernel void spmm_bmm_coo_grouped(
    device const long*   rows         [[buffer(0)]],
    device const long*   cols         [[buffer(1)]],
    device const float*  vals         [[buffer(2)]],
    device const float*  dense        [[buffer(3)]],
    device float*        out          [[buffer(4)]],
    device const long*   batch_ptr    [[buffer(5)]],
    constant uint4&      dims         [[buffer(6)]],
    uint3                tid          [[thread_position_in_grid]],
    uint3                ltid         [[thread_position_in_threadgroup]],
    uint3                tptg         [[threads_per_threadgroup]]
)
{
  const uint K = dims.w;
  const uint J = dims.z;
  const uint I = dims.y;
  const uint B = dims.x;

  const uint lane = ltid.x;
  const uint tgW = tptg.x;

  // we launch exactly one threadgroup per batch along Y, with 1 row-group on Z,
  // because threadsPerThreadgroup.y == 1, tid.y directly encodes the batch
  const uint b = tid.y;

  const uint base = (uint)batch_ptr[b];
  const uint lim  = (uint)batch_ptr[b + 1];
  if (base >= lim) {
    return; // no nnz in this batch
  }

  uint p = base;
  while (p < lim) {
    const long rowL = rows[p];

    // find end of this row segment
    uint q = p + 1;
    while (q < lim && rows[q] == rowL) {
      ++q;
    }

    // Compute y[b, rowL, 0..K) in parallel across the threadgroup lanes
    for (uint k = lane; k < K; k += tgW) {
      float acc = 0.0f;

      for (uint t = p; t < q; ++t) {
        const uint c = (uint)cols[t];
        const uint dense_off = ((b * J) + c) * K + k;
        acc = fma(vals[t], dense[dense_off], acc);
      }

      const uint out_off = ((b * I) + (uint)rowL) * K + k;
      out[out_off] = acc;
    }

    p = q; // next row segment
  }
}

// have to do this to support both float and float2
template <typename T>
inline float to_compute_dtype(T x) { return static_cast<float>(x); }
inline float2 to_compute_dtype(float2 x){ return x; }

inline float cmadd(float acc, float a, float b) {
  return fma(a, b, acc);
}
inline float2 cmadd(float2 acc, float2 a, float2 b) {
  float real = a.x * b.x - a.y * b.y;
  float imag = a.x * b.y + a.y * b.x;
  return acc + float2(real, imag);
}


template <typename T>
kernel void spmm_addmm_coo(
    device const long*   indices2d   [[buffer(0)]],
    device const T*      vals        [[buffer(1)]],
    device const T*      dense       [[buffer(2)]],
    device const T*      t_in        [[buffer(3)]],
    device T*            out         [[buffer(4)]],
    constant uint3&      dims        [[buffer(5)]],
    constant float2&     alpha_beta  [[buffer(6)]],
    constant uint&       nnz         [[buffer(7)]],
    uint3                tid         [[thread_position_in_grid]])
{
  const uint K = dims.z;
  const uint k = tid.x;
  const uint i = tid.z;
  const float alpha = alpha_beta.x;
  const float beta = alpha_beta.y;

  device const long* rows = indices2d;
  device const long* cols = indices2d + nnz;

  const uint start = lower_bound_i64(rows, 0u, nnz, (long)i);
  const uint end = upper_bound_i64(rows, 0u, nnz, (long)i);

  // accumulator is float for scalar/half/bfloat and float2 for float2
  auto acc = to_compute_dtype(T(0));

  for (uint p = start; p < end; ++p) {
    const uint c = (uint)cols[p];
    const auto v = to_compute_dtype(vals[p]);
    const uint dense_off = c * K + k;
    const auto d = to_compute_dtype(dense[dense_off]);
    acc = cmadd(acc, v, d);
  }

  const uint off = i * K + k;
  const auto base = (beta != 0.0f) ? (to_compute_dtype(t_in[off]) * beta) : to_compute_dtype(T(0));
  const auto y = base + alpha * acc;
  out[off] = static_cast<T>(y);
}


#define INSTANTIATE_DENSE_SPARSE_MUL(DTYPE)                                 \
  template [[host_name("dense_sparse_mul_kernel_" #DTYPE)]] kernel void     \
  dense_sparse_mul_kernel<DTYPE>(                                           \
      device const DTYPE* dense         [[buffer(0)]],                      \
      device const DTYPE* values        [[buffer(1)]],                      \
      device DTYPE* out_values          [[buffer(2)]],                      \
      device const long* indices        [[buffer(3)]],                      \
      device const long* sizes          [[buffer(4)]],                      \
      constant uint3& sparse_params     [[buffer(5)]],                      \
      uint3 gid                         [[thread_position_in_grid]]);

INSTANTIATE_DENSE_SPARSE_MUL(float);
INSTANTIATE_DENSE_SPARSE_MUL(half);
INSTANTIATE_DENSE_SPARSE_MUL(bfloat);

#define INSTANTIATE_FUSED_GATHER_MUL(DTYPE)                                  \
  template [[host_name("fused_gather_mul_kernel_" #DTYPE)]] kernel void      \
  fused_gather_mul_kernel<DTYPE>(                                            \
      device const DTYPE* lhs_vals      [[buffer(0)]],                       \
      device const DTYPE* rhs_vals      [[buffer(1)]],                       \
      device const long*  lhs_sel       [[buffer(2)]],                       \
      device const long*  rhs_sel       [[buffer(3)]],                       \
      device const long*  lhs_indices   [[buffer(4)]],                       \
      device long*        out_indices   [[buffer(5)]],                       \
      device DTYPE*       out_vals      [[buffer(6)]],                       \
      constant uint2&     dims_input    [[buffer(7)]],                       \
      constant uint2&     dims_output   [[buffer(8)]],                       \
      uint3               gid           [[thread_position_in_grid]]);

INSTANTIATE_FUSED_GATHER_MUL(float);
INSTANTIATE_FUSED_GATHER_MUL(half);
INSTANTIATE_FUSED_GATHER_MUL(bfloat);


#define INSTANTIATE_SPMM_ADDMM_COO(DTYPE) \
  template [[host_name("spmm_addmm_coo_" #DTYPE)]] kernel void  \
  spmm_addmm_coo<DTYPE>(                                        \
    device const long*   indices2d   [[buffer(0)]],             \
    device const DTYPE*  vals        [[buffer(1)]],             \
    device const DTYPE*  dense       [[buffer(2)]],             \
    device const DTYPE*  t_in        [[buffer(3)]],             \
    device DTYPE*        out         [[buffer(4)]],             \
    constant uint3&      dims        [[buffer(5)]],             \
    constant float2&     alpha_beta  [[buffer(6)]],             \
    constant uint&       nnz         [[buffer(7)]],             \
    uint3                tid         [[thread_position_in_grid]]);

INSTANTIATE_SPMM_ADDMM_COO(float);
INSTANTIATE_SPMM_ADDMM_COO(half);
INSTANTIATE_SPMM_ADDMM_COO(bfloat);
INSTANTIATE_SPMM_ADDMM_COO(float2);
