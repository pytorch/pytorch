#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
using namespace c10::metal;
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

template <typename T>
kernel void spmm_bmm_coo_rows_grouped(
    device const long*   cols      [[buffer(1)]],
    device const T*      vals      [[buffer(2)]],
    device const T*      dense     [[buffer(3)]],
    device T*            out       [[buffer(4)]],
    device const long*   row_ptr   [[buffer(5)]],
    constant uint4&      dims      [[buffer(6)]],
    uint3                tid       [[thread_position_in_grid]],
    uint3                ltid      [[thread_position_in_threadgroup]],
    uint3                tptg      [[threads_per_threadgroup]])
{
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
    auto acc = static_cast<accum_t<T>>(T(0));
    for (uint p = start; p < end; ++p) {
      const uint c = (uint)cols[p];
      const auto v = static_cast<accum_t<T>>(vals[p]);
      const uint d_off = ((b * J) + c) * K + k;
      const auto d = static_cast<accum_t<T>>(dense[d_off]);
      acc += mul(v, d);
    }
    const uint y_off = ((b * I) + i) * K + k;
    out[y_off] = static_cast<T>(acc);
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

  const auto a = static_cast<accum_t<T>>(values[val_idx]);
  const auto b = static_cast<accum_t<T>>(dense[dense_idx]);
  out_values[val_idx] = static_cast<T>(mul(a, b));
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

    const auto a = static_cast<accum_t<T>>(lhs_vals[offL]);
    const auto b = static_cast<accum_t<T>>(rhs_vals[offR]);
    out_vals[offO] = static_cast<T>(mul(a, b));
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
    constant uint2&    nnz_B         [[buffer(2)]],
    uint3              tid           [[thread_position_in_grid]])
{
  uint b = tid.x;
  uint nnz = nnz_B.x;
  uint batch = nnz_B.y;

  if (b == batch) {
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
  auto acc = static_cast<accum_t<T>>(T(0));

  for (uint p = start; p < end; ++p) {
    const uint c = (uint)cols[p];
    const auto v = static_cast<accum_t<T>>(vals[p]);
    const uint dense_off = c * K + k;
    const auto d = static_cast<accum_t<T>>(dense[dense_off]);
    acc += mul(v, d);
  }

  const uint off = i * K + k;
  const auto base = (beta != 0.0f) ? (static_cast<accum_t<T>>(t_in[off]) * beta) : static_cast<accum_t<T>>(T(0));
  const auto y = base + alpha * acc;
  out[off] = static_cast<T>(y);
}


kernel void mark_segments(
    device const int64_t* indices [[buffer(0)]],
    device int*           mask    [[buffer(1)]],
    uint                  tid     [[thread_position_in_grid]])
{
    mask[tid] = (tid == 0 || indices[tid] != indices[tid - 1]) ? 1 : 0;
}

kernel void compute_offsets_and_counts(
    device const int* scan           [[buffer(0)]],
    device int*       offsets        [[buffer(1)]],
    device int*       counts         [[buffer(2)]],
    constant uint&    total_elements [[buffer(3)]],
    uint              tid            [[thread_position_in_grid]])
{
    int num_pools = scan[total_elements - 1];

    int target = int(tid) + 1;
    int lo = 0;
    int hi = int(total_elements);
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (scan[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int start = lo;
    offsets[tid] = start;

    int end;
    if (int(tid) == num_pools - 1) {
        end = int(total_elements);
    } else {
        target = int(tid) + 2;
        lo = start + 1;
        hi = int(total_elements);
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (scan[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        end = lo;
    }

    counts[tid] = end - start;
}

template <typename T>
kernel void softmax_sparse_forward(
    device const T*    values       [[buffer(0)]],
    device T*          output       [[buffer(1)]],
    device const int*  pool_offsets [[buffer(2)]],
    device const int*  pool_sizes   [[buffer(3)]],
    device const int*  scan         [[buffer(4)]],
    constant uint2&    nnz_nvalues  [[buffer(5)]],
    constant bool&     is_log       [[buffer(6)]],
    uint               tid          [[thread_position_in_grid]])
{
    uint nnz = nnz_nvalues.x;
    uint nvalues = nnz_nvalues.y;
    int num_pools = scan[nnz - 1];
    if (tid >= uint(num_pools)) return;

    int start = pool_offsets[tid];
    int count = pool_sizes[tid];

    for (uint j = 0; j < nvalues; ++j) {
        float max_val = -INFINITY;
        for (int i = 0; i < count; ++i) {
            float val = static_cast<float>(values[(start + i) * nvalues + j]);
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < count; ++i) {
            float val = static_cast<float>(values[(start + i) * nvalues + j]);
            sum_exp += exp(val - max_val);
        }

        float log_sum = is_log ? log(sum_exp) : 0.0f;
        float inv_sum = is_log ? 0.0f : (1.0f / sum_exp);

        for (int i = 0; i < count; ++i) {
            uint idx = (start + i) * nvalues + j;
            float val = static_cast<float>(values[idx]);

            if (is_log) {
                output[idx] = static_cast<T>(val - max_val - log_sum);
            } else {
                output[idx] = static_cast<T>(exp(val - max_val) * inv_sum);
            }
        }
    }
}

template <typename T>
kernel void softmax_sparse_backward(
    device const T*    grad_output  [[buffer(0)]],
    device const T*    output       [[buffer(1)]],
    device T*          grad_input   [[buffer(2)]],
    device const int*  offsets      [[buffer(3)]],
    device const int*  counts       [[buffer(4)]],
    device const int*  scan         [[buffer(5)]],
    constant uint2&    nnz_nvalues  [[buffer(6)]],
    constant bool&     is_log       [[buffer(7)]],
    uint               tid          [[thread_position_in_grid]])
{
    uint nnz = nnz_nvalues.x;
    uint nvalues = nnz_nvalues.y;
    int num_pools = scan[nnz - 1];
    if (tid >= uint(num_pools)) return;

    int start = offsets[tid];
    int count = counts[tid];

    for (uint j = 0; j < nvalues; ++j) {
        float sum_val = 0.0f;

        for (int i = 0; i < count; ++i) {
            uint idx = (start + i) * nvalues + j;
            float g = static_cast<float>(grad_output[idx]);
            if (is_log) {
                sum_val += g;
            } else {
                float y = static_cast<float>(output[idx]);
                sum_val += g * y;
            }
        }

        for (int i = 0; i < count; ++i) {
            uint idx = (start + i) * nvalues + j;
            float g = static_cast<float>(grad_output[idx]);
            float y = static_cast<float>(output[idx]);
            float res;

            if (is_log) {
                res = g - exp(y) * sum_val;
            } else {
                res = y * (g - sum_val);
            }
            grad_input[idx] = static_cast<T>(res);
        }
    }
}

#define INSTANTIATE_SOFTMAX_SPARSE_FORWARD(DTYPE)                           \
  template [[host_name("softmax_sparse_forward_" #DTYPE)]] kernel void      \
  softmax_sparse_forward<DTYPE>(                                            \
      device const DTYPE* values       [[buffer(0)]],                       \
      device DTYPE*       output       [[buffer(1)]],                       \
      device const int*   pool_offsets [[buffer(2)]],                       \
      device const int*   pool_sizes   [[buffer(3)]],                       \
      device const int*   scan         [[buffer(4)]],                       \
      constant uint2&     nnz_nvalues  [[buffer(5)]],                       \
      constant bool&      is_log       [[buffer(6)]],                       \
      uint                tid          [[thread_position_in_grid]]);

#define INSTANTIATE_SOFTMAX_SPARSE_BACKWARD(DTYPE)                          \
  template [[host_name("softmax_sparse_backward_" #DTYPE)]] kernel void     \
  softmax_sparse_backward<DTYPE>(                                           \
      device const DTYPE* grad         [[buffer(0)]],                       \
      device const DTYPE* output       [[buffer(1)]],                       \
      device DTYPE*       grad_input   [[buffer(2)]],                       \
      device const int*   pool_offsets [[buffer(3)]],                       \
      device const int*   pool_sizes   [[buffer(4)]],                       \
      device const int*   scan         [[buffer(5)]],                       \
      constant uint2&     nnz_nvalues  [[buffer(6)]],                       \
      constant bool&      is_log       [[buffer(7)]],                       \
      uint                tid          [[thread_position_in_grid]]);


INSTANTIATE_FOR_FLOAT_TYPES(INSTANTIATE_SOFTMAX_SPARSE_FORWARD);
INSTANTIATE_FOR_FLOAT_TYPES(INSTANTIATE_SOFTMAX_SPARSE_BACKWARD);


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
INSTANTIATE_DENSE_SPARSE_MUL(long);
INSTANTIATE_DENSE_SPARSE_MUL(float2);

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

INSTANTIATE_FOR_ALL_TYPES(INSTANTIATE_FUSED_GATHER_MUL);


#define INSTANTIATE_SPMM_BMM_COO_ROWS_GROUPED(DTYPE)                         \
  template [[host_name("spmm_bmm_coo_rows_grouped_" #DTYPE)]] kernel void    \
  spmm_bmm_coo_rows_grouped<DTYPE>(                                          \
      device const long*   cols      [[buffer(1)]],                          \
      device const DTYPE*  vals      [[buffer(2)]],                          \
      device const DTYPE*  dense     [[buffer(3)]],                          \
      device DTYPE*        out       [[buffer(4)]],                          \
      device const long*   row_ptr   [[buffer(5)]],                          \
      constant uint4&      dims      [[buffer(6)]],                          \
      uint3                tid       [[thread_position_in_grid]],            \
      uint3                ltid      [[thread_position_in_threadgroup]],     \
      uint3                tptg      [[threads_per_threadgroup]]);

INSTANTIATE_FOR_ALL_TYPES(INSTANTIATE_SPMM_BMM_COO_ROWS_GROUPED);

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

INSTANTIATE_FOR_ALL_TYPES(INSTANTIATE_SPMM_ADDMM_COO);
