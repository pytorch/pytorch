#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;

template <typename T>
static T bool_to_float(bool b) {
  return static_cast<T>(b);
}

template <>
half2 bool_to_float(bool b) {
  return half2(b ? 1 : 0, 0);
}

template <>
float2 bool_to_float(bool b) {
  return float2(b ? 1 : 0, 0);
}

constant constexpr auto kMaxThreadsPerThreadgroup = 1024;
constant constexpr auto kMaxSIMDGroups =
    kMaxThreadsPerThreadgroup / c10::metal::simdgroup_size;

// Combines a block of reflectors into one factor, one column per threadgroup,
// so the host can apply the whole block with two matmuls.
template <typename T>
kernel void householder_block(
    device const T* A,
    device const T* tau,
    device T* V,
    device T* W,
    constant GeqrfParams<>& params,
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  auto dims = params.num_batch_dims;
  auto m = params.A_sizes[dims];
  auto n = params.A_sizes[dims + 1];
  auto row_stride = params.A_strides[dims];
  auto col_stride = params.A_strides[dims + 1];
  auto tau_stride = params.tau_strides[dims];
  auto batch = tgid / n;
  for (int dim = dims - 1; dim >= 0; --dim) {
    auto index = batch % params.A_sizes[dim];
    A += index * params.A_strides[dim];
    tau += index * params.tau_strides[dim];
    batch /= params.A_sizes[dim];
  }
  auto col = tgid % n;
  V += uint64_t(tgid) * m;
  W += uint64_t(tgid) * m;
  threadgroup T scratch[kMaxSIMDGroups];
  for (auto row = tid; row < m; row += tptg) {
    auto value = row > col ? A[row * row_stride + col * col_stride]
                           : bool_to_float<T>(row == col);
    V[row] = value;
    W[row] = c10::metal::mul(tau[col * tau_stride], value);
  }
  threadgroup_barrier(mem_flags::mem_device);
  // H_0 ... H_{n-1} = I - W V^H.
  for (int i = int(col) - 1; i >= 0; --i) {
    T dot = 0;
    for (auto row = uint(i) + tid; row < m; row += tptg) {
      auto value = row == uint(i) ? bool_to_float<T>(true)
                                  : A[row * row_stride + i * col_stride];
      dot += c10::metal::mul(c10::metal::conj(value), W[row]);
    }
    auto factor = c10::metal::mul(
        tau[i * tau_stride],
        c10::metal::threadgroup_sum(scratch, dot, tid, tptg));
    for (auto row = uint(i) + tid; row < m; row += tptg) {
      auto value = row == uint(i) ? bool_to_float<T>(true)
                                  : A[row * row_stride + i * col_stride];
      W[row] -= c10::metal::mul(value, factor);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

template <typename T>
static T calc_H_irc(
    device T* A,
    uint32_t A_stride_r,
    uint32_t A_stride_c,
    device const T* tau,
    uint32_t tau_stride,
    uint32_t r,
    uint32_t c,
    uint32_t i) {
  T I_val = bool_to_float<T>(r == c);
  T tau_val = tau[i * tau_stride];

  T A_ci = c10::metal::conj(A[c * A_stride_r + i * A_stride_c]);
  T A_ri = A[r * A_stride_r + i * A_stride_c];

  T c_eq_i = bool_to_float<T>(c == i);
  T r_eq_i = bool_to_float<T>(r == i);

  T A_ci_ = (c > i) ? A_ci : c_eq_i;
  T A_ri_ = (r > i) ? A_ri : r_eq_i;

  return I_val - c10::metal::mul(tau_val, c10::metal::mul(A_ci_, A_ri_));
}

// Calculate (A @ B)[r, c], the element in the r-th row and c-th column of the
// result of matrix multiplying A and B together. A and B must be size m-by-m
// and have the same strides. The formula for this operation, written in Python
// syntax, is:
//   (A @ B)[r, c] = A[r, :].dot(B[:, c])
template <typename T>
static T calc_matmul_rc(
    device T* A,
    device T* B,
    uint32_t stride_r,
    uint32_t stride_c,
    uint32_t m,
    uint32_t r,
    uint32_t c) {
  T AB_rc = 0;
  auto A_row_offset = r * stride_r;
  auto B_col_offset = c * stride_c;

  uint32_t A_col_offset = 0;
  uint32_t B_row_offset = 0;

  for (uint32_t j = 0; j < m;
       j++, A_col_offset += stride_c, B_row_offset += stride_r) {
    AB_rc += c10::metal::mul(
        A[A_row_offset + A_col_offset], B[B_row_offset + B_col_offset]);
  }
  return AB_rc;
}

template <typename T>
kernel void orgqr(
    device T* A [[buffer(0)]],
    device const T* tau [[buffer(1)]],
    device T* H [[buffer(2)]],
    device T* H_prod [[buffer(3)]],
    device T* H_prod_work [[buffer(4)]],
    constant OrgqrParams<>& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  constant auto& A_strides = params.A_strides;
  constant auto& tau_strides = params.tau_strides;
  constant auto& H_strides = params.H_strides;
  constant auto& H_sizes = params.H_sizes;

  auto num_batch_dims = params.num_batch_dims;
  auto m = params.m;
  auto m2 = params.m2;
  auto n = params.n;
  auto k = params.k;

  auto batch_idx = tgid;

  // Find the matrices for this thread's batch index
  uint32_t A_offset = 0;
  uint32_t tau_offset = 0;
  uint32_t H_offset = 0;

  for (auto dim = num_batch_dims - 1; dim >= 0; dim--) {
    auto dim_size = H_sizes[dim];
    auto dim_idx = batch_idx % dim_size;

    A_offset += dim_idx * A_strides[dim];
    tau_offset += dim_idx * tau_strides[dim];
    H_offset += dim_idx * H_strides[dim];

    batch_idx /= dim_size;
  }

  A += A_offset;
  tau += tau_offset;
  H += H_offset;
  H_prod += H_offset;
  H_prod_work += H_offset;

  auto A_stride_r = A_strides[num_batch_dims];
  auto A_stride_c = A_strides[num_batch_dims + 1];
  auto tau_stride = tau_strides[num_batch_dims];
  auto H_stride_r = H_strides[num_batch_dims];
  auto H_stride_c = H_strides[num_batch_dims + 1];

  for (uint32_t i = 0; i < k; i++) {
    // Calculate and write H_i
    for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
      auto r = matrix_idx / m;
      auto c = matrix_idx % m;
      T H_irc = calc_H_irc(A, A_stride_r, A_stride_c, tau, tau_stride, r, c, i);

      if (i == 0) {
        H_prod[r * H_stride_r + c * H_stride_c] = H_irc;
      } else {
        H[r * H_stride_r + c * H_stride_c] = H_irc;
      }
    }

    if (i > 0) {
      // Need this sync because the below matmul requires all threads to finish
      // writing their entries to `H_prod` and `H`.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Calculate H_prod @ H_i, and write result to H_prod_work
      for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
        auto r = matrix_idx / m;
        auto c = matrix_idx % m;

        T H_prod_0_to_i_rc =
            calc_matmul_rc(H_prod, H, H_stride_r, H_stride_c, m, r, c);

        H_prod_work[r * H_stride_r + c * H_stride_c] = H_prod_0_to_i_rc;
      }

      // Need this sync because the above matmul uses the current values in
      // `H_prod`, and we don't want to overwrite those until all threads are
      // finished using them.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Copy H_prod_work into H_prod
      for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
        auto r = matrix_idx / m;
        auto c = matrix_idx % m;
        H_prod[r * H_stride_r + c * H_stride_c] =
            H_prod_work[r * H_stride_r + c * H_stride_c];
      }
    }
  }

  for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
    auto r = matrix_idx / m;
    auto c = matrix_idx % m;
    if (c < n) {
      A[r * A_stride_r + c * A_stride_c] =
          H_prod[r * H_stride_r + c * H_stride_c];
    }
  }
}

// Unblocked QR of a panel held entirely in registers, RowsPerThread rows per
// lane, so every reduction stays within one SIMD group.
template <uint RowsPerThread>
[[max_total_threads_per_threadgroup(kMaxThreadsPerThreadgroup)]]
kernel void geqrf_panel(
    device float* A,
    device float* tau,
    constant GeqrfParams<>& params,
    uint tid [[thread_position_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  auto dims = params.num_batch_dims;
  auto m = params.A_sizes[dims];
  auto n = params.A_sizes[dims + 1];
  for (int dim = dims - 1; dim >= 0; --dim) {
    auto index = batch % params.A_sizes[dim];
    A += index * params.A_strides[dim];
    tau += index * params.tau_strides[dim];
    batch /= params.A_sizes[dim];
  }
  auto lane = tid % c10::metal::simdgroup_size;
  auto col = tid / c10::metal::simdgroup_size;
  auto row_stride = params.A_strides[dims];
  auto col_stride = params.A_strides[dims + 1];
  threadgroup float v[RowsPerThread * c10::metal::simdgroup_size];
  threadgroup float tau_shared;
  // One SIMD group per column; n <= SIMD width keeps each pivot in r[0].
  float r[RowsPerThread];
  for (uint j = 0; j < RowsPerThread; ++j) {
    auto row = lane + c10::metal::simdgroup_size * j;
    r[j] = row < m ? A[row * row_stride + col * col_stride] : 0;
  }
  for (uint k = 0; k < n; ++k) {
    if (col == k) {
      float norm_sq = 0;
      float max_abs = 0;
      for (uint j = 0; j < RowsPerThread; ++j) {
        if (lane + c10::metal::simdgroup_size * j >= k) {
          norm_sq = fma(r[j], r[j], norm_sq);
          max_abs = c10::metal::max(max_abs, fabs(r[j]));
        }
      }
      norm_sq = simd_sum(norm_sq);
      constexpr auto safe_min =
          numeric_limits<float>::min() / numeric_limits<float>::epsilon();
      float scale = 1;
      if (norm_sq < m * safe_min || !isfinite(norm_sq)) {
        scale = c10::metal::simd_max(max_abs);
        scale = scale == 0 ? 1 : scale;
        norm_sq = 0;
        for (uint j = 0; j < RowsPerThread; ++j) {
          if (lane + c10::metal::simdgroup_size * j >= k) {
            r[j] /= scale;
            norm_sq = fma(r[j], r[j], norm_sq);
          }
        }
        norm_sq = simd_sum(norm_sq);
      }
      auto norm = precise::sqrt(norm_sq);
      auto alpha = simd_broadcast(r[0], k);
      auto beta = copysign(norm, alpha);
      auto t = norm == 0 ? 0 : 1 + fabs(alpha) / norm;
      auto inv = t == 0 ? 0 : 1 / (alpha + beta);
      if (lane == 0) {
        tau[k * params.tau_strides[dims]] = t;
        tau_shared = t;
      }
      for (uint j = 0; j < RowsPerThread; ++j) {
        auto row = lane + c10::metal::simdgroup_size * j;
        v[row] = row == k ? 1 : row > k ? r[j] * inv : 0;
        if (t != 0 && row >= k) {
          r[j] = row == k ? -beta * scale : v[row];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col > k && tau_shared != 0) {
      float dot = 0;
      for (uint j = 0; j < RowsPerThread; ++j) {
        auto row = lane + c10::metal::simdgroup_size * j;
        if (row >= k) {
          dot = fma(r[j], v[row], dot);
        }
      }
      auto factor = tau_shared * simd_sum(dot);
      for (uint j = 0; j < RowsPerThread; ++j) {
        auto row = lane + c10::metal::simdgroup_size * j;
        if (row >= k) {
          r[j] -= v[row] * factor;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < RowsPerThread; ++j) {
    auto row = lane + c10::metal::simdgroup_size * j;
    if (row < m) {
      A[row * row_stride + col * col_stride] = r[j];
    }
  }
}

// geqrf_kernel_mps picks the smallest of these that holds the whole panel.
#define REGISTER_GEQRF_PANEL(Rows)                                            \
  template [[host_name("geqrf_panel_" #Rows)]] kernel void geqrf_panel<Rows>( \
      device float*, device float*, constant GeqrfParams<>&, uint, uint);
REGISTER_GEQRF_PANEL(8);
REGISTER_GEQRF_PANEL(16);
REGISTER_GEQRF_PANEL(32);

// Unblocked Householder QR with one threadgroup per matrix. Reflectors are
// normalized in parallel, and each trailing column gets as many simdgroups as
// the threadgroup can spare.
template <typename T>
kernel void geqrf(
    device T* R [[buffer(0)]],
    device T* tau [[buffer(1)]],
    constant GeqrfParams<>& params [[buffer(2)]],
    device T* v_work [[buffer(3)]],
    uint thread_pos [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint tg_pos [[threadgroup_position_in_grid]]) {
  using opmath_t = c10::metal::opmath_t<T>;

  const uint32_t tid = thread_pos;
  const uint32_t group_size = tpg;
  const uint32_t m = params.A_sizes[params.num_batch_dims];
  const uint32_t n = params.A_sizes[params.num_batch_dims + 1];
  const uint32_t K = min(m, n);

  // Batch indexing
  uint32_t batch_idx = tg_pos;
  const uint32_t v_stride = m;
  device T* v_batch = v_work + batch_idx * v_stride;

  // Find the matrices for this thread's batch index
  uint32_t A_offset = 0;
  uint32_t tau_offset = 0;

  for (int32_t dim = params.num_batch_dims - 1; dim >= 0; dim--) {
    auto dim_size = params.A_sizes[dim];
    auto dim_idx = batch_idx % dim_size;

    A_offset += dim_idx * params.A_strides[dim];
    tau_offset += dim_idx * params.tau_strides[dim];

    batch_idx /= dim_size;
  }

  device T* R_batch = R + A_offset;
  device T* tau_batch = tau + tau_offset;

  const uint32_t tau_stride = params.tau_strides[params.num_batch_dims];
  const uint32_t R_stride_r = params.A_strides[params.num_batch_dims];
  const uint32_t R_stride_c = params.A_strides[params.num_batch_dims + 1];

  threadgroup opmath_t scratch[kMaxSIMDGroups];
  threadgroup opmath_t tau_shared;
  threadgroup opmath_t inv_u1;

  for (uint32_t k = 0; k < K; k++) {
    uint32_t R_k_offset = k * R_stride_c;
    uint32_t tau_k_offset = k * tau_stride;
    opmath_t norm_sq = 0;
    opmath_t max_abs = 0;
    for (uint32_t i = k + tid; i < m; i += group_size) {
      auto value = static_cast<opmath_t>(R_batch[i * R_stride_r + R_k_offset]);
      v_batch[i] = static_cast<T>(value);
      norm_sq = fma(value, value, norm_sq);
      max_abs = c10::metal::max(max_abs, fabs(value));
    }
    norm_sq = c10::metal::threadgroup_sum(scratch, norm_sq, tid, group_size);
    // Recompute with scaling only when squaring overflowed or underflowed.
    constexpr auto safe_min =
        numeric_limits<opmath_t>::min() / numeric_limits<opmath_t>::epsilon();
    opmath_t scale = 1;
    if (norm_sq < m * safe_min || !isfinite(norm_sq)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scale = c10::metal::threadgroup_max(scratch, max_abs, tid, group_size);
      scale = scale == 0 ? 1 : scale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      norm_sq = 0;
      for (uint32_t i = k + tid; i < m; i += group_size) {
        auto value = static_cast<opmath_t>(v_batch[i]) / scale;
        v_batch[i] = static_cast<T>(value);
        norm_sq = fma(value, value, norm_sq);
      }
      norm_sq = c10::metal::threadgroup_sum(scratch, norm_sq, tid, group_size);
    }
    const auto norm = precise::sqrt(norm_sq);

    if (tid == 0) {
      // LAPACK leaves the last row's sign unchanged.
      if (norm == 0 || k == m - 1) {
        tau_shared = 0;
      } else {
        auto alpha = static_cast<opmath_t>(v_batch[k]);
        auto beta = copysign(norm, alpha);
        tau_shared = 1 + fabs(alpha) / norm;
        inv_u1 = 1 / (alpha + beta);
        R_batch[k * R_stride_r + R_k_offset] = static_cast<T>(-beta * scale);
      }
      tau_batch[tau_k_offset] = static_cast<T>(tau_shared);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    const auto tau_val = tau_shared;
    if (tau_val == 0) {
      continue;
    }
    if (tid == 0) {
      v_batch[k] = static_cast<T>(1.0);
    }
    for (uint32_t i = k + 1 + tid; i < m; i += group_size) {
      auto value = static_cast<T>(static_cast<opmath_t>(v_batch[i]) * inv_u1);
      v_batch[i] = value;
      R_batch[i * R_stride_r + R_k_offset] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 3: apply reflection to trailing columns of R
    uint32_t threads_per_col = c10::metal::simdgroup_size;
    while (uint64_t(threads_per_col) * 2 * max(n - k - 1, 1u) <= group_size) {
      threads_per_col *= 2;
    }
    uint32_t col_lane = tid % threads_per_col;
    uint32_t col_group = tid / threads_per_col;
    uint32_t num_col_groups = group_size / threads_per_col;
    uint32_t simds_per_col = threads_per_col / c10::metal::simdgroup_size;

    for (uint32_t j_base = k + 1; j_base < n; j_base += num_col_groups) {
      uint32_t j = j_base + col_group;
      uint32_t R_j_offset = j * R_stride_c;
      opmath_t dot = 0.0;
      if (j < n) {
        for (uint32_t i = k + col_lane; i < m; i += threads_per_col) {
          opmath_t v_i = static_cast<opmath_t>(v_batch[i]);
          opmath_t r_ij =
              static_cast<opmath_t>(R_batch[i * R_stride_r + R_j_offset]);
          dot = fma(v_i, r_ij, dot);
        }
      }
      opmath_t vt_col = simd_sum(dot);
      if (simds_per_col > 1) {
        if (tid % c10::metal::simdgroup_size == 0) {
          scratch[tid / c10::metal::simdgroup_size] = vt_col;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        vt_col = 0;
        for (uint32_t s = 0; s < simds_per_col; ++s) {
          vt_col += scratch[col_group * simds_per_col + s];
        }
      }
      if (j < n) {
        opmath_t factor = tau_val * vt_col;

        // Update column
        for (uint32_t i = k + col_lane; i < m; i += threads_per_col) {
          opmath_t v_i = static_cast<opmath_t>(v_batch[i]);
          opmath_t r_ij =
              static_cast<opmath_t>(R_batch[i * R_stride_r + R_j_offset]);
          R_batch[i * R_stride_r + R_j_offset] =
              static_cast<T>(r_ij - v_i * factor);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

#define REGISTER_GEQRF(T)              \
  template [[host_name("geqrf_" #T)]]  \
  kernel void geqrf<T>(                \
      device T * R,                    \
      device T * tau,                  \
      constant GeqrfParams<> & params, \
      device T * v_work,               \
      uint tid,                        \
      uint tpg,                        \
      uint tg_pos);

REGISTER_GEQRF(float);

#define REGISTER_ORGQR(T)                           \
  template [[host_name("orgqr_" #T)]]               \
  kernel void orgqr<T>(                             \
      device T * A [[buffer(0)]],                   \
      device const T* tau [[buffer(1)]],            \
      device T* H [[buffer(2)]],                    \
      device T* H_prod [[buffer(3)]],               \
      device T* H_prod_work [[buffer(4)]],          \
      constant OrgqrParams<>& params [[buffer(5)]], \
      uint tid [[thread_position_in_threadgroup]],  \
      uint tptg [[threads_per_threadgroup]],        \
      uint tgid [[threadgroup_position_in_grid]]);

REGISTER_ORGQR(float);
REGISTER_ORGQR(half);
REGISTER_ORGQR(bfloat);
REGISTER_ORGQR(float2);
REGISTER_ORGQR(half2);

#define REGISTER_HOUSEHOLDER_BLOCK(T)             \
  template [[host_name("householder_block_" #T)]] \
  kernel void householder_block<T>(               \
      device const T*,                            \
      device const T*,                            \
      device T*,                                  \
      device T*,                                  \
      constant GeqrfParams<>&,                    \
      uint,                                       \
      uint,                                       \
      uint);

REGISTER_HOUSEHOLDER_BLOCK(float);
REGISTER_HOUSEHOLDER_BLOCK(float2);
