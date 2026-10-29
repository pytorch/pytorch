#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <metal_stdlib>

using namespace metal;

// See Note [linalg.eig for MPS]
// Helper functions and kernels for linalg.eig. Matrices are column-major:
// element (i, j) of an n x n matrix M is `M[i + j * n]`.
namespace eig {

constant constexpr float kEps = numeric_limits<float>::epsilon();
// Magnitudes below kSafeMin * n count as zero, as with LAPACK's SMLNUM
constant constexpr float kSafeMin = numeric_limits<float>::min() / kEps;

// |re| + |im|, which is cheap and within a factor sqrt(2) of |z|
inline float abs1(float2 z) {
  return fabs(z.x) + fabs(z.y);
}

// A Givens rotation G(i, j, c, s), with real c and c^2 + |s|^2 = 1 [Arbenz,
// eq. (4.4), with complex s so that G is unitary], represents a rotation of a
// vector in the plane spanned by basis vectors e_i and e_j. It is the identity
// matrix except for four elements:
//   G[i, i] = c
//   G[i, j] = s
//   G[j, i] = -s^*
//   G[j, j] = c
// However, we don't have to form the full Givens matrix and calculate the full
// `G H G^*` matrix multiplications when applying a Givens rotation. Instead we
// can just apply the rotation directly to the rows or columns that the Givens
// rotation will affect.
struct Givens {
  float c;
  float2 s;
};

// The rotation with G [x; y] = [r; 0] [Arbenz, eq. (4.5)]: c = |x| / rho and
// s = (x / |x|) y^* / rho, where rho = ||(x, y)||
inline Givens make_givens(float2 x, float2 y) {
  const float x_abs = c10::metal::hypot(x.x, x.y);
  const float y_abs = c10::metal::hypot(y.x, y.y);
  if (y_abs == 0) {
    return {1.0f, float2(0)};
  }
  if (x_abs == 0) {
    return {0.0f, c10::metal::conj(y) / y_abs};
  }
  const float rho = c10::metal::hypot(x_abs, y_abs);
  return {x_abs / rho, c10::metal::mul(x / x_abs, c10::metal::conj(y)) / rho};
}

// (a, b) <- G (a, b): G applied to a pair of rows
inline void rotate_rows(Givens g, thread float2& a, thread float2& b) {
  const float2 a_new = g.c * a + c10::metal::mul(g.s, b);
  b = g.c * b - c10::metal::mul(c10::metal::conj(g.s), a);
  a = a_new;
}

// (a, b) <- (a, b) G^*: G^* applied to a pair of columns
inline void rotate_columns(Givens g, thread float2& a, thread float2& b) {
  const float2 a_new = g.c * a + c10::metal::mul(c10::metal::conj(g.s), b);
  b = g.c * b - c10::metal::mul(g.s, a);
  a = a_new;
}

inline packed_float3 pack_givens(Givens g) {
  return packed_float3(g.c, g.s.x, g.s.y);
}

inline Givens unpack_givens(packed_float3 v) {
  return {v.x, float2(v.y, v.z)};
}

// Orders accesses to a matrix in threadgroup memory (P = threadgroup float2*)
// or device memory (P = device float2*)
template <typename P>
inline void matrix_barrier() {
  if IF_CONSTEXPR (::metal::is_same_v<P, threadgroup float2*>) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    threadgroup_barrier(mem_flags::mem_device);
  }
}

// Returns the top l of the active window H[l:ihi+1, l:ihi+1]: the largest
// l <= ihi whose subdiagonal entry H[l, l-1] is negligible, or 0. That entry is
// set to zero, deflating the problem [Arbenz, Algorithm 4.4, with the
// convergence test of Algorithm 4.5].
//
// Like LAPACK's xLAHQR, an entry that passes that test must also pass the more
// conservative one of Ahues and Kressner, which compares it with the rest of
// its 2x2 block.
template <typename P, typename I>
int find_active_window(
    P H,
    I n,
    int ihi,
    threadgroup int* scratch,
    uint tid,
    uint nthreads) {
  const float small_num = kSafeMin * n;
  int l = 0;
  for (int k = 1 + tid; k <= ihi; k += nthreads) {
    const float2 a = H[(k - 1) + (k - 1) * n];
    const float2 d = H[k + k * n];
    const float h = abs1(H[k + (k - 1) * n]);
    if (h <= small_num) {
      l = k;
    } else if (h <= kEps * (abs1(a) + abs1(d))) {
      const float b = abs1(H[(k - 1) + k * n]);
      const float ab = max(h, b);
      const float ba = min(h, b);
      const float aa = max(abs1(d), abs1(a - d));
      const float bb = min(abs1(d), abs1(a - d));
      const float s = aa + ab;
      if (ba * (ab / s) <= max(small_num, kEps * (bb * (aa / s)))) {
        l = k;
      }
    }
  }
  // The previous reduction may still be reading scratch
  threadgroup_barrier(mem_flags::mem_threadgroup);
  l = c10::metal::threadgroup_max(scratch, l, tid, nthreads);
  if (tid == 0 && l > 0) {
    H[l + (l - 1) * n] = float2(0);
  }
  return l;
}

// The shift mu for a QR step on the active window ending at row ihi. Most of
// the time, this is the Wilkinson shift (See Note [linalg.eig for MPS]). But
// the Wilkinson shift alone does not always converge, so for every 10
// iterations where deflation has not occurred yet, we use an exceptional shift
// to break out of cycles.
template <typename P, typename I>
float2 qr_shift(P H, I n, int ihi, int its) {
  // Pick out the lower right 2x2 corner of the active window:
  //  . . . .
  //  . . . .
  //  . . a b
  //  . . c d
  const float2 d = H[ihi + ihi * n];
  const float2 c = H[ihi + (ihi - 1) * n];
  if (its > 0 && its % 10 == 0) {
    // 10th iteration without deflation, apply exceptional shift, based on
    // LAPACK's xLAHQR.
    return d + float2(0.75f * abs1(c), 0.0f);
  }
  // Eigenvalues of the 2x2 corner are:
  //   l (+/-) r
  // where
  //   l = (a + d) / 2
  //   r = sqrt((a - d)^2 + 4 b c) / 2
  // As in LAPACK's xLAHQR, u = sqrt(b) sqrt(c) stands in for sqrt(b c), and x
  // and u are divided by s before squaring them, so that tiny or huge entries
  // can't underflow or overflow.
  const float2 a = H[(ihi - 1) + (ihi - 1) * n];
  const float2 u = c10::metal::mul(
      c10::metal::sqrt(H[(ihi - 1) + ihi * n]), c10::metal::sqrt(c));
  const float2 x = 0.5f * (a - d);
  const float s = max(abs1(x), abs1(u));
  if (s == 0) {
    return d;
  }
  const float2 xs = x / s;
  const float2 us = u / s;
  float2 y =
      s * c10::metal::sqrt(c10::metal::mul(xs, xs) + c10::metal::mul(us, us));
  if (dot(x, y) < 0) {
    y = -y;
  }
  return d - c10::metal::mul(u, c10::metal::div(u, x + y));
}

// One implicitly shifted QR step on the active window H[l:ihi+1, l:ihi+1]
// [Arbenz, Section 4.6.2, which shows it for tridiagonal matrices; it works the
// same way for Hessenberg ones]. The first rotation G_l is the one that starts
// the QR factorization of H - mu I [Arbenz, Algorithm 4.2]. Applying it to both
// sides of H puts a bulge at H[l+2, l], and each following rotation G_k zeroes
// the bulge at H[k+1, k-1], which moves it one row and column down, until it
// falls out of the window. By the implicit Q theorem [Arbenz, Theorem 4.5] this
// gives the same result as the explicit step in Note [linalg.eig for MPS],
// without forming H - mu I.
//
// The rotations are applied one after another, each in parallel over the rows
// or columns it touches, but only inside the window. They are recorded in
// `rotations` for apply_rotations_outside_window to apply to the rest of H and
// to U.
template <typename P, typename I>
void chase_bulge(
    P H,
    I n,
    int l,
    int ihi,
    float2 mu,
    device packed_float3* rotations,
    uint tid,
    uint nthreads) {
  for (int k = l; k < ihi; ++k) {
    float2 x, y;
    if (k == l) {
      x = H[l + l * n] - mu;
      y = H[(l + 1) + l * n];
      // Column l is rotated below, so all threads must read it first
      matrix_barrier<P>();
    } else {
      x = H[k + (k - 1) * n];
      y = H[(k + 1) + (k - 1) * n];
    }
    const Givens g = make_givens(x, y);
    // G_k on rows k and k+1. Column k-1, where it zeroes the bulge, is written
    // after the next barrier, once all threads have read x and y.
    for (int j = k + tid; j <= ihi; j += nthreads) {
      float2 a = H[k + j * n];
      float2 b = H[(k + 1) + j * n];
      rotate_rows(g, a, b);
      H[k + j * n] = a;
      H[(k + 1) + j * n] = b;
    }
    matrix_barrier<P>();
    // G_k^* on columns k and k+1, which are nonzero down to row k+2
    for (int i = l + tid; i <= min(k + 2, ihi); i += nthreads) {
      float2 a = H[i + k * n];
      float2 b = H[i + (k + 1) * n];
      rotate_columns(g, a, b);
      H[i + k * n] = a;
      H[i + (k + 1) * n] = b;
    }
    if (tid == 0) {
      if (k > l) {
        rotate_rows(g, x, y);
        H[k + (k - 1) * n] = x;
        H[(k + 1) + (k - 1) * n] = float2(0);
      }
      rotations[k] = pack_givens(g);
    }
    matrix_barrier<P>();
  }
}

// Applies rotations[l:ihi] to a row (from the right) or a column (from the
// left) whose entry j is v[j * stride]
template <typename P, typename I>
void rotate_vector(
    P v,
    I stride,
    int l,
    int ihi,
    device const packed_float3* rotations,
    bool is_column) {
  float2 a = v[l * stride];
  for (int k = l; k < ihi; ++k) {
    float2 b = v[(k + 1) * stride];
    if (is_column) {
      rotate_rows(unpack_givens(rotations[k]), a, b);
    } else {
      rotate_columns(unpack_givens(rotations[k]), a, b);
    }
    v[k * stride] = a;
    a = b;
  }
  v[ihi * stride] = a;
}

// Applies the rotations of chase_bulge where it didn't: from the right to rows
// H[:l, :] and to all rows of U, and from the left to columns H[:, ihi+1:].
// Each of these rows and columns only depends on its own entries, so threads
// work through them independently, without barriers.
template <typename P, typename I>
void apply_rotations_outside_window(
    P H,
    device float2* U,
    I n,
    int l,
    int ihi,
    device const packed_float3* rotations,
    bool update_u,
    uint tid,
    uint nthreads) {
  const I h_rows = l;
  const I h_columns = n - 1 - ihi;
  const I u_rows = update_u ? n : 0;
  for (I t = tid; t < h_rows + h_columns + u_rows; t += nthreads) {
    if (t < h_rows) {
      rotate_vector(H + t, n, l, ihi, rotations, false);
    } else if (t < h_rows + h_columns) {
      rotate_vector(
          H + (ihi + 1 + t - h_rows) * n, I(1), l, ihi, rotations, true);
    } else {
      rotate_vector(U + (t - h_rows - h_columns), n, l, ihi, rotations, false);
    }
  }
}

// Reduces the Hessenberg H to upper triangular T = Q^* H Q and updates
// U <- U Q, by QR steps with deflation.
//
// Like LAPACK's xLAHQR returns 0 on success, if all subdiagonal elements
// converged close to 0; otherwise, returns the number of dimensions remaining
// if it did not converge after `max_iterations`.
template <typename P, typename I>
int qr_iterate(
    P H,
    device float2* U,
    device packed_float3* rotations,
    I n,
    bool update_u,
    threadgroup int* scratch,
    uint tid,
    uint nthreads) {
  const int max_iterations = 30 * max(10, int(n));
  int ihi = int(n) - 1;
  while (ihi > 0) {
    int l;
    for (int its = 0;; ++its) {
      l = find_active_window(H, n, ihi, scratch, tid, nthreads);
      if (l == ihi) {
        break;
      }
      if (its == max_iterations) {
        return ihi + 1;
      }
      const float2 mu = qr_shift(H, n, ihi, its);
      chase_bulge(H, n, l, ihi, mu, rotations, tid, nthreads);
      threadgroup_barrier(mem_flags::mem_device);
      apply_rotations_outside_window(
          H, U, n, l, ihi, rotations, update_u, tid, nthreads);
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    ihi = l - 1;
  }
  return 0;
}

// Upper bound on intermediate magnitudes in eig_vectors [Adlerborn et al.,
// Section 3.1]. It leaves room below FLT_MAX for the constant factors that come
// from measuring complex numbers with abs1.
constant constexpr float kOmega = 1e36f;

// ProtectDivision [Adlerborn et al., Algorithm 3]: a factor alpha in (0, 1]
// such that alpha b / d can't exceed kOmega, given the magnitudes b and d of
// the numerator and divisor
inline float protect_division(float b, float d) {
  if (d < 1 / kOmega) {
    if (b > d * kOmega) {
      return d * kOmega / b;
    }
  } else if (d < 1) {
    if (b > d * kOmega) {
      return 1 / b;
    }
  }
  return 1;
}

// ProtectUpdate [Adlerborn et al., Algorithm 4]: a factor zeta such that
// zeta y - c (zeta x) can't exceed kOmega, given bounds on the magnitudes of
// y, c and x
inline float protect_update(float y, float c, float x) {
  if (x <= 1) {
    if (c * x > kOmega - y) {
      return 0.5f;
    }
  } else if (c > (kOmega - y) / x) {
    return 0.5f / x;
  }
  return 1;
}

// Whether row (or column) i of A is zero off the diagonal within columns (or
// rows) first..last
template <typename I>
bool is_isolated(
    device float2* A,
    I n,
    I i,
    I first,
    I last,
    bool is_row,
    threadgroup float* scratch,
    uint tid,
    uint nthreads) {
  float nonzero = 0;
  for (I m = first + tid; m <= last; m += nthreads) {
    const float2 a = is_row ? A[i + m * n] : A[m + i * n];
    if (m != i && (a.x != 0 || a.y != 0)) {
      nonzero = 1;
    }
  }
  // The previous reduction may still be reading scratch
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return c10::metal::threadgroup_max(scratch, nonzero, tid, nthreads) == 0;
}

// Swaps rows i and j and columns i and j of A, and entries i and j of perm
template <typename I>
void swap_indices(
    device float2* A,
    device long* perm,
    I n,
    I i,
    I j,
    uint tid,
    uint nthreads) {
  for (I m = tid; m < n; m += nthreads) {
    const float2 a = A[i + m * n];
    A[i + m * n] = A[j + m * n];
    A[j + m * n] = a;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (I m = tid; m < n; m += nthreads) {
    const float2 a = A[m + i * n];
    A[m + i * n] = A[m + j * n];
    A[m + j * n] = a;
  }
  if (tid == 0) {
    const long p = perm[i];
    perm[i] = perm[j];
    perm[j] = p;
  }
  threadgroup_barrier(mem_flags::mem_device);
}

// MPS arithmetic flushes subnormal floats to zero, so these work on the bit
// patterns instead.

// floor(log2(x)) for the bits of a positive float x
inline int exponent_of(uint bits) {
  return bits < 0x00800000u ? -118 - int(clz(bits)) : int(bits >> 23) - 127;
}

// x * 2^e, rounded toward zero if the result is subnormal
inline float ldexp_exact(float x, int e) {
  const uint bits = as_type<uint>(x);
  const uint sign = bits & 0x80000000u;
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude == 0) {
    return x;
  }
  // |x| = mantissa * 2^exponent, with mantissa in [2^23, 2^24)
  uint mantissa;
  int exponent;
  if (magnitude < 0x00800000u) {
    const uint shift = clz(magnitude) - 8;
    mantissa = magnitude << shift;
    exponent = -149 - int(shift);
  } else {
    mantissa = (magnitude & 0x007fffffu) | 0x00800000u;
    exponent = int(magnitude >> 23) - 150;
  }
  const int biased_exponent = exponent + e + 150;
  if (biased_exponent >= 255) {
    return as_type<float>(sign | 0x7f800000u);
  }
  if (biased_exponent > 0) {
    return as_type<float>(
        sign | (uint(biased_exponent) << 23) | (mantissa & 0x007fffffu));
  }
  const int shift = 1 - biased_exponent;
  return as_type<float>(sign | (shift < 24 ? mantissa >> shift : 0u));
}

inline float2 ldexp_exact(float2 z, int e) {
  return float2(ldexp_exact(z.x, e), ldexp_exact(z.y, e));
}

} // namespace eig

// Scales A by 2^scale_exponent, then balances A <- D^-1 P^T A P D (see Note
// [linalg.eig for MPS]).
//
// Like LAPACK's xGEEV, the scaling brings the largest entry of A between 2^-40
// and 2^40, which keeps intermediate values in the kernels from underflowing or
// overflowing. eig_qr_iterate undoes it for the eigenvalues.
//
// The balancing is like LAPACK's xGEBAL. The permutation P, with
// (P^T A P)[i, j] = A[perm[i], perm[j]], moves
// rows and columns that are zero off the diagonal to the bottom and left, which
// isolates eigenvalues that are then computed exactly, e.g. those of triangular
// matrices. The elements of D are powers of 2, so the scaling is exact. It
// sweeps over the rows until no scaling changes the sum of their norms by more
// than 5%.
// See LAPACK documentation: https://netlib.org/lapack/lug/node51.html
template <typename I>
kernel void eig_balance(
    device float2* A [[buffer(0)]],
    device float* D [[buffer(1)]],
    device long* perm [[buffer(2)]],
    device int* scale_exponent [[buffer(3)]],
    constant EigParams& params [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  constexpr int kMaxSweeps = 32;
  // Limits on the norms that keep the scaling from overflowing
  const float min_norm = 2 * eig::kSafeMin;
  const float max_norm = 1 / min_norm;
  const I n = static_cast<I>(params.n);
  device float2* A_b = A + static_cast<long>(batch) * n * n;
  device float* D_b = D + static_cast<long>(batch) * n;
  device long* perm_b = perm + static_cast<long>(batch) * n;
  threadgroup float column_scratch[32];
  threadgroup float row_scratch[32];
  threadgroup uint largest_scratch[32];

  // The bits of non-negative floats compare like their values
  uint largest = 0;
  for (I i = tid; i < n * n; i += nthreads) {
    const uint2 bits = as_type<uint2>(A_b[i]) & 0x7fffffffu;
    largest = max(largest, max(bits.x, bits.y));
  }
  largest =
      c10::metal::threadgroup_max(largest_scratch, largest, tid, nthreads);
  const int exponent = largest == 0 ? 0 : eig::exponent_of(largest);
  const int scale = clamp(exponent, -40, 40) - exponent;
  if (scale != 0) {
    for (I i = tid; i < n * n; i += nthreads) {
      A_b[i] = eig::ldexp_exact(A_b[i], scale);
    }
  }
  if (tid == 0) {
    scale_exponent[batch] = scale;
  }

  for (I i = tid; i < n; i += nthreads) {
    // Initialize D to identity so that `D^-1 A D = A`
    D_b[i] = 1;
    // Initialize perm to arange(n) so that `A[perm[i], perm[j]] == A[i, j]`
    perm_b[i] = i;
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Move rows that are zero off the diagonal within columns 0..high to row
  // high, then columns that are zero off the diagonal within rows low..high to
  // column low
  I low = 0;
  I high = n - 1;
  for (bool swapped = true; swapped && high > 0;) {
    swapped = false;
    for (I i = high; i >= 0 && high > 0; --i) {
      if (eig::is_isolated(
              A_b, n, i, I(0), high, true, column_scratch, tid, nthreads)) {
        eig::swap_indices(A_b, perm_b, n, i, high, tid, nthreads);
        --high;
        swapped = true;
      }
    }
  }
  for (bool swapped = true; swapped;) {
    swapped = false;
    for (I j = low; j <= high; ++j) {
      if (eig::is_isolated(
              A_b, n, j, low, high, false, column_scratch, tid, nthreads)) {
        eig::swap_indices(A_b, perm_b, n, j, low, tid, nthreads);
        ++low;
        swapped = true;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  bool converged = false;
  for (int sweep = 0; sweep < kMaxSweeps && !converged; ++sweep) {
    converged = true;
    for (I i = 0; i < n; ++i) {
      // 1-norms of column i and row i without the diagonal
      float column_part = 0;
      float row_part = 0;
      for (I m = tid; m < n; m += nthreads) {
        if (m != i) {
          column_part += eig::abs1(A_b[m + i * n]);
          row_part += eig::abs1(A_b[i + m * n]);
        }
      }
      const auto norms = c10::metal::threadgroup_sum2(
          column_scratch, row_scratch, column_part, row_part, tid, nthreads);
      // The next reduction must not overwrite scratch before all threads read
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float c = norms[0];
      float r = norms[1];
      if (c == 0 || r == 0) {
        continue;
      }
      // The power of 2, f, that brings c * f and r / f closest together
      const float s = c + r;
      float f = 1;
      float g = r / 2;
      while (c < g && max(f, c) < max_norm && min(r, g) > min_norm) {
        f *= 2;
        c *= 2;
        r /= 2;
        g /= 2;
      }
      g = c / 2;
      while (g >= r && r < max_norm && min(min(f, c), g) > min_norm) {
        f /= 2;
        c /= 2;
        g /= 2;
        r *= 2;
      }
      if (c + r >= 0.95f * s) {
        continue;
      }
      converged = false;
      if (tid == 0) {
        D_b[i] *= f;
      }
      for (I m = tid; m < n; m += nthreads) {
        if (m != i) {
          A_b[i + m * n] /= f;
          A_b[m + i * n] *= f;
        }
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
  }
}

// Reduces A to upper Hessenberg form H = P^* A P [Arbenz, Algorithm 4.3],
// storing each v_k[1:] in A[k+2:, k] and tau_k in `tau`, as described in Note
// [linalg.eig for MPS].
template <typename I>
kernel void eig_hessenberg(
    device float2* A [[buffer(0)]],
    device float2* tau [[buffer(1)]],
    constant EigParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]],
    uint batch [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const I n = static_cast<I>(params.n);
  const uint num_simd_groups = nthreads / c10::metal::simdgroup_size;
  device float2* A_b = A + static_cast<long>(batch) * n * n;
  device float2* tau_b = tau + static_cast<long>(batch) * (n - 2);
  threadgroup float scratch[32];

  for (I k = 0; k < n - 2; ++k) {
    // x = x_k = A[k+1:, k], which has m entries, and alpha = x_k[0]
    const I m = n - k - 1;
    device float2* x = A_b + (k + 1) + k * n;
    const float2 alpha = x[0];
    // ||x_k|| is computed with x_k scaled by 2^-exponent, near its largest
    // entry, so that the squares can't underflow or overflow, like LAPACK's
    // xNRM2. Scaling by a power of 2 is exact, so otherwise the result is the
    // same as without scaling.
    float tail_max = 0;
    for (I i = 1 + tid; i < m; i += nthreads) {
      tail_max = max(tail_max, max(fabs(x[i].x), fabs(x[i].y)));
    }
    tail_max = c10::metal::threadgroup_max(scratch, tail_max, tid, nthreads);
    int exponent;
    frexp(max(tail_max, max(fabs(alpha.x), fabs(alpha.y))), exponent);
    float tail_part = 0;
    for (I i = 1 + tid; i < m; i += nthreads) {
      const float2 x_i =
          float2(ldexp(x[i].x, -exponent), ldexp(x[i].y, -exponent));
      tail_part += dot(x_i, x_i);
    }
    // The previous reduction may still be reading scratch
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tail_norm2 =
        c10::metal::threadgroup_sum(scratch, tail_part, tid, nthreads);

    // P_k reflects x_k onto beta e_0, with beta = -e^{i arg(alpha)} ||x_k||, so
    // u_k'[0] = alpha - beta. Then v_k = u_k' / (alpha - beta), and tau_k
    // simplifies to 1 + |alpha| / ||x_k||.
    float t = 0;
    float2 beta = alpha;
    if (tail_max > 0) {
      const float alpha_abs = c10::metal::hypot(alpha.x, alpha.y);
      const float alpha_scaled = ldexp(alpha_abs, -exponent);
      const float x_norm = ldexp(
          precise::sqrt(alpha_scaled * alpha_scaled + tail_norm2), exponent);
      const float2 phase = alpha_abs > 0 ? alpha / alpha_abs : float2(1, 0);
      beta = -x_norm * phase;
      const float2 inv = c10::metal::div(float2(1, 0), alpha - beta);
      t = 1 + alpha_abs / x_norm;
      for (I i = 1 + tid; i < m; i += nthreads) {
        x[i] = c10::metal::mul(x[i], inv);
      }
      // Hold v_k[0] = 1 in x[0] while P_k is applied, then replace it with beta
      if (tid == 0) {
        x[0] = float2(1, 0);
      }
    }
    if (tid == 0) {
      tau_b[k] = float2(t, 0);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
      continue;
    }

    // A <- P_k A, with one SIMD group per column j > k
    for (I j = k + 1 + simd_group; j < n; j += num_simd_groups) {
      device float2* column = A_b + (k + 1) + j * n;
      float2 w = float2(0);
      for (I i = simd_lane; i < m; i += c10::metal::simdgroup_size) {
        w += c10::metal::mul(c10::metal::conj(x[i]), column[i]);
      }
      w = t * c10::metal::simd_sum(w);
      for (I i = simd_lane; i < m; i += c10::metal::simdgroup_size) {
        column[i] -= c10::metal::mul(x[i], w);
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // A <- A P_k, with one thread per row
    for (I r = tid; r < n; r += nthreads) {
      device float2* row = A_b + r + (k + 1) * n;
      float2 z = float2(0);
      for (I i = 0; i < m; ++i) {
        z += c10::metal::mul(row[i * n], x[i]);
      }
      z *= t;
      for (I i = 0; i < m; ++i) {
        row[i * n] -= c10::metal::mul(z, c10::metal::conj(x[i]));
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
      x[0] = beta;
    }
  }
}

// Reduces the Hessenberg H to upper triangular T = Q^* H Q. If
// params.compute_eigenvectors, updates U <- U Q.
// See eig::qr_iterate.
template <typename I>
kernel void eig_qr_iterate(
    device float2* H [[buffer(0)]],
    device float2* U [[buffer(1)]],
    device packed_float3* rotations [[buffer(2)]],
    device float2* eigenvalues [[buffer(3)]],
    device int* info [[buffer(4)]],
    device const int* scale_exponent [[buffer(5)]],
    constant EigParams& params [[buffer(6)]],
    threadgroup float2* H_staged [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  const I n = static_cast<I>(params.n);
  const bool update_u = params.compute_eigenvectors;
  const long matrix_offset = static_cast<long>(batch) * n * n;
  device float2* H_b = H + matrix_offset;
  device float2* U_b = U + (update_u ? matrix_offset : 0);
  device packed_float3* rotations_b = rotations + static_cast<long>(batch) * n;
  device float2* eigenvalues_b = eigenvalues + static_cast<long>(batch) * n;
  threadgroup int scratch[32];

  int status;
  if (params.stage_h) {
    // Keeping small matrices in threadgroup memory makes the barriers between
    // the rotations of chase_bulge cheaper
    for (I i = tid; i < n * n; i += nthreads) {
      H_staged[i] = H_b[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    status = eig::qr_iterate(
        H_staged, U_b, rotations_b, n, update_u, scratch, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (I i = tid; i < n * n; i += nthreads) {
      H_b[i] = H_staged[i];
    }
  } else {
    status = eig::qr_iterate(
        H_b, U_b, rotations_b, n, update_u, scratch, tid, nthreads);
  }
  threadgroup_barrier(mem_flags::mem_device);

  // Undo eig_balance's scaling
  for (I i = tid; i < n; i += nthreads) {
    eigenvalues_b[i] = eig::ldexp_exact(H_b[i + i * n], -scale_exponent[batch]);
  }
  if (tid == 0) {
    info[batch] = status;
  }
}

// Computes the eigenvectors of an upper triangular matrix T.  The output X is
// also upper triangular with T X = X diag(T), so each column k solves
// (T - T[k, k] I) X[:, k] = 0, scaled so that the largest |Re| + |Im| in each
// column is 1.
//
// Algorithm 10 of [Adlerborn et al.] computes X one row at a time from the
// bottom up. Each row j is divided by T[j, j] - T[k, k] in every column k > j,
// with a thread per column, and then T[:j, j] X[j, :] is subtracted from the
// rows above, with a thread per row.
//
// Overflow is prevented column by column as in RobustScalarBacksolve [Adlerborn
// et al., Algorithm 5]: before a division or update that could exceed kOmega,
// the column is scaled down by the factor from ProtectDivision or
// ProtectUpdate. The host normalizes the eigenvectors afterwards, so these
// factors don't need to be kept. Instead of recomputing the norm of the part of
// the column above row j after each update like Algorithm 5, we keep an upper
// bound of it, which is all ProtectUpdate needs. Magnitudes are measured with
// abs1, which bounds the real and imaginary parts of complex sums and products
// the way |.| does for real ones, so that Theorem 1 applies to complex numbers.
//
// Algorithm 10 assumes distinct eigenvalues. As in LAPACK's xTREVC, divisors
// T[j, j] - T[k, k] smaller than eps |T[k, k]| are increased to that size, so
// that repeated eigenvalues still give finite eigenvectors.
template <typename I>
kernel void eig_vectors(
    device const float2* T [[buffer(0)]],
    device float2* X [[buffer(1)]],
    device float* bounds [[buffer(2)]],
    constant EigParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  const I n = static_cast<I>(params.n);
  const long matrix_offset = static_cast<long>(batch) * n * n;
  device const float2* T_b = T + matrix_offset;
  device float2* X_b = X + matrix_offset;
  // x_bound[k] bounds the magnitudes of column k of X above the current row,
  // and column_norm[j] is the largest magnitude in T[:j, j]
  device float* x_bound = bounds + static_cast<long>(batch) * 2 * n;
  device float* column_norm = x_bound + n;

  for (I i = tid; i < n * n; i += nthreads) {
    X_b[i] = float2(0);
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (I k = tid; k < n; k += nthreads) {
    float norm = 0;
    for (I i = 0; i < k; ++i) {
      norm = max(norm, eig::abs1(T_b[i + k * n]));
    }
    column_norm[k] = norm;
    x_bound[k] = 0;
    X_b[k + k * n] = float2(1, 0);
  }
  threadgroup_barrier(mem_flags::mem_device);

  const float small_num = eig::kSafeMin * n;
  for (I j = n - 1; j >= 0; --j) {
    const float2 t_jj = T_b[j + j * n];
    for (I k = j + tid; k < n; k += nthreads) {
      device float2* x = X_b + k * n;
      float2 x_j = x[j];
      // The column is scaled by s, the product of both protection factors
      float s = 1;
      if (k > j) {
        const float2 t_kk = T_b[k + k * n];
        const float min_divisor = max(eig::kEps * eig::abs1(t_kk), small_num);
        float2 divisor = t_jj - t_kk;
        if (eig::abs1(divisor) < min_divisor) {
          divisor = float2(min_divisor, 0);
        }
        s = eig::protect_division(eig::abs1(x_j), eig::abs1(divisor));
        x_j = c10::metal::div(s * x_j, divisor);
      }
      const float s_update =
          eig::protect_update(s * x_bound[k], column_norm[j], eig::abs1(x_j));
      x_j *= s_update;
      s *= s_update;
      if (s != 1) {
        for (I i = 0; i <= k; ++i) {
          x[i] *= s;
        }
        x_bound[k] *= s;
      }
      x[j] = x_j;
      x_bound[k] += column_norm[j] * eig::abs1(x_j);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // X[:j, j:] -= T[:j, j] X[j, j:]
    for (I i = tid; i < j; i += nthreads) {
      const float2 t_ij = T_b[i + j * n];
      for (I k = j; k < n; ++k) {
        X_b[i + k * n] -= c10::metal::mul(t_ij, X_b[j + k * n]);
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  // Entries can be as large as kOmega, so scale each column to a largest
  // magnitude of 1 to let the host compute norms without overflow
  for (I k = tid; k < n; k += nthreads) {
    device float2* x = X_b + k * n;
    float largest = 0;
    for (I i = 0; i <= k; ++i) {
      largest = max(largest, eig::abs1(x[i]));
    }
    for (I i = 0; i <= k; ++i) {
      x[i] /= largest;
    }
  }
}

#define REGISTER_EIG_KERNELS(I)                                               \
  template [[host_name("eig_balance_" #I)]] kernel void eig_balance<I>(       \
      device float2*,                                                         \
      device float*,                                                          \
      device long*,                                                           \
      device int*,                                                            \
      constant EigParams&,                                                    \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("eig_hessenberg_" #I)]] kernel void eig_hessenberg<I>( \
      device float2*,                                                         \
      device float2*,                                                         \
      constant EigParams&,                                                    \
      uint,                                                                   \
      uint,                                                                   \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("eig_qr_iterate_" #I)]] kernel void eig_qr_iterate<I>( \
      device float2*,                                                         \
      device float2*,                                                         \
      device packed_float3*,                                                  \
      device float2*,                                                         \
      device int*,                                                            \
      device const int*,                                                      \
      constant EigParams&,                                                    \
      threadgroup float2*,                                                    \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("eig_vectors_" #I)]] kernel void eig_vectors<I>(       \
      device const float2*,                                                   \
      device float2*,                                                         \
      device float*,                                                          \
      constant EigParams&,                                                    \
      uint,                                                                   \
      uint,                                                                   \
      uint);

REGISTER_EIG_KERNELS(int);
REGISTER_EIG_KERNELS(long);
