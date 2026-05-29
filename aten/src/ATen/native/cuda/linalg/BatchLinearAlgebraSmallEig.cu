// Batched small-matrix eig kernel (n <= 3), one thread per matrix.
// Avoids cuSolver Xgeev's sequential per-matrix launch overhead.
// https://github.com/pytorch/pytorch/issues/183806
//
// Eigenvalue layout follows LAPACK dgeev: [re_0..re_{n-1}, im_0..im_{n-1}].
// Eigenvectors are column-major, LAPACK dgeev convention for complex pairs.

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>

namespace at::native {

namespace {

// solve cubic x^3 + a*x^2 + b*x + c = 0 using trigonometric method for 3 real roots,
// or Cardano's formula for mixed real/complex roots
template <typename T>
__device__ void solve_cubic(T a, T b, T c, T* re, T* im) {
  T p = b - a * a / T(3);
  T q = T(2) * a * a * a / T(27) - a * b / T(3) + c;
  T disc = q * q / T(4) + p * p * p / T(27);

  if (disc <= T(0)) {
    // three real roots (Viete's trigonometric solution)
    T m = sqrt(fmax(-p / T(3), T(0)));
    T theta = atan2(sqrt(fmax(-disc, T(0))), -q / T(2)) / T(3);
    T shift = -a / T(3);
    re[0] = shift + T(2) * m * cos(theta);
    re[1] = shift + T(2) * m * cos(theta - T(2) * M_PI / T(3));
    re[2] = shift + T(2) * m * cos(theta + T(2) * M_PI / T(3));
    im[0] = im[1] = im[2] = T(0);
  } else {
    // one real root + complex conjugate pair (Cardano)
    T sd = sqrt(disc);
    T u = cbrt(-q / T(2) + sd);
    T v = cbrt(-q / T(2) - sd);
    T shift = -a / T(3);
    re[0] = shift + u + v;
    im[0] = T(0);
    re[1] = shift - (u + v) / T(2);
    im[1] = sqrt(T(3)) / T(2) * (u - v);
    re[2] = re[1];
    im[2] = -im[1];
  }
}

// compute eigenvector for a real eigenvalue of a real 3x3 matrix
// A is column-major, eigenvector written to v (length 3)
template <typename T>
__device__ void eigvec_real_3x3(const T* A, T lambda, T* v) {
  T B[9];
  for (int i = 0; i < 9; i++) B[i] = A[i];
  B[0] -= lambda;
  B[4] -= lambda;
  B[8] -= lambda;

  // rows of B (column-major: row i = [B[i], B[i+3], B[i+6]])
  T r0[3] = {B[0], B[3], B[6]};
  T r1[3] = {B[1], B[4], B[7]};
  T r2[3] = {B[2], B[5], B[8]};

  // cross products of pairs of rows, pick largest norm for stability
  T c01[3] = {r0[1]*r1[2] - r0[2]*r1[1], r0[2]*r1[0] - r0[0]*r1[2], r0[0]*r1[1] - r0[1]*r1[0]};
  T c02[3] = {r0[1]*r2[2] - r0[2]*r2[1], r0[2]*r2[0] - r0[0]*r2[2], r0[0]*r2[1] - r0[1]*r2[0]};
  T c12[3] = {r1[1]*r2[2] - r1[2]*r2[1], r1[2]*r2[0] - r1[0]*r2[2], r1[0]*r2[1] - r1[1]*r2[0]};

  T n01 = c01[0]*c01[0] + c01[1]*c01[1] + c01[2]*c01[2];
  T n02 = c02[0]*c02[0] + c02[1]*c02[1] + c02[2]*c02[2];
  T n12 = c12[0]*c12[0] + c12[1]*c12[1] + c12[2]*c12[2];

  T* best = c01;
  T nbest = n01;
  if (n02 > nbest) { best = c02; nbest = n02; }
  if (n12 > nbest) { best = c12; nbest = n12; }

  T norm = sqrt(nbest);
  if (norm > T(0)) {
    v[0] = best[0] / norm;
    v[1] = best[1] / norm;
    v[2] = best[2] / norm;
  } else {
    v[0] = T(1); v[1] = T(0); v[2] = T(0);
  }
}

template <typename scalar_t>
__global__ void batched_small_eig_real_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ values,
    scalar_t* __restrict__ vectors,
    int* __restrict__ infos,
    int64_t batch_size,
    int n,
    int64_t input_stride,
    int64_t values_stride,
    int64_t vectors_stride,
    bool compute_eigenvectors) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) return;

  const scalar_t* A = input + idx * input_stride;
  scalar_t* W = values + idx * values_stride;
  scalar_t* V = compute_eigenvectors ? vectors + idx * vectors_stride : nullptr;
  infos[idx] = 0;

  if (n == 1) {
    W[0] = A[0];
    W[1] = scalar_t(0);
    if (compute_eigenvectors) V[0] = scalar_t(1);
    return;
  }

  if (n == 2) {
    scalar_t a00 = A[0], a10 = A[1], a01 = A[2], a11 = A[3];
    scalar_t tr = a00 + a11;
    scalar_t det = a00 * a11 - a01 * a10;
    scalar_t disc = tr * tr - scalar_t(4) * det;

    if (disc >= scalar_t(0)) {
      scalar_t sd = sqrt(disc);
      W[0] = (tr + sd) / scalar_t(2);
      W[1] = (tr - sd) / scalar_t(2);
      W[2] = scalar_t(0);
      W[3] = scalar_t(0);

      if (compute_eigenvectors) {
        for (int k = 0; k < 2; k++) {
          scalar_t lam = W[k];
          scalar_t b00 = a00 - lam, b01 = a01, b10 = a10, b11 = a11 - lam;
          if (fabs(b00) + fabs(b01) >= fabs(b10) + fabs(b11)) {
            if (fabs(b00) >= fabs(b01)) {
              scalar_t t = -b01 / (fabs(b00) > scalar_t(0) ? b00 : scalar_t(1));
              scalar_t norm = sqrt(t * t + scalar_t(1));
              V[k * 2 + 0] = t / norm;
              V[k * 2 + 1] = scalar_t(1) / norm;
            } else {
              scalar_t t = -b00 / b01;
              scalar_t norm = sqrt(scalar_t(1) + t * t);
              V[k * 2 + 0] = scalar_t(1) / norm;
              V[k * 2 + 1] = t / norm;
            }
          } else {
            if (fabs(b10) >= fabs(b11)) {
              scalar_t t = -b11 / (fabs(b10) > scalar_t(0) ? b10 : scalar_t(1));
              scalar_t norm = sqrt(t * t + scalar_t(1));
              V[k * 2 + 0] = t / norm;
              V[k * 2 + 1] = scalar_t(1) / norm;
            } else {
              scalar_t t = -b10 / b11;
              scalar_t norm = sqrt(scalar_t(1) + t * t);
              V[k * 2 + 0] = scalar_t(1) / norm;
              V[k * 2 + 1] = t / norm;
            }
          }
        }
      }
    } else {
      scalar_t sd = sqrt(-disc);
      W[0] = tr / scalar_t(2);
      W[1] = tr / scalar_t(2);
      W[2] = sd / scalar_t(2);
      W[3] = -sd / scalar_t(2);

      if (compute_eigenvectors) {
        scalar_t lam_re = W[0], lam_im = W[2];
        scalar_t b00 = a00 - lam_re;
        scalar_t vr0, vr1, vi0, vi1;
        if (fabs(a01) > scalar_t(0)) {
          vr0 = scalar_t(1); vr1 = -b00 / a01;
          vi0 = scalar_t(0); vi1 = lam_im / a01;
        } else if (fabs(a10) > scalar_t(0)) {
          scalar_t b11 = a11 - lam_re;
          vr0 = -b11 / a10; vr1 = scalar_t(1);
          vi0 = lam_im / a10; vi1 = scalar_t(0);
        } else {
          vr0 = scalar_t(1); vr1 = scalar_t(0);
          vi0 = scalar_t(0); vi1 = scalar_t(1);
        }
        scalar_t norm = sqrt(vr0*vr0 + vr1*vr1 + vi0*vi0 + vi1*vi1);
        if (norm > scalar_t(0)) {
          V[0] = vr0/norm; V[1] = vr1/norm;
          V[2] = vi0/norm; V[3] = vi1/norm;
        } else {
          V[0] = scalar_t(1); V[1] = scalar_t(0);
          V[2] = scalar_t(0); V[3] = scalar_t(1);
        }
      }
    }
    return;
  }

  if (n == 3) {
    scalar_t a0 = A[0], a1 = A[1], a2 = A[2];
    scalar_t a3 = A[3], a4 = A[4], a5 = A[5];
    scalar_t a6 = A[6], a7 = A[7], a8 = A[8];

    scalar_t tr = a0 + a4 + a8;
    scalar_t cof = (a0*a4 - a3*a1) + (a0*a8 - a6*a2) + (a4*a8 - a7*a5);
    scalar_t det = a0*(a4*a8 - a7*a5) - a1*(a3*a8 - a6*a5) + a2*(a3*a7 - a6*a4);

    scalar_t re[3], im_parts[3];
    solve_cubic(-tr, cof, -det, re, im_parts);

    W[0] = re[0]; W[1] = re[1]; W[2] = re[2];
    W[3] = im_parts[0]; W[4] = im_parts[1]; W[5] = im_parts[2];

    if (compute_eigenvectors) {
      scalar_t a_arr[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
      bool has_complex = (im_parts[0] != scalar_t(0) || im_parts[1] != scalar_t(0) || im_parts[2] != scalar_t(0));

      if (!has_complex) {
        for (int k = 0; k < 3; k++) {
          eigvec_real_3x3(a_arr, re[k], &V[k * 3]);
        }
      } else {
        eigvec_real_3x3(a_arr, re[0], &V[0]);

        // Complex eigenvector via cross product of rows of C = B - i*lam_im*I,
        // where B = A - lam_re*I. Decomposed into real arithmetic.
        scalar_t lam_re = re[1], lam_im = im_parts[1];
        scalar_t B[9] = {a0 - lam_re, a1, a2, a3, a4 - lam_re, a5, a6, a7, a8 - lam_re};

        scalar_t Br0[3] = {B[0], B[3], B[6]};
        scalar_t Br1[3] = {B[1], B[4], B[7]};
        scalar_t Br2[3] = {B[2], B[5], B[8]};

        // Helper: cross(a,b) for 3-vectors
        #define CROSS(a, b, out) do { \
          out[0] = a[1]*b[2] - a[2]*b[1]; \
          out[1] = a[2]*b[0] - a[0]*b[2]; \
          out[2] = a[0]*b[1] - a[1]*b[0]; \
        } while(0)

        // Compute complex cross products for all 3 row pairs, pick largest.
        // For rows j,k of C = (Br + i*Bi):
        //   real = Br_j x Br_k - Bi_j x Bi_k
        //   imag = Br_j x Bi_k + Bi_j x Br_k
        // where Bi_k = -lam_im * e_k (only diagonal is nonzero).
        // The Bi x Bi terms reduce to lam_im^2 * (e_j x e_k) and the
        // mixed terms simplify since each Bi_k has a single nonzero entry.
        scalar_t best_vr[3], best_vi[3];
        scalar_t best_norm2 = scalar_t(-1);

        #define CROSS(a, b, out) do { \
          out[0] = a[1]*b[2] - a[2]*b[1]; \
          out[1] = a[2]*b[0] - a[0]*b[2]; \
          out[2] = a[0]*b[1] - a[1]*b[0]; \
        } while(0)

        #define TRY_PAIR(vr, vi) do { \
          scalar_t n2 = vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]+ \
                        vi[0]*vi[0]+vi[1]*vi[1]+vi[2]*vi[2]; \
          if (n2 > best_norm2) { \
            best_norm2 = n2; \
            for (int j = 0; j < 3; j++) { best_vr[j] = vr[j]; best_vi[j] = vi[j]; } \
          } \
        } while(0)

        // Pair (0,1): Bi0 x Bi1 = lam_im^2 * e2
        {
          scalar_t cr_re[3], cr_im[3];
          CROSS(Br0, Br1, cr_re);
          cr_re[2] -= lam_im * lam_im;
          cr_im[0] = lam_im*Br0[2];
          cr_im[1] = lam_im*Br1[2];
          cr_im[2] = -lam_im*(Br0[0] + Br1[1]);
          TRY_PAIR(cr_re, cr_im);
        }
        // Pair (0,2): Bi0 x Bi2 = -lam_im^2 * e1
        {
          scalar_t cr_re[3], cr_im[3];
          CROSS(Br0, Br2, cr_re);
          cr_re[1] += lam_im * lam_im;
          cr_im[0] = -lam_im*Br0[1];
          cr_im[1] = lam_im*(Br0[0] + Br2[2]);
          cr_im[2] = -lam_im*Br2[1];
          TRY_PAIR(cr_re, cr_im);
        }
        // Pair (1,2): Bi1 x Bi2 = lam_im^2 * e0
        {
          scalar_t cr_re[3], cr_im[3];
          CROSS(Br1, Br2, cr_re);
          cr_re[0] -= lam_im * lam_im;
          cr_im[0] = -lam_im*(Br1[1] + Br2[2]);
          cr_im[1] = lam_im*Br1[0];
          cr_im[2] = lam_im*Br2[0];
          TRY_PAIR(cr_re, cr_im);
        }

        #undef CROSS
        #undef TRY_PAIR

        // normalize
        scalar_t cnorm = sqrt(best_norm2);
        if (cnorm > scalar_t(0)) {
          for (int r = 0; r < 3; r++) {
            best_vr[r] /= cnorm;
            best_vi[r] /= cnorm;
          }
        }

        // LAPACK convention: col 1 = real part, col 2 = imag part
        for (int r = 0; r < 3; r++) {
          V[r + 1*3] = best_vr[r];
          V[r + 2*3] = best_vi[r];
        }
      }
    }
    return;
  }
}

} // anonymous namespace

void linalg_eig_batched_small_real(
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    const Tensor& input,
    const Tensor& infos,
    bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linalg_eig_batched_small", [&] {
    auto n = input.size(-1);
    auto batch_size = batchCount(input);

    Tensor A_fortran = input.mT().contiguous();
    auto input_stride = matrixStride(A_fortran);
    auto values_stride = eigenvalues.size(-1);
    auto vectors_stride = compute_eigenvectors ? matrixStride(eigenvectors) : 0;

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    batched_small_eig_real_kernel<scalar_t><<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        A_fortran.const_data_ptr<scalar_t>(),
        eigenvalues.data_ptr<scalar_t>(),
        compute_eigenvectors ? eigenvectors.data_ptr<scalar_t>() : nullptr,
        infos.data_ptr<int>(),
        batch_size,
        static_cast<int>(n),
        input_stride,
        values_stride,
        vectors_stride,
        compute_eigenvectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

} // namespace at::native
