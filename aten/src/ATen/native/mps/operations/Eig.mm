//  Copyright © 2022 Apple Inc.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/common.h>

#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/zeros.h>
#endif

#include <limits>
#include <string>

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Eig_metallib.h>
#endif
} // namespace

// Encodes one of the linalg.eig kernels, which run one threadgroup per matrix.
// Must be called on the stream's queue.
template <typename... Args>
static void encode_eig_kernel(MPSStream* stream,
                              const std::string& name,
                              const Tensor& profiled,
                              int64_t batch,
                              int64_t n,
                              int64_t threadgroup_memory,
                              const Args&... args) {
  auto encoder = stream->commandEncoder();
  const bool use_32bit_index = n * n <= std::numeric_limits<int32_t>::max();
  auto pso = lib.getPipelineStateForFunc(fmt::format("{}_{}", name, use_32bit_index ? "int" : "long"));
  getMPSProfiler().beginProfileKernel(pso, name, {profiled}, stream);
  [encoder setComputePipelineState:pso];
  mtl_setArgs(encoder, args...);
  if (threadgroup_memory > 0) {
    [encoder setThreadgroupMemoryLength:threadgroup_memory atIndex:0];
  }
  constexpr int64_t simd_size = c10::metal::simdgroup_size;
  const NSUInteger max_threads = pso.maxTotalThreadsPerThreadgroup / simd_size * simd_size;
  const NSUInteger threads = std::min<NSUInteger>(max_threads, ceil_div(n, simd_size) * simd_size);
  [encoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  getMPSProfiler().endProfileKernel(pso, stream);
}

// Note [linalg.eig for MPS]
//
// Calculates the eigenvalues and eigenvectors of the input, based on numerical
// linear algebra methods described in:
//
// [Arbenz] "Lecture Notes on Solving Large Scale Eigenvalue Problems", Chapter 4,
//   by Prof. Dr. Peter Arbenz:
//   <https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf>.
//
// [Adlerborn et al.] "Towards Highly Parallel and Compute-Bound Computation of
//   Eigenvectors of Matrices in Schur Form" by Adlerborn et al:
//   <https://webapps.cs.umu.se/uminf/reports/2017/010/part1.pdf>
//
// Eigenvalues are calculated using the following steps:
//
//   1. "Balance" the input by transforming it with `B = D^-1 (P_b)^-1 A P_b D`,
//      where D is a diagonal matrix and P_b is a permutation matrix. Based on
//      LAPACK xGEBAL: https://netlib.org/lapack/lug/node51.html
//
//      P_b is defined by `((P_b)^-1 A P_b)[i, j] = A[perm[i], perm[j]]`. The
//      purpose of the permutation is to attempt to upper-triangularize as many
//      of the rows/columns as possible, in order to limit the amount of
//      computations and error in the later steps.
//
//      The elements of D are chosen to make the norms of each row of B and its
//      corresponding column similar, which improves the accuracy of the final
//      output.
//
//   2. Reduce B to Hessenberg form, `H = P^* B P`, which is almost an upper
//      triangular matrix, except that the elements directly below the diagonal
//      can be nonzero. P is a product of householder reflector matrices, each
//      chosen to zero out the elements `row_index > column_index + 1` in one
//      column. A householder reflector is any matrix that can be written in
//      the form `I - 2 u u^*`, where `u` is a unit vector.
//
//      We choose one householder reflector for all but the last two columns of
//      `B`. We apply the householder reflector for each column `k` in sequence,
//      `H_{k+1} = (P_k)^* H_k P_k`, starting with `H_0 = B`. The householder
//      reflector will zero out the elements `H_k[k+2:, k]`, which is done like
//      so:
//
//        x_k = H_k[k+1:, k]
//        if |x_k[0]| == 0:
//          u_k'[0] = ||x_k||
//        else:
//          u_k'[0] = x_k[0] (1 + ||x_k|| / |x_k[0]|)
//        u_k'[1:] = x_k[1:]
//        u_k = u_k' / ||u_k'||
//        P_k' = I - 2 u_k (u_k)^*
//
//      If `B` is an n x n matrix, `P_k'` is (n-k-1) x (n-k-1), so we pad the
//      first `k + 1` dimension with an identity matrix:
//
//        P_k = pad(P_k') = I  0
//                          0  P_k'
//
//      However we avoid forming the entire `P_k` matrix and calculating the two
//      full matrix multiplies, `(P_k)^* H_k P_k`. Instead, we simplify it to
//      two matrix-vector products and two vector outer products, based on the
//      fact that `H_k P_k = H_k (I - 2 u u^*) = H_k - 2 H_k u u^*`.
//
//   3. Perform Hessenberg QR iteration to convert H into an upper triangular
//      matrix T. The method we use is based on the following "explicit"
//      Hessenberg QR iteration with a Wilkinson shift:
//
//        Each iteration aims to bring one of the elements, called the target
//        element, below the diagonal closer to 0. It starts with the
//        bottom-rightmost subdiagonal element, iterates until it is near zero,
//        and moves on to the next element to the above-left. Each iteration
//        applies the transformation:
//
//          Q, R = decompQR(H[:m, :m] - μI)
//          H[:m, :m] := R Q + μI
//
//        `H[m-1, m-2]` marks the target element, and `m` is decremented each
//        time the current target element reaches near zero. `μ` is a Wilkinson
//        shift value. To calculate it, we take the 2x2 matrix at the bottom
//        right corner of `H[:m, :m]`, which is `H[m-2:m, m-2:m]`, calculate its
//        two eigenvalues and choose whichever one is closest to the bottom
//        right element, `H[m-1, m-1]`.
//
//      However, we actually avoid calculating the QR decomposition directly.
//      Instead we apply a series of Givens rotations which mathematically gives
//      the same result as one iteration of explicit Hessenberg QR iteration
//      with less computational work.
//
// The final output, T, is upper triangular, so its diagonal elements are its
// eigenvalues. Since T was obtained only by applying similarity
// transformations, which preserve the eigenvalues, its diagonal elements are
// the eigenvalues of the original input A.
//
// If `compute_eigenvectors` is turned on, we compute the Schur vectors of B,
// which are the columns of the matrix `(prod_k P_k) (prod_j Q_j)`, where `P_k`
// is one of the padded householder reflectors and `Q_j` is one of the Q
// matrices from QR decomposition during QR iteration. Then we convert the Schur
// vectors into the eigenvectors of B and undo the balancing we did at the very
// beginning, and finally we normalize the eigenvectors.
//
// This is accomplished in the following steps:
//
//   1. After the reduction of B to Hessenberg form, recover the product of
//      the householder reflectors, since we didn't calculate it explicitly.
//      This is done efficiently with `at::linalg_householder_product`, since it
//      avoids explicitly forming each reflector matrix. Given a matrix
//      containing `v_k` and vector containing `tau_k` for each iteration k of
//      the Hessenberg reduction, it calculates:
//
//        P = prod_k pad(P_k')
//          = prod_k pad(I - 2 u_k (u_k)^*)
//          = prod_k pad(I - tau_k v_k (v_k)^*)
//
//      `v_k` is defined as a vector where `v_k[0] = 1`. In terms of `u_k'` from
//      earlier:
//
//        v_k = u_k' / u_k'[0]
//        tau_k = 2 |u_k'[0]|^2 / ||u_k'||^2
//
//      `tau_k` and `v_k` are both calculated during the Hessenberg reduction
//      step itself.  `tau_k` is stored in its own tensor, but `v_k[1:]` is
//      actually written into `H[k+2:, k]`, the elements of H that were zeroed
//      out. Since those elements are known to be zero, they are reused rather
//      than creating a separate tensor for `v_k`. We don't need to keep track
//      of `v_k[0]` since it is always 1.
//
//      After the Householder product is calculated, we zero out `H[k+2:, k]`,
//      since we no longer want `v_k` to be stored there.
//
//   2. During each step `k` of Hessenberg QR iteration, accumulate:
//
//        U = P prod_k Q_k
//
//   3. The columns of U are now the Schur vectors of B. In order to find the
//      eigenvectors of B, we first find the eigenvectors of T (the upper
//      triangular matrix output of QR iteration) and put them into the columns
//      of a matrix X. Calculating the eigenvectors of a triangular matrix is a
//      special case that is much simpler than that of the original dense matrix
//      A. Algorithm 1 in [Adlerborn et al.] describes how to find a single
//      eigenvector of T. It simplifies down to the following:
//
//      An eigenvector `x` for the eigenvalue `T[s, s]` can be constructed one
//      at a time by:
//        x[s+1:] = 0
//        x[s] = any nonzero value
//        for i in [s-1, s-2, ..., 0]:
//            x[i] = T[i, i+1:s+1].dot(x[i+1:s+1]) / (T[s, s] - T[i, i])
//
//      Our implementation is based on Algorithm 10 from [Adlerborn et al.],
//      which is mathematically equivalent to the above, but it computes all the
//      eigenvectors simultaneously and handles repeated eigenvalues (cases
//      where T[s, s] == T[i, i]).
//
//   4. Calculate the eigenvectors of B, undo the initial balancing, and undo
//      the permutation:
//
//        V' = eigenvectors(A) = P_b D eigenvectors(B) = P_b D U X
//
//   5. Finally, normalize the eigenvectors. We rotate the complex values of
//      each eigenvector so that the largest element of the output is real and
//      positive.
//
//        j = argmax(|V'[:, k]|)
//        m = V'[j, k]
//        V"[:, k] = V'[:, k] (m)^*
//        V[:, k] = V"[:, k] / ||V"[:, k]||
//
// The operation uses four specialized Metal kernels to do much of the work:
//  eig_balance     Balance the input, `B = D^-1 A D`.
//  eig_hessenberg  Reduce B to Hessenberg form, `H = P^* B P`.
//  eig_qr_iterate  Apply the Hessenberg QR algorithm to convert H to triangular
//                  form T.
//  eig_vectors     Obtain the eigenvectors X of T.
static void linalg_eig_kernel_mps(Tensor& eigenvalues,
                                  Tensor& eigenvectors,
                                  Tensor& infos,
                                  const Tensor& input,
                                  bool compute_eigenvectors) {
  using namespace mps;
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() == kComplexFloat,
                              "\"linalg_eig_kernel_mps\" not implemented for '",
                              input.scalar_type(),
                              "'");
  if (input.numel() == 0) {
    return;
  }
  const int64_t n = input.size(-1);
  const int64_t batch = batchCount(input);
  const auto options = input.options();

  const int64_t h_bytes = n * n * static_cast<int64_t>(sizeof(c10::complex<float>));
  const int64_t threadgroup_limit =
      static_cast<int64_t>([MPSDevice::getInstance()->device() maxThreadgroupMemoryLength]);
  // eig_qr_iterate keeps H in threadgroup memory if it fits next to the kernel's
  // own threadgroup variables
  constexpr int64_t kReservedThreadgroupBytes = 1024;
  const bool stage_h = h_bytes + kReservedThreadgroupBytes <= threadgroup_limit;
  const EigParams params{n, compute_eigenvectors, stage_h};

  // Working copy of A, which will be overwritten with B, then H, then T.
  Tensor H = cloneBatchedColumnMajor(input.resolve_conj().resolve_neg()).reshape({batch, n, n});
  // D is a diagonal matrix, so we only track the diagonal entries
  Tensor D = at::empty({batch, n}, options.dtype(kFloat));
  Tensor perm = at::empty({batch, n}, options.dtype(kLong));
  Tensor scale_exponent = at::empty({batch}, options.dtype(kInt));
  Tensor tau = at::empty({batch, std::max<int64_t>(n - 2, 0)}, options);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode_eig_kernel(stream, "eig_balance", H, batch, n, 0, H, D, perm, scale_exponent, params);
      if (n > 2) {
        encode_eig_kernel(stream, "eig_hessenberg", H, batch, n, 0, H, tau, params);
      }
    }
  });

  Tensor U;
  if (compute_eigenvectors) {
    // U starts as P. Dropping the first row of H puts each implied v_k[0] = 1
    // on the diagonal, where linalg_householder_product expects it.
    U = at::zeros({batch, n, n}, options).mT();
    U.select(-1, 0).select(-1, 0).fill_(1);
    if (n > 1) {
      auto reflectors = H.narrow(-2, 1, n - 1).narrow(-1, 0, n - 1);
      U.narrow(-2, 1, n - 1).narrow(-1, 1, n - 1).copy_(at::linalg_householder_product(reflectors, tau));
    }
  }
  // Remove the v_k stored below the subdiagonal
  H.triu_(-1);

  Tensor rotations = at::empty({batch, n, 3}, options.dtype(kFloat));
  Tensor U_arg = compute_eigenvectors ? U : at::empty({1}, options);
  const int64_t threadgroup_memory = stage_h ? ceil_div<int64_t>(h_bytes, 16) * 16 : 16;
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode_eig_kernel(stream,
                        "eig_qr_iterate",
                        H,
                        batch,
                        n,
                        threadgroup_memory,
                        H,
                        U_arg,
                        rotations,
                        eigenvalues,
                        infos,
                        scale_exponent,
                        params);
    }
  });
  if (!compute_eigenvectors) {
    return;
  }

  Tensor X = at::empty({batch, n, n}, options).mT();
  Tensor bounds = at::empty({batch, 2 * n}, options.dtype(kFloat));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      encode_eig_kernel(stream, "eig_vectors", H, batch, n, 0, H, X, bounds, params);
    }
  });

  Tensor V = at::matmul(U, X).mul_(D.unsqueeze(-1));
  // Undo eig_balance's permutation: row i of V is row perm[i] of the result
  V = at::empty_like(V).scatter_(-2, perm.unsqueeze(-1).expand_as(V), V);
  // Multiplying by m^* / |m| / |m| instead of just m^* keeps the norm
  // computation from overflowing however D scales the columns. conj_physical
  // is used because a lazy conj is ignored when the Conjugate dispatch key is
  // excluded, as it is under torch.compile.
  Tensor largest = V.gather(-2, V.abs().argmax(-2, /*keepdim=*/true));
  Tensor largest_abs = largest.abs();
  V.mul_(largest.conj_physical().div_(largest_abs).div_(largest_abs));
  V.div_(at::linalg_vector_norm(V, 2, {-2}, /*keepdim=*/true));
  eigenvectors.copy_(V.view(eigenvectors.sizes()));
}

} // namespace mps

REGISTER_DISPATCH(linalg_eig_stub, mps::linalg_eig_kernel_mps);

} // namespace at::native
