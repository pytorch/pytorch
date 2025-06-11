#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <c10/util/Exception.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>
#include <ATen/native/cpu/zmath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_linalg_check_errors.h>
#endif

namespace at::native {
#if defined(BUILD_LAZY_CUDA_LINALG)
// All registrations with PyTorch runtime should be done dynamically
// so if library is lazy loaded it must not export anything, otherwise
// it can result in symbol clashes
namespace lazy_linalg {
#endif

#define ALLOCATE_ARRAY(name, type, size) \
  auto storage_##name = pin_memory<type>(size); \
  name = static_cast<type*>(storage_##name.mutable_data());

namespace {

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {

  return ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_solve: complex tensors with hermitian=True flag are not supported on CUDA.");
  }

  ldl_solve_cusolver(LD, pivots, B, upper);
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
  return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
      cholesky_helper_cusolver(input, upper, info);
}

REGISTER_CUDA_DISPATCH(cholesky_stub, &cholesky_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& cholesky_inverse_kernel_impl(Tensor &result, Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_cholesky_inverse'
  return cholesky_inverse_kernel_impl_cusolver(result, infos, upper);
}


REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

static void lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  auto batch_size = batchCount(input);
  (void) batch_size; // Silence unused warning in some builds
  auto m = input.size(-2);
  auto n = input.size(-1);

#ifdef USE_LINALG_SOLVER
  const auto lu_factor_cusolver = [batch_size, m, n](const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
    // In CUDA 10.2, lu_factor_looped_cusolver does not finish the computations when the input
    // matrix is exactly singular. The returned pivots contain garbage. This breaks linalg.det
    // Now, batched_cublas does not handle rectangular matrices, so we still dispatch to
    // looped_cusolver even if m != n.
#ifdef USE_ROCM
    constexpr bool looped_correct = true;
#else
    constexpr bool looped_correct = CUSOLVER_VERSION >= 11100;
#endif
    if (m != n || (looped_correct && (batch_size == 1 || m >= 512))) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
  };
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#endif // ifdef USE_LINALG_SOLVER
  
  // We return the trivial permutation of pivots starting with 1 (FORTRAN indexing)
  if (!compute_pivots) {
    auto k = std::min(input.size(-2), input.size(-1));
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    pivots.copy_(pivots_tmp);
  }
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // For batches smaller than 8 and matrix sizes larger than 64x64 cuBLAS forloop is faster than batched version
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_cublas(A, B, left, upper, transpose, unitriangular);
  } else {
    triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
  }
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
#ifdef USE_LINALG_SOLVER
  return orgqr_helper_cusolver(result, tau); // cusolver
#else
  TORCH_CHECK(false, "Calling torch.orgqr on a CUDA tensor requires compiling ",
    "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#ifdef USE_LINALG_SOLVER
  ormqr_cusolver(input, tau, other, left, transpose);
#else
  TORCH_CHECK(false,
      "Calling torch.ormqr on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_CUDA_DISPATCH(ormqr_stub, &ormqr_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

void geqrf_kernel(const Tensor& input, const Tensor& tau) {
  auto geqrf_cusolver_backend = [](const Tensor& input, const Tensor& tau) {
      // For the benchmarks see
      // https://github.com/pytorch/pytorch/pull/56253#discussion_r622851107
      if (input.size(-2) <= 256 && batchCount(input) >= std::max<int64_t>(2, input.size(-2) / 16)) {
        return geqrf_batched_cublas(input, tau);
      } else {
        return geqrf_cusolver(input, tau);
      }
      return geqrf_batched_cublas(input, tau);
  };

  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
    default:
      return geqrf_cusolver_backend(input, tau);
  }
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(geqrf_stub, &geqrf_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
    default:
      linalg_eigh_cusolver(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

/*
Computes the eigenvalues and eigenvectors of n-by-n matrix 'input'.
This is an in-place routine, content of 'input', 'values', 'vectors' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
*/
template <typename scalar_t>
void apply_linalg_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
  TORCH_CHECK(false, "not supported");
}

// This is a type dispatching helper function for 'apply_linalg_eig'
void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  // This function calculates the non-symmetric eigendecomposition in-place
  // tensors should be in batched column major memory format
  // the content of eigenvalues, eigenvectors and infos is overwritten by 'apply_linalg_eig'

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_cuda());
  Tensor input_working_copy = at::empty(input.sizes(), input.options().device(kCPU));
  input_working_copy.transpose_(-2, -1);  // make input_working_copy to have Fortran contiguous memory layout
  input_working_copy.copy_(input);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cuda", [&]{
    apply_linalg_eig<scalar_t>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
  });
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const std::optional<std::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& info) {
    // svd_cusolver computes V rather than Vh, so we pass a view of Vh.mT
    // and then conjugate Vh in-place
    svd_cusolver(A, full_matrices, compute_uv, driver, U, S, compute_uv ? Vh.mT() : Vh, info);
    if (compute_uv && Vh.is_complex()) {
      Vh._set_conj(!Vh.is_conj());
    }
  }
} // anonymous namespace

REGISTER_CUDA_DISPATCH(svd_stub, &svd_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

c10::MaybeOwned<Tensor> maybe_expand_lu(const Tensor& B, const Tensor& LU) {
  // B and LU have the same number of dimensions
  if (batchCount(B) != batchCount(LU)) {
        auto n = B.dim();
    auto expand_shape = DimVector(B.sizes().slice(0, n - 2));
    expand_shape.append({LU.size(-2), LU.size(-1)});
    return c10::MaybeOwned<Tensor>::owned(
        cloneBatchedColumnMajor(LU.expand(expand_shape)));
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(LU);
  }
}

c10::MaybeOwned<Tensor> maybe_expand_pivots(const Tensor& B, const Tensor& pivots) {
  // B and pivots have the same number of dimensions
  if (batchCount(B) != batchCount(pivots.unsqueeze(-1))) {
    auto expand_shape = DimVector(B.sizes().slice(0, B.dim() - 2));
    expand_shape.push_back(pivots.size(-1));
    return c10::MaybeOwned<Tensor>::owned(pivots.expand(expand_shape).contiguous());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(pivots);
  }
}

static void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Trivial case. Remove it once `torch.solve` is removed, as linalg.solve already shortcuts this case
  if (B.numel() == 0) {
    return;
  }

  auto b = batchCount(B);
  auto n = LU.size(-2);
  auto k = B.size(-1);
  auto lu_solve_triangular = [n](const Tensor& LU, const Tensor& pivots, const Tensor& B, const TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    // LAPACK / cublas / etc returns the permutation in an odd format
    // Here we transform it to a vector representing a permutation, i.e. a (batch of) vectors st. P(i) = j
    auto perm = at::arange(n, pivots_->options().dtype(kLong)).expand(pivots_->sizes()).contiguous();
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .declare_static_shape(pivots_->sizes(), /*squash_dim=*/pivots_->dim() - 1)
      .add_output(perm)
      .add_const_input(*pivots_)
      .build();
    unpack_pivots_stub(pivots_->device().type(), iter, n, n);

    if (trans == TransposeType::NoTranspose) {
      // Get the inverse permutation
      // This is an insertion sort, and it's equivalent to
      // perm = at::argsort(perm);
      // but more parallelisable and O(n), exploiting that perm is a permutation
      auto id_perm = at::arange(n, perm.options()).expand(perm.sizes());
      auto inv_perm = perm.scatter(-1, perm, id_perm);
      // B1 = P^T @ B  (must be done out-of-place as B is both source and target)
      auto B1 = B.scatter(-2, inv_perm.unsqueeze(-1).expand_as(B), B);
      // B = L^{-1} @ B1
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B1, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
      // B = U^{-1} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B, /*upper=*/true);
    } else {
      auto LU_H = LU_->mH();
      // B = U^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/false);
      // B = L^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/true, /*left=*/true, /*unitriangular=*/true);
      // B = P @ B
      B.scatter_(-2, perm.unsqueeze(-1).expand_as(B), B.clone());
    }
  };

#ifdef USE_LINALG_SOLVER
  auto lu_solve_batched_cublas_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    lu_solve_batched_cublas(*LU_, *pivots_, B, trans);
  };
#endif


    if (b <= 2 && n >= 64) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    }
    return;


  // Heuristic
  //if (n == k) {
  // if (k <= 16) batched_cublas
  // else solve_triag
  //} else {
  //if (n <= 8) {
  // batched_cusolver
  //} else if (n <= 32) {
  //  b <= 2 looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 64) {
  //  b <= 2 && (k <= 64 || adjoint) looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 128) {
  //  if (b <= 2 && k <= 2) looped_cusolver
  //  else if (k <= 2) batched_cusolver
  //  else solve_triag
  //} else { // n > 128
  //  solve_triag
  //}
  //}


#ifdef USE_LINALG_SOLVER
  // Particular case when multiplying A^{-1}B where B is square
  // In this case doing two triangular solves is almost always fastest
  if (n == k) {
    if (n <= 16) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
      return;
    }
    lu_solve_triangular(LU, pivots, B, trans);
    return;
  }

if (n <= 8) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
} else if (n <= 64) {
  if (b <= 2 && (k <= 64 || trans != TransposeType::NoTranspose || n <= 32)) {
    lu_solve_looped_cusolver(LU, pivots, B, trans);
  } else if (k <= 8) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else {
    lu_solve_triangular(LU, pivots, B, trans);
  }
} else if (n <= 128) {
  if (b <= 2 && k <= 2)  {
    lu_solve_looped_cusolver(LU, pivots, B, trans);
  } else if (k <= 2)  {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else {
    lu_solve_triangular(LU, pivots, B, trans);
  }
} else { // n > 128
  lu_solve_triangular(LU, pivots, B, trans);
}
#else
  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef USE_LINALG_SOLVER
}

} // anonymous namespace


REGISTER_CUDA_DISPATCH(lu_solve_stub, &lu_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

void linalg_lstsq_gels(const Tensor& A, const Tensor& B, const Tensor& /*infos*/) {
  // The steps for using the QR decomposition for solving least squares problems
  // are outlined here https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto mn = std::min(m, n);

  // explicitly broadcast the batch dimensions of A
  // TODO: revisit this later to use batch_iterator_with_broadcasting in triangular_solve
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);

  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  Tensor tau = at::empty(tau_shape, A.options());

  if (m >= n) {
    // Step 1: compute QR factorization using geqrf
    geqrf_kernel(A, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {A.size(-2), A.size(-1)});
    Tensor A_expanded = A.expand({A_expand_batch});
    bool is_fortran_contiguous = A_expanded.mT().is_contiguous();
    Tensor A_broadcasted = is_fortran_contiguous ? A_expanded : cloneBatchedColumnMajor(A_expanded);
    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 2: B <- Q^H B
    ormqr_kernel(A_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/true);

    // Step 3: solve R X = B
    triangular_solve_kernel(
        A_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/TransposeType::NoTranspose,
        /*unitriangular=*/false);
  } else { // underdetermined case
    Tensor Ah = cloneBatchedColumnMajor(A.mH());

    // Step 1: compute QR factorization of conjugate transpose of A using geqrf
    geqrf_kernel(Ah, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {Ah.size(-2), Ah.size(-1)});
    Tensor Ah_expanded = Ah.expand({A_expand_batch});
    bool is_fortran_contiguous = Ah_expanded.mT().is_contiguous();
    Tensor Ah_broadcasted = is_fortran_contiguous ? Ah_expanded : cloneBatchedColumnMajor(Ah_expanded);

    // Step 2: R^H Z = B
    const auto trans = Ah_broadcasted.is_complex() ? TransposeType::ConjTranspose
                                                   : TransposeType::Transpose;
    triangular_solve_kernel(
        Ah_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/trans,
        /*unitriangular=*/false);

    // B matrix has the size max(m, n) x nrhs
    // triangular_solve_kernel writes its output into the first m rows of B leaving the rest untouched
    // we need to set the rest of the rows to zero so that the multiplication from step 3 is correct
    B.narrow(-2, m, n - m).zero_();

    auto tau_expand_batch = std::move(expand_batch_portion);
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 3: X <- Q Z
    ormqr_kernel(Ah_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/false);
  }
}

void gels_looped(const Tensor& a, Tensor& b, Tensor& infos) {
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
    default:
      // linalg_lstsq_gels is a generic function that is implemented using
      // geqrf_stub, ormqr_stub, and triangular_solve_stub
      // It dispatches to cuSOLVER for CUDA inputs if USE_LINALG_SOLVER is defined
      return linalg_lstsq_gels(a, b, infos);
  }
}

void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& /*rank*/, Tensor& /*singular_values*/, Tensor& infos, double /*rcond*/, std::string /*driver_name*/)  {
  auto m = a.size(-2);
  auto n = a.size(-1);

  // first handle the underdetermined case (m < n)
  // this case is not supported by cuBLAS
  if (m < n) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
    linalg_lstsq_gels(a, b, infos);
#else
    TORCH_CHECK(
        false,
        "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA. ",
        "Please rebuild with cuSOLVER.");
#endif
  } else { // m >= n
    // On CUDA platform we use either cuBLAS or cuSOLVER here
    // the batched vs looped dispatch is implemented based on the following performance results
    // https://github.com/pytorch/pytorch/pull/54725#issuecomment-832234456
    if (m <= 256 && batchCount(b) >= std::max<int64_t>(2, m / 16)) {
      gels_batched_cublas(a, b, infos);
    } else {
      gels_looped(a, b, infos);
    }
  }
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(lstsq_stub, &lstsq_kernel)


#if defined(BUILD_LAZY_CUDA_LINALG)
struct DispatchInitializer {
  DispatchInitializer() {
    cuda::detail::LinalgDispatch disp{_cholesky_solve_helper_cuda};
    cuda::detail::registerLinalgDispatch(disp);
  };
} initializer;

}  // namespace lazy_linalg
#endif

}  // namespace at::native

#undef ALLOCATE_ARRAY
