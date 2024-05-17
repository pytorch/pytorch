// LinearAlgebraStubs.cpp
// Mostly a no-op unless BUILD_LAZY_CUDA_LINALG is defined
// In that case load library is dynamically loaded when first linalg call is made
// This helps reduce size of GPU memory context if linear algebra functions are not used
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/TransposeType.h>
#if defined(BUILD_LAZY_CUDA_LINALG)
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

#if AT_MAGMA_ENABLED()
#include <ATen/cuda/detail/CUDAHooks.h>

namespace {
struct MagmaInitializer {
  MagmaInitializer() {
    ::at::cuda::detail::set_magma_init_fn([]{ });
  };
} initializer;
}  // namespace (anonymous)
#endif
#endif
namespace at::native {
#if defined(BUILD_LAZY_CUDA_LINALG)
namespace {
cuda::detail::LinalgDispatch disp = {_cholesky_solve_helper_cuda};

at::DynamicLibrary& getTorchLinalgLibrary() {
  static at::DynamicLibrary lib("libtorch_cuda_linalg.so", nullptr, true);
  return lib;
}

// Lazy dispatches do nothing but load linalg library and call the stub
// Loading the library should override the registration of those with the proper implementation
// getTorchLinalgLibrary() throws an exception if library is not found,
// which makes it unnecessary to have an explicit error checking
// But make sure that this function is called only once, to avoid infinite recursion
void loadLazyTorchLinalgLibrary() {
  static int invoke_count = 0;
  getTorchLinalgLibrary();
  TORCH_CHECK(invoke_count++ == 0, "lazy wrapper should be called at most once");
}

void lazy_cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  loadLazyTorchLinalgLibrary();
  cholesky_stub(DeviceType::CUDA, input, info, upper);
}

Tensor& lazy_cholesky_inverse_kernel(Tensor &result, Tensor& infos, bool upper) {
  loadLazyTorchLinalgLibrary();
  return cholesky_inverse_stub(DeviceType::CUDA, result, infos, upper);
}

void lazy_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  loadLazyTorchLinalgLibrary();
  lu_factor_stub(DeviceType::CUDA, input, pivots, infos, compute_pivots);
}

void lazy_triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  loadLazyTorchLinalgLibrary();
  triangular_solve_stub(DeviceType::CUDA, A, B, left, upper, transpose, unitriangular);
}

Tensor& lazy_orgqr_kernel(Tensor& result, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  return orgqr_stub(DeviceType::CUDA, result, tau);
}

void lazy_ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  loadLazyTorchLinalgLibrary();
  ormqr_stub(DeviceType::CUDA, input, tau, other, left, transpose);
}

void lazy_geqrf_kernel(const Tensor& input, const Tensor& tau) {
  loadLazyTorchLinalgLibrary();
  geqrf_stub(DeviceType::CUDA, input, tau);
}

void lazy_linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  loadLazyTorchLinalgLibrary();
  linalg_eigh_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

void lazy_linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  getTorchLinalgLibrary();
  linalg_eig_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, input, compute_eigenvectors);
}

void lazy_svd_kernel(const Tensor& A,
                     const bool full_matrices,
                     const bool compute_uv,
                     const std::optional<c10::string_view>& driver,
                     const Tensor& U,
                     const Tensor& S,
                     const Tensor& Vh,
                     const Tensor& info) {
  getTorchLinalgLibrary();
  svd_stub(DeviceType::CUDA, A, full_matrices, compute_uv, driver, U, S, Vh, info);
}

void lazy_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  getTorchLinalgLibrary();
  lu_solve_stub(DeviceType::CUDA, LU, pivots, B, trans);
}

void lazy_lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name)  {
  getTorchLinalgLibrary();
  lstsq_stub(DeviceType::CUDA, a, b, rank, singular_values, infos, rcond, driver_name);
}

void lazy_ldl_factor(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  ldl_factor_stub(DeviceType::CUDA, LD, pivots, info, upper, hermitian);
}

void lazy_ldl_solve(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  loadLazyTorchLinalgLibrary();
  ldl_solve_stub(DeviceType::CUDA, LD, pivots, B, upper, hermitian);
}

REGISTER_CUDA_DISPATCH(cholesky_stub, &lazy_cholesky_kernel)
REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &lazy_cholesky_inverse_kernel);
REGISTER_CUDA_DISPATCH(lu_factor_stub, &lazy_lu_factor);
REGISTER_CUDA_DISPATCH(ldl_factor_stub, &lazy_ldl_factor);
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &lazy_ldl_solve);
REGISTER_CUDA_DISPATCH(triangular_solve_stub, &lazy_triangular_solve_kernel);
REGISTER_CUDA_DISPATCH(orgqr_stub, &lazy_orgqr_kernel);
REGISTER_CUDA_DISPATCH(ormqr_stub, &lazy_ormqr_kernel);
REGISTER_CUDA_DISPATCH(geqrf_stub, &lazy_geqrf_kernel);
REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &lazy_linalg_eigh_kernel);
REGISTER_CUDA_DISPATCH(linalg_eig_stub, &lazy_linalg_eig_kernel);
REGISTER_CUDA_DISPATCH(svd_stub, &lazy_svd_kernel)
REGISTER_CUDA_DISPATCH(lu_solve_stub, &lazy_lu_solve);
REGISTER_CUDA_DISPATCH(lstsq_stub, &lazy_lstsq_kernel);
} // anonymous namespace

// Old style dispatches
// torch_cuda_linalg dynamic library should have a global constructor
// that calls regiserLinaglDispatch so in order ot lazy bind
// old style dispatch all one have to do is to load library and call disp.func_name
// Protect from infinite recursion by initializing dispatch to self and checking
// that values are different after linalg library were loaded

namespace cuda {
namespace detail {
void registerLinalgDispatch(const LinalgDispatch& disp_) {
  disp = disp_;
}
}} //namespace cuda::detail

Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
    getTorchLinalgLibrary();
    TORCH_CHECK(disp.cholesky_solve_helper != _cholesky_solve_helper_cuda, "Can't find _cholesky_solve_helper_cuda");
    return disp.cholesky_solve_helper(self, A, upper);
}

#endif /*defined(BUILD_LAZY_CUDA_LINALG)*/

} // namespace at::native
