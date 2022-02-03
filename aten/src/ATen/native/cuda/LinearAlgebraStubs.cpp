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
#if AT_MAGMA_ENABLED()
#include <ATen/cuda/detail/CUDAHooks.h>
#include <iostream>

#if defined(BUILD_LAZY_CUDA_LINALG)
namespace {
struct MagmaInitializer {
  MagmaInitializer() {
    ::at::cuda::detail::set_magma_init_fn([]{ });
  };
} initializer;
}  // namespace (anonymous)
#endif

#endif
namespace at {
namespace native {
#if !defined(BUILD_LAZY_CUDA_LINALG)
std::tuple<Tensor, Tensor> legacy_lstsq_cuda_impl(const Tensor &B, const Tensor &A);
std::tuple<Tensor, Tensor> _solve_helper_cuda_impl(const Tensor& self, const Tensor& A);
std::tuple<Tensor, Tensor> _symeig_helper_cuda_impl(const Tensor& self, bool eigenvectors, bool upper);
Tensor _cholesky_solve_helper_cuda_impl(const Tensor& self, const Tensor& A, bool upper);
std::tuple<Tensor, Tensor> _linalg_qr_helper_cuda_impl(const Tensor& input, c10::string_view mode);
Tensor& _linalg_inv_out_helper_cuda_impl(Tensor &result, Tensor& infos_lu, Tensor& infos_getri);
#else
namespace {
std::tuple<Tensor, Tensor> (*legacy_lstsq_cuda_impl)(const Tensor &B, const Tensor &A) = nullptr;
std::tuple<Tensor, Tensor> (*_solve_helper_cuda_impl)(const Tensor& self, const Tensor& A) = nullptr;
std::tuple<Tensor, Tensor> (*_symeig_helper_cuda_impl)(const Tensor& self, bool eigenvectors, bool upper) = nullptr;
Tensor (*_cholesky_solve_helper_cuda_impl)(const Tensor& self, const Tensor& A, bool upper) = nullptr;
std::tuple<Tensor, Tensor> (*_linalg_qr_helper_cuda_impl)(const Tensor& input, c10::string_view mode) = nullptr;
Tensor& (*_linalg_inv_out_helper_cuda_impl)(Tensor &result, Tensor& infos_lu, Tensor& infos_getri) = nullptr;
}

at::DynamicLibrary& getTorchLinalgLibrary() {
  static at::DynamicLibrary lib("libtorch_cuda_linalg.so");
  return lib;
}

Tensor& _linalg_inv_out_helper_cuda(Tensor &result, Tensor& infos_lu, Tensor& infos_getri) {
    if (!_linalg_inv_out_helper_cuda_impl) {
          _linalg_inv_out_helper_cuda_impl = reinterpret_cast<decltype(&_linalg_inv_out_helper_cuda)>(getTorchLinalgLibrary().sym("_ZN2at6native32_linalg_inv_out_helper_cuda_implERNS_6TensorES2_S2_"));
          TORCH_CHECK(_linalg_inv_out_helper_cuda_impl != nullptr, "Can't find");
        }
    return _linalg_inv_out_helper_cuda_impl(result, infos_lu, infos_getri);
}

namespace {
void lazy_cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  getTorchLinalgLibrary();
  cholesky_stub(DeviceType::CUDA, input, info, upper);
}

Tensor& lazy_cholesky_inverse_kernel(Tensor &result, Tensor& infos, bool upper) {
  getTorchLinalgLibrary();
  return cholesky_inverse_stub(DeviceType::CUDA, result, infos, upper);
}

void lazy_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  getTorchLinalgLibrary();
  lu_factor_stub(DeviceType::CUDA, input, pivots, infos, compute_pivots);
}
void lazy_triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  getTorchLinalgLibrary();
  triangular_solve_stub(DeviceType::CUDA, A, B, left, upper, transpose, unitriangular);
}
Tensor& lazy_orgqr_kernel(Tensor& result, const Tensor& tau) {
  getTorchLinalgLibrary();
  return orgqr_stub(DeviceType::CUDA, result, tau);
}
void lazy_ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  getTorchLinalgLibrary();
  ormqr_stub(DeviceType::CUDA, input, tau, other, left, transpose);
}

void lazy_geqrf_kernel(const Tensor& input, const Tensor& tau) {
  getTorchLinalgLibrary();
  geqrf_stub(DeviceType::CUDA, input, tau);
}

void lazy_linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  getTorchLinalgLibrary();
  linalg_eigh_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

std::tuple<Tensor, Tensor> lazy_eig_kernel(const Tensor& self, bool& eigenvectors) {
  getTorchLinalgLibrary();
  return eig_stub(DeviceType::CUDA, self, eigenvectors);
}

void lazy_linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  getTorchLinalgLibrary();
  linalg_eig_stub(DeviceType::CUDA, eigenvalues, eigenvectors, infos, input, compute_eigenvectors);
}

void lazy_svd_kernel(const Tensor& A,
                     const bool full_matrices,
                     const bool compute_uv,
                     const Tensor& U,
                     const Tensor& S,
                     const Tensor& Vh,
                     const Tensor& info) {
  getTorchLinalgLibrary();
  svd_stub(DeviceType::CUDA, A, full_matrices, compute_uv, U, S, Vh, info);
}

void lazy_lu_solve_trans(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType trans) {
  getTorchLinalgLibrary();
  lu_solve_trans_stub(DeviceType::CUDA, b, lu, pivots, trans);
}

void lazy_lu_solve(const Tensor& b, const Tensor& lu, const Tensor& pivots) {
  getTorchLinalgLibrary();
  lu_solve_stub(DeviceType::CUDA, b, lu, pivots);
}

void lazy_lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name)  {
  getTorchLinalgLibrary();
  lstsq_stub(DeviceType::CUDA, a, b, rank, singular_values, infos, rcond, driver_name);
}

REGISTER_CUDA_DISPATCH(cholesky_stub, &lazy_cholesky_kernel)
REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &lazy_cholesky_inverse_kernel);
REGISTER_CUDA_DISPATCH(lu_factor_stub, &lazy_lu_factor);
REGISTER_CUDA_DISPATCH(triangular_solve_stub, &lazy_triangular_solve_kernel);
REGISTER_CUDA_DISPATCH(orgqr_stub, &lazy_orgqr_kernel);
REGISTER_CUDA_DISPATCH(ormqr_stub, &lazy_ormqr_kernel);
REGISTER_CUDA_DISPATCH(geqrf_stub, &lazy_geqrf_kernel);
REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &lazy_linalg_eigh_kernel);
REGISTER_CUDA_DISPATCH(eig_stub, &lazy_eig_kernel);
REGISTER_CUDA_DISPATCH(linalg_eig_stub, &lazy_linalg_eig_kernel);
REGISTER_CUDA_DISPATCH(svd_stub, &lazy_svd_kernel)
REGISTER_CUDA_DISPATCH(lu_solve_trans_stub, &lazy_lu_solve_trans);
REGISTER_CUDA_DISPATCH(lu_solve_stub, &lazy_lu_solve);
REGISTER_CUDA_DISPATCH(lstsq_stub, &lazy_lstsq_kernel);
} // anonymous namespace
#endif /*defined(BUILD_LAZY_CUDA_LINALG)*/

std::tuple<Tensor, Tensor> legacy_lstsq_cuda(const Tensor &B, const Tensor &A) {
    return legacy_lstsq_cuda_impl(B, A);
}
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
    return _cholesky_solve_helper_cuda_impl(self, A, upper);
}
std::tuple<Tensor, Tensor> _linalg_qr_helper_cuda(const Tensor& input, c10::string_view mode) {
    return _linalg_qr_helper_cuda_impl(input, mode);
}
std::tuple<Tensor, Tensor> _symeig_helper_cuda(const Tensor& self, bool eigenvectors, bool upper) {
    return _symeig_helper_cuda_impl(self, eigenvectors, upper);
}
std::tuple<Tensor, Tensor> _solve_helper_cuda(const Tensor& self, const Tensor& A) {
    return _solve_helper_cuda_impl(self, A);
}

std::tuple<Tensor&, Tensor&> legacy_lstsq_out_cuda(
    const Tensor& B, const Tensor& A, Tensor& B_out, Tensor& A_out) {
  const auto dtype = A.scalar_type();
  TORCH_CHECK(B.scalar_type() == dtype, "exepected A and B dtypes to match but found ",
              A.scalar_type(), " and ", B.scalar_type());
  TORCH_CHECK(A_out.scalar_type() == dtype, "A_out to have scalar type ", dtype,
              " but found", A_out.scalar_type());
  TORCH_CHECK(B_out.scalar_type() == dtype, "A_out to have scalar type ", dtype,
              " but found", B_out.scalar_type());
  Tensor A_tmp, B_tmp;
  std::tie(B_tmp, A_tmp) = native::legacy_lstsq_cuda(B, A);
  resize_output(A_out, A_tmp.sizes());
  A_out.copy_(A_tmp);
  resize_output(B_out, B_tmp.sizes());
  B_out.copy_(B_tmp);
  return std::tuple<Tensor&, Tensor&>(B_out, A_out);
}

}} // namespace at::native
