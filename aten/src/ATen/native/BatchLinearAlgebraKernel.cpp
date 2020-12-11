#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>

#include <TH/TH.h>  // for USE_LAPACK

namespace at { namespace native {

namespace {

template <typename scalar_t>
void apply_eig(const Tensor& self, bool eigenvectors, Tensor& vals_, Tensor& vecs_, int64_t* info_ptr) {
#ifndef USE_LAPACK
  TORCH_CHECK(false, "Calling torch.eig on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  char jobvr = eigenvectors ? 'V' : 'N';
  int64_t n = self.size(-1);
  auto self_data = self.data_ptr<scalar_t>();

  auto vals_data = vals_.data_ptr<scalar_t>();
  scalar_t* wr = vals_data;
  scalar_t* wi = vals_data + n;

  scalar_t* vecs_data = eigenvectors ? vecs_.data_ptr<scalar_t>() : nullptr;
  int ldvr = eigenvectors ? n : 1;

  if (n > 0) {
    // call lapackEig once to get the optimal size for work data
    scalar_t wkopt;
    int info;
    lapackEig<scalar_t>('N', jobvr, n, self_data, n, wr, wi,
      nullptr, 1, vecs_data, ldvr, &wkopt, -1, &info);
    int lwork = static_cast<int>(wkopt);

    // call again to do the actual work
    Tensor work = at::empty({lwork}, self.dtype());
    lapackEig<scalar_t>('N', jobvr, n, self_data, n, wr, wi,
      nullptr, 1, vecs_data, ldvr, work.data_ptr<scalar_t>(), lwork, &info);
    *info_ptr = info;
  }
#endif
}

std::tuple<Tensor, Tensor> eig_kernel_impl(const Tensor& self, bool& eigenvectors) {
  int64_t n = self.size(-1);
  // lapackEig function expects the input to be column major, or stride {1, n},
  // so we must set the stride manually since the default stride for tensors is
  // row major, {n, 1}
  Tensor self_ = at::empty_strided(
      {n, n},
      {1, n},
      at::TensorOptions(self.dtype()));
  self_.copy_(self);

  auto options = self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor vals_ = at::empty_strided({n, 2}, {1, n}, options);
  Tensor vecs_ = eigenvectors
                 ? at::empty_strided({n, n}, {1, n}, options)
                 : Tensor();

  int64_t info;
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "eig_cpu", [&]{
    apply_eig<scalar_t>(self_, eigenvectors, vals_, vecs_, &info);
  });
  singleCheckErrors(info, "eig_cpu");
  return std::tuple<Tensor, Tensor>(vals_, vecs_);
}

} // anonymous namespace

REGISTER_ARCH_DISPATCH(eig_stub, DEFAULT, &eig_kernel_impl);
REGISTER_AVX_DISPATCH(eig_stub, &eig_kernel_impl);
REGISTER_AVX2_DISPATCH(eig_stub, &eig_kernel_impl);

}} // namespace at::native
