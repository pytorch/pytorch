#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>

namespace at { namespace native {

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

template<typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda >= std::max<int64_t>(1L, m);
}

Tensor &addmv_impl_cpu(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_) {
  auto r_stride = result.stride(0);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, mat.scalar_type(), "addmv_impl_cpu", [&] {
    auto beta = beta_.to<scalar_t>();
    auto alpha = alpha_.to<scalar_t>();
    if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
      gemv<scalar_t>('n', mat.size(0), mat.size(1), alpha, mat.data_ptr<scalar_t>(), mat.stride(1),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
      gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, mat.data_ptr<scalar_t>(), mat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
    else {
      Tensor cmat = mat.contiguous();
      gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, cmat.data_ptr<scalar_t>(), cmat.stride(0),
          vec.data_ptr<scalar_t>(), vec.stride(0), beta, result.data_ptr<scalar_t>(), r_stride);
    }
  });
  return result;
}

Tensor &addmv_out(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  { // scope of NoNamesGuard

  at::NoNamesGuard guard;
  result.resize_({mat.size(0)});

  Tensor self_ = self;
  if (self.dim() == 0 || self.size(0) == 1) {
    self_ = self.expand({mat.size(0)});
  }

  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self_.dim() == 1),
    "vector + matrix @ vector expected, got ", self_.dim(), ", ", mat.dim(), ", ", vec.dim());
  TORCH_CHECK((mat.size(1) == vec.size(0) && mat.size(0) == self_.size(0)),
    "size mismatch, get ", self_.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));

  if (mat.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      at::cpu::mul_out(
          result,
          self,
          at::native::scalar_tensor(
              beta, self.scalar_type(), c10::nullopt /* layout */, at::kCPU, c10::nullopt /* pin_memory */));
    }
  } else {
    if (!result.is_same(self_)) {
      at::native::copy_(result, self_);
    }
    if (result.numel() != 0) {
      at::_addmv_impl_(result, self_, mat, vec, beta, alpha);
    }
  }

  } // scope of NoNamesGuard
  at::namedinference::propagate_names_for_addmv(result, mat, vec, self);
  return result;
}

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({mat.size(0)}, mat.options());
  return native::addmv_out(self, mat, vec, beta, alpha, result);
}

Tensor &addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha) {
  return native::addmv_out(self, mat, vec, beta, alpha, self);
}

Tensor &mv_out(const Tensor &self, const Tensor &vec, Tensor& result) {
  return native::addmv_out(result, self, vec, 0, 1, result);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return native::mv_out(self, vec, result);
}

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");

  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());

  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
}

Tensor dot(const Tensor &self, const Tensor &other){
  at::NoNamesGuard guard;

  dot_check(self, other);

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, self.scalar_type(), "dot", [&] {
    Tensor result = at::empty({}, self.options());
    result.fill_(dot_impl<scalar_t>(self.numel(), self.data_ptr<scalar_t>(), self.stride(0), other.data_ptr<scalar_t>(), other.stride(0)));
    return result;
  });
}

Tensor vdot(const Tensor &self, const Tensor &other){
  at::NoNamesGuard guard;

  // Dispatch to `dot` for real dtypes.
  if (!self.is_complex()){
    return at::dot(self, other);
  }

  // For complex dtypes.
  dot_check(self, other);
  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());
    result.fill_(vdot_impl<scalar_t>(self.numel(), self.data_ptr<scalar_t>(), self.stride(0), other.data_ptr<scalar_t>(), other.stride(0)));
    return result;
  });

}

}}  // namespace at::native
