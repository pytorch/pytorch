#include <ATen/ATen.h>
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

template<typename scalar_t>
void addr_impl(int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);

constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda > std::max<int64_t>(1L, m);
}

Tensor &addmv_impl_cpu(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta_, Scalar alpha_) {
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

Tensor &addmv_out(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
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
      at::native::mul_out(result, self, at::native::scalar_tensor(beta, at::device(at::kCPU).dtype(self.scalar_type())));
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

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({mat.size(0)}, mat.options());
  return native::addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor &addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return native::addmv_out(self, self, mat, vec, beta, alpha);
}

Tensor &mv_out(Tensor& result, const Tensor &self, const Tensor &vec) {
  return native::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return native::mv_out(result, self, vec);
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

// NOTE: this is to port the below LDA_COND macro from the TH codebase
//   #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))
// lda_cond function in this file is not used
// because it requires `lda > std::max<int64_t>(1L, m)`
// instead of `>=` as defined in the TH implementation
// this function is to check whether the input `m` and `n`
// could be used as the row and column numbers for the blas functions
constexpr inline bool lda_cond_addr(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda >= std::max<int64_t>(1L, m);
}

template<typename scalar_t>
Tensor &addr_impl_cpu(Tensor &result, const Tensor &self,
                    const Tensor& vec1, const Tensor& vec2,
                    scalar_t beta, scalar_t alpha) {
  if (&result != &self) {
    at::native::resize_as_(result, self);
    // if beta is zero, no need to copy the input tensor to result
    if (beta == 0.0) {
      at::native::zero_(result);
    } else {
      at::native::copy_(result, self);
    }
  }

  if (beta == 0.0) {
    at::native::zero_(result);
  } else if (beta != 1.0) {
    at::native::mul_(result, beta);
  }

  // prepare the parameters following the style of blas level 2 routines
  // sger and dger to calculate the outer product of vec1 and vec2
  if (result.stride(0) == 1 &&
    lda_cond_addr(vec1.size(0), vec2.size(0), result.stride(1))) {
    addr_impl<scalar_t>(
      vec1.size(0), vec2.size(0),
      alpha, vec1.data_ptr<scalar_t>(), vec1.stride(0),
      vec2.data_ptr<scalar_t>(), vec2.stride(0),
      result.data_ptr<scalar_t>(), result.stride(1)
    );
  } else if (result.stride(1) == 1 &&
    lda_cond_addr(vec2.size(0), vec1.size(0), result.stride(0))) {
    addr_impl<scalar_t>(
      vec2.size(0), vec1.size(0),
      alpha, vec2.data_ptr<scalar_t>(), vec2.stride(0),
      vec1.data_ptr<scalar_t>(), vec1.stride(0),
      result.data_ptr<scalar_t>(), result.stride(0)
    );
  } else {
    Tensor cr = result.clone();
    addr_impl<scalar_t>(
      vec2.size(0), vec1.size(0),
      alpha, vec2.data_ptr<scalar_t>(), vec2.stride(0),
      vec1.data_ptr<scalar_t>(), vec1.stride(0),
      cr.data_ptr<scalar_t>(), cr.stride(0)
    );
    result.set_(cr);
  }
  return result;
}

Tensor& addr_out_cpu(Tensor &result, const Tensor& self,
                      const Tensor& vec1, const Tensor& vec2,
                      Scalar beta, Scalar alpha) {
  TORCH_CHECK(vec1.dim() == 1 && vec2.dim() == 1,
              "vec1 and vec2 should be 1-dimensional vectors. Got dimensions ",
              vec1.dim(), " and ", vec2.dim());

  Tensor self_;
  if (&result != &self) {
    std::tie(self_) = expand_size(self, {vec1.size(0), vec2.size(0)}, "addr");
  } else {
    self_ = self;
  }

  TORCH_CHECK(result.device() == self_.device() &&
              result.device() == vec1.device() &&
              result.device() == vec2.device(),
              "Expected all tensors to be on the same device. Found: ",
              result.device(), ", ", self_.device(), ", ",
              vec1.device(), " and ", vec2.device());
  TORCH_CHECK(self_.dim() == 2,
              "2D tensor expected, got ", self_.dim(), "D tensor for input");
  TORCH_CHECK(self_.size(0) == vec1.size(0) && self_.size(1) == vec2.size(0),
              "size mismatch",
              ", input: ", self_.sizes(),
              ", v1: ", vec1.sizes(),
              ", v2: ", vec2.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self_.scalar_type(), "addr_out_cpu", [&] {
    addr_impl_cpu<scalar_t>(
      result, self_,
      vec1, vec2,
      beta.to<scalar_t>(), alpha.to<scalar_t>()
    );
  });
  return result;
}

Tensor& addr__cpu(Tensor& self,
                   const Tensor& vec1, const Tensor& vec2,
                   Scalar beta, Scalar alpha) {
  addr_out_cpu(self, self, vec1, vec2, beta, alpha);
  return self;
}

Tensor addr_cpu(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 Scalar beta, Scalar alpha) {
  Tensor result = at::empty({0}, self.options());
  addr_out_cpu(result, self, vec1, vec2, beta, alpha);
  return result;
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
