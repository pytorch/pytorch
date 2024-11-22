#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Config.h>

#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/native/mkldnn/Linear.h>
#include <ATen/native/Resize.h>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/matmul.h>
#endif

namespace at::meta {
TORCH_META_FUNC(addmv)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
    "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(), ", ", vec.dim());

  TORCH_CHECK(mat.size(1) == vec.size(0) && (mat.size(0) == self.numel() || self.numel() == 1),
    "size mismatch, got input (", self.size(0), "), mat (", mat.size(0), "x", mat.size(1), "), vec (", vec.size(0), ")");
  auto names = at::namedinference::propagate_names_for_addmv(mat, vec, self);
  set_output_raw_strided(0, IntArrayRef(mat.sizes().data(), 1), {}, vec.options(), names);
}
} // namespace at::meta

namespace at::native {

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

template<typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

constexpr inline bool lda_cond(int64_t m, int64_t n, int64_t lda) {
  return n == 1 || lda >= std::max<int64_t>(1L, m);
}




TORCH_IMPL_FUNC(addmv_out_cpu)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();
  if (mat.numel() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (betaval == 0.0) {
      result.zero_();
    } else {
      at::cpu::mul_out(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), std::nullopt /* layout */, at::kCPU, std::nullopt /* pin_memory */));
    }
  } else {
    if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents is ignored
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      at::native::copy_(const_cast<Tensor&>(result), *self_);
    }
    if (result.numel() != 0) {

      NoNamesGuard guard;
      if (use_mkldnn_matmul(mat, vec, /*result=*/Tensor())){
        mkldnn_matmul(mat, vec, result, beta_.to<float>(), alpha_.to<float>());
        return;
      }

      auto r_stride = result.stride(0);
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, mat.scalar_type(), "addmv_impl_cpu", [&] {
        auto beta = beta_.to<scalar_t>();
        auto alpha = alpha_.to<scalar_t>();
        if (mat.stride(0) == 1 && lda_cond(mat.size(0), mat.size(1), mat.stride(1))) {
          gemv<scalar_t>('n', mat.size(0), mat.size(1), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(1),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        else if (mat.stride(1) == 1 && lda_cond(mat.size(1), mat.size(0), mat.stride(0))) {
          gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(0),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        else {
          Tensor cmat = mat.contiguous();
          gemv<scalar_t>('t', mat.size(1), mat.size(0), alpha, cmat.const_data_ptr<scalar_t>(), cmat.stride(0),
              vec.const_data_ptr<scalar_t>(), vec.stride(0), beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
      });
    }
  }
}

Tensor &mv_out(const Tensor &self, const Tensor &vec, Tensor& result) {
  //self arg sent to addmv_out cannot be resized
  //here we use result as self argument for addmv, and result is user supplied and can be wrong size
  //it's not a hard error, because we allow resizing result, but it becomes a hard error
  //in addmv, because addmv expects self to satisfy proper conditions
  //to avoid this, supply correctly sized self, its contents doesn't matter because beta is 0
  if (result.dim() > 1 || (result.numel() != self.size(0) || result.numel() !=1)) {
    Tensor self_addmv = at::empty({self.size(0)}, vec.options());
    return at::addmv_out(result, self_addmv, self, vec, 0, 1);
  }
  return at::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result = at::empty({self.size(0)}, vec.options());
  //inplace version is more efficient if we can use it
  return at::addmv_(result, self, vec, 0, 1);
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
  if (self.is_complex()) {
    if (self.is_conj()) {
      if (other.is_conj()) {
        return (at::native::dot(self.conj(), other.conj())).conj();
       } else {
         return at::native::vdot(self.conj(), other);
       }
    } else if (other.is_conj()) {
      return at::native::vdot(other.conj(), self);
    }
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  if (use_mkldnn_matmul(self, other, /*result=*/Tensor())){
    // mkldnn matmul expect result have sizes info to create ideep tensor
    auto r =  at::empty({1, 1}, self.options());
    mkldnn_matmul(self, other, r, /*beta=*/0);
    return r;
  }

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(), "dot", [&] {
    Tensor result = at::empty({}, self.options());
    result.fill_(dot_impl<scalar_t>(self.numel(), const_cast<scalar_t*>(self.const_data_ptr<scalar_t>()), self.stride(0), const_cast<scalar_t*>(other.const_data_ptr<scalar_t>()), other.stride(0)));
    return result;
  });
}

Tensor vdot(const Tensor &self, const Tensor &other){
  // Dispatch to `dot` for real dtypes.
  if (!self.is_complex()){
    return at::dot(self, other);
  }

  if (self.is_conj()) {
    if (other.is_conj()) {
      return at::native::vdot(other.conj(), self.conj());
    } else {
      return at::native::dot(self.conj(), other);
    }
  } else if (other.is_conj()) {
    return (at::native::dot(self, other.conj())).conj();
  }

  at::NoNamesGuard guard;
  // For complex dtypes.
  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());
    result.fill_(vdot_impl<scalar_t>(self.numel(), const_cast<scalar_t*>(self.const_data_ptr<scalar_t>()), self.stride(0), const_cast<scalar_t *>(other.const_data_ptr<scalar_t>()), other.stride(0)));
    return result;
  });

}

static Tensor&
_scaled_mm_out_cpu_emulated(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  TORCH_INTERNAL_ASSERT((scale_a.numel() == 1 && scale_b.numel() == 1), "Now _scaled_mm only supports per-tensor scaling for CPU backend.");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());

  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());

  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();
  IntArrayRef mat1_sizes = mat1_c.sizes();
  IntArrayRef mat2_sizes = mat2_c.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  float input_scale = scale_a.item<float>();
  float weight_scale = scale_b.item<float>();
  auto fp32_mat1 = at::mul(mat1.to(kFloat), input_scale);
  auto fp32_mat2 = at::mul(mat2_c.to(kFloat), weight_scale);
  auto out_tmp = at::matmul(fp32_mat1, fp32_mat2);
  if (bias) {
    out_tmp.add_(bias.value());
  }
  out_tmp = out_tmp.to(out.scalar_type());
  out.copy_(out_tmp);
  return out;
}

Tensor&
_scaled_mm_out_cpu(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
#if AT_MKLDNN_ENABLED()
  if (at::globalContext().userEnabledMkldnn() && cpuinfo_has_x86_amx_int8()) {
    return mkldnn_scaled_mm(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
  } else
#endif
  {
  return _scaled_mm_out_cpu_emulated(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
  }
}

Tensor
_scaled_mm_cpu(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out_cpu(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

}  // namespace at::native
