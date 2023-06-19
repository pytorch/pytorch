#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

namespace at::native {

namespace {

// TODO: https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(const Tensor& tensor, bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
      transpose_tensor = tensor.is_contiguous();
      return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) && (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if ((tensor_strides[1] == 1) && (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // namespace

c10::MaybeOwned<Tensor> prepare_batch_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor, int64_t& ld_tensor, bool transpose_result, int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = resolve_conj_if_indicated(tensor, true);
    ld_tensor = tensor_->strides()[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = resolve_conj_if_indicated(tensor, false);
    ld_tensor = tensor_->strides()[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero = tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = resolve_conj_if_indicated(tensor, transpose_result);
    } else {
      tensor_ = c10::MaybeOwned<Tensor>::owned(tensor.clone(at::MemoryFormat::Contiguous));
    }
    ld_tensor = tensor_->strides()[1];
  }

  return tensor_;
}

namespace {

enum class Activation {
  None,
  RELU,
  GELU,
};

#if !defined(USE_ROCM) && !defined(_MSC_VER)
cuda::blas::GEMMAndBiasActivationEpilogue activation_to_gemm_and_blas_arg(Activation a) {
  switch (a) {
    case Activation::None:
      return cuda::blas::GEMMAndBiasActivationEpilogue::None;
    case Activation::RELU:
      return cuda::blas::GEMMAndBiasActivationEpilogue::RELU;
    case Activation::GELU:
      return cuda::blas::GEMMAndBiasActivationEpilogue::GELU;
    default:
      TORCH_CHECK(false);
      return cuda::blas::GEMMAndBiasActivationEpilogue::None;
  }
}
#endif

static bool getDisableAddmmCudaLt() {
    static const char* env_value = std::getenv("DISABLE_ADDMM_CUDA_LT");
    if (env_value != nullptr && strcmp(env_value, "1") == 0) {
      return true;
    }
    return false;
}

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  TensorArg args[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  bool useLtInterface = false;
  static bool disable_addmm_cuda_lt = getDisableAddmmCudaLt();
  at::ScalarType scalar_type = self.scalar_type();
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040 && !defined(_MSC_VER)
    // Strangely, if mat2 has only 1 row or column, we get
    // CUBLAS_STATUS_INVALID_VALUE error from cublasLtMatmulAlgoGetHeuristic.
    // self.dim() == 1 && result.dim() == 2 && self.sizes()[0] == mat2_sizes[1]
    // is to use lt interface only when self is bias.
    // for cuda 11.4, cublasLtMatmul is activated
    // the last two conditions is to skip 16b transA and non-trans-B having
    // leading dim >> rows when they are sliced from a large tensor
    // see fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul
    if (!disable_addmm_cuda_lt) {
      useLtInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
          result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
          self.is_contiguous() && result.is_contiguous() &&
          (scalar_type == at::ScalarType::Double ||
           scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1 &&
          mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32 &&
          mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32 &&
          // avoid leaing dim >> rows bugs
          ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
           (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16)) &&
          ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
           (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16));
    }
#endif
    if (!useLtInterface) {
      self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    }
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&result != &self) {
    at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
    if (beta.toComplexDouble() != 0.0 && !useLtInterface) {
      at::native::copy_(result, *self_);
    }
  }


  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  auto mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
  auto mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    // TODO: We could squeeze some perf by calling at::cuda::mul_out here instead, to bypass the dispatcher.
    // That requires some fixing some internal build dependencies though.
    return at::mul_out(
        result,
        self,
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */));
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

#if !defined(USE_ROCM) && !defined(_MSC_VER)
  if (useLtInterface) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
          at::cuda::blas::gemm_and_bias<scalar_t>(
              transpose_mat1,
              transpose_mat2,
              m,
              n,
              k,
              alpha.to<at::opmath_type<scalar_t>>(),
              mat1_->data_ptr<scalar_t>(),
              mat1_ld,
              mat2_->data_ptr<scalar_t>(),
              mat2_ld,
              self.const_data_ptr<scalar_t>(),
              result_->data_ptr<scalar_t>(),
              result_ld,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
              activation_to_gemm_and_blas_arg(activation)
#else
              // GELU is not supported (and does not compile!) prior
              // to CUDA 11.4. Have observed accuracy issues with
              // GELU epilogue in 11.4; disabling the GELU epilogue
              // path for CUDA version < 11.8.
              activation != Activation::GELU
              ? activation_to_gemm_and_blas_arg(activation)
              : cuda::blas::GEMMAndBiasActivationEpilogue::None
#endif
          );
        });
  } else
#endif
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda",
        [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t alpha_val = alpha.to<opmath_t>();
          opmath_t beta_val = beta.to<opmath_t>();
          const scalar_t* mat1_ptr = mat1_->const_data_ptr<scalar_t>();
          const scalar_t* mat2_ptr = mat2_->const_data_ptr<scalar_t>();
          scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
          at::cuda::blas::gemm<scalar_t>(
              transpose_mat1 ? mat1_->is_conj() ? 'c' : 't' : 'n',
              transpose_mat2 ? mat2_->is_conj() ? 'c' : 't' : 'n',
              m,
              n,
              k,
              alpha_val,
              mat1_ptr,
              mat1_ld,
              mat2_ptr,
              mat2_ld,
              beta_val,
              result_ptr,
              result_ld);
        });
    switch (activation) {
      case Activation::RELU:
        at::relu_(const_cast<Tensor&>(*result_));
        break;
      case Activation::GELU:
        at::gelu_(const_cast<Tensor&>(*result_));
        break;
      default: break;
    }
  }

// Preprocessor gate here needs to match the inverse of the check
// gating activation_to_gemm_and_blas_arg above; here we are manually
// performing a post-GELU because we weren't able to use the GELU
// epilogue above.
#if !defined(CUDA_VERSION) || CUDA_VERSION < 11080
  if (useLtInterface && activation == Activation::GELU) {
    at::gelu_(const_cast<Tensor&>(*result_));
  }
#endif

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

const Tensor& baddbmm_out_cuda_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  IntArrayRef batch1_sizes = batch1.sizes();

  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = resolve_conj_if_indicated(result, true);
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ = c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(at::MemoryFormat::Contiguous).transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t alpha_val = alpha.to<opmath_t>();
    opmath_t beta_val = beta.to<opmath_t>();
    const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
    const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
    at::cuda::blas::bgemm<scalar_t>(
      transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n',
      transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n',
      m, n, k,
      alpha_val,
      batch1_ptr, lda, batch1_->strides()[0],
      batch2_ptr, ldb, batch2_->strides()[0],
      beta_val,
      result_ptr, ldc, result_->strides()[0],
      num_batches
    );
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // anonymous namespace

TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha);
}

TORCH_IMPL_FUNC(addmm_activation_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha, use_gelu ? Activation::GELU : Activation::RELU);
}

TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
}

TORCH_IMPL_FUNC(baddbmm_out_cuda)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  {
    at::NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, self, batch1, batch2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(bmm_out_cuda)(const Tensor& batch1, const Tensor& batch2, const Tensor &result) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
}

namespace {

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
  TORCH_CHECK(
      self.device() == other.device(),
      "expected all tensors to be on the same device. Found: ",
      self.device(),
      ", ",
      other.device());
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

} // anonymous namespace

Tensor dot_cuda(const Tensor& self, const Tensor& other) {
  if (self.is_complex()) {
    if (self.is_conj()) {
      if (other.is_conj()) {
        return (dot_cuda(self.conj(), other.conj())).conj();
       } else {
         return vdot_cuda(self.conj(), other);
       }
    } else if (other.is_conj()) {
      return vdot_cuda(other.conj(), self);
    }
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

if (self._is_zerotensor() || other._is_zerotensor()) {
  return at::_efficientzerotensor({}, self.options());
}

return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "dot",
      [&] {
        Tensor result = at::empty({}, self.options());

        auto handle = at::cuda::getCurrentCUDABlasHandle();
        at::cuda::blas::PointerModeGuard pointerModeGuard(handle, CUBLAS_POINTER_MODE_DEVICE);
        at::cuda::blas::dot<scalar_t>(
            handle,
            n,
            self.const_data_ptr<scalar_t>(),
            incx,
            other.const_data_ptr<scalar_t>(),
            incy,
            result.mutable_data_ptr<scalar_t>());

        return result;
      });
}

Tensor vdot_cuda(const Tensor& self, const Tensor& other) {
  if (!self.is_complex()) {
    return dot_cuda(self, other);
  }

  if (self.is_conj()) {
    if (other.is_conj()) {
      return vdot_cuda(other.conj(), self.conj());
    } else {
      return dot_cuda(self.conj(), other);
    }
  } else if (other.is_conj()) {
    return (dot_cuda(self, other.conj())).conj();
  }

  at::NoNamesGuard guard;
  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  const int n = static_cast<int>(self.numel());
  int incx = static_cast<int>(self.stride(0));
  int incy = static_cast<int>(other.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, self.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointerModeGuard(
        handle, CUBLAS_POINTER_MODE_DEVICE);
    at::cuda::blas::vdot<scalar_t>(
        handle,
        n,
        self.const_data_ptr<scalar_t>(),
        incx,
        other.const_data_ptr<scalar_t>(),
        incy,
        result.mutable_data_ptr<scalar_t>());

    return result;
  });
}

TORCH_IMPL_FUNC(addmv_out_cuda)(const Tensor &self, const Tensor &mat, const Tensor &vec, const Scalar& beta_, const Scalar& alpha_, const Tensor& result) {
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();
  if (mat.numel() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (betaval == 0.0) {
      result.zero_();
    } else {
      at::mul_out(
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), c10::nullopt /* layout */, at::kCPU, c10::nullopt /* pin_memory */));
    }
  } else {
    if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents will be zeroed later
      at::native::copy_(const_cast<Tensor&>(result), *self_);
    }
    if (result.numel() != 0) {
      auto r_stride = result.stride(0);
      auto vec_stride = vec.stride(0);

      // Check for contiguity of `vec` and update `vec_stride` accordingly
      const auto vec_contiguous = vec_stride == 0 ? vec.contiguous() : vec;
      // A vector can be contiguous and have a stride of zero if it has it is of length 1
      vec_stride = std::max<int64_t>(vec_contiguous.stride(0), 1LL);

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "addmv_impl_cuda", [&] {
        auto beta = beta_.to<scalar_t>();
        auto alpha = alpha_.to<scalar_t>();
        if (mat.stride(0) == 1 && mat.stride(1) >= std::max<int64_t>(1, mat.size(0))) {
          at::cuda::blas::gemv<scalar_t>('n',
            mat.size(0), mat.size(1), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(1), vec_contiguous.const_data_ptr<scalar_t>(),
            vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        else if (mat.stride(1) == 1 && mat.stride(0) >= std::max<int64_t>(1, mat.size(1))) {
          at::cuda::blas::gemv<scalar_t>('t',
            mat.size(1), mat.size(0), alpha, mat.const_data_ptr<scalar_t>(), mat.stride(0),
            vec_contiguous.const_data_ptr<scalar_t>(), vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
        else {
          Tensor cmat = mat.contiguous();
          at::cuda::blas::gemv<scalar_t>('t',
              mat.size(1), mat.size(0), alpha, cmat.const_data_ptr<scalar_t>(), cmat.stride(0),
              vec_contiguous.const_data_ptr<scalar_t>(), vec_stride, beta, result.mutable_data_ptr<scalar_t>(), r_stride);
        }
      });
    }
  }
}

Tensor& _int_mm_out_cuda(const Tensor& self, const Tensor& mat2, Tensor& result) {
  // NOTE: cuBLAS is currently broken for some combination of transposed inputs.
  TORCH_CHECK(self.dim() == 2, "Expected self to be of dimension 2 but got ", self.dim());
  TORCH_CHECK(mat2.dim() == 2, "Expected mat2 to be of dimension 2 but got ", mat2.dim());
  TORCH_CHECK(self.size(0) > 16, "self.size(0) needs to be greater than 16, but got ", self.size(0));
  TORCH_CHECK(self.size(1) > 0 && self.size(1) % 8 == 0, "self.size(1) needs to be greater than 0 and a multiple of 8, but got ", self.size(1));
  TORCH_CHECK(self.size(1) == mat2.size(0), "self.size(1) needs to match mat2.size(0) but got ", self.size(1), " and ", mat2.size(0));
  TORCH_CHECK(mat2.size(1) > 0 && mat2.size(1) % 8 == 0, "mat2.size(1) needs to be greater than 0 and a multiple of 8, but got ", mat2.size(1));

  TORCH_CHECK(result.dtype() == at::kInt, "Expected result dtype to be of type kInt but got ", result.dtype());
  TORCH_CHECK(result.size(0) == self.size(0), "Expected result.size(0) to be ", self.size(0), " but got ", result.size(0));
  TORCH_CHECK(result.size(1) == mat2.size(1), "Expected result.size(1) to be ", mat2.size(1), " but got ", result.size(1));

  TORCH_CHECK(result.dim() == 2, "Expected result to be of dimension 2 but got ", result.dim());

  TORCH_CHECK(result.is_contiguous(), "Expected result to be contiguous.");

#if !defined(USE_ROCM) && !defined(_MSC_VER) && defined(CUDA_VERSION) && CUDA_VERSION == 11070
  auto mat1 = self;
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  bool transpose_result;
  c10::MaybeOwned<Tensor> result_ = prepare_matrix_for_cublas(result, transpose_result);
  bool transpose_mat1;
  bool transpose_mat2;
  c10::MaybeOwned<Tensor> mat1_ = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
  c10::MaybeOwned<Tensor> mat2_ = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);

  if (transpose_result) {
    transpose_mat1 = !transpose_mat1;
    transpose_mat2 = !transpose_mat2;
    mat1_sizes = mat1_->sizes();
    mat2_sizes = mat2_->sizes();
  }

  int64_t m = mat1_sizes[transpose_result ? 1 : 0];
  int64_t k = mat1_sizes[transpose_result ? 0 : 1];
  int64_t n = mat2_sizes[transpose_result ? 0 : 1];
  int64_t mat1_ld = mat1_->stride((transpose_mat1 == transpose_result) ? 1 : 0);
  int64_t mat2_ld = mat2_->stride((transpose_mat2 == transpose_result) ? 1 : 0);
  int64_t result_ld = result_->stride(transpose_result ? 0 : 1);

  at::cuda::blas::int8_gemm(
      transpose_mat1,
      transpose_mat2,
      m,
      n,
      k,
      mat1_->data_ptr<int8_t>(),
      mat1_ld,
      mat2_->data_ptr<int8_t>(),
      mat2_ld,
      result_->data_ptr<int32_t>(),
      result_ld);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#else
#if !defined(USE_ROCM) && !defined(_MSC_VER) && defined(CUDA_VERSION)
  TORCH_CHECK(false, "_int_mm_out_cuda not compiled for CUDA ", CUDA_VERSION);
#else
  TORCH_CHECK(false, "_int_mm_out_cuda not compiled for this platform.");
#endif
#endif

  return result;
}

Tensor _int_mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({self.size(0), mat2.size(1)}, self.options().dtype(at::kInt));
  return _int_mm_out_cuda(self, mat2, result);
}

} // namespace at::native
