#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAScaledBlas.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/cuBlasCommonArgs.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#include <ATen/ceil_div.h>

#ifdef USE_FBGEMM_GENAI
#include <fbgemm_gpu/torch_ops.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

namespace at::native {

using at::blas::ScalingType;
using at::blas::SwizzleType;

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

/*
 * Checks whether DISABLE_ADDMM_CUDA_LT is set.
 * Additionally, for ROCM we test whether the architecture supports the Lt.
 */
static bool isGloballyDisabledAddmmCudaLt(const at::Device& device) {
  // When hipBLASLt is not supported on the architecture, return true
  #ifdef USE_ROCM
  static const std::vector<std::string> archs = {
        "gfx90a", "gfx942",
    #if ROCM_VERSION >= 60300
        "gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx908",
    #endif
    #if ROCM_VERSION >= 70000
        "gfx950", "gfx1150", "gfx1151"
    #endif
  };
  const auto is_hipblas_lt_arch_supported = at::detail::getCUDAHooks().isGPUArch(archs, device.index());
  if (!is_hipblas_lt_arch_supported) {
    return true;
  }
  #endif

  // Check whether it is disabled in the env
  static const auto is_addmm_cuda_lt_disabled = c10::utils::get_env("DISABLE_ADDMM_CUDA_LT");
  if (is_addmm_cuda_lt_disabled == "1") {
    return true;
  }

  return false;
}

/*
 * Check whether for the given input we want to enable the Lt interface
 */
static bool isInputCompliesAddmmCudaLt(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Activation activation
) {
  #ifdef USE_ROCM
  // Implies 2D bias which we currently not send through Lt.
  // TODO: this check is done pre col-major input preparation,
  // so, this condition can be ralexed in cases when a col-major
  // copy of result is needed.
  if (self.is_same(result) || self.dim() == 2) {
    return false;
  }
  #endif

  #if defined(USE_ROCM) && ROCM_VERSION == 60400
  // hipblaslt TT fp32 regression on ROCm 6.4, cannot use
  const auto args = cublasCommonArgs(mat1, mat2, result);
  if (args.transa == 't' && args.transb == 't') {
    return false;
  }
  #endif

  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();
  #if defined(CUDA_VERSION) || defined(USE_ROCM)
  const auto scalar_type = mat1.scalar_type();
  return (beta.toComplexDouble() == 1.0
    // NOTE: row-major result is important when bias is 1D.
    // This is because Lt broadcasts 1D bias over the columns
    // while the aten::addmm API broadcasts it over the rows,
    // and this is in conjuction with the data preparation
    // procedure that does not transpose arguments with
    // col-major result. For col-major result we need
    // to explicitly transpose the problem so that bias is
    // correctly applied.
    // TODO: enable col-major result if needed.
    // TODO: no need to check result's layout when
    // !result.is_same(self) and self.dim() == 2, because
    // self needs to be copied into result and the bias ptr
    // will be ignored.
    && result.dim() == 2 && result.is_contiguous()
    && (
      ( // Conditions for bias to be fusable -- implies direct Lt path without copies.
        self.is_contiguous() &&
        // NOTE: fine to have 1-len dims to the left from the right-most one
        (self.dim() == 1 || self.squeeze().dim() == 1) &&
        self.sizes().back() == mat2_sizes[1]
      )
      || ( // 2D bias restrictions. self.is_contiguous() is implicit when result.is_same(self),
        // and we need to copy self into result otherwise, so the self's layout becomes irrelevant.
        // See also TODO from above.
        activation != Activation::None && // Lt is faster when activation is fused
        (self.dim() == 2 && at::is_expandable_to(self.sizes(), {mat1_sizes[0], mat2_sizes[1]}))
      )
    )
    && ( // some dtype restrictions
      #ifndef USE_ROCM
      scalar_type == at::ScalarType::Double ||
      #endif
      scalar_type == at::ScalarType::Float ||
      scalar_type == at::ScalarType::Half ||
      scalar_type == at::ScalarType::BFloat16
    )
    && ( // some shape/stride restrictions
      // Strangely, if mat2 has only 1 row or column, we get
      // CUBLAS_STATUS_INVALID_VALUE error from cublasLtMatmulAlgoGetHeuristic.
      // NOTE: extension to mat1 because mat1/mat2 can be swapped based off
      // their row-/col-majorness.
      mat1_sizes[0] > 1 && mat1_sizes[1] > 1 &&
      mat2_sizes[0] > 1 && mat2_sizes[1] > 1
      // The last conditions is to skip 16b transA and non-trans-B having
      // leading dim >> rows when they are sliced from a large tensor
      // see fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul
      #if !(defined(CUDA_VERSION) && CUDA_VERSION >= 12010 || defined(USE_ROCM))
      // Related to avoiding the leading stride >> leading dim problematic case
      // with 16b dtypes described above. For such dtypes we only allow inputs
      // which are either row- or col-major (i.e. non-overlapping, compact memory layout).
      // In that case the leading stride will be equal to the outer dim len.
      // Why do we catch this case here? The following `prepare_matrix_for_cublas` method
      // does not modify inputs as long as there is a stride of length 1
      // and the leading stride is at least max(1, other dim length), so we might
      // end up with contiguous cols but not rows (i.e. holes between different rows)
      // and vice versa.
      && mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32
      && mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32
      && (
        // filter by dtype
        (scalar_type != at::ScalarType::Half && scalar_type != at::ScalarType::BFloat16) ||
        // check mat1/mat2 is row-/col-major
        (mat1.is_non_overlapping_and_dense() && mat2.is_non_overlapping_and_dense())
      )
      #endif
    )
  );
  #endif

  // no compliance by default
  return false;
}

template <typename scalar_t>
void launchTunableGemmAndBias(cublasCommonArgs &args, const Scalar& alpha, const scalar_t* bias, cuda::blas::GEMMAndBiasActivationEpilogue activation) {
  bool transa_ = ((args.transa != 'n') && (args.transa != 'N'));
  bool transb_ = ((args.transb != 'n') && (args.transb != 'N'));
  at::cuda::tunable::GemmAndBiasParams<scalar_t> params;
  params.transa = args.transa;
  params.transb = args.transb;
  params.m = args.m;
  params.n = args.n;
  params.k = args.k;
  params.alpha = alpha.to<at::opmath_type<scalar_t>>();
  params.a = args.mata->const_data_ptr<scalar_t>();
  params.lda = args.lda;
  params.b = args.matb->const_data_ptr<scalar_t>();
  params.ldb = args.ldb;
  params.c = args.result->data_ptr<scalar_t>();
  params.ldc = args.result_ld;
  params.bias = bias;
  params.activation = activation;
  if (transa_ && transb_) {
    static at::cuda::tunable::GemmAndBiasTunableOp<scalar_t, at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else if (transa_ && !transb_) {
    static at::cuda::tunable::GemmAndBiasTunableOp<scalar_t, at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
  else if (!transa_ && transb_) {
    static at::cuda::tunable::GemmAndBiasTunableOp<scalar_t, at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else if (!transa_ && !transb_) {
    static at::cuda::tunable::GemmAndBiasTunableOp<scalar_t, at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
  else {
    TORCH_CHECK(false, "unreachable");
  }
}

template <typename scalar_t, typename res_scalar_t = scalar_t>
bool launchGemmAndBiasCublasLt(
    // args contains result which is modified
    cublasCommonArgs& args,
    const Tensor& self,
    const Scalar& alpha,
    Activation activation = Activation::None
) {
  // We apply bias in the epilogue only when it is 1D,
  // or when it can be squeezed to 1D.
  // self_ptr == nullptr implies ignore bias epilogue
  // and use standard gemm-like API.
  const auto* self_ptr = [&]() -> auto {
    if (self.dim() == 1 || self.squeeze().dim() == 1) {
      return self.const_data_ptr<scalar_t>();
    }
    return static_cast<const scalar_t*>(nullptr);
  }();

  const auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    // TODO: maybe also return some success state?
    launchTunableGemmAndBias<scalar_t>(
      args, alpha, self_ptr, activation_to_gemm_and_blas_arg(activation)
    );
    return true;
  }

  return at::cuda::blas::gemm_and_bias<scalar_t, res_scalar_t>(
    args.transa == 't',
    args.transb == 't',
    args.m,
    args.n,
    args.k,
    alpha.to<at::opmath_type<scalar_t>>(),
    args.mata->const_data_ptr<scalar_t>(),
    args.lda,
    args.matb->const_data_ptr<scalar_t>(),
    args.ldb,
    self_ptr,
    args.result->data_ptr<res_scalar_t>(),
    args.result_ld,
    activation_to_gemm_and_blas_arg(activation)
  );
}

template <typename scalar_t, typename res_scalar_t = scalar_t>
bool launchGemmCublas(
    // args contains result which is modified
    cublasCommonArgs& args,
    const Scalar& alpha,
    const Scalar& beta
) {
  at::cuda::blas::gemm<scalar_t, res_scalar_t>(
    args.transa,
    args.transb,
    args.m,
    args.n,
    args.k,
    alpha.to<at::opmath_type<scalar_t>>(),
    args.mata->const_data_ptr<scalar_t>(),
    args.lda,
    args.matb->const_data_ptr<scalar_t>(),
    args.ldb,
    beta.to<at::opmath_type<scalar_t>>(),
    args.result->data_ptr<res_scalar_t>(),
    args.result_ld
  );
  return true; // success!
}

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None, bool disable_addmm_cuda_lt_override=false) {
  // Shape checks {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  if (result.is_same(self)) {
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self.sizes()[0] == mat1.sizes()[0], "self dim 0 must match mat1 dim 0");
    TORCH_CHECK(self.sizes()[1] == mat2.sizes()[1], "self dim 1 must match mat2 dim 1");
  }
  // } Shape checks

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, targs);

  // Handle whether to use the Lt interface {
  static bool persistent_disable_addmm_cuda_lt = isGloballyDisabledAddmmCudaLt(self.device());
  // if lt path fails, we recurse back into this function here and force the lt path to off
  // we cannot update variable disable_addmm_cuda_lt from above since it is static and would be permanent
  bool disable_addmm_cuda_lt = persistent_disable_addmm_cuda_lt || disable_addmm_cuda_lt_override;
  #ifdef USE_ROCM
  // Conditioned on the device index, which is not persistent
  disable_addmm_cuda_lt = isGloballyDisabledAddmmCudaLt(self.device()) || disable_addmm_cuda_lt;
  #endif
  // Condition on the input
  disable_addmm_cuda_lt = !isInputCompliesAddmmCudaLt(result, self, mat1, mat2, beta, alpha, activation) || disable_addmm_cuda_lt;
  // }

  at::ScalarType scalar_type = mat1.scalar_type();
  bool is_float_output_with_half_input = (scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::BFloat16) && result.scalar_type() == at::ScalarType::Float;

  // Handle result/self shapes
  if (!result.is_same(self)) {
    at::native::resize_output(result, {mat1.sizes()[0], mat2.sizes()[1]});

    // We use bias ptr in the Lt path only when bias is 1D
    const auto use_bias_ptr_lt = (self.dim() == 1) && !disable_addmm_cuda_lt;
    const auto self_maybe_expanded = [&]() -> c10::MaybeOwned<Tensor> {
      if (!use_bias_ptr_lt) {
        // We do expand self even before
        // check for beta != 0.0 to make sure that
        // test_sparse_csr.py::TestSparseCSRCUDA::test_addmm_errors_*
        // runs green.
        return expand_size(self, result.sizes(), "addmm");
      }
      return c10::MaybeOwned<Tensor>::borrowed(self);
    }();
    // We do not copy bias only when we need the bias ptr
    if (beta.toComplexDouble() != 0.0 && !use_bias_ptr_lt) {
      // NOTE: self should broadcast over result
      at::native::copy_(result, *self_maybe_expanded);
    }
  }

  // Short circuit on empty result
  if (result.numel() == 0) {
    return result;
  }

  // Short circuit if the reduction dim is empty
  if (mat1.sizes()[1] == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    // TODO: We could squeeze some perf by calling at::cuda::mul_out here instead, to bypass the dispatcher.
    // That requires some fixing some internal build dependencies though.
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::scalar_tensor(
          beta,
          self.scalar_type(),
          std::nullopt /* layout */,
          at::kCPU,
          std::nullopt /* pin_memory */
        )
    );
  }

  cublasCommonArgs args(mat1, mat2, result);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!args.result->is_conj());

  // The Lt path
  if (!disable_addmm_cuda_lt) {
    bool lt_success = false;
    if (is_float_output_with_half_input) {
      #ifdef USE_ROCM
      TORCH_CHECK(false, "float output with half input is not enabled for ROCm");
      #else
      if (at::cuda::tunable::getTuningContext()->IsTunableOpEnabled()) {
       TORCH_CHECK(false, "Tunable GEMM is not supported for float output with reduced float input");
      }
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
        scalar_type,
        "addmm_cuda_lt",
        [&] {
          lt_success = launchGemmAndBiasCublasLt<scalar_t, float>(args, self, alpha, activation);
        }
      );
      #endif
    } else {
      // !is_float_output_with_half_input
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
          lt_success = launchGemmAndBiasCublasLt<scalar_t>(args, self, alpha, activation);
        }
      );
    } // end is_float_output_with_half_input

    if (!lt_success) {
    // lt path failed; recurse but disable lt path
      return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha, activation, true);
    }
    // end Lt path
  } else {
    // No Lt, we use a GEMM instead
    if (is_float_output_with_half_input) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
        scalar_type,
        "addmm_cuda",
        [&] {
          launchGemmCublas<scalar_t, float>(args, alpha, beta);
        }
      );
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda",
        [&] {
          launchGemmCublas<scalar_t>(args, alpha, beta);
        }
      );
    }

    // Apply epilogue
    switch (activation) {
      case Activation::RELU:
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        at::relu_(const_cast<Tensor&>(*args.result));
        break;
      case Activation::GELU:
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        at::gelu_(const_cast<Tensor&>(*args.result), "tanh");
        break;
      default: break;
    }
  } // end GEMM path

// Preprocessor gate here needs to match the inverse of the check
// gating activation_to_gemm_and_blas_arg above; here we are manually
// performing a post-GELU because we weren't able to use the GELU
// epilogue above.
#if !defined(CUDA_VERSION) && !defined(USE_ROCM)
  if (!disable_addmm_cuda_lt && activation == Activation::GELU) {
    at::gelu_(const_cast<Tensor&>(*args.result), "tanh");
  }
#endif

  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
  }
  return result;
}

const Tensor& baddbmm_out_cuda_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
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

  int64_t lda = 0, ldb = 0, ldc = 0;
  bool transpose_batch1 = false, transpose_batch2 = false;
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());
  bool is_float_output_with_half_input = (batch1.scalar_type() == at::ScalarType::Half || batch1.scalar_type() == at::ScalarType::BFloat16) && result.scalar_type() == at::ScalarType::Float;

  if (is_float_output_with_half_input) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(batch1.scalar_type(), "baddbmm_cuda", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_t alpha_val = alpha.to<opmath_t>();
      opmath_t beta_val = beta.to<opmath_t>();
      const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
      const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
      const auto transa = transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n';
      const auto transb = transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n';

      float* result_ptr = result_->mutable_data_ptr<float>();

      // If batch is 1 call gemm rather than bgemm
      if (num_batches == 1) {
          at::cuda::blas::gemm<scalar_t, float>(
              transa, transb,
              m, n, k,
              alpha_val,
              batch1_ptr, lda,
              batch2_ptr, ldb,
              beta_val,
              result_ptr, ldc);
        } else {
          at::cuda::blas::bgemm<scalar_t, float>(
            transa, transb,
            m, n, k,
            alpha_val,
            batch1_ptr, lda, batch1_->strides()[0],
            batch2_ptr, ldb, batch2_->strides()[0],
            beta_val,
            result_ptr, ldc, result_->strides()[0],
            num_batches
          );
        }
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, batch1.scalar_type(), "baddbmm_cuda", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_t alpha_val = alpha.to<opmath_t>();
      opmath_t beta_val = beta.to<opmath_t>();
      const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
      const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
      const auto transa = transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n';
      const auto transb = transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n';
      scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
      // If batch is 1 call gemm rather than bgemm
      if (num_batches == 1) {
        at::cuda::blas::gemm<scalar_t>(
            transa, transb,
            m, n, k,
            alpha_val,
            batch1_ptr, lda,
            batch2_ptr, ldb,
            beta_val,
            result_ptr, ldc);
      } else {
        at::cuda::blas::bgemm<scalar_t>(
          transa, transb,
          m, n, k,
          alpha_val,
          batch1_ptr, lda, batch1_->strides()[0],
          batch2_ptr, ldb, batch2_->strides()[0],
          beta_val,
          result_ptr, ldc, result_->strides()[0],
          num_batches
        );
      }
    });
  }
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // anonymous namespace

TORCH_IMPL_FUNC(addmm_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha);
}

TORCH_IMPL_FUNC(addmm_activation_out_cuda)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  addmm_out_cuda_impl(const_cast<Tensor&>(result), self, mat1, mat2, beta, alpha, use_gelu ? Activation::GELU : Activation::RELU);
}

TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
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
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_, self.scalar_type(), std::nullopt /* layout */, at::kCPU, std::nullopt /* pin_memory */));
    }
  } else {
    if (!result.is_same(*self_) && betaval != 0.0) { //if beta is 0, result contents will be zeroed later
                                                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
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

#if defined(CUDA_VERSION) || defined(USE_ROCM)
  cublasCommonArgs args(self, mat2, result);

  at::cuda::blas::int8_gemm(
      args.transa == 't',
      args.transb == 't',
      args.m,
      args.n,
      args.k,
      args.mata->data_ptr<int8_t>(),
      args.lda,
      args.matb->data_ptr<int8_t>(),
      args.ldb,
      args.result->data_ptr<int32_t>(),
      args.result_ld);

  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
  }
#else
#if !defined(USE_ROCM) && defined(CUDA_VERSION)
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

static void baddbmm_bmm_out_dtype_checks(const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, const at::ScalarType out_dtype, bool is_bmm, const std::optional<Tensor>& self_baddbmm = std::nullopt) {
  // ref ATen/native/LinearAlgebra.cpp common_checks_baddbmm_bmm
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  std::vector<int64_t> output_size {bs, res_rows, res_cols};

  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");

  TORCH_CHECK(batch1.scalar_type() == batch2.scalar_type(), "batch1 and batch2 must have the same dtype");

  TORCH_CHECK(out_dtype == batch1.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (batch1.scalar_type() == at::ScalarType::Half || batch1.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");

  if (!is_bmm && self_baddbmm.has_value()) {
    const auto& self = self_baddbmm.value();
    TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
    TORCH_CHECK(self.sizes() == output_size, "self must have the same shape as the output");
  }
}

Tensor _bmm_dtype_cuda(const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype) {
  IntArrayRef batch1_sizes = batch1.sizes();
  IntArrayRef batch2_sizes = batch2.sizes();

  Tensor out = at::empty({batch1_sizes[0], batch1_sizes[1], batch2_sizes[2]}, batch1.options().dtype(out_dtype));
  return _bmm_out_dtype_cuda(batch1, batch2, out_dtype, out);
}

Tensor& _bmm_out_dtype_cuda(const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype, Tensor &out) {
  baddbmm_bmm_out_dtype_checks(batch1, batch2, 0.0, 1.0, out_dtype, true);
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(out, out, batch1, batch2, beta, alpha);
  }

  return out;
}

Tensor _baddbmm_dtype_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha) {
  // We need to copy the tensor
  Tensor out = self.clone().to(self.options().dtype(out_dtype));

  return _baddbmm_out_dtype_cuda(out, batch1, batch2, out_dtype, beta, alpha, out);
}

Tensor& _baddbmm_out_dtype_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha, Tensor &out) {
  baddbmm_bmm_out_dtype_checks(batch1, batch2, beta, alpha, out_dtype, false, self);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(out, out, batch1, batch2, beta, alpha);
  }

  return out;
}

Tensor _mm_dtype_cuda(const Tensor& self, const Tensor& mat2, const at::ScalarType out_dtype) {
  Tensor result = at::empty({self.size(0), mat2.size(1)}, self.options().dtype(out_dtype));
  return _mm_dtype_out_cuda(self, mat2, out_dtype, result);
}

Tensor& _mm_dtype_out_cuda(const Tensor& self, const Tensor& mat2, const at::ScalarType out_dtype, Tensor &out) {
  TORCH_CHECK(self.dim() == 2,  "self must be a matrix, got ", self.dim(), "-D tensor");
  TORCH_CHECK(mat2.dim() == 2,  "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x", self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "input dtypes must be the same");
  TORCH_CHECK(out_dtype == self.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");


  addmm_out_cuda_impl(out, out, self, mat2, 0, 1);

  return out;
}

Tensor _addmm_dtype_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty(self.sizes(), self.options().dtype(out_dtype));
  return _addmm_dtype_out_cuda(self, mat1, mat2, out_dtype, beta, alpha, result);
}

Tensor& _addmm_dtype_out_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha, Tensor &out) {
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype, but got ", self.scalar_type(), " and ", mat2.scalar_type());
  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type());
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");
  TORCH_CHECK(out_dtype == self.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");

  addmm_out_cuda_impl(out, self, mat1, mat2, beta, alpha);

  return out;
}


} // namespace at::native
