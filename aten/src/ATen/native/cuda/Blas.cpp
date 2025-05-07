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
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>

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


/**
 * @brief Prepares matrices for CUBLAS operation
 *
 * This constructor prepares tensors for CUBLAS
 * The main difference is that PyTorch uses row-major as the default and
 * CUBLAS expects column-major.
 *
 * @details
 * To enable row-major output while using CUBLAS,
 * we use the mathematical identity that (A × B)^T = B^T × A^T.
 *
 * Transpose in this context refers to Cublas's(Fortran) definition of transpose (row-major)
 * T = row-major, N = col-major
 *
 * Example:
 * For matrices A (M×K)(row-major) and B (K×N)(row-major):
 *   - Standard multiplication: A × B = (M×K) × (K×N) = M×N result (row-major)
 *   - Using our transpose trick: (B^T × A^T) = (N×K)(T) × (K×M)(T) = N×M(N)
 *   - However, since the output form cublas is column-major this is
 *   - equivalent to an output of size MxN row-major as expected
 *
 * The transpose flags are derived from the layouts of the passed in tensors
 *
 * If the operands are in packed float4 format, `k`, `lda` and `ldb` are adjusted
 * to their unpacked values to match what cuBLAS expects.
 *
 * @param mat1 First input matrix
 * @param mat2 Second input matrix
 * @param c Output matrix (result)
 * @param scale_a Optional scaling factor for first matrix
 * @param scale_b Optional scaling factor for second matrix
 * @param scale_result Optional scaling factor for result
 */
struct cublasCommonArgs {
  cublasCommonArgs(
      const Tensor& mat1,
      const Tensor& mat2,
      Tensor& c,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt) {
    bool transpose_result = false, transpose_a = false, transpose_b = false;
    result = prepare_matrix_for_cublas(c, transpose_result);
    mata = prepare_matrix_for_cublas(transpose_result ? mat2 : mat1, transpose_a, transpose_result);
    matb = prepare_matrix_for_cublas(transpose_result ? mat1 : mat2, transpose_b, transpose_result);

    // Handle scale tensors if provided
    if (scale_a && scale_b) {
      // By default since we return in row-major we run the gemm
      // as B.T @ A.T, check transpose_result to determine if we flip the scales
      scale_mata_ptr = transpose_result ? scale_b->data_ptr() : scale_a->data_ptr();
      scale_mata_dtype = transpose_result ? scale_b->scalar_type() : scale_a->scalar_type();
      scale_matb_ptr = transpose_result ? scale_a->data_ptr() : scale_b->data_ptr();
      scale_matb_dtype = transpose_result ? scale_a->scalar_type() : scale_b->scalar_type();
    }

    if (scale_result) {
      scale_result_ptr = scale_result->data_ptr();
      scale_result_dtype = scale_result->scalar_type();
    }

    // Update transpose flags
    if (transpose_result) {
      transpose_a = !transpose_a;
      transpose_b = !transpose_b;
    }

    auto sizes_a = mata->sizes();
    auto sizes_b = matb->sizes();

    m = sizes_a[transpose_result ? 1 : 0];
    k = sizes_a[transpose_result ? 0 : 1];
    n = sizes_b[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_a == transpose_result) ? 1 : 0);
    ldb = matb->stride((transpose_b == transpose_result) ? 1 : 0);
    result_ld = result->stride(transpose_result ? 0 : 1);
    transa = transpose_a ? mata->is_conj() ? 'c' : 't' : 'n';
    transb = transpose_b ? matb->is_conj() ? 'c' : 't' : 'n';

    // cuBLAS expects unpacked values of `k`, `lda` and `ldb`, adjust for 4x2 packing
    // if the gemm operands are in packed float4
    if (mat1.dtype() == at::kFloat4_e2m1fn_x2 && mat2.dtype() == at::kFloat4_e2m1fn_x2) {
      k = k * 2;
      lda = lda * 2;
      ldb = ldb * 2;
    }
  }

  // Matrix members
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;

  // Scale members
  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  std::optional<c10::ScalarType> scale_mata_dtype;
  std::optional<c10::ScalarType> scale_matb_dtype;
  std::optional<c10::ScalarType> scale_result_dtype;
};
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

static bool getDisableAddmmCudaLt() {
    static const auto env_value = c10::utils::get_env("DISABLE_ADDMM_CUDA_LT");
    if (env_value == "1") {
      return true;
    }
    return false;
}

#ifdef USE_ROCM
static bool isSupportedHipLtROCmArch(int index) {
    static const std::vector<std::string> archs = {
        "gfx90a", "gfx942",
#if ROCM_VERSION >= 60300
        "gfx1100", "gfx1101", "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
        "gfx950"
#endif
    };
    return at::detail::getCUDAHooks().isGPUArch(archs, index);
}
#endif

template <typename scalar_t>
static void launchTunableGemmAndBias(cublasCommonArgs &args, const Scalar& alpha, const scalar_t* bias, cuda::blas::GEMMAndBiasActivationEpilogue activation) {
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

Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None, bool disable_addmm_cuda_lt_override=false) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, targs);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  bool useLtInterface = false;
#if defined(USE_ROCM)
  // When hipBLASLt is not supported on the architecture,
  // disable_addmm_cuda_lt will always be to set to true
  static bool disable_addmm_cuda_lt =
    !isSupportedHipLtROCmArch(self.device().index()) || getDisableAddmmCudaLt();
#else
  static bool disable_addmm_cuda_lt = getDisableAddmmCudaLt();
#endif
  // if lt path fails, we recurse back into this function here and force the lt path to off
  disable_addmm_cuda_lt |= disable_addmm_cuda_lt_override;
  at::ScalarType scalar_type = mat1.scalar_type();
  bool is_float_output_with_half_input = (scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::BFloat16) && result.scalar_type() == at::ScalarType::Float;
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11040)) || defined(USE_ROCM)
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
#ifdef USE_ROCM
          (scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#else
          (scalar_type == at::ScalarType::Double ||
           scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12010 || defined(USE_ROCM))
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1;
#else
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1 &&
          mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32 &&
          mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32 &&
          // avoid leading dim >> rows bugs
          ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
           (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16)) &&
          ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
           (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16));
#endif
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

  cublasCommonArgs args(mat1, mat2, result);

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
        self.expand(result.sizes()),
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            std::nullopt /* layout */,
            at::kCPU,
            std::nullopt /* pin_memory */));
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!args.result->is_conj());

  if (useLtInterface) {
#if defined(USE_ROCM)
    bool okay = true;
    if (is_float_output_with_half_input) {
      TORCH_CHECK(false, "float output with half input is not enabled for ROCm");
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          launchTunableGemmAndBias<scalar_t>(
              args,
              alpha,
              (&result != &self) ? self.const_data_ptr<scalar_t>() : nullptr,
              activation_to_gemm_and_blas_arg(activation));
        }

        okay = at::cuda::blas::gemm_and_bias<scalar_t>(
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
            // This condition is needed for mm case on ROCm for hipblasLt path.
            // Passing the bias ptr as null to avoid accuracy issues for mm case.
            (&result != &self) ? self.const_data_ptr<scalar_t>() : nullptr,
            args.result->data_ptr<scalar_t>(),
            args.result_ld,
            activation_to_gemm_and_blas_arg(activation)
        );
      });
    }
    if (!okay) {
      // lt path failed; recurse but disable lt path
      return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha, activation, true);
    }
#else
    auto activation_epilogue = activation_to_gemm_and_blas_arg(activation);
#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11080))
    // GELU is not supported (and does not compile!) prior
    // to CUDA 11.4. Have observed accuracy issues with
    // GELU epilogue in 11.4; disabling the GELU epilogue
    // path for CUDA version < 11.8.
    if (activation == Activation::GELU)
      activation_epilogue = cuda::blas::GEMMAndBiasActivationEpilogue::None;
#endif

    bool okay = true;
    if (is_float_output_with_half_input) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
        scalar_type,
        "addmm_cuda_lt",
        [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          TORCH_CHECK(false, "Tunable GEMM is not supported for float output with reduced float input");
        }
        else {
          okay = at::cuda::blas::gemm_and_bias<scalar_t, float>(
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
              self.const_data_ptr<scalar_t>(),
              args.result->data_ptr<float>(),
              args.result_ld,
              activation_epilogue
          );
        }});
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          launchTunableGemmAndBias<scalar_t>(
              args,
              alpha,
              self.const_data_ptr<scalar_t>(),
              activation_epilogue);
        }
        else {
          okay = at::cuda::blas::gemm_and_bias<scalar_t>(
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
              self.const_data_ptr<scalar_t>(),
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_epilogue
          );
      }});
    }
    if (!okay) {
      // lt path failed; recurse but disable lt path
      return addmm_out_cuda_impl(result, self, mat1, mat2, beta, alpha, activation, true);
    }
#endif
  } else
  {
    if (is_float_output_with_half_input) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
        scalar_type,
        "addmm_cuda",
        [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t alpha_val = alpha.to<opmath_t>();
          opmath_t beta_val = beta.to<opmath_t>();
          const scalar_t* mat1_ptr = args.mata->const_data_ptr<scalar_t>();
          const scalar_t* mat2_ptr = args.matb->const_data_ptr<scalar_t>();

          float* result_ptr = args.result->mutable_data_ptr<float>();
          at::cuda::blas::gemm<scalar_t, float>(
              args.transa,
              args.transb,
              args.m,
              args.n,
              args.k,
              alpha_val,
              mat1_ptr,
              args.lda,
              mat2_ptr,
              args.ldb,
              beta_val,
              result_ptr,
              args.result_ld);
        });
    } else {
      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda",
        [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t alpha_val = alpha.to<opmath_t>();
          opmath_t beta_val = beta.to<opmath_t>();
          const scalar_t* mat1_ptr = args.mata->const_data_ptr<scalar_t>();
          const scalar_t* mat2_ptr = args.matb->const_data_ptr<scalar_t>();
          scalar_t* result_ptr = args.result->mutable_data_ptr<scalar_t>();
          at::cuda::blas::gemm<scalar_t>(
              args.transa,
              args.transb,
              args.m,
              args.n,
              args.k,
              alpha_val,
              mat1_ptr,
              args.lda,
              mat2_ptr,
              args.ldb,
              beta_val,
              result_ptr,
              args.result_ld);
        });
    }
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
  }

// Preprocessor gate here needs to match the inverse of the check
// gating activation_to_gemm_and_blas_arg above; here we are manually
// performing a post-GELU because we weren't able to use the GELU
// epilogue above.
#if !(defined(CUDA_VERSION) && CUDA_VERSION >= 11080) && !defined(USE_ROCM)
  if (useLtInterface && activation == Activation::GELU) {
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

#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11070)) || defined(USE_ROCM)
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

static bool _scaled_mm_allowed_device(bool sm90_only=false) {
#ifdef USE_ROCM
    static const std::vector<std::string> archs = {
        "gfx942",
#if ROCM_VERSION >= 60300
        "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
        "gfx950"
#endif
    };
    return at::detail::getCUDAHooks().isGPUArch(archs);
#else
    auto dprops = at::cuda::getCurrentDeviceProperties();
    if (sm90_only) {
      return dprops->major == 9;
    } else {
      return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
    }
#endif
}

#ifdef USE_ROCM
static bool _scaled_mm_is_fnuz() {
    return at::detail::getCUDAHooks().isGPUArch({"gfx942"});
}
#endif

namespace{

enum class ScalingType : std::uint8_t {
  TensorWise,
  RowWise,
  BlockWise,
  Error
};
/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale tensors are both `Float8_e8m0fnu` or `Float8_e4m3fn`:
 *   - Returns BlockWise (with additional size checks).
 *
 * - If scale_a.numel() == 1 && scale_b.numel() == 1:
 *   - Returns TensorWise.
 *
 * - Else if scale_a.dim() == 2 && scale_a.size(0) == dim_m && scale_b.size(0) == dim_n:
 *   - Returns RowWise.
 *
 * - Otherwise:
 *   - Returns Error.
 */

// Validates the scale tensors to scaled_mm
// And returns the type of scaling/which kernel to use
ScalingType get_scaling_type(
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t dim_m,
    int64_t dim_k,
    int64_t dim_n) {
  // Check for BlockWise scaling (FP8_E8M0 and FP8_E4M3 types)
  if ((scale_a.scalar_type() == scale_b.scalar_type()) &&
      ((scale_a.scalar_type() == at::kFloat8_e8m0fnu) || (scale_a.scalar_type() == at::kFloat8_e4m3fn))) {
    const bool is_nvfp4 = scale_a.scalar_type() == at::kFloat8_e4m3fn;

    // cuBLAS's mxfp8 gemm: block_size is 1 scale per 32 elements
    // cuBLAS's nvfp4 gemm: block_size is 1 scale per 16 unpacked elements.
    const auto BLOCK_SIZE_K = is_nvfp4 ? 16 : 32;

    constexpr int64_t BLOCK_SIZE_MN = 128;

    // adjust for fp4x2 packing if necessary
    const auto dim_k_unpacked = is_nvfp4 ? dim_k * 2 : dim_k;

    auto ceil_div = [](auto a, auto b) { return (a + b - 1) / b; };
    auto num_k_blocks = ceil_div(dim_k_unpacked, BLOCK_SIZE_K);
    auto padded_num_k_blocks = ceil_div(num_k_blocks, 4) * 4;

    // TODO: We might want to enforce some structure on the shapes of the scale
    // tensors

    // Check expected sizes for block-wise scaling
    auto expected_a_size =
        BLOCK_SIZE_MN * ceil_div(dim_m, BLOCK_SIZE_MN) * padded_num_k_blocks;
    auto expected_b_size =
        BLOCK_SIZE_MN * ceil_div(dim_n, BLOCK_SIZE_MN) * padded_num_k_blocks;

    TORCH_CHECK(scale_a.numel() == expected_a_size,
                "For BlockWise scaling: Expected scale_a size to be ",
                expected_a_size, " but got ", scale_a.numel());
    TORCH_CHECK(scale_b.numel() == expected_b_size,
                "For BlockWise scaling: Expected scale_b size to be ",
                expected_b_size, " but got ", scale_b.numel());

    TORCH_CHECK(
        scale_a.is_contiguous() && scale_b.is_contiguous(),
        "For BlockWise scaling: Both scale_a and scale_b must be contiguous");

    return ScalingType::BlockWise;
  }
  // Both Per-Tensor and Row-wise scaling expect fp32 tensors
  TORCH_CHECK(
      scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "Both scale_a and scale_b must be float (fp32) tensors.");

  // Check the singluar scale case for per-tensor scaling
  if (scale_a.numel() == 1 && scale_b.numel() == 1) {
    return ScalingType::TensorWise;
  }

  // For non-TensorWise scaling, enforce 2D input tensors
  TORCH_CHECK(
      scale_a.dim() == 2 && scale_b.dim() == 2,
      "For non-TensorWise scaling, scale tensors must be 2-dimensional, "
      "but got scale_a.dim()=",
      scale_a.dim(),
      " and scale_b.dim()=",
      scale_b.dim());

  // Check for RowWise scaling
  if (scale_a.size(0) == dim_m && scale_a.size(1) == 1 &&
      scale_b.size(0) == 1 && scale_b.size(1) == dim_n) {
#if (!defined(USE_ROCM) && !defined(_MSC_VER)) || \
    (defined(USE_ROCM) && defined(HIPBLASLT_VEC_EXT))
    TORCH_CHECK(
        scale_a.is_contiguous() && scale_b.is_contiguous(),
        "Both scale_a and scale_b must be contiguous for RowWise scaling.");
    return ScalingType::RowWise;
#else
    TORCH_CHECK(false, "Per-row scaling is not supported for this platform!");
    return ScalingType::Error;
#endif
  }

  // If we reach here, the input doesn't match any valid scaling type
  TORCH_CHECK(
      false,
      "Invalid scaling configuration. For TensorWise scaling, both scales should be scalar. "
      "For RowWise scaling, scale_a should be (",
      dim_m,
      ", 1) and scale_b should be (1, ",
      dim_n,
      "). "
      "Got scale_a.size()=(",
      scale_a.size(0),
      ", ",
      scale_a.size(1),
      ") and ",
      "scale_b.size()=(",
      scale_b.size(0),
      ", ",
      scale_b.size(1),
      ")");

  return ScalingType::Error;
}

} // namespace

// Computes matrix multiply + bias while applying scaling to input and output matrices
// Scales are only applicable when matrices are of Float8 type and assumed to be equal to 1.0 by default.
// If output matrix type is 16 or 32-bit type, scale_result is not applied.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
//  - If 1-dimensional tensors are used then scale_a should be size = mat1.size(0)
//    and scale_b should have size = to mat2.size(1)
//  Arguments:
//    - `mat1`: the first operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `mat2`: the second operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `bias`: the bias, can be type `torch.float16` or `torch.bfloat16`
//    - `out_dtype`: the output dtype, can either be a float8 or a higher precision floating point type
//    - `scale_a`: a scalar or 1-dimensional tensor with the inverse scale of `mat1`, only needed if `mat1` is a float8 type
//    - `scale_b`: a scalar or 1-dimensional tensor with the inverse scale of `mat2`, only needed if `mat2` is a float8 type
//    - `scale_result`: a scalar tensor with the scale of the output, only utilized if the output is a float8 type
//    - `use_fast_accum`: if true, enables fast float8 accumulation
//    - `out`: a reference to the output tensor

Tensor&
_scaled_mm_out_cuda(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  // Check sizes
  bool allowed_device = _scaled_mm_allowed_device();
  TORCH_CHECK(allowed_device, "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+");
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  // Check what type of scaling we are doing based on inputs
  ScalingType scaling_choice = get_scaling_type(scale_a, scale_b, mat1.size(0), mat1.size(1), mat2.size(1));
  TORCH_INTERNAL_ASSERT(scaling_choice != ScalingType::Error, "Scaling type not supported");

  TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
       "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0, "mat2 shape (", mat2.sizes()[0], "x",
       mat2.sizes()[1], ") must be divisible by 16");
  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()) || mat1.scalar_type() == ScalarType::Float4_e2m1fn_x2, "Expected mat1 to be Float8 or Float4_x2 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()) || mat2.scalar_type() == ScalarType::Float4_e2m1fn_x2, "Expected mat2 to be Float8 or Float4_x2 matrix got ", mat2.scalar_type());
#ifndef USE_ROCM
  // Type restrictions imposed by CuBLASLt as of CUDA-12.1
  TORCH_CHECK(mat1.scalar_type() != ScalarType::Float8_e5m2 || mat2.scalar_type() != ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported");
#endif
  if (use_fast_accum) {
    TORCH_CHECK(mat1.scalar_type() != ScalarType::Float4_e2m1fn_x2 && mat2.scalar_type() != ScalarType::Float4_e2m1fn_x2, "`use_fast_accum` is not supported when `mat1` or `mat2` tensors have the `Float4_e2m1fn_x2` dtype.");
  }
  if (bias) {
    TORCH_CHECK(out.scalar_type() != kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 || bias->scalar_type() == ScalarType::Half,
         "Bias must be either Half or BFloat16, but got ", bias->scalar_type());
    TORCH_CHECK((out.scalar_type() != kFloat && out.scalar_type() != ScalarType::BFloat16) ||
          bias->scalar_type() == ScalarType::BFloat16,
          "Bias must be BFloat16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
    TORCH_CHECK(out.scalar_type() != ScalarType::Half || bias->scalar_type() == ScalarType::Half,
          "Bias must be Float16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
  }
  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{{out, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2},
                      {bias_, "bias", 3}, {scale_a, "scale_a", 4}, {scale_b, "scale_b", 5},
                      {scale_result_, "scale_result", 6}};
    checkAllSameGPU(__func__, targs);
  }
  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm kernels
  // do not support this case).
  if (mat1_sizes[0] == 0 || mat1_sizes[1] == 0 || mat2_sizes[1] == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    if (mat1_sizes[1] == 0) {
      out.zero_();
    }

    return out;
  }

  // ROCm's hipblaslt supports rowwise, so skip this check that sends this to cutlass.
#ifndef USE_ROCM
  // We are doing row-wise scaling
  if (scaling_choice == ScalingType::RowWise) {
    TORCH_CHECK(out.dtype() == kBFloat16, "Only bf16 high precision output types are supported for row-wise scaling.");
    at::cuda::detail::f8f8bf16_rowwise(
        mat1,
        mat2,
        scale_a,
        scale_b,
        bias,
        use_fast_accum,
        out);
    return out;
  }
#else
  if (scaling_choice == ScalingType::RowWise) {
    // For ROCm, match behavior of f8f8bf16_rowwise type checking, for unit test purposes.
    Tensor b = mat2;
    if (_scaled_mm_is_fnuz()) {
      TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fnuz);
    }
    else {
      TORCH_CHECK(b.dtype() == at::kFloat8_e4m3fn);
    }
    // Until more than bf16 is supported.
    TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16,
         "hipblaslt rowwise _scaled_mm only supports BFloat16 output but got ", out.scalar_type());
  }
#endif

  cublasCommonArgs args(mat1, mat2, out, scale_a, scale_b, scale_result);
  const auto out_dtype_ = args.result->scalar_type();
  TORCH_CHECK(args.transa == 't' && args.transb == 'n', "Only multiplication of row-major and column-major matrices is supported by cuBLASLt");

#ifdef USE_ROCM
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
#define TUNABLE_DISPATCH(BLASOP_A, BLASOP_B)                            \
        if (mat1.scalar_type() == ScalarType::Float8_e4m3fnuz) {        \
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fnuz, at::Float8_e4m3fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fnuz, at::Float8_e5m2fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
        }                                                               \
        else if (mat1.scalar_type() == ScalarType::Float8_e5m2fnuz) {   \
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2fnuz, at::Float8_e4m3fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2fnuz, at::Float8_e5m2fnuz, scalar_t,     \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
        }                                                               \
        else if (mat1.scalar_type() == ScalarType::Float8_e4m3fn) {     \
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fn) {        \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fn, at::Float8_e4m3fn, scalar_t,         \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2) {     \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e4m3fn, at::Float8_e5m2, scalar_t,           \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
        }                                                               \
        else if (mat1.scalar_type() == ScalarType::Float8_e5m2) {       \
          if (mat2.scalar_type() == ScalarType::Float8_e4m3fn) {        \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2, at::Float8_e4m3fn, scalar_t,           \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
          else if (mat2.scalar_type() == ScalarType::Float8_e5m2) {     \
            static at::cuda::tunable::ScaledGemmTunableOp<              \
                at::Float8_e5m2, at::Float8_e5m2, scalar_t,             \
                BLASOP_A, BLASOP_B> scaledgemm{};                       \
            scaledgemm(&params);                                        \
          }                                                             \
        }
    AT_DISPATCH_V2(out_dtype_, "_tunable_scaled_gemm", AT_WRAP([&] {
      bool transa_ = ((args.transa != 'n') && (args.transa != 'N'));
      bool transb_ = ((args.transb != 'n') && (args.transb != 'N'));
      at::cuda::tunable::ScaledGemmParams<scalar_t> params;
      params.transa = args.transa;
      params.transb = args.transb;
      params.m = args.m;
      params.n = args.n;
      params.k = args.k;
      params.a = args.mata->data_ptr();
      params.a_scale_ptr = args.scale_mata_ptr;
      params.lda = args.lda;
      params.a_dtype = args.mata->scalar_type();
      params.b = args.matb->data_ptr();
      params.b_scale_ptr = args.scale_matb_ptr;
      params.ldb = args.ldb;
      params.b_dtype = args.matb->scalar_type();
      params.bias_ptr = bias ? bias->data_ptr(): nullptr;
      params.bias_dtype = bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_;
      params.c = args.result->data_ptr();
      params.c_scale_ptr = args.scale_result_ptr;
      params.ldc = args.result_ld;
      params.c_dtype = out_dtype_;
      params.use_fast_accum = use_fast_accum;
      params.use_rowwise = scaling_choice == ScalingType::RowWise;
      if (transa_ && transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::T)
      }
      else if (transa_ && !transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::N)
      }
      else if (!transa_ && transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::T)
      }
      else if (!transa_ && !transb_) {
        TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::N)
      }
      else {
        TORCH_CHECK(false, "unreachable");
      }
    }),
    kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_FLOATING_TYPES));
#undef TUNABLE_DISPATCH
  }
  else
#endif
 {
    at::cuda::blas::scaled_gemm(
        args.transa,
        args.transb,
        args.m,
        args.n,
        args.k,
        args.mata->data_ptr(),
        args.scale_mata_ptr,
        args.lda,
        args.mata->scalar_type(),
        args.scale_mata_dtype.value(),
        args.matb->data_ptr(),
        args.scale_matb_ptr,
        args.ldb,
        args.matb->scalar_type(),
        args.scale_matb_dtype.value(),
        bias ? bias->data_ptr(): nullptr,
        bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_,
        args.result->data_ptr(),
        args.scale_result_ptr,
        args.result_ld,
        out_dtype_,
        use_fast_accum,
        scaling_choice == ScalingType::RowWise);
  }

  return out;
}

namespace {
  c10::SmallVector<int64_t, 3> compute_grouped_gemm_output_size(const Tensor& mat_a,
  const Tensor& mat_b,
  const std::optional<at::Tensor>& offs
  ) {
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    if (a_is_2d) {
      if (b_is_2d) {
        return {offs->size(0), mat_a.size(0), mat_b.size(1)};
      } else {
        TORCH_CHECK(offs->size(0) == mat_b.size(0), "matrix batch sizes have to match");
        return {mat_a.size(0), mat_b.size(-1)};
      }
    } else {
      if (b_is_2d) {
        // this case is not actually encountered for MoE gemms
        TORCH_CHECK(offs->size(0) == mat_a.size(0), "matrix batch sizes have to match");
        return {mat_a.size(1), mat_b.size(1)};
      } else { // regular bmm
        TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
        return {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
      }
    }
  }

  bool check_valid_strides_and_return_transposed(const Tensor& mat) {
    IntArrayRef tensor_strides = mat.strides();
    IntArrayRef tensor_sizes = mat.sizes();
    int end_dim = mat.dim() - 1;
    int alignment = 16 / mat.element_size();
    TORCH_CHECK(uint64_t(mat.data_ptr()) % 16 ==0, "expected data_ptr to be aligned to 16 bytes\n");
    if ((tensor_strides[end_dim - 1] == 1) && (tensor_strides[end_dim] >= std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
      TORCH_CHECK(tensor_strides[end_dim] % alignment == 0, "strides should be multiple of 16 bytes");
      return true;
    } else if ((tensor_strides[end_dim] == 1) && (tensor_strides[end_dim - 1] >= std::max<int64_t>(1, tensor_sizes[end_dim]))) {
      TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0, "strides should be multiple of 16 bytes");
      return false;
    } else {
      TORCH_CHECK(false, "Tensor should have a contiguous dimension and not be self-overlapping, got ", mat.strides(), " for strides and ", mat.sizes(), " for sizes");
    }
  }

  void check_scale(const Tensor& mat, const Tensor& scale, const int dim, const int arg_idx, const int scale_multiplier=1) {
    if (mat.dim() == 2) {
      TORCH_CHECK(
          scale.dim() == 1,
          "scale must be a 1D tensor, but got ",
          scale.dim(),
          "D, arg ",
          arg_idx);
      TORCH_CHECK(
          scale.is_contiguous(), "scale_a must be contiguous for arg ", arg_idx);
      TORCH_CHECK(
          scale.size(0) == mat.size(dim) * scale_multiplier,
          "scale must have the same length as mat for arg ",
          arg_idx);
    } else {
      TORCH_CHECK(
          scale.dim() == 2,
          "scale must be a 2D tensor, but got ",
          scale.dim(),
          "D for arg ",
          arg_idx);
      TORCH_CHECK(
          scale.stride(1),
          "scale_a must be contiguous in the last dimension for arg ",
          arg_idx);
      TORCH_CHECK(
          scale.size(0) == mat.size(0),
          "scale must have the same batch dimension as mat for arg ",
          arg_idx);
      TORCH_CHECK(
          scale.size(1) == mat.size(1 + dim),
          "scale must have the same first dimension as mat for arg ",
          arg_idx);
    }
}


}

Tensor
_scaled_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out_cuda(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}


Tensor
_scaled_grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
const Tensor& scale_a, const Tensor& scale_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
const std::optional<at::Tensor>& scale_result,
std::optional<c10::ScalarType> out_dtype,
bool use_fast_accum) {
#ifndef USE_ROCM
  bool allowed_device = _scaled_mm_allowed_device(/*sm90_only*/true);
  TORCH_CHECK(allowed_device, "torch._scaled_grouped_mm is only supported on CUDA devices with compute capability = 9.0");

  TORCH_CHECK(mat_a.dtype() == at::kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3 matrix got ", mat_a.scalar_type());
  TORCH_CHECK(mat_b.dtype() == at::kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3 matrix got ", mat_b.scalar_type());
  TORCH_CHECK(!check_valid_strides_and_return_transposed(mat_a), "Expected mat1 to not be transposed");
  TORCH_CHECK(check_valid_strides_and_return_transposed(mat_b), "Expected mat2 to be transposed");
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  TORCH_CHECK(
    mat_a.size(-1) % 16 == 0,
    "Expected trailing dimension of mat_a to be divisible by 16 ",
    "but got mat1 shape: (",
    mat_a.sizes(),
    ").");
  TORCH_CHECK(mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
    "Expected mat_b shape to be divisible by 16 ",
    "but got mat_b shape: (",
    mat_b.sizes(),
    ").");


  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");
  TORCH_CHECK(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }

  // Both Per-Tensor and Row-wise scaling expect fp32 tensors
  TORCH_CHECK(
      scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "Both scale_a and scale_b must be float (fp32) tensors.");

  const int scale_multiplier = (mat_a.dim() == 2 && mat_b.dim() == 2) ? offs->size(0) : 1;
  check_scale(mat_a, scale_a, 0 ,0, scale_multiplier);
  check_scale(mat_b, scale_b, 1, 1, scale_multiplier);

  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  TORCH_CHECK(out_dtype_ == kBFloat16, "Only bf16 high precision output types are supported for grouped gemm");
  const auto out_size = compute_grouped_gemm_output_size(mat_a, mat_b, offs);
  Tensor out = at::empty(out_size, mat_a.options().dtype(out_dtype_));


  at::cuda::detail::f8f8bf16_grouped_mm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs,
      bias,
      use_fast_accum,
      out);
    return out;




#else
  TORCH_CHECK(false, "grouped gemm is not supported on ROCM")
#endif

}

Tensor _grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
#ifndef USE_ROCM
  bool allowed_device = _scaled_mm_allowed_device(/*sm90_only*/true);
  TORCH_CHECK(allowed_device, "torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0");

  TORCH_CHECK(mat_a.dtype() == at::kBFloat16, "Expected mat_a to be BFloat16 matrix got ", mat_a.scalar_type());
  TORCH_CHECK(mat_b.dtype() == at::kBFloat16, "Expected mat_a to be BFloat16 matrix got ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  // check that the strides are valid, the fn will throw an error if not
  check_valid_strides_and_return_transposed(mat_a);
  check_valid_strides_and_return_transposed(mat_b);
  TORCH_CHECK(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  TORCH_CHECK(out_dtype_ == kBFloat16, "Only bf16 high output type is supported for grouped gemm");
  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");

  const auto out_size = compute_grouped_gemm_output_size(mat_a, mat_b, offs);
  Tensor out = at::empty(out_size, mat_a.options().dtype(out_dtype_));
  at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  return out;
#else
  TORCH_CHECK(false, "grouped gemm is not supported on ROCM")
#endif
}

Tensor _bmm_dtype_cuda(const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype) {
  IntArrayRef batch1_sizes = batch1.sizes();
  IntArrayRef batch2_sizes = batch2.sizes();

  Tensor out = at::empty({batch1_sizes[0], batch1_sizes[1], batch2_sizes[2]}, batch1.options().dtype(out_dtype));
  return _bmm_out_dtype_cuda(batch1, batch2, out_dtype, out);
}

Tensor& _bmm_out_dtype_cuda(const Tensor& batch1, const Tensor& batch2, const at::ScalarType out_dtype, Tensor &out) {
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");

  TORCH_CHECK(out_dtype == batch1.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (batch1.scalar_type() == at::ScalarType::Half || batch1.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");

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
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");

  TORCH_CHECK(out_dtype == batch1.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (batch1.scalar_type() == at::ScalarType::Half || batch1.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");

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
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "input dtypes must be the same");
  TORCH_CHECK(out_dtype == self.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");


  addmm_out_cuda_impl(const_cast<Tensor&>(out), out, self, mat2, 0, 1);

  return out;
}

Tensor _addmm_dtype_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty(self.sizes(), self.options().dtype(out_dtype));
  return _addmm_dtype_out_cuda(self, mat1, mat2, out_dtype, beta, alpha, result);
}

Tensor& _addmm_dtype_out_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const at::ScalarType out_dtype, const Scalar& beta, const Scalar& alpha, Tensor &out) {
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");
  TORCH_CHECK(out_dtype == self.scalar_type() ||
    (out_dtype == at::ScalarType::Float && (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::BFloat16)),
    "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs");
  TORCH_CHECK(out_dtype == out.scalar_type(), "out_dtype must be the same as the dtype of the provided out tensor");

  addmm_out_cuda_impl(out, self, mat1, mat2, beta, alpha);

  return out;
}


} // namespace at::native
