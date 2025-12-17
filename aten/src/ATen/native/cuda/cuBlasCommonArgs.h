#pragma once

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>

namespace at::native {

using at::blas::ScalingType;
using at::blas::SwizzleType;

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

std::optional<c10::MaybeOwned<Tensor>> inline maybe_prepare_matrix_with_layout(
    const Tensor& t,
    bool make_row_major_like,
    const IntArrayRef& result_sizes) {
  int64_t ld = make_row_major_like ? 0 : 1;
  IntArrayRef strides = t.strides();
  IntArrayRef sizes = t.sizes();
  if (strides[1 - ld] == 1 && strides[ld] >= std::max<int64_t>(1, sizes[1 - ld])) {
    // Means already complies with being row-/col-major-like
    return t.sizes() == result_sizes
      ? c10::MaybeOwned<Tensor>::borrowed(t)
      : c10::MaybeOwned<Tensor>::owned(*expand_size(t, result_sizes, "maybe_prepare_matrix_with_layout"));
  } else {
    // No compliance, it is best to copy bias into result
    return std::nullopt;
  }
}

} // namespace

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
      const std::optional<Tensor>& self = std::nullopt,
      const std::optional<Scalar>& beta = std::nullopt,
      const std::optional<Tensor>& scale_a = std::nullopt,
      const std::optional<Tensor>& scale_b = std::nullopt,
      const std::optional<Tensor>& scale_result = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_a = std::nullopt,
      const std::optional<ScalingType>& scaling_choice_b = std::nullopt) {
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
      scaling_mata_type = transpose_result ? scaling_choice_b : scaling_choice_a;
      scale_matb_ptr = transpose_result ? scale_a->data_ptr() : scale_b->data_ptr();
      scale_matb_dtype = transpose_result ? scale_a->scalar_type() : scale_b->scalar_type();
      scaling_matb_type = transpose_result ? scaling_choice_a : scaling_choice_b;
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

    this->beta = beta;
    // Prepare bias if it is different from result and beta != 0
    if (!self.has_value() || c.is_same(*self) || is_beta_zero()) {
      return;
    }
    if (self->scalar_type() == c.scalar_type() && self->scalar_type() != mat1.scalar_type()) {
      // NOTE: self and result are different tensors here
      // NOTE: possible Lt dispatch with self.dtype == result.dtype == Float and
      // mat1.dtype/mat2.dtype if of reduced float type -- we do not have
      // such a kernel specialization yet.
      return;
    }
    const bool can_use_1D_bias_no_copy = (
        transpose_result // required so that bias properly broadcasts
        && self->is_contiguous()
        // == m -> should match the rows, hence transpose_result is essential
        && (self->dim() >= 1 && self->sizes().back() == m && self->numel() == m) // shapes [1, ..., 1, m] are fine
    );
    const bool can_use_bias_in_epilogue = (
        can_use_1D_bias_no_copy
        && is_beta_one() // no scaling for bias in epilogue
        && !result->is_complex() // no Epilogue support for complex types
    );
    if (can_use_bias_in_epilogue) { // Case for bias in epilogue
      bias = c10::MaybeOwned<Tensor>::borrowed(*self);
    } else { // Case for, potentially, an out-of-place GEMM
      if (can_use_1D_bias_no_copy) { // 1D bias
        // Bias expanded to a matrix with the leading dimension 0
        bias = c10::MaybeOwned<Tensor>::borrowed(*self);
        bias_ld = static_cast<int64_t>(0);
      }
      if (self->dim() == 2) { // 2D bias
        // Bias should match the result's layout
        bias = maybe_prepare_matrix_with_layout(*self, /*make_row_major_like=*/transpose_result, result->sizes());
        if (bias.has_value()) {
          bias_ld = (*bias)->stride(transpose_result ? 0 : 1);
        }
      }
    }
  }

  bool is_beta_zero() const {
    return beta.has_value() && beta->toComplexDouble() == 0.0;
  }

  bool is_beta_one() const {
    return !beta.has_value() || (beta.has_value() && beta->toComplexDouble() == 1.0);
  }

  bool can_use_bias_epilogue() const {
    return bias.has_value() && !bias_ld.has_value() && is_beta_one();
  }

  bool must_copy_bias_into_result() const {
    return !bias.has_value() && beta.has_value() && !is_beta_zero();
  }

  // Matrix members
  char transa, transb;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;

  // Bias
  std::optional<int64_t> bias_ld = std::nullopt;
  std::optional<c10::MaybeOwned<Tensor>> bias = std::nullopt;
  std::optional<Scalar> beta = std::nullopt;

  // Scale members
  void* scale_mata_ptr = nullptr;
  void* scale_matb_ptr = nullptr;
  void* scale_result_ptr = nullptr;
  std::optional<c10::ScalarType> scale_mata_dtype;
  std::optional<ScalingType> scaling_mata_type;
  std::optional<c10::ScalarType> scale_matb_dtype;
  std::optional<ScalingType> scaling_matb_type;
  std::optional<c10::ScalarType> scale_result_dtype;
};

} // namespace at::native
