#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/SparseCsrTensorUtils.h>

// Required for checking whether Triton kernels are available
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <ATen/core/dispatch/Dispatcher.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_triton_bsr_dense_mm.h>
#endif

namespace at {
namespace native {
namespace sparse {
namespace impl {

Tensor& _compressed_row_strided_mm_out(const Tensor& compressed, const Tensor& strided, Tensor& result) {
  const auto compressed_layout = compressed.layout();
  const auto compressed_layout_str = at::sparse_csr::layoutToString(compressed_layout);

  // Device restrictions
  TORCH_CHECK(compressed.device() == strided.device()
      && compressed.device() == result.device(),
      "spmm_out(): all input arguments are expected to be on the same device.");

  // Layout restrictions.
  TORCH_CHECK(compressed_layout == kSparseCsr || compressed_layout == kSparseBsr,
      "spmm(", compressed_layout_str, ", Strided): only Csr and Bsr formats are supported for the sparse argument.");
  TORCH_CHECK(result.layout() == kStrided,
      "spmm_out(): out argument is expected to be strided.");

  // Dtype restrictions.
  TORCH_CHECK(compressed.scalar_type() == strided.scalar_type(),
      "spmm(", compressed_layout_str, ", Strided): arguments expected to have the same dtype.");

  // Dim restrictions.
  TORCH_CHECK(compressed.dim() == 2,
      "spmm(", compressed_layout_str, ", Strided): sparse arguments which are not 2D are not supported.");
  TORCH_CHECK(strided.dim() >= 2,
      "spmm(", compressed_layout_str, ", Strided): expects strided inputs to be at least 2D.");

  const auto m = compressed.sizes()[0];
  const auto k = compressed.sizes()[1];
  const auto n = strided.size(-1);
  // Matrix product size compatibility.
  TORCH_CHECK(strided.size(-2) == k,
      "spmm(", compressed_layout_str, "Strided): argument sizes are not compatible for matrix multiplication. ",
      "Got ", compressed_layout_str, ".sizes(-1) == ", k, " is not equal to ",
      "Strided.sizes(-2) == ", strided.size(-2), ".");

  // We assume that result is properly resized.
  auto result_expected_size = at::DimVector(strided.sizes().slice(0, strided.dim() - 2));
  result_expected_size.push_back(m);
  result_expected_size.push_back(n);
  TORCH_CHECK(result.sizes() == result_expected_size,
      "spmm_out(): out argument has wrong size. ",
      "Expected (", result_expected_size, ") but got (", result.sizes(), ").");

  auto values = compressed.values();

  using Blocksize = std::array<int64_t, 2>;
  // We refer to these as (b0, b1) in the comments below.
  Blocksize blocksize = {1, 1};
  if (compressed_layout == kSparseBsr) {
    blocksize = {values.size(-2), values.size(-1)};
  }

  // Triton works only with blocksizes which are powers of 2.
  const auto is_power_of_2 = [](int64_t v) -> bool {
    return !(v & (v - 1));
  };

  // Dtype and blocksize checks for potential Triton usage.
  if ((strided.scalar_type() == ScalarType::Half
    || strided.scalar_type() == ScalarType::BFloat16)
   && is_power_of_2(blocksize[0]) && is_power_of_2(blocksize[1])
   && (blocksize[0] >= 16) && (blocksize[1] >= 16)) {
    const auto triton_kernel = c10::Dispatcher::singleton()
      .findOp(torch::jit::parseName("aten::_triton_bsr_dense_mm"));
    // Call Triton only if dispatch key was overwritten.
    if (triton_kernel->hasKernelForDispatchKey(c10::DispatchKey::SparseCsrCUDA)) {
      return at::_triton_bsr_dense_mm_out(result, compressed, strided);
    }
  }

  // (..., r, c) -> (..., r / b0, c / b1, b0, b1)
  // NOTE: this function ALWAYS creates a view upon successful execution.
  const auto tile_tensor = [compressed_layout](
      const Tensor& t, Blocksize blocksize) -> Tensor {
    if (compressed_layout == kSparseCsr) {
      return t.unsqueeze(-1).unsqueeze_(-1);
    }
    else {
      const auto size_neg_2_blocked = t.size(-2) / blocksize[0];
      const auto size_neg_1_blocked = t.size(-1) / blocksize[1];
      auto tiled_sizes = at::DimVector(t.sizes().slice(0, t.dim() - 2));
      tiled_sizes.push_back(size_neg_2_blocked);
      tiled_sizes.push_back(blocksize[0]);
      tiled_sizes.push_back(size_neg_1_blocked);
      tiled_sizes.push_back(blocksize[1]);
      return t.reshape(tiled_sizes).transpose(-3, -2);
    }
  };

  // Note that sparse values are (..., b0, b1). This means that
  // the strided input has to be "tilable" to (..., b1, x) with
  // any x >= 1 such that all the shapes are (block) matrix product
  // compatible. The matrix product will then have shape (..., b0, x).
  // This in turn means the the result has to be "tilable" to
  // (..., b0, x).
  //
  // These observations imply the following restrictions:
  // 1. strided.size(-2) has to be divisible by b1.
  // 2. result.size(-2) has to be divisible by b0.
  // 3. both strided.size(-1) and result.size(-1)
  //    have to be divisible by x.
  //
  // Restrictions 1 and 2 are trivially satisfied.
  // Regarding restriction 3:
  // it would make sense to take the largest possible x for better
  // performance since it is very likely that the last dimension
  // is contiguous. As such, this value is exactly
  // x = strided.size(-1), since strided.size(-1) == result.size(-1)

  // See the comments above. This is our x.
  const auto outer_blocksize = n;

  Blocksize strided_blocksize = {blocksize[1], outer_blocksize};
  const auto strided_tiled = tile_tensor(strided, strided_blocksize);

  // Left argument is (..., b0, b1) and right is (..., b1, x).
  // This naturally implies the result should be "tilable" as
  // (..., b0, x).
  Blocksize result_blocksize = {blocksize[0], outer_blocksize};
  auto result_tiled = tile_tensor(result, result_blocksize);

  if (compressed_layout == kSparseCsr) {
    values.unsqueeze_(-1).unsqueeze_(-1);
  }

  Tensor compressed_indices, plain_indices;
  std::tie(compressed_indices, plain_indices) = at::sparse_csr::getCompressedPlainIndices(compressed);

  // Select block rows of the strided input that intersect with the block colums of the sparse input.
  auto strided_tiled_selected_rows = strided_tiled.index_select(-4, plain_indices);

  // Promote to float if output is half or bfloat16 for better precision
  const auto mm_dtype = (result.scalar_type() == kHalf || result.scalar_type() == kBFloat16)
    ? kFloat : result.scalar_type();
  // Now that we know which block rows intersect with which block columns,
  // we can perform matrix products between pairs of blocks.
  // NOTE: .to is a no-op when result.scalar_type() == mm_dtype.
  const auto pairwise_block_mm = values.unsqueeze(-3).to(mm_dtype)
    .matmul(strided_tiled_selected_rows.to(mm_dtype));

  // Having pairwise block matrix products stored in pairwise_block_mm,
  // it is sufficient to sum all the block products that share the same row
  // encoded in the sparse index. Since the reduction step is done via
  // advanced indexing methods, the compressed index ought to get converted
  // to the COO format.
  const auto compressed_indices_coo = at::_convert_indices_from_csr_to_coo(
      compressed_indices,
      plain_indices,
      compressed_indices.scalar_type() == kInt).select(0, 0);

  // Reduction step.
  // If result is neither half nor bfloat16, do everyting in-place.
  if (result.scalar_type() == mm_dtype) {
    // Zero out and sum over the blocks that share the same row indices.
    result_tiled.zero_();
    result_tiled.index_add_(
        /*dim=*/-4,
        /*index=*/compressed_indices_coo,
        /*source=*/pairwise_block_mm);
  }
  // Otherwise accumulate into a buffer and then copy.
  else {
    // No need to zero out, sum over the blocks goes into a buffer
    // followed by a copy into result.
    auto promoted_result_tiled = at::zeros(
        result_tiled.sizes(),
        result_tiled.options().dtype(mm_dtype));
    promoted_result_tiled.index_add_(
        /*dim=*/-4,
        /*index=*/compressed_indices_coo,
        /*source=*/pairwise_block_mm);
    result_tiled.copy_(promoted_result_tiled);
  }

  return result;
}

Tensor& _compressed_row_strided_addmm_out(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // If result is not the same as self, it could always be used as out argument to mm.
  if (!result.is_same(self)) {
    _compressed_row_strided_mm_out(mat1, mat2, result).mul_(alpha);

    // Process beta
    if (beta.toComplexDouble() != 0.) {
      result.add_(self.mul(beta));
    }
  }
  // Otherwise we need to allocate external memory for mm if beta != 0.
  else {
    // Process beta
    if (beta.toComplexDouble() != 0.) {
      result.mul_(beta);
      auto mm = at::empty_like(result);
      _compressed_row_strided_mm_out(mat1, mat2, mm);
      mm.mul_(alpha);
      result.add_(mm);
    }
    else {
      _compressed_row_strided_mm_out(mat1, mat2, result).mul_(alpha);
    }
  }

  return result;
}

namespace cpu {

/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
#endif
}

/*
  Computes a sum of two sparse matrices defined as
  result <- mat1 + alpha*mat2

  Args:
  * `mat1` - CSR Tensor storing sparse m x n matrix.
  * `mat2` - CSR Tensor storing sparse m x n matrix.
  * `result` - [in] CSR Tensor storing sparse m x n matrix.
               [out] result of the operation.
*/
void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::add_out_sparse_csr(mat1, mat2, alpha, result);
#endif
}

void triangular_solve_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling triangular_solve on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == kSparseCsr || A.layout() == kSparseBsr);
  sparse::impl::mkl::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
#endif
}

} // namespace cpu
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
