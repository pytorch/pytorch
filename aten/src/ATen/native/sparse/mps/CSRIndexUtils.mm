#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/mps/CSRIndexUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#include <ATen/TensorOperators.h>

#include <ATen/ops/arange.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/repeat_interleave_native.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_native.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace at::native::mps::csr {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SparseTensorMath_metallib.h>
#endif

static int64_t index_count(IntArrayRef sizes) {
  int64_t result = 1;
  for (const auto& size : sizes) {
    result *= size;
  }
  return result;
}

static void validate_compressed_sparse_indices_mps_host(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  if (is_crow) {
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "crow_indices have wrong shape: ",
        "crow_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "nrows + 1 = ",
        cdim + 1);
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "col_indices have wrong shape: ",
        "col_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  } else {
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "ccol_indices have wrong shape: ",
        "ccol_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "ncols + 1 = ",
        cdim + 1);
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "row_indices have wrong shape: ",
        "row_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  }

  AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), "compressed_index_invariance_checks_mps", [=]() {
    if (is_crow) {
      TORCH_CHECK(
          static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
          sizeof(index_t) * 8,
          "-bit integer overflow in column dimension = ",
          dim);
      TORCH_CHECK(
          static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
          sizeof(index_t) * 8,
          "-bit integer overflow in row dimension = ",
          cdim);
    } else {
      TORCH_CHECK(
          static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
          sizeof(index_t) * 8,
          "-bit integer overflow in row dimension = ",
          dim);
      TORCH_CHECK(
          static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
          sizeof(index_t) * 8,
          "-bit integer overflow in column dimension = ",
          cdim);
    }
    TORCH_CHECK(
        static_cast<int64_t>(static_cast<index_t>(nnz)) == nnz,
        sizeof(index_t) * 8,
        "-bit integer overflow in nnz = ",
        nnz);
  });
}

static void launch_validate_plain_idx_bounds_mps_kernel(
    const bool is_crow,
    const Tensor& idx,
    const int64_t dim) {
  const int64_t total_work = idx.numel();
  if (total_work == 0) {
    return;
  }

  TORCH_CHECK(
      total_work <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "_validate_compressed_sparse_indices_mps: too many plain indices for Metal kernel launch");

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      const auto kernel_name = "validate_plain_idx_bounds_" + mps::scalarToMetalTypeString(idx);
      auto pipeline = lib.getPipelineStateForFunc(kernel_name);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(
          encoder,
          idx,
          idx.sizes(),
          idx.strides(),
          static_cast<uint32_t>(idx.dim()),
          dim,
          is_crow,
          stream->getErrorBuffer());
      mtl_dispatch1DJob(encoder, pipeline, static_cast<NSUInteger>(total_work));
    }
  });
  stream->synchronize(SyncType::COMMIT_AND_WAIT);
}

static void launch_validate_compressed_sparse_indices_mps_kernel(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  const auto batch_dims = cidx.sizes().slice(0, cidx.dim() - 1);
  const int64_t batch_count = index_count(batch_dims);
  if (batch_count == 0) {
    return;
  }

  const int64_t slices_per_batch = std::max<int64_t>(cdim, 1);
  const int64_t total_work = batch_count * slices_per_batch;
  TORCH_CHECK(
      total_work <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
      "_validate_compressed_sparse_indices_mps: too many work items for Metal kernel launch");

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      const auto kernel_name =
          "validate_compressed_sparse_indices_" + mps::scalarToMetalTypeString(idx);
      auto pipeline = lib.getPipelineStateForFunc(kernel_name);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(
          encoder,
          cidx,
          idx,
          cidx.sizes(),
          cidx.strides(),
          idx.sizes(),
          idx.strides(),
          std::array<uint32_t, 4>{
              static_cast<uint32_t>(cidx.dim()),
              static_cast<uint32_t>(idx.dim()),
              static_cast<uint32_t>(batch_count),
              0},
          std::array<int64_t, 4>{cdim, dim, nnz, 0},
          is_crow,
          stream->getErrorBuffer());
      mtl_dispatch1DJob(encoder, pipeline, static_cast<NSUInteger>(total_work));
    }
  });
  stream->synchronize(SyncType::COMMIT_AND_WAIT);
}

void expand_csr_rows_to_coo_out(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool transpose,
    const Tensor& coo_indices) {
  TORCH_CHECK(
      crow_indices.is_mps() && col_indices.is_mps() && coo_indices.is_mps(),
      "expand_csr_rows_to_coo: expected MPS tensors");
  const auto crow_dtype = crow_indices.scalar_type();
  const auto col_dtype = col_indices.scalar_type();
  TORCH_CHECK(
      (crow_dtype == at::kInt || crow_dtype == at::kLong) &&
          crow_dtype == col_dtype,
      "expand_csr_rows_to_coo: crow_indices and col_indices must share the same int32 or int64 dtype");

  TORCH_CHECK(crow_indices.dim() >= 1, "expand_csr_rows_to_coo: expected batched crow_indices");

  if (col_indices.numel() == 0) {
    coo_indices.zero_();
    return;
  }

  const int64_t rows_plus_one = crow_indices.size(-1);
  TORCH_CHECK(
      rows_plus_one == rows_per_batch + 1,
      "expand_csr_rows_to_coo: crow_indices last dimension must equal rows_per_batch + 1");

  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t batch_size = std::accumulate(
      batch_shape.begin(), batch_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  TORCH_CHECK(
      col_indices.numel() % batch_size == 0,
      "expand_csr_rows_to_coo: col_indices elements must be divisible by batch count");
  const int64_t nnz_per_batch = col_indices.numel() / batch_size;

  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  const int64_t total_nnz = col_indices.numel();

  TORCH_CHECK(
      coo_indices.dim() == 2 &&
      coo_indices.size(0) == expected_rows &&
      coo_indices.size(1) == total_nnz,
      "expand_csr_rows_to_coo: output must have shape [",
      expected_rows,
      ", ",
      total_nnz,
      "]");

  auto options_long = crow_indices.options().dtype(at::kLong);
  Tensor crow_flat = crow_indices.reshape({batch_size, rows_plus_one}).contiguous();
  if (crow_flat.scalar_type() != at::kLong) {
    crow_flat = crow_flat.to(at::kLong);
  }
  Tensor row_counts =
      crow_flat.slice(/*dim=*/1, /*start=*/1, /*end=*/rows_plus_one) -
      crow_flat.slice(/*dim=*/1, /*start=*/0, /*end=*/rows_plus_one - 1);
  Tensor row_ids = at::arange(rows_per_batch, options_long)
                       .view({1, rows_per_batch})
                       .expand({batch_size, rows_per_batch})
                       .reshape({-1});
  Tensor rows_flat = at::_ops::repeat_interleave_self_Tensor::call(
      row_ids,
      row_counts.reshape({-1}),
      /*dim=*/0,
      ::std::optional<int64_t>{});
  Tensor cols_flat = col_indices.reshape({total_nnz}).contiguous();

  Tensor linear_matrix = at::arange(batch_size, options_long).unsqueeze(1).expand({batch_size, nnz_per_batch});
  Tensor linear_flat = linear_matrix.reshape({total_nnz});

  std::vector<int64_t> strides(batch_ndim);
  int64_t stride_acc = 1;
  for (int64_t i = batch_ndim - 1; i >= 0; --i) {
    strides[i] = stride_acc;
    stride_acc *= batch_shape[i];
  }

  for (int64_t dim_idx = 0; dim_idx < batch_ndim; ++dim_idx) {
    int64_t size = batch_shape[dim_idx];
    if (size == 1) {
      coo_indices.select(0, dim_idx).zero_();
      continue;
    }
    int64_t stride = strides[dim_idx];
    Tensor coord = at::floor_divide(linear_flat, stride);
    coord = at::remainder(coord, size);
    coo_indices.select(0, dim_idx).copy_(coord.to(coo_indices.scalar_type()));
  }

  auto assign_row = [&](int64_t idx, const Tensor& src) {
    Tensor tmp = src.scalar_type() == coo_indices.scalar_type()
        ? src
        : src.to(coo_indices.scalar_type());
    coo_indices.select(0, idx).copy_(tmp);
  };

  if (transpose) {
    assign_row(batch_ndim, cols_flat);
    assign_row(batch_ndim + 1, rows_flat);
  } else {
    assign_row(batch_ndim, rows_flat);
    assign_row(batch_ndim + 1, cols_flat);
  }
}

Tensor expand_csr_rows_to_coo(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose) {
  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t total_nnz = col_indices.numel();
  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  auto options = crow_indices.options().dtype(out_int32 ? at::kInt : at::kLong);
  Tensor coo_indices = at::empty({expected_rows, total_nnz}, options);
  if (total_nnz == 0) {
    coo_indices.zero_();
    return coo_indices;
  }
  expand_csr_rows_to_coo_out(
      crow_indices,
      col_indices,
      rows_per_batch,
      transpose,
      coo_indices);
  return coo_indices;
}

} // namespace at::native::mps::csr

namespace at::native {

void _validate_compressed_sparse_indices_mps(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  mps::csr::validate_compressed_sparse_indices_mps_host(
      is_crow, cidx, idx, cdim, dim, nnz);
  // Invariants 5.4 and 5.5
  mps::csr::launch_validate_plain_idx_bounds_mps_kernel(
      is_crow, idx, dim);
  // Invariants 5.1, 5.2, 5.3, and 5.6
  mps::csr::launch_validate_compressed_sparse_indices_mps_kernel(
      is_crow, cidx, idx, cdim, dim, nnz);
}

} // namespace at::native
