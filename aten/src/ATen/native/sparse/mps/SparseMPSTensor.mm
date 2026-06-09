#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/sparse/mps/CSRIndexUtils.h>

#include <functional>
#include <numeric>
#include <optional>
#include <c10/util/OptionalArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#endif

namespace at::native {

using namespace mps;
using namespace at::sparse;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Coalesce_metallib.h>
#endif

static Tensor compute_output_positions(const Tensor& is_unique) {

  int64_t nnz = is_unique.size(0);
  if (nnz == 0) {
    return at::empty({0}, TensorOptions().device(kMPS).dtype(kInt));
  }

  Tensor positions = at::empty({nnz}, TensorOptions().device(kMPS).dtype(kInt));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("compute_output_positions_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, is_unique, positions);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  return positions;
}

static Tensor compute_output_positions_parallel(const Tensor& is_unique) {

  int64_t nnz = is_unique.size(0);
  if (nnz == 0) {
    return at::empty({0}, TensorOptions().device(kMPS).dtype(kInt));
  }

  // for small arrays, use simple kernel
  // speed of the naive kernel drops off after 4096 nnz elements
  if (nnz <= 4096) {
    return compute_output_positions(is_unique);
  }
  auto stream = getCurrentMPSStream();
  Tensor positions = is_unique.to(kInt);
  // Kogge-Stone parallel prefix sum
  Tensor positions_cloned = positions.clone();

  for (int64_t stride = 1; stride < nnz; stride *= 2) {
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto pipeline = lib.getPipelineStateForFunc("kogge_stone_step");
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder, positions, positions_cloned, stride);
        mtl_dispatch1DJob(encoder, pipeline, nnz);
      }
    });
    std::swap(positions, positions_cloned);
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("shift_right_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, positions, positions_cloned);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  return positions_cloned;
}

static std::pair<Tensor, int32_t> mark_unique_and_count(const Tensor& flat_indices) {

  int64_t nnz = flat_indices.size(0);
  if (nnz == 0) {
    return {at::empty({0}, flat_indices.options().dtype(kBool)), 0};
  }

  Tensor is_unique = at::empty({nnz}, flat_indices.options().dtype(kBool));
  Tensor count_result = at::zeros({1}, flat_indices.options().dtype(kInt));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("mark_unique_positions_and_count_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder, flat_indices, is_unique, count_result);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  int32_t num_unique = count_result.item<int32_t>();

  return {is_unique, num_unique};
}

SparseTensor _coalesce_sparse_mps(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor indices = self._indices();
  Tensor values = self._values();

  Tensor flat_indices = flatten_indices(indices, self.sizes());
  Tensor sorted_order = flat_indices.argsort();
  Tensor flat_indices_sorted = flat_indices.index({sorted_order});
  values = values.index({sorted_order});
  indices = indices.index_select(1, sorted_order);

  auto unique_info = mark_unique_and_count(flat_indices_sorted);
  Tensor is_unique = unique_info.first;
  int32_t newNnz = unique_info.second;

  Tensor output_positions = compute_output_positions_parallel(is_unique);

  Tensor out_indices = at::empty({indices.size(0), newNnz}, indices.options());
  auto outValuesSize = values.sizes().vec();
  outValuesSize[0] = newNnz;
  Tensor out_values = at::zeros(outValuesSize, values.options());

  Tensor is_unique_local = is_unique;
  int64_t sparse_dim = indices.size(0);

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("coalesce_with_positions_kernel_" + scalarToMetalTypeString(values));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t valueSize = static_cast<uint32_t>(values.numel() / nnz);
      mtl_setArgs(encoder,
                  flat_indices_sorted,
                  indices,
                  values,
                  is_unique_local,
                  output_positions,
                  out_indices,
                  out_values,
                  numThreads,
                  valueSize,
                  sparse_dim,
                  newNnz);
      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });

  Tensor tensor = at::native::_sparse_coo_tensor_unsafe_symint(
      out_indices,
      out_values,
      self.sym_sizes(),
      std::optional<ScalarType>(values.scalar_type()),
      std::optional<Layout>(self.layout()),
      std::optional<Device>(self.device()),
      std::nullopt,
      /*is_coalesced=*/true);
  SparseTensor result = tensor._coalesced_(true);
  return result;
}

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_mps)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  TORCH_CHECK(
      input.is_mps() && result.is_mps(),
      "_convert_indices_from_coo_to_csr: expected MPS tensors");
  TORCH_CHECK(
      result.scalar_type() == (out_int32 ? kInt : kLong),
      "_convert_indices_from_coo_to_csr: output dtype mismatch");

  TORCH_CHECK(size >= 0, "_convert_indices_from_coo_to_csr: size must be non-negative");

  result.zero_();

  const int64_t nnz = input.numel();
  if (nnz == 0 || size == 0) {
    return;
  }

  Tensor rows = input.contiguous();
  if (rows.scalar_type() != kLong) {
    rows = rows.to(kLong);
  }

  auto options = rows.options().dtype(kLong);
  Tensor batch_ptr = at::zeros({2}, options);
  batch_ptr.narrow(0, 1, 1).fill_(nnz);

  Tensor row_ptr_long = out_int32 ? at::empty(result.sizes(), options) : result;

  mps::csr::build_row_ptr_per_batch_mps(
      rows,
      batch_ptr,
      /*batch_count=*/1,
      /*rows_per_batch=*/size,
      row_ptr_long);

  if (out_int32) {
    result.copy_(row_ptr_long.to(kInt));
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_mps)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  TORCH_CHECK(
      crow_indices.is_mps() && col_indices.is_mps() && result.is_mps(),
      "_convert_indices_from_csr_to_coo: expected MPS tensors");

  TORCH_CHECK(
      result.scalar_type() == (out_int32 ? kInt : kLong),
      "_convert_indices_from_csr_to_coo: output dtype mismatch");

  if (result.numel() == 0) {
    result.zero_();
    return;
  }

  int64_t rows_per_batch = crow_indices.size(-1) - 1;
  TORCH_CHECK(
      rows_per_batch >= 0,
      "_convert_indices_from_csr_to_coo: invalid crow_indices shape");

  mps::csr::expand_csr_rows_to_coo_out(
      crow_indices,
      col_indices,
      rows_per_batch,
      transpose,
      result);
}

} // namespace at::native