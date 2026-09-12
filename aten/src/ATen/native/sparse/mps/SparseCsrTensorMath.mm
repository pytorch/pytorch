#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/_validate_compressed_sparse_indices_native.h>
#include <ATen/ops/repeat_interleave.h>
#endif

namespace at::native {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/SparseCsrConversions_metallib.h>
#endif

namespace {

// The sparse compressed invariants restrict index tensors to int32 and int64,
// so the kernels are only instantiated for those two.
std::string index_type_name(const Tensor& t, const char* what) {
  TORCH_CHECK(t.scalar_type() == kInt || t.scalar_type() == kLong,
              "convert_indices: expected ", what, " to be int32 or int64, but got ", t.scalar_type());
  return mps::scalarToMetalTypeString(t);
}

} // namespace

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_mps)
(const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result) {
  const int64_t numel = input.numel();
  if (numel == 0) {
    result.zero_();
    return;
  }

  const std::string func = "convert_indices_from_coo_to_csr_" + index_type_name(input, "input") + "_" +
      index_type_name(result, "result");
  auto input_ = input.expect_contiguous();

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc(func);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, *input_, result, numel);
      mtl_dispatch1DJob(encoder, pipeline, size + 1);
    }
  });
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_mps)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  const int64_t nrows = crow_indices.size(-1) - 1;
  const int64_t nnz = col_indices.size(-1);
  if (nrows == 0 || nnz == 0) {
    result.zero_();
    return;
  }

  const int64_t total_nnz = col_indices.numel();
  const int64_t batch_ndim = crow_indices.dim() - 1;
  if (batch_ndim > 0) {
    auto batch_indices = result.narrow(0, 0, batch_ndim);
    batch_indices.copy_(
        at::sparse::full_coo_indices(crow_indices.sizes().slice(0, batch_ndim), result.options())
            .repeat_interleave(nnz, 1));
  }

  TORCH_INTERNAL_ASSERT(result.is_contiguous());
  auto row0 = result.select(0, transpose ? batch_ndim + 1 : batch_ndim + 0);
  auto row1 = result.select(0, transpose ? batch_ndim + 0 : batch_ndim + 1);
  auto col_indices_ = col_indices.expect_contiguous();
  row1.copy_(col_indices_->view({-1}));

  const std::string func = "convert_indices_from_csr_to_coo_" + index_type_name(crow_indices, "crow_indices") + "_" +
      index_type_name(result, "result");
  auto crow_indices_ = crow_indices.expect_contiguous();
  const int64_t nthreads = nrows * (total_nnz / nnz);

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc(func);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder, *crow_indices_, row0, nrows, nnz, nthreads);
      mtl_dispatch1DJob(encoder, pipeline, nthreads);
    }
  });
}

// The invariant checks are written as an elementwise kernel over an arbitrary
// C++ predicate, which has no Metal equivalent. They run off by default and are
// turned on explicitly through `torch.sparse.check_sparse_tensor_invariants`, so
// this delegates to the CPU kernel rather than leaving the checks unavailable on
// MPS. Only the index tensors cross the bus, never the values.
void _validate_compressed_sparse_indices_mps(const bool is_crow,
                                             const Tensor& cidx,
                                             const Tensor& idx,
                                             const int64_t cdim,
                                             const int64_t dim,
                                             const int64_t nnz) {
  at::_validate_compressed_sparse_indices(is_crow, cidx.cpu(), idx.cpu(), cdim, dim, nnz);
}

} // namespace at::native
