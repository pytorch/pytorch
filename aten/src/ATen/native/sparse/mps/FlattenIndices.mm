#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/zeros_native.h>
#endif

namespace at::native {
namespace {

using namespace mps;
using namespace at::sparse;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FlattenIndices_metallib.h>
#endif

Tensor flatten_indices_mps(const Tensor& indices, IntArrayRef size) {
  TORCH_CHECK(indices.dim() == 2, "flatten_indices: indices must be 2D");
  TORCH_CHECK(static_cast<size_t>(indices.size(0)) == size.size(),
              "flatten_indices: indices.size(0) must equal size.size()");

  const int64_t sparse_dim = indices.size(0);
  const int64_t nnz = indices.size(1);

  if (nnz == 0) {
    return at::empty({0}, indices.options().dtype(kLong));
  }

  // Row-major multipliers for flattening: mul[d] = prod_{j>d}(size[j])
  std::vector<int64_t> row_muls(sparse_dim);
  row_muls[sparse_dim - 1] = 1;
  for (int64_t i = sparse_dim - 2; i >= 0; --i) {
    row_muls[i] = row_muls[i + 1] * size[i + 1];
  }

  auto flat_indices = at::empty({nnz}, indices.options().dtype(kLong));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("flatten_indices_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      mtl_setArgs(encoder,
                  indices,
                  row_muls,
                  flat_indices,
                  static_cast<uint>(sparse_dim),
                  indices.strides()
      );

      mtl_dispatch1DJob(encoder, pipeline, nnz);
    }
  });
  return flat_indices;
}

} // namespace
REGISTER_MPS_DISPATCH(flatten_indices_stub, &flatten_indices_mps)
} // namespace at::native