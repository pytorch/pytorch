#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/SparseTensorUtils.h>

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

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Sparse_metallib.h>
#endif


using namespace at::sparse;

SparseTensor _coalesce_sparse_mps(const SparseTensor& self) {
  using namespace mps;
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
  if (nnz < 2) {
    // If 0 or 1 non-zero, just clone and mark coalesced
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  // Extract indices and values
  Tensor indices = self._indices();          // shape: (sparse_dim, nnz)
  Tensor values = self._values();            // shape: (nnz, ... dense dims ...)
  // Flatten multi-dimensional indices into a single linear index for sorting
  Tensor flat_indices = flatten_indices(indices, self.sizes(), /*flatten_dim=*/true);

  // Allocate output buffers on MPS (max size = nnz)
  Tensor out_flat_indices = at::empty({nnz}, flat_indices.options());            // int64 (long) tensor
  auto outValuesSize = values.sizes().vec(); 
  outValuesSize[0] = nnz;
  Tensor out_values = at::empty(outValuesSize, values.options());                // same dtype as input values
  Tensor unique_count = at::zeros({1}, TensorOptions().device(kMPS).dtype(kInt)); // int32 counter for unique nnz

  // Launch Metal kernel to coalesce duplicates
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("coalesce_kernel_" + scalarToMetalTypeString(values));
      const uint32_t maxThreadsPerGroup = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

      // Set kernel arguments and dispatch
      mtl_setArgs(encoder, 
                  flat_indices,   // buffer(0): input flattened indices (ulong*)
                  values,         // buffer(1): input values (T*)
                  out_flat_indices, // buffer(2): output unique flattened indices (ulong*)
                  out_values,     // buffer(3): output values summed for duplicates (T*)
                  numThreads,     // buffer(4): nnz (constant uint)
                  unique_count    // buffer(5): atomic counter for unique count (atomic_uint*)
      );
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  // Retrieve the number of unique non-zero entries from the atomic counter
  int32_t newNnz = unique_count.cpu().item<int32_t>();
  // Narrow the output tensors to the actual number of unique elements
  out_flat_indices = out_flat_indices.narrow(0, 0, newNnz);
  out_values = out_values.narrow(0, 0, newNnz);

  // Unflatten the indices back to multi-dimensional (sparse_dim x newNnz)
  int64_t sparse_dim = self.sparse_dim();
  Tensor new_indices;
  if (sparse_dim == 1) {
    // If 1-dimensional indices, just reshape the flat indices
    new_indices = out_flat_indices.reshape({1, newNnz});
  } else {
    // Compute each coordinate from the flat index
    Tensor flat_cpu = out_flat_indices.cpu();  // bring to CPU for index math
    auto* flat_data = flat_cpu.data_ptr<int64_t>();
    new_indices = at::empty({sparse_dim, newNnz}, indices.options());
    auto* new_idx_data = new_indices.data_ptr<int64_t>();
    for (int64_t idx = 0; idx < newNnz; ++idx) {
      int64_t remaining = flat_data[idx];
      // Extract coordinates from last dimension to first
      for (int64_t d = sparse_dim - 1; d >= 0; --d) {
        int64_t size_d = self.size(d);
        int64_t coord = remaining % size_d;
        remaining /= size_d;
        new_idx_data[d * newNnz + idx] = coord;
      }
    }
  }

  // Assemble the coalesced sparse tensor
  SparseTensor result = _sparse_coo_tensor_unsafe_symint(new_indices, out_values, self.sym_sizes())._coalesced_(true);
  
  return result;
}

} // namespace at::native::mps