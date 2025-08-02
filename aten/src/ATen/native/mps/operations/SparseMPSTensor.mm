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


// Add this function after mark_unique_and_count
static Tensor compute_output_positions(const Tensor& is_unique) {
  using namespace mps;
  
  int64_t nnz = is_unique.size(0);
  if (nnz == 0) {
    return at::empty({0}, TensorOptions().device(kMPS).dtype(kInt));
  }
  
  // Allocate tensor for output positions
  Tensor positions = at::empty({nnz}, TensorOptions().device(kMPS).dtype(kInt));
  
  // Compute prefix sum to get output positions
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("compute_output_positions_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      
      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);
      
      mtl_setArgs(encoder,
                  is_unique,        // buffer(0): input marking unique positions
                  positions,        // buffer(1): output positions
                  nnz              // buffer(2): number of elements
      );
      
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });
  
  return positions;
}

// Add this function before _coalesce_sparse_mps
static Tensor flatten_indices(const Tensor& indices, IntArrayRef size, bool /*flatten_dim*/) {
  using namespace mps;
  
  TORCH_CHECK(indices.dim() == 2, "flatten_indices: indices must be 2D");
  TORCH_CHECK(static_cast<size_t>(indices.size(0)) == size.size(), 
              "flatten_indices: indices.size(0) must equal size.size()");
  
  int64_t sparse_dim = indices.size(0);
  int64_t nnz = indices.size(1);
  
  if (nnz == 0) {
    return at::empty({0}, indices.options().dtype(kLong));
  }
  
  // Compute strides for row-major (C-contiguous) layout
  std::vector<int64_t> strides(sparse_dim);
  strides[sparse_dim - 1] = 1;
  for (int64_t i = sparse_dim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * size[i + 1];
  }
  
  // Create stride tensor on MPS device
  auto stride_tensor = at::empty({sparse_dim}, TensorOptions().device(kMPS).dtype(kLong));
  stride_tensor.copy_(at::tensor(strides, TensorOptions().dtype(kLong)));
  
  // Allocate output tensor for flattened indices
  Tensor flat_indices = at::empty({nnz}, indices.options().dtype(kLong));
  
  // Launch Metal kernel
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("flatten_indices_kernel"); //
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      
      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);
      
      mtl_setArgs(encoder,
                  indices,          // buffer(0): input indices (sparse_dim x nnz)
                  stride_tensor,    // buffer(1): strides for each dimension
                  flat_indices,     // buffer(2): output flattened indices
                  sparse_dim,       // buffer(3): number of sparse dimensions
                  nnz              // buffer(4): number of non-zero elements
      );
      
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });
  
  return flat_indices;
}

// Replace the mark_unique_and_prefix_sum function with this simpler version
// Add this debug version of mark_unique_and_count
static std::pair<Tensor, int32_t> mark_unique_and_count(const Tensor& flat_indices) {
  using namespace mps;
  
  int64_t nnz = flat_indices.size(0);
  if (nnz == 0) {
    return {at::empty({0}, flat_indices.options().dtype(kBool)), 0};
  }
  
  // Allocate tensor for marking unique positions
  Tensor is_unique = at::empty({nnz}, flat_indices.options().dtype(kBool));
  
  // First pass: mark unique positions
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("mark_unique_positions_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];
      
      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);
      
      mtl_setArgs(encoder,
                  flat_indices,     // buffer(0): sorted flat indices
                  is_unique,        // buffer(1): output marking unique positions
                  nnz              // buffer(2): number of elements
      );
      
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });
  
  // Count unique elements on CPU to avoid atomic issues
  Tensor is_unique_cpu = is_unique.cpu();
  int32_t num_unique = 0;
  auto is_unique_data = is_unique_cpu.data_ptr<bool>();
  for (int64_t i = 0; i < nnz; i++) {
    // std::cout << "CAME HERE " << i << std::endl;
    // if (is_unique_data[i]){
    //   std::cout << "INCREMENTED " << std::endl;
    //   num_unique++;
    // } else {
    //   std::cout << "NOT INCREMENTED INDICES ARE " << flat_indices[i - 3] << std::endl;
    //   std::cout << "NOT INCREMENTED INDICES ARE " << flat_indices[i - 2] << std::endl;
    //   std::cout << "NOT INCREMENTED INDICES ARE " << flat_indices[i] << std::endl;
    //   std::cout << "NOT INCREMENTED INDICES ARE " << flat_indices[i - 1] << std::endl;
    //   std::cout << "NOT INCREMENTED INDICES ARE " << flat_indices[i + 1] << std::endl;
    //   std::cout << "NOT INCREMENTED " << std::endl;
    // }
  }
  
  return {is_unique, num_unique};
}

// Modified _coalesce_sparse_mps function
// Modified _coalesce_sparse_mps function
SparseTensor _coalesce_sparse_mps(const SparseTensor& self) {
  using namespace mps;
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor indices = self._indices();
  Tensor values = self._values();
  
  // Flatten and sort
  Tensor flat_indices = flatten_indices(indices, self.sizes(), /*flatten_dim=*/true);
  Tensor sorted_order = flat_indices.argsort();
  auto sorted_sorted = sorted_order.sort();
  Tensor flat_indices_sorted = flat_indices.index_select(0, sorted_order);
  values = values.index_select(0, sorted_order);
  indices = indices.index_select(1, sorted_order);

  // Mark unique positions and count them
  auto unique_info = mark_unique_and_count(flat_indices);
  Tensor is_unique = unique_info.first;
  int32_t newNnz = unique_info.second;
  
  // Compute output positions for each unique element
  Tensor output_positions = compute_output_positions(is_unique);
  
  // Allocate output buffers
  Tensor out_indices = at::empty({indices.size(0), newNnz}, indices.options());
  auto outValuesSize = values.sizes().vec(); 
  outValuesSize[0] = newNnz;
  Tensor out_values = at::zeros(outValuesSize, values.options());

  // Create local copies for capture in the block
  Tensor is_unique_local = is_unique;
  int64_t sparse_dim = indices.size(0);

  // Launch kernel to coalesce duplicates
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("coalesce_with_positions_kernel_" + scalarToMetalTypeString(values));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t maxThreadsPerGroup = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);
      const uint32_t valueSize = static_cast<uint32_t>(values.numel() / nnz);
      mtl_setArgs(encoder, 
                  flat_indices,          // buffer(0)
                  indices,               // buffer(1): original multi-dim indices
                  values,                // buffer(2)
                  is_unique_local,       // buffer(3)
                  output_positions,      // buffer(4): precomputed output positions
                  out_indices,           // buffer(5)
                  out_values,            // buffer(6)
                  numThreads,            // buffer(7)
                  valueSize,             // buffer(8)
                  sparse_dim,            // buffer(9): sparse_dim
                  newNnz                 // buffer(10): total unique count
      );
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  SparseTensor result = _sparse_coo_tensor_unsafe_symint(out_indices, out_values, self.sym_sizes())._coalesced_(true);
  return result;
}

} // namespace at::native::mps