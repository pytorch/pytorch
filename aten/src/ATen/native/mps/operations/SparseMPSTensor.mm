#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/mps/OperationUtils.h>

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

static Tensor flatten_indices(const Tensor& indices, IntArrayRef size) {
  using namespace mps;

  TORCH_CHECK(indices.dim() == 2, "flatten_indices: indices must be 2D");
  TORCH_CHECK(static_cast<size_t>(indices.size(0)) == size.size(),
              "flatten_indices: indices.size(0) must equal size.size()");

  int64_t sparse_dim = indices.size(0);
  int64_t nnz = indices.size(1);

  if (nnz == 0) {
    return at::empty({0}, indices.options().dtype(kLong));
  }

  std::vector<int64_t> strides(sparse_dim);
  strides[sparse_dim - 1] = 1;
  for (int64_t i = sparse_dim - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * size[i + 1];
  }

  Tensor flat_indices = at::empty({nnz}, indices.options().dtype(kLong));

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("flatten_indices_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

      mtl_setArgs(encoder, indices, strides, flat_indices, sparse_dim, nnz);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  return flat_indices;
}

static Tensor compute_output_positions(const Tensor& is_unique) {
  using namespace mps;

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

      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

      mtl_setArgs(encoder, is_unique, positions);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  return positions;
}

static Tensor compute_output_positions_parallel(const Tensor& is_unique) {
  using namespace mps;

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

        const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
        const uint32_t numThreads = static_cast<uint32_t>(nnz);
        const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);

        MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

        mtl_setArgs(encoder, positions, positions_cloned, nnz, stride);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
      }
    });
    std::swap(positions, positions_cloned);
  }

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pipeline = lib.getPipelineStateForFunc("shift_right_kernel");
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pipeline];

      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

      mtl_setArgs(encoder, positions, positions_cloned, nnz);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  return positions_cloned;
}

static std::pair<Tensor, int32_t> mark_unique_and_count(const Tensor& flat_indices) {
  using namespace mps;

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

      const uint32_t maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);

      mtl_setArgs(encoder, flat_indices, is_unique, count_result, nnz);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  int32_t num_unique = count_result.item<int32_t>();

  return {is_unique, num_unique};
}

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

      const uint32_t maxThreadsPerGroup = static_cast<uint32_t>(pipeline.maxTotalThreadsPerThreadgroup);
      const uint32_t numThreads = static_cast<uint32_t>(nnz);
      const uint32_t threadsPerTG = std::min(numThreads, maxThreadsPerGroup);
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(threadsPerTG, 1, 1);
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
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });

  SparseTensor result = _sparse_coo_tensor_unsafe_symint(out_indices, out_values, self.sym_sizes())._coalesced_(true);
  return result;
}

} // namespace at::native