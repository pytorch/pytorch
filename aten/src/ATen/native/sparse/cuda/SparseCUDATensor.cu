#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <THC/THCThrustAllocator.cuh>
#include <THC/THCTensorSort.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace at { namespace native {

using namespace at::sparse;

// --------------------------------------------------------------------
// see NOTE [Sparse Coalesce]
//
// coalesce sum
// --------------------------------------------------------------------
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t> sparse_coalesce_common_cuda(const SparseTensor& self) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  // For indices, a simple sort + unique suffices
  // For values, we use a custom kernel for segmented reduction (can't use Thrust due to indirection).

  int64_t nnz = self._nnz();
  Tensor values = self._values();
  IntList sizes = self.sizes();
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();

  // indices will be modified by Thrust, so we have to clone or use new storage
  // here.
  LongTensor indices1D = flatten_indices(self._indices(), self.sizes(), true);
  LongTensor origIndices = at::empty({nnz}, self._indices().options());
  LongTensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  typedef thrust::device_ptr<int64_t> thrust_ptr;
  thrust_ptr indicesIter(indices1D.data<int64_t>());
  thrust_ptr origIndicesIter(origIndices.data<int64_t>());
  thrust_ptr uniqueOffsetsIter(uniqueOffsets.data<int64_t>());

  // Fill sortedOrigIndices with sequential indices, so that
  //  origIndicesIter = uniqueOffsetsIter = (0, 1, 2, ..., nnz)
  thrust::counting_iterator<int64_t> countIterI(TH_INDEX_BASE);
  thrust::counting_iterator<int64_t> countIterO(TH_INDEX_BASE);
  thrust::copy(policy, countIterI, countIterI + nnz, origIndicesIter);
  thrust::copy(policy, countIterO, countIterO + nnz, uniqueOffsetsIter);

  thrust::sort_by_key(policy,
    indicesIter, indicesIter + nnz,
    origIndicesIter, ThrustLTOp<int64_t>()
  );

  // after unique_by_key, uniqueOffsetsIter holds strided indices, where
  // difference of two indices = #same consecutive indices
  thrust::pair<thrust_ptr, thrust_ptr> newEnd = thrust::unique_by_key(policy,
    indicesIter, indicesIter + nnz,
    uniqueOffsetsIter
  );
  int64_t new_nnz = newEnd.first - indicesIter;

  indices1D.resize_({1, new_nnz});
  auto newValues_size = values.sizes().vec();
  newValues_size[0] = new_nnz;
  Tensor newValues = at::empty(newValues_size, values.options());

  // unflatten indices if necessary
  LongTensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, new_nnz}, origIndices.options());
    if (TH_INDEX_BASE != 0) {
      indices1D.add_(-1);
    }
    for (int64_t d = sparse_dim - 1; d >= 0; d--) {
      // NB: Not a select, so I can preserve the outer dimension
      LongTensor indicesSlice = newIndices.narrow(0, d, 1);
      // Note for the porting guide: THCTensor_(copy) does NOT do normal
      // broadcasting logic; instead, it will blast the elements from one
      // to the other so long as the numel is the same
      indicesSlice.copy_(indices1D);
      indices1D.div_(self.size(d));
      indicesSlice.add_(indices1D, -self.size(d));
    }
    if (TH_INDEX_BASE != 0) {
      indices1D.add_(1); // "lol"
    }
  }

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t>(uniqueOffsets, origIndices, newValues, newIndices, indices1D, new_nnz);
}

SparseTensor coalesce_sum_cuda(const SparseTensor& self) {
  int64_t nnz = self._nnz();

  if (self.is_coalesced()) {
    return self;
  }

  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (nnz < 2) {
    SparseTensor out = self.clone();
    out._coalesced_(true);
    return out;
  }

  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  IntList sizes = self.sizes();

  Tensor uniqueOffsets, origIndices, newValues, newIndices, indices1D;
  int64_t new_nnz = 0;

  std::tie(uniqueOffsets, origIndices, newValues, newIndices, indices1D, new_nnz) = sparse_coalesce_common_cuda(self);

  Tensor values = self._values().contiguous();
  int64_t stride = at::prod_intlist(values.sizes().slice(1));

  // If there is no values to copy, save running the kernel.
  if (newValues.numel() > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    dim3 grid(THCCeilDiv(new_nnz, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
    dim3 block(32, 4);
    AT_DISPATCH_ALL_TYPES_AND_HALF(values.type(), "coalesce_sum_cuda", [&] {
      apply::coalesce_sum_kernel<scalar_t><<<grid, block, 0, stream>>>(
        uniqueOffsets.data<int64_t>(),
        origIndices.data<int64_t>(),
        values.data<scalar_t>(),
        newValues.data<scalar_t>(),
        nnz,
        new_nnz,
        stride
      );
    });
  }

  SparseTensor out = at::_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, sizes, newIndices, newValues, self.options());
  out._coalesced_(true);

  THCudaCheck(cudaGetLastError());
  return out;
}

SparseTensor coalesce_sparse_cuda(const SparseTensor& self) {
  return coalesce_sum_cuda(self);
}

// --------------------------------------------------------------------
// see NOTE [Sparse Coalesce]
//
// coalesce max / min
// --------------------------------------------------------------------
std::tuple<SparseTensor, Tensor> coalesce_maxmin_common_cuda(const SparseTensor& self, CoalesceReductionType reduction_type) {
  int64_t nnz = self._nnz();
  LongTensor indices = self._indices();
  Tensor values = self._values().contiguous();
  // see NOTE [Reduction Indices at Coalesce]
  LongTensor reduction_indices;

  if (self.is_coalesced()) {
    reduction_indices = at::arange(0, nnz, indices.options()).reshape({nnz, 1}).repeat({1, values.stride(0)});
    return std::tuple<SparseTensor, Tensor>(self, reduction_indices);
  }

  if (nnz < 2) {
    reduction_indices = at::arange(0, nnz, indices.options()).reshape({nnz, 1}).repeat({1, values.stride(0)});
    // see NOTE [Coalesce SparseTensor]
    SparseTensor out = self.clone();
    out._coalesced_(true);
    return std::tuple<SparseTensor, Tensor>(out, reduction_indices);
  }

  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  IntList sizes = self.sizes();

  Tensor uniqueOffsets, origIndices, newValues, new_indices, indices1D;
  int64_t new_nnz = 0;

  std::tie(uniqueOffsets, origIndices, newValues, new_indices, indices1D, new_nnz) = sparse_coalesce_common_cuda(self);
  reduction_indices = at::empty({new_nnz, values.stride(0)}, new_indices.options());

  int64_t stride = at::prod_intlist(values.sizes().slice(1));

  // If there is no values to copy, save running the kernel.
  if (newValues.numel() > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    dim3 grid(THCCeilDiv(new_nnz, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
    dim3 block(32, 4);
    AT_DISPATCH_ALL_TYPES_AND_HALF(values.type(), "coalesce_maxmin_common_cuda", [&] {
      if (reduction_type == CoalesceReductionType::MAX) {
        apply::coalesce_max_kernel<scalar_t><<<grid, block, 0, stream>>>(
          uniqueOffsets.data<int64_t>(),
          origIndices.data<int64_t>(),
          reduction_indices.data<int64_t>(),
          values.data<scalar_t>(),
          newValues.data<scalar_t>(),
          nnz,
          new_nnz,
          stride
        );
      }
      else if (reduction_type == CoalesceReductionType::MIN) {
        apply::coalesce_min_kernel<scalar_t><<<grid, block, 0, stream>>>(
          uniqueOffsets.data<int64_t>(),
          origIndices.data<int64_t>(),
          reduction_indices.data<int64_t>(),
          values.data<scalar_t>(),
          newValues.data<scalar_t>(),
          nnz,
          new_nnz,
          stride
        );
      }
      else {
        AT_ERROR("expected CoalesceReductionType MAX and MIN, but other type is found.");
      }
    });
  }

  SparseTensor out = at::_sparse_coo_tensor_with_dims_and_tensors(
    sparse_dim, dense_dim, sizes,
    new_indices, newValues, self.options()
  )._coalesced_(true);

  THCudaCheck(cudaGetLastError());

  return std::tuple<SparseTensor, Tensor>(out, reduction_indices);
}

std::tuple<SparseTensor, Tensor> coalesce_max_cuda(const SparseTensor& self) {
  return coalesce_maxmin_common_cuda(self, CoalesceReductionType::MAX);
}

std::tuple<SparseTensor, Tensor> coalesce_min_cuda(const SparseTensor& self) {
  return coalesce_maxmin_common_cuda(self, CoalesceReductionType::MIN);
}

}} // namespace at::native
