#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseUtils.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

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

SparseTensor coalesce_sparse_cuda(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  if (self.is_coalesced()) {
    return self;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    _get_sparse_impl(dst)->set_coalesced(true);
    return dst;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  // Replace instances with

  // For indices, a simple sort + unique suffices
  // For values, we use a custom kernel for segmented reduction (can't use Thrust due to indirection).

  // TODO: I'm not sure if this could ever be non-contiguous
  LongTensor values = self._values().contiguous();

  int64_t sparseDims = self._sparseDims();
  int64_t stride = values.stride(0);

  // indices will be modified by Thrust, so we have to clone or use new storage
  // here.
  LongTensor indices1D = _newFlattenedIndices(self, true);

  LongTensor origIndices = at::empty({nnz}, self._indices().options());
  LongTensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  typedef thrust::device_ptr<int64_t> thrust_ptr;
  thrust_ptr indicesIter(indices1D.data<int64_t>());
  thrust_ptr origIndicesIter(origIndices.data<int64_t>());
  thrust_ptr uniqueOffsetsIter(uniqueOffsets.data<int64_t>());


  // Fill sortedOrigIndices with sequential indices
  thrust::counting_iterator<int64_t> countIterI(TH_INDEX_BASE);
  thrust::counting_iterator<int64_t> countIterO(TH_INDEX_BASE);

  thrust::copy(policy, countIterI, countIterI + nnz, origIndicesIter);
  thrust::copy(policy, countIterO, countIterO + nnz, uniqueOffsetsIter);

  thrust::sort_by_key(policy,
    indicesIter, indicesIter + nnz,
    origIndicesIter, ThrustLTOp<int64_t>()
  );

  // this forces device-host synchronization!
  thrust::pair<thrust_ptr, thrust_ptr> newEnd = thrust::unique_by_key(policy,
    indicesIter, indicesIter + nnz,
    uniqueOffsetsIter
  );
  int64_t newNnz = newEnd.first - indicesIter;

  indices1D.resize_({1, newNnz});
  auto newValues_size = values.sizes().vec();
  newValues_size[0] = newNnz;
  Tensor newValues = at::empty(newValues_size, values.options());

  dim3 grid(THCCeilDiv(newNnz, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      values.type(), "coalesce_sparse_cuda", [&] {
        using cuda_accscalar_t = acc_type<scalar_t, /* is_cuda */ true>;
        apply::coalesceValuesKernel<scalar_t, cuda_accscalar_t><<<grid, block, 0, stream>>>(
          uniqueOffsets.data<int64_t>(),
          origIndices.data<int64_t>(),
          values.data<scalar_t>(),
          newValues.data<scalar_t>(),
          nnz,
          newNnz,
          stride
        );
      });

// this grid-strided version is slower but probably more flexible
  // to different sizes
  // int64_t blockX = min(stride, (int64_t) 512);
  // dim3 block(blockX, 512 / blockX);
  // int64_t grid = min((int64_t) 1024, THCCeilDiv((int64_t) newNnz * stride, (int64_t) block.x * block.y));
  // THCSTensor_coalesceValuesKernel_gridStrided<real, accreal><<<grid, block, 0, stream>>>(
  //   THCIndexTensor_(data)(state, uniqueOffsets),
  //   THCIndexTensor_(data)(state, origIndices),
  //   THCTensor_(data)(state, values),
  //   THCTensor_(data)(state, newValues),
  //   nnz,
  //   newNnz,
  //   stride
  // );

  ////////////////////////////////////////////////////////////
  // unflatten indices if necessary
  LongTensor newIndices;
  if (sparseDims == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparseDims, newNnz}, origIndices.options());
    if (TH_INDEX_BASE != 0) {
      indices1D.add_(-1);
    }
    for (int64_t d = sparseDims - 1; d >= 0; d--) {
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
  ////////////////////////////////////////////////////////////

  SparseTensor dst = ::at::native::sparse_coo_tensor(newIndices, newValues, self.sizes());
  _get_sparse_impl(dst)->set_coalesced(true);

  THCudaCheck(cudaGetLastError());
  return dst;
}

}} // namespace at::native
