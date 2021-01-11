#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
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
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/binary_search.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

using namespace at::sparse;

SparseTensor coalesce_sparse_cuda(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  if (self.is_coalesced()) {
    return self;
  }
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  // Replace instances with

  // For indices, a simple sort + unique suffices
  // For values, we use a custom kernel for segmented reduction (can't use Thrust due to indirection).

  Tensor values = self._values();

  int64_t sparse_dim = self.sparse_dim();

  // indices will be modified by Thrust, so we have to clone or use new storage
  // here.
  Tensor indices1D = flatten_indices(self._indices(), self.sizes(), true);

  Tensor origIndices = at::empty({nnz}, self._indices().options());
  Tensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  typedef thrust::device_ptr<int64_t> thrust_ptr;
  thrust_ptr indicesIter(indices1D.data_ptr<int64_t>());
  thrust_ptr origIndicesIter(origIndices.data_ptr<int64_t>());
  thrust_ptr uniqueOffsetsIter(uniqueOffsets.data_ptr<int64_t>());


  // Fill sortedOrigIndices with sequential indices
  thrust::counting_iterator<int64_t> countIterI(0);
  thrust::counting_iterator<int64_t> countIterO(0);

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

  // If there is no values to copy, save running the kernel.
  if (newValues.numel() > 0) {
    const int SZ = 4;
    values = values.contiguous();
    int64_t stride = at::prod_intlist(values.sizes().slice(1));
    dim3 grid(THCCeilDiv(newNnz, (int64_t) SZ), THCCeilDiv(stride, (int64_t) C10_WARP_SIZE*SZ));
    dim3 block(C10_WARP_SIZE, SZ);
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "coalesce_sparse_cuda", [&] {
        using cuda_accscalar_t = acc_type<scalar_t, /* is_cuda */ true>;
        apply::coalesceValuesKernel<scalar_t, cuda_accscalar_t><<<grid, block, 0, stream>>>(
          uniqueOffsets.data_ptr<int64_t>(),
          origIndices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          newValues.data_ptr<scalar_t>(),
          nnz,
          newNnz,
          stride
        );
      });
  }

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
  Tensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, newNnz}, origIndices.options());
    for (int64_t d = sparse_dim - 1; d >= 0; d--) {
      // NB: Not a select, so I can preserve the outer dimension
      Tensor indicesSlice = newIndices.narrow(0, d, 1);
      // Note for the porting guide: THCTensor_(copy) does NOT do normal
      // broadcasting logic; instead, it will blast the elements from one
      // to the other so long as the numel is the same
      indicesSlice.copy_(indices1D);
      indices1D.floor_divide_(self.size(d));
      indicesSlice.add_(indices1D, -self.size(d));
    }
  }
  ////////////////////////////////////////////////////////////
  // We can use unsafe sparse tensor constructor because the indices do not
  // need to be revalidated as we do not add or change indices, just remove
  // duplicates.
  SparseTensor dst = ::at::native::_sparse_coo_tensor_unsafe(newIndices, newValues, self.sizes())._coalesced_(true);

  THCudaCheck(cudaGetLastError());
  return dst;
}


Tensor sparse_mask_helper_cuda(const SparseTensor& t, const Tensor& mask_indices) 
{
  /*
    This is a helper function which filter values from `t._values()` using the `mask_indices`.
    This CUDA implementation uses `thrust::lower_bound` operation to find the intersection 
    of the `mask_indices` and the `t._indices()` to then filter the values.  

    Inputs:
      `t`             - tensor input 
      `mask_indices`  - mask indices tensor
  */
  int64_t r_nnz = mask_indices.size(1); 
  auto t_v = t._values().contiguous();
  auto full_size = t.sizes();
  auto vsize = t_v.sizes().vec();
  vsize[0] = r_nnz;
 
  Tensor r_values = at::zeros({vsize}, t_v.options());

  auto t_i = t._indices().contiguous();
  auto t_nnz = t._nnz();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto t_n_cols = t.size(1);

  // Step 1: flatten the sparse indices `t._indices()` tensor into a 1D indices tensor `t_flatten_indices`.  
  auto t_flatten_indices = at::sparse::flatten_indices(t_i, full_size);

  // Step 2: flatten the sparse indices `mask_indices` tensor into a 1D indices tensor `mask_flatten_indices`.  
  // Note: This could be not sorted if the input indices in the constructor are not coalesced form
  auto flattened_mask_indices = at::sparse::flatten_indices(mask_indices, full_size);

  Tensor lower_bound_values = at::empty({t_nnz}, mask_indices.options());

  thrust::lower_bound(
    policy, 
    t_flatten_indices.data_ptr<int64_t>(),
    t_flatten_indices.data_ptr<int64_t>() + t_nnz,
    flattened_mask_indices.data_ptr<int64_t>(),
    flattened_mask_indices.data_ptr<int64_t>() + t_nnz,
    lower_bound_values.data_ptr<int64_t>()
  );
  Tensor lower_bound_values_host = lower_bound_values.cpu();
  at::parallel_for(0, r_nnz, 0, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      auto j = lower_bound_values_host.data_ptr<int64_t>()[i];
      if (j < t_nnz) {
        r_values[i] = t_v[j];
      }
    }
  });
 return r_values;
}

}} // namespace at::native
