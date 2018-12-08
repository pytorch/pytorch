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

template <typename scalar_t, typename func_t>
static void sparse_reduction_kernel_cuda(const SparseTensor& self, SparseTensor& out, const func_t& op) {
  int64_t nnz = self._nnz();

  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is false,
  // we should keep the original tensor intact and do coalesce on a copy of the tensor
  if (nnz < 2) {
    out._coalesced_(true);
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  // For indices, a simple sort + unique suffices
  // For values, we use a custom kernel for segmented reduction (can't use Thrust due to indirection).

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

  // If there is no values to copy, save running the kernel.
  if (newValues.numel() > 0) {
    values = values.contiguous();
    int64_t stride = at::prod_intlist(values.sizes().slice(1));
    dim3 grid(THCCeilDiv(newNnz, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
    dim3 block(32, 4);
    // AT_DISPATCH_ALL_TYPES_AND_HALF(
        // values.type(), "coalesce_sparse_cuda", [&] {
          // using cuda_accscalar_t = acc_type<scalar_t, /* is_cuda */ true>;
          apply::coalesceValuesKernel<scalar_t, func_t><<<grid, block, 0, stream>>>(
            uniqueOffsets.data<int64_t>(),
            origIndices.data<int64_t>(),
            values.data<scalar_t>(),
            newValues.data<scalar_t>(),
            nnz,
            newNnz,
            stride,
            op
          );
        // });
  }

  // unflatten indices if necessary
  LongTensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, newNnz}, origIndices.options());
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

  get_sparse_impl(out)->resize_and_clear_(sparse_dim, dense_dim, sizes);
  get_sparse_impl(out)->set_indices_and_values_unsafe(newIndices, newValues);
  out._coalesced_(true);
  // SparseTensor dst = at::native::sparse_coo_tensor(newIndices, newValues, self.sizes())._coalesced_(true);

  THCudaCheck(cudaGetLastError());
}

template <typename scalar_t>
void sparse_sum_kernel_impl(const SparseTensor& self, SparseTensor& out) {
  sparse_reduction_kernel_cuda<scalar_t>(self, out, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
    return a + b;
  });
}

SparseTensor coalesce_sparse_cuda(const SparseTensor& self) {
  if (self.is_coalesced()) {
    return self;
  }

  SparseTensor out = at::empty_like(self);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self._values().type(), "coalesce_sparse_cuda", [&] {
    // using cuda_accscalar_t = acc_type<scalar_t, /* is_cuda */ true>;
    sparse_sum_kernel_impl<scalar_t>(self, out);
    // sparse_reduction_kernel_cuda<scalar_t>(self, out, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
    //   return a + b;
    // });
  });
  return out;
}

}} // namespace at::native
