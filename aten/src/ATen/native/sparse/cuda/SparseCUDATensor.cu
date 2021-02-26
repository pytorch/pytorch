#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/accumulate.h>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/binary_search.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

using namespace at::sparse;
using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;

namespace {

template <typename scalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void _sparse_mask_copy_kernel(
  int64_t total_threads,
  int64_t t_nnz,
  const TensorInfo<int64_t, int64_t> t_indices_ti,
  const TensorInfo<int64_t, int64_t> mask_indices_ti,
  const TensorInfo<int64_t, int64_t> t_indices_pos_ti,
  const TensorInfo<scalar_t, int64_t> t_values_ti,
  TensorInfo<scalar_t, int64_t> r_values_ti
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_threads) return;
  const int64_t j = t_indices_pos_ti.data[i];

  bool has_match = false;
  if (j >= 0 &&  j < t_nnz && t_indices_ti.data[j] == mask_indices_ti.data[i]) {
    has_match = true;
  }

  int64_t values_stride0 = r_values_ti.strides[0];
  int64_t out_start = i * values_stride0;
  int64_t out_end = (i + 1) * values_stride0;
  int64_t in_start = j * t_values_ti.strides[0];

  if (has_match) {
    for (int64_t out_i = out_start, in_i = in_start; out_i < out_end; out_i++, in_i++) {
      r_values_ti.data[out_i] = t_values_ti.data[in_i];
    }
  }
}

} // end namespace

SparseTensor _coalesce_sparse_cuda(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
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
    int64_t stride = c10::multiply_integers(values.sizes().slice(1));
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }

// this grid-strided version is slower but probably more flexible
  // to different sizes
  // int64_t blockX = min(stride, (int64_t) 512);
  // dim3 block(blockX, 512 / blockX);
  // int64_t grid = min((int64_t) 1024, THCCeilDiv((int64_t) newNnz * stride, (int64_t) block.x * block.y));
  // THCSTensor_coalesceValuesKernel_gridStrided<real, accreal><<<grid, block, 0, stream> >>(
  //   THCIndexTensor_(data)(state, uniqueOffsets),
  //   THCIndexTensor_(data)(state, origIndices),
  //   THCTensor_(data)(state, values),
  //   THCTensor_(data)(state, newValues),
  //   nnz,
  //   newNnz,
  //   stride
  // );
  // C10_CUDA_KERNEL_LAUNCH_CHECK();

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
      indices1D.divide_(self.size(d), "trunc");
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

Tensor sparse_mask_helper_cuda(
    const SparseTensor& t,
    const Tensor& mask_indices) {
  /*
    This is a helper function which filter values from `t._values()` using the
    `mask_indices`. This CUDA implementation uses `thrust::lower_bound`
    operation to find the intersection of the `mask_indices` and the
    `t._indices()` to then filter the values.

    Inputs:
      `t`             - coalesced sparse tensor input
      `mask_indices`  - mask indices tensor

    Note: The nnz in the output tensor will be same as the `mask_indices`. So it will
    works independently if the mask is coalesced or not.
  */
  TORCH_CHECK(t.is_sparse(), "t: input is not a sparse tensor");
  TORCH_CHECK(t.is_coalesced(), "t:  input is uncoalesced");
  TORCH_CHECK(mask_indices.dim() == t._indices().dim(), "mask_indices: operands have incompatible indices dim; self has dim ",
      t._indices().dim(), " but mask has dim ", mask_indices.dim());
  TORCH_CHECK(mask_indices.is_contiguous(), "mask_indices: mask is not contiguous");

  int64_t r_nnz = mask_indices.size(1);
  auto t_values = t._values().contiguous();
  auto full_size = t.sizes();
  auto vsize = t_values.sizes().vec();
  vsize[0] = r_nnz;


  if (t.sparse_dim() == 0) {
    Tensor t_values_expand = t_values;
    t_values_expand = t_values_expand.expand(vsize).contiguous();
    return t_values_expand;
  }
  Tensor r_values = at::zeros({vsize}, t_values.options());
  auto t_indices = t._indices().contiguous();
  auto t_nnz = t._nnz();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  // Step 1: flatten the sparse indices `t._indices()` tensor into a 1D indices
  // tensor `t_flatten_indices`.
  auto t_flatten_indices = at::sparse::flatten_indices(t_indices, full_size).contiguous();

  // Step 2: flatten the sparse indices `mask_indices` tensor into a 1D indices
  // tensor `mask_flatten_indices`. Note: This could be not sorted if the input
  // indices in the constructor are not in a coalesced form
  auto flattened_mask_indices =
      at::sparse::flatten_indices(mask_indices, full_size);

  Tensor t_indices_pos = at::empty({r_nnz}, mask_indices.options());

  // Step 3: Match the flattened `mask_indices` with the flattened
  // `t._indices()` using the `thrust::lower_bound`
  thrust::lower_bound(
      policy,
      t_flatten_indices.data_ptr<int64_t>(),
      t_flatten_indices.data_ptr<int64_t>() + t_nnz,
      flattened_mask_indices.data_ptr<int64_t>(),
      flattened_mask_indices.data_ptr<int64_t>() + r_nnz,
      t_indices_pos.data_ptr<int64_t>());

  // Step 4: Copy the Filtered `t._values()` using the matches at `t_indices_pos`
  if (r_nnz > 0 && t_values.numel() > 0) {
    int64_t block_size = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
    auto grid_size = cuda::ATenCeilDiv(r_nnz, block_size);

    auto t_indices_ti = getTensorInfo<int64_t, int64_t>(t_flatten_indices);
    auto mask_indices_ti =
        getTensorInfo<int64_t, int64_t>(flattened_mask_indices);
    auto t_indices_pos_ti =
        getTensorInfo<int64_t, int64_t>(t_indices_pos);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        r_values.scalar_type(), "sparse_mask_helper_cuda", [&] {
          auto t_values_ti = getTensorInfo<scalar_t, int64_t>(t_values);
          auto r_values_ti =
              getTensorInfo<scalar_t, int64_t>(r_values);

          _sparse_mask_copy_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
              r_nnz,
              t_nnz,
              t_indices_ti,
              mask_indices_ti,
              t_indices_pos_ti,
              t_values_ti,
              r_values_ti);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return r_values;
}
}} // namespace at::native
