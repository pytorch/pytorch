#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/SparseTensorUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#endif

#include <thrust/adjacent_difference.h>
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

namespace at::native {

using namespace at::sparse;

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
  at::cuda::ThrustAllocator allocator;
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
    origIndicesIter, LTOp<int64_t>()
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
    int warp_size = at::cuda::warp_size();
#ifdef USE_ROCM
    const int64_t BATCHING_SEGMENT = 4096;
    int64_t nsegments = ceil_div(newNnz, (int64_t) SZ);
    int64_t s_batch = ceil_div(nsegments, BATCHING_SEGMENT);
    dim3 grid(s_batch, (s_batch == 1) ? nsegments : BATCHING_SEGMENT, ceil_div(stride, (int64_t) warp_size*SZ));
#else
    dim3 grid(ceil_div(newNnz, (int64_t) SZ), ceil_div(stride, (int64_t) warp_size*SZ));
#endif
    dim3 block(warp_size, SZ);
#ifdef USE_ROCM
    // Must duplicate the whole section otherwise does not compile on Windows
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf, at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
      values.scalar_type(), "coalesce_sparse_cuda", [&] {
        using cuda_accscalar_t = acc_type<scalar_t, /* is_cuda */ true>;
        apply::coalesceValuesKernel<scalar_t, cuda_accscalar_t><<<grid, block, 0, stream>>>(
          uniqueOffsets.data_ptr<int64_t>(),
          origIndices.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          newValues.data_ptr<scalar_t>(),
          nnz,
          newNnz,
          nsegments,
          stride
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
#else
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf, at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
      values.scalar_type(), "coalesce_sparse_cuda", [&] {
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
#endif
  }

// this grid-strided version is slower but probably more flexible
  // to different sizes
  // int64_t blockX = min(stride, (int64_t) 512);
  // dim3 block(blockX, 512 / blockX);
  // int64_t grid = min((int64_t) 1024, ceil_div((int64_t) newNnz * stride, (int64_t) block.x * block.y));
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

  AT_CUDA_CHECK(cudaGetLastError());
  return dst;
}

template<typename scalar_t>
__global__ void _get_real_imag_kernel(
    const int64_t* __restrict__ last_dim_indices,
    const scalar_t* __restrict__ values,
    c10::complex<scalar_t>* __restrict__ complex_values,
    int64_t nnz) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < nnz) {
    int64_t last_idx = last_dim_indices[idx]; // indices[last_dim][idx]

    if (last_idx == 0) {
      // Real part
      complex_values[idx] = c10::complex<scalar_t>(values[idx], scalar_t(0));
    } else {
      // Imaginary part
      complex_values[idx] = c10::complex<scalar_t>(scalar_t(0), values[idx]);
    }
  }
}

Tensor view_as_complex_sparse_cuda(const Tensor& self) {
  auto new_sizes = self.sizes().vec();
  TORCH_CHECK(!new_sizes.empty(), "Input tensor must have one or more dimensions");
  TORCH_CHECK(new_sizes[new_sizes.size() - 1] == 2, "Tensor must have a last dimension of size 2");
  new_sizes.pop_back();

  auto values = self._values();
  auto indices = self._indices();
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);

  auto new_indices = indices.slice(/*dim=*/0, /*start=*/0, /*end=*/ndim-1);
  const auto complex_type = c10::toComplexType(self.scalar_type());
  auto options = values.options().dtype(complex_type).layout(kStrided);
  auto complex_values = at::empty(nnz, options);

  if (nnz == 0) {
    // Handle empty tensor
    return at::_sparse_coo_tensor_with_dims_and_tensors(
        self.sparse_dim() - 1,
        self.dense_dim(),
        new_sizes,
        new_indices,
        complex_values,
        self.options().dtype(complex_type),
        self.is_coalesced()
    );
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // values after complex_values in the same position are combined
  Tensor reduced_values = at::empty({nnz}, complex_values.options());

  at::cuda::ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  // Create a flattened hash for sorting
  Tensor indices_hash = at::sparse::flatten_indices(new_indices, new_sizes);

  // Sort columns by new_indices and reorder indices/values
  auto sort_result = indices_hash.sort();
  Tensor sorted_hash = std::get<0>(sort_result);
  Tensor sort_perm = std::get<1>(sort_result);
  new_indices = new_indices.index_select(1, sort_perm);

  typedef thrust::device_ptr<int64_t> int_ptr;
  int_ptr hash_begin(sorted_hash.data_ptr<int64_t>());
  int_ptr hash_end = hash_begin + nnz;

  Tensor unique_hash = at::empty({nnz}, sorted_hash.options());
  int_ptr unique_hash_begin(unique_hash.data_ptr<int64_t>());

  int64_t new_nnz = 0;

  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "view_as_complex_sparse_cuda", [&] {
    using complex_t = c10::complex<scalar_t>;

    int64_t threads = 256;
    int64_t blocks = (nnz + threads - 1) / threads;
    auto last_dim_indices = indices.select(0, ndim-1);

    _get_real_imag_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        last_dim_indices.data_ptr<int64_t>(),
        values.data_ptr<scalar_t>(),
        complex_values.data_ptr<complex_t>(),
        nnz
      );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    complex_values = complex_values.index_select(0, sort_perm);

    typedef thrust::device_ptr<complex_t> complex_ptr;
    complex_ptr vals_begin(complex_values.data_ptr<complex_t>());
    complex_ptr reduced_vals_begin(reduced_values.data_ptr<complex_t>());

    // Use Thrust to group by indices and sum complex values
    auto new_end = thrust::reduce_by_key(
        policy,
        hash_begin,
        hash_end,
        vals_begin,
        unique_hash_begin,
        reduced_vals_begin,
        thrust::equal_to<int64_t>(),
        thrust::plus<complex_t>()
    );
    new_nnz = new_end.first - unique_hash_begin;
  });

  reduced_values = reduced_values.narrow(0, 0, new_nnz);

  // Create mask to remove indices with the same position.
  Tensor keep_mask = at::empty({nnz}, at::TensorOptions().dtype(at::kBool).device(self.device()));
  typedef thrust::device_ptr<bool> bool_ptr;
  bool_ptr mask_begin(keep_mask.data_ptr<bool>());

  // Mark positions where hash changes (keep first of each group)
  thrust::adjacent_difference(
      policy,
      hash_begin,
      hash_end,
      mask_begin,
      thrust::not_equal_to<int64_t>()
  );
  // First element is always kept
  thrust::fill_n(policy, mask_begin, 1, true);

  // Below does the same as kept_positions = keep_mask.nonzero().squeeze(1);
  // but keeps data in GPU (nonzero() moves data to CPU).
  Tensor kept_positions = at::empty({nnz}, indices.options());
  auto positions = at::arange(nnz, indices.options());
  thrust::copy_if(
      policy,
      thrust::device_ptr<int64_t>(positions.data_ptr<int64_t>()),
      thrust::device_ptr<int64_t>(positions.data_ptr<int64_t>()) + nnz,
      mask_begin,
      thrust::device_ptr<int64_t>(kept_positions.data_ptr<int64_t>()),
      thrust::identity<bool>()
  );
  kept_positions = kept_positions.narrow(0, 0, new_nnz);

  new_indices = new_indices.index_select(1, kept_positions);

  return at::_sparse_coo_tensor_with_dims_and_tensors(
      self.sparse_dim() - 1,
      self.dense_dim(),
      new_sizes,
      new_indices,
      reduced_values,
      self.options().dtype(complex_type),
      true
  );
}

REGISTER_CUDA_DISPATCH(view_as_complex_sparse_stub, &view_as_complex_sparse_cuda)

} // namespace at::native
