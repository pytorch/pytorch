#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#endif

#include <cuda_runtime.h>
#include <type_traits>


#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
#include <ATen/native/sparse/cuda/SparseCUDATensorMath.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

namespace at {
namespace native {

namespace {

template <typename input_t, typename output_t>
__global__ void convert_indices_from_coo_to_csr_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t size, const int64_t numel) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int64_t i = 0; i <= data_in[0]; i++)
      data_out[i] = static_cast<output_t>(0);
  } else if (tid < numel) {
    for (int64_t i = data_in[tid - 1]; i < data_in[tid]; i++)
      data_out[i + 1] = static_cast<output_t>(tid);
  } else if (tid == numel) {
    for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
      data_out[i] = static_cast<output_t>(numel);
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cuda(const Tensor& result, const Tensor& input, const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  // Run (numel + 1) threads...
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (numel + THREADS) / THREADS;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  convert_indices_from_coo_to_csr_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, data_in, size, numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename output_t>
__global__ void convert_indices_from_csr_to_coo_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t nrows) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < nrows) {
    for (int64_t i = data_in[tid]; i < data_in[tid + 1]; i++)
      data_out[i] = static_cast<output_t>(tid);
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cuda(const Tensor& indices, const Tensor& crow_indices, const Tensor& col_indices, const bool transpose=false) {
  int64_t nrows = crow_indices.numel() - 1;
  if (nrows == 0) {
    indices.zero_();
    return;
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose?1:0);
  auto row1 = indices.select(0, transpose?0:1);
  output_t* data_out = row0.data_ptr<output_t>();

  // Run nrows threads...
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (nrows + THREADS) / THREADS;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  row1.copy_(*col_indices.expect_contiguous());
  convert_indices_from_csr_to_coo_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, crow_indices_data_in, nrows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

Tensor& add_out_dense_sparse_csr_cuda(
    Tensor& output,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
  TORCH_INTERNAL_ASSERT(src.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(dense.is_cuda());

  TORCH_CHECK(
      output.is_contiguous(),
      "out argument must be contiguous, but got: ",
      output.suggest_memory_format());
  TORCH_CHECK(
      output.is_cuda(),
      "add: expected 'out' to be CUDA tensor, but got tensor on device: ",
      output.device());

  TORCH_CHECK(
      src.is_cuda(),
      "add: expected 'other' to be a CUDA tensor, but got tensor on device: ",
      src.device());

  TORCH_CHECK(
      dense.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      src.sizes(),
      " (FYI: dense-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());
  TORCH_CHECK(
      canCast(commonDtype, output.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      output.scalar_type(),
      " in add operation");

  Tensor src_values = src.values();
  Tensor src_crow_indices = src.crow_indices();
  Tensor src_col_indices = src.col_indices();

  resize_output(output, dense.sizes());

  Tensor resultBuffer = output;
  Tensor valuesBuffer = src_values.to(commonDtype);
  if (output.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(output, dense)) {
    resultBuffer.copy_(dense);
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBool, kBFloat16,
      commonDtype,
      "add_out_op2_sparse_csr",
      [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
        AT_DISPATCH_INDEX_TYPES(
            src_crow_indices.scalar_type(),
            "csr_add_out_crow_indices",
              [&valuesBuffer, &resultBuffer, &alpha, &src_crow_indices, &src_col_indices]() {
                scalar_t* values_accessor = valuesBuffer.data_ptr<scalar_t>();
                scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
                scalar_t cast_value = alpha.to<scalar_t>();

                index_t* crow_indices_accessor = src_crow_indices.data_ptr<index_t>();
                index_t* col_indices_accessor = src_col_indices.data_ptr<index_t>();
                int64_t out_storage_offset = resultBuffer.storage_offset();

                auto out_strides = resultBuffer.strides();
                int64_t out_strides0 = out_strides[0];
                int64_t out_strides1 = out_strides[1];

                cudaStream_t stream = at::cuda::getCurrentCUDAStream();
                at::cuda::ThrustAllocator allocator;
                auto policy = thrust::cuda::par(allocator).on(stream);

               // Note that this could be wildly imbalanced if the sparsity pattern varies a lot between rows.
               thrust::for_each(
                    policy,
                    thrust::make_counting_iterator(int64_t(0)),
                    thrust::make_counting_iterator(int64_t(src_crow_indices.size(0) - 1)),
                    [values_accessor,
                    crow_indices_accessor,
                    col_indices_accessor,
                    out_ptr,
                    out_storage_offset,
                    out_strides0,
                    cast_value,
                    out_strides1
                    ]__device__(int64_t irow) {
                        index_t start_index = crow_indices_accessor[irow];
                        index_t end_index = crow_indices_accessor[irow + 1];

                        for (index_t i = start_index; i < end_index; ++i) {
                            auto icol = col_indices_accessor[i];
                            auto index = out_storage_offset + irow * out_strides0 + icol * out_strides1;
                            out_ptr[index] += cast_value * values_accessor[i];
                        }
                    });
              });
      });
  if (output.scalar_type() != commonDtype) {
    output.copy_(resultBuffer);
  }
  return output;
}

Tensor& add_out_sparse_csr_cuda(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    add_out_dense_sparse_csr_cuda(out, self, other, alpha);
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());
    at::native::resize_as_sparse_csr_(out, self);
    sparse::impl::cuda::add_out_sparse_csr(self, other, Scalar(1), alpha, out);
  }
  return out;
}

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cuda) (
  const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result
) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int>(result, input, size);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int64_t>(result, input, size);
    });
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_cuda) (
  const Tensor& crow_indices, const Tensor& col_indices, const bool out_int32, const bool transpose, const Tensor& result
) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cuda", [&] {
      convert_indices_from_csr_to_coo_cuda<scalar_t, int32_t>(result, crow_indices, col_indices, transpose);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cuda", [&] {
      convert_indices_from_csr_to_coo_cuda<scalar_t, int64_t>(result, crow_indices, col_indices, transpose);
    });
  }
}

} // namespace native
} // namespace at
