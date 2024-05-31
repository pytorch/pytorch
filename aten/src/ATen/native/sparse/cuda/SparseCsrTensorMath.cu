#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <algorithm>
#include <ATen/AccumulateType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <cuda_runtime.h>
#include <type_traits>


#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
#include <ATen/native/sparse/cuda/SparseCUDATensorMath.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

namespace at::native {

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
  const input_t* data_in = input.const_data_ptr<input_t>();
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
__global__ void convert_indices_from_csr_to_coo_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t nrows, const int64_t nnz, const int64_t nbatches) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < nrows * nbatches) {
    int64_t b = tid / nrows;
    int64_t i_ = b * (nrows + 1) + tid % nrows;
    for (int64_t i = data_in[i_]; i < data_in[i_ + 1]; i++) {
      data_out[b * nnz + i] = static_cast<output_t>(tid % nrows);
    }
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cuda(const Tensor& indices, const Tensor& crow_indices, const Tensor& col_indices, const bool transpose=false) {
  int64_t nrows = crow_indices.size(-1) - 1;
  int64_t nnz = col_indices.size(-1);
  if (nrows == 0 || nnz == 0) {
    indices.zero_();
    return;
  }
  int64_t total_nnz = col_indices.numel();
  int64_t batch_ndim = crow_indices.dim() - 1;
  if (batch_ndim > 0) {
    auto batch_indices = indices.narrow(0, 0, batch_ndim);
    batch_indices.copy_(at::sparse::full_coo_indices(crow_indices.sizes().slice(0, batch_ndim), indices.options())
                        .repeat_interleave(nnz, 1));
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->const_data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose?batch_ndim + 1:batch_ndim + 0);
  auto row1 = indices.select(0, transpose?batch_ndim + 0:batch_ndim + 1);
  auto col_indices_ = col_indices.expect_contiguous();
  row1.copy_(col_indices_->view({-1}));
  output_t* data_out = row0.data_ptr<output_t>();

  // Run nrows * nbatches threads...
  int64_t nbatches = total_nnz / nnz;
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (nrows * nbatches + THREADS) / THREADS;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  convert_indices_from_csr_to_coo_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, crow_indices_data_in, nrows, nnz, nbatches);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

using namespace at::sparse_csr;
// certain utility functions are usable from sparse COO.
using namespace at::sparse;

Tensor& add_out_dense_sparse_compressed_cuda(
    Tensor& output,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
  TORCH_INTERNAL_ASSERT(
      src.layout() == kSparseCsr || src.layout() == kSparseCsc);
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

  resize_output(output, dense.sizes());

  Tensor resultBuffer = output;

  if (output.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(output, dense)) {
    resultBuffer.copy_(dense);
  }

  if (src._nnz() == 0) {
    return output;
  }

  auto valuesBuffer = src_values.to(commonDtype).reshape({-1, src_values.size(-1)}).contiguous();
  resultBuffer = resultBuffer.view({-1, output.size(-2), output.size(-1)});
  Tensor src_compressed_indices;
  Tensor src_plain_indices;
  std::tie(src_compressed_indices, src_plain_indices) =
      at::sparse_csr::getCompressedPlainIndices(src);
  src_compressed_indices =
      src_compressed_indices.reshape({-1, src_compressed_indices.size(-1)});
  src_plain_indices =
      src_plain_indices.reshape({-1, src_plain_indices.size(-1)});
  auto src_layout = src.layout();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      commonDtype,
      "add_out_op2_sparse_csr",
      [&valuesBuffer,
       &resultBuffer,
       &alpha,
       &src_compressed_indices,
       &src_plain_indices,
       &src_layout]() {
        AT_DISPATCH_INDEX_TYPES(
            src_compressed_indices.scalar_type(),
            "csr_add_out_crow_indices",
            [&valuesBuffer,
             &resultBuffer,
             &alpha,
             &src_compressed_indices,
             &src_plain_indices,
             &src_layout]() {
              auto batch_count =
                  resultBuffer.dim() > 2 ? resultBuffer.size(-3) : 1;
              scalar_t* values_accessor = valuesBuffer.data_ptr<scalar_t>();
              scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
              scalar_t cast_value = alpha.to<scalar_t>();

              index_t* compressed_indices_accessor =
                  src_compressed_indices.data_ptr<index_t>();
              index_t* plain_indices_accessor =
                  src_plain_indices.data_ptr<index_t>();
              int64_t out_storage_offset = resultBuffer.storage_offset();

              auto out_strides = resultBuffer.strides();
              auto const out_stride_batch = out_strides[0];
              auto const out_stride_compressed =
                  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
                      src_layout,
                      "add_out_dense_sparse_compressed_cpu",
                      [&out_strides] { return out_strides[1]; },
                      [&out_strides] { return out_strides[2]; });
              auto const out_stride_plain =
                  AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
                      src_layout,
                      "add_out_dense_sparse_compressed_cpu",
                      [&out_strides] { return out_strides[2]; },
                      [&out_strides] { return out_strides[1]; });
              auto compressed_stride0 = src_compressed_indices.stride(0);
              auto plain_stride0 = src_plain_indices.stride(0);
              auto val_stride0 = valuesBuffer.stride(0);

              cudaStream_t stream = at::cuda::getCurrentCUDAStream();
              at::cuda::ThrustAllocator allocator;
              auto policy = thrust::cuda::par(allocator).on(stream);

              // Note that this could be wildly imbalanced if the sparsity
              // pattern varies a lot between slices along the compressed
              // dimension.
              thrust::for_each(
                  policy,
                  thrust::make_counting_iterator(int64_t(0)),
                  thrust::make_counting_iterator(
                      int64_t(src_compressed_indices.size(-1) - 1)),
                  [values_accessor,
                   compressed_indices_accessor,
                   plain_indices_accessor,
                   out_ptr,
                   cast_value,
                   out_stride_batch,
                   out_stride_compressed,
                   out_stride_plain,
                   compressed_stride0,
                   plain_stride0,
                   val_stride0,
                   batch_count] __device__(int64_t i_compressed) {
                    for (index_t batch_idx = 0; batch_idx < batch_count;
                         batch_idx++) {
                      index_t start_index = compressed_indices_accessor
                          [batch_idx * compressed_stride0 + i_compressed];
                      index_t end_index = compressed_indices_accessor
                          [batch_idx * compressed_stride0 + i_compressed + 1];

                      for (index_t i = start_index; i < end_index; ++i) {
                        auto i_plain = plain_indices_accessor
                            [batch_idx * plain_stride0 + i];
                        auto index = batch_idx * out_stride_batch +
                            i_compressed * out_stride_compressed +
                            i_plain * out_stride_plain;
                        out_ptr[index] += cast_value *
                            values_accessor[batch_idx * val_stride0 + i];
                      }
                    }
                  });
            });
      });
  if (output.scalar_type() != commonDtype) {
    output.copy_(resultBuffer);
  }
  return output;
}

Tensor& add_out_sparse_compressed_cuda(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    add_out_dense_sparse_compressed_cuda(out, self, other, alpha);
  } else if (other.layout() == kStrided) {
    add_out_dense_sparse_compressed_cuda(out, other, self, alpha);
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());
    TORCH_CHECK(
      self.is_cuda(),
      "add: expected 'self' to be CUDA tensor, but got tensor on device: ",
      self.device());
    TORCH_CHECK(
      other.is_cuda(),
      "add: expected 'other' to be CUDA tensor, but got tensor on device: ",
      other.device());
    TORCH_CHECK(
      out.is_cuda(),
      "add: expected 'out' to be CUDA tensor, but got tensor on device: ",
      out.device());

    if (only_sparse_compressed_add_trivial_cases(self, other, alpha, out)) {
      return out;
    }

    at::native::resize_as_sparse_compressed_(out, self);
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

  /*
    Reductions on sparse CSR tensors using masked semantics.

    - To support a reduction operator on a CSR tensor with CUDA storage, define

template <typename scalar_t>
struct Reduction...Op {
  __device__ __forceinline__ scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a ... b;
  }
  __device__ __forceinline__ scalar_t identity() const { return ...; }
  __forceinline__ scalar_t identity_cpu() const { return ...; }
};


Tensor _sparse_csr_..._cuda(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
  ...
      result = reduce_sparse_csr_cuda_template<scalar_t>(input_, dims_to_sum, keepdim, Reduction...Op<scalar_t>());
  ...
  return result;
}

      and add the following

        - func: _sparse_csr_op.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
          dispatch:
            SparseCsrCUDA: _sparse_csr_..._cuda

      to native_functions.yaml
  */

namespace {

template <typename scalar_t, typename index_t, typename ReductionOp, typename acc_t>
__global__ void reduce_sparse_csr_dim0_cuda_kernel(acc_t* new_values,
                                                   const index_t* new_col_indices,
                                                   const int64_t new_nnz,
                                                   const scalar_t* values,
                                                   const index_t* col_indices,
                                                   const int64_t nnz,
                                                   ReductionOp rop
                                                   ) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < new_nnz) {
    index_t col = new_col_indices[tid];
    acc_t v = rop.identity();
    for (int64_t j=0; j < nnz; j++) {
      if (col == col_indices[j]) {
        v = rop(v, acc_t(values[j]));
      }
    }
    new_values[tid] = v;
  }
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim0_cuda_template(const Tensor& sparse, ReductionOp rop) {
  /*
    Consider the following sparse tensor:

      1 * * * *
      * * * 2 *
      * * 3 * *
      * * * * *
      4 * 5 * *

    that has CSR representation

      crow_indices = [0, 1, 2, 3, 3, 5]
      col_indices = [0, 3, 2, 0, 2]
      values = [1, 2, 3, 4, 5]

    Reduction with dim=0 results:

      rop(1,4) * rop(3,5) 2 *

    that has CSR representation

      new_crow_indices = [0, 3]
      new_col_indices = [0, 2, 3]
      new_values = [rop(1, 4], rop(3, 5), 2]

    In general, the CSR representation data can be computed as follows:

      nnz = col_indices.numel()
      new_col_indices = col_indices.unique(sorted=True, return_inverse=False)
      new_nnz = new_col_indices.numel()
      new_crow_indices = [0, new_nnz]
      new_values.resize(new_nnz)

      for i in range(new_nnz):
          v = identity
          col = new_col_indices[i]
          for j in range(nnz):
              if col == col_indices[j]:
                  v = rop(v, values[j])
          new_values[i] = v

    Notice this algorithm is different from the one used on CPU data.
  */

  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto ncols = sparse.size(1);
  auto nnz = col_indices.numel();
  Tensor new_col_indices;

  std::tie(new_col_indices, std::ignore) = at::_unique(col_indices, true, false);
  auto new_nnz = new_col_indices.numel();
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, new_nnz}, col_indices.options());

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), new_nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (new_nnz + THREADS) / THREADS;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "reduce_sparse_csr_dim0_cuda_indices",
                          [&]() {
                            index_t* col_indices_ptr = col_indices.data_ptr<index_t>();
                            index_t* new_col_indices_ptr = new_col_indices.data_ptr<index_t>();
                            reduce_sparse_csr_dim0_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(new_values_acc_ptr,
                                                                                               new_col_indices_ptr,
                                                                                               new_nnz,
                                                                                               values_ptr,
                                                                                               col_indices_ptr,
                                                                                               nnz,
                                                                                               rop
                                                                                               );
                          });
  copy_from_acc_buffer(new_values, new_values_acc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, ncols},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename index_t>
__global__ void reduce_crow_indices_dim1_cuda_kernel(index_t* new_crow_indices,
                                                     index_t* row_map,
                                                     const index_t* crow_indices,
                                                     const int64_t nrows
                                                     ) {
  int64_t nnz = 0;
  new_crow_indices[0] = 0;
  for(int64_t i=0; i<nrows; i++) {
    if (crow_indices[i] != crow_indices[i + 1]) {
      row_map[i] = nnz;
      nnz++;
    }
    new_crow_indices[i + 1] = nnz;
  }
}

template <typename scalar_t, typename index_t, typename ReductionOp, typename acc_t>
__global__ void reduce_sparse_csr_dim1_cuda_kernel(acc_t* new_values,
                                                   const scalar_t* values,
                                                   const index_t* crow_indices,
                                                   const index_t* row_map,
                                                   const int64_t nrows,
                                                   ReductionOp rop
                                                   ) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < nrows) {
    index_t i_start = crow_indices[tid];
    index_t i_end = crow_indices[tid+1];
    if (i_start != i_end) {
      acc_t acc = rop.identity();
      for (index_t i = i_start; i < i_end; i++) {
        acc = rop(acc, acc_t(values[i]));
      }
      new_values[row_map[tid]] = acc;
    }
  }
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim1_cuda_template(const Tensor& sparse, ReductionOp rop) {
  /*
    The algorithm of computing reduce of a CSR tensor along the last
    dimension is explained in the comment of the
    reduce_sparse_csr_dim1_cpu_template function.
  */
  Tensor crow_indices = sparse.crow_indices();
  auto ioptions = crow_indices.options();
  Tensor values = sparse.values();
  auto nrows = sparse.size(0);
  auto numel = values.numel();

  Tensor new_crow_indices = at::empty({crow_indices.numel()}, ioptions);
  Tensor new_col_indices = at::empty({}, ioptions);
  Tensor row_map = at::empty({nrows}, ioptions);

  // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
  // of float should be float in current scenario. In CUDA, float is the accumulate type
  // of float, while in CPU, double is the accumulate type of float.
  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type());
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (nrows + THREADS) / THREADS;

  AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "reduce_sparse_csr_dim1_cuda_indices",
                          [&]() {
                            index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
                            index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
                            index_t* row_map_ptr = row_map.data_ptr<index_t>();
                            reduce_crow_indices_dim1_cuda_kernel<<<1, 1, 0, stream>>>(new_crow_indices_ptr,
                                                                                      row_map_ptr,
                                                                                      crow_indices_ptr,
                                                                                      nrows);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                            index_t new_nnz = new_crow_indices[-1].item<index_t>();
                            new_col_indices.resize_(new_nnz);
                            new_col_indices.fill_(index_t(0));
                            new_values.resize_(new_nnz);
                            new_values_acc.resize_(new_nnz);

                            scalar_t* values_ptr = values.data_ptr<scalar_t>();
                            acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
                            reduce_sparse_csr_dim1_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(new_values_acc_ptr,
                                                                                               values_ptr,
                                                                                               crow_indices_ptr,
                                                                                               row_map_ptr,
                                                                                               nrows,
                                                                                               rop);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                          });

  copy_from_acc_buffer(new_values, new_values_acc);
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {sparse.size(0), 1},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_cuda_template(const Tensor& sparse, ReductionOp rop) {

  auto ioptions = sparse.col_indices().options();
  Tensor values = sparse.values();
  auto numel = values.numel();
  auto nnz = std::min<int64_t>(1, numel);

  auto result_dtype = at::isIntegralType(values.scalar_type(), /*includeBool=*/true) ? ScalarType::Long : values.scalar_type();
  Tensor new_values, new_values_acc;
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));
    new_values_acc = at::empty({1}, values.options());
    auto iter = TensorIterator::reduce_op(new_values_acc, values);
    gpu_reduce_kernel<scalar_t, scalar_t>(iter, func_wrapper<scalar_t>(rop), rop.identity_cpu());
    new_values.copy_(new_values_acc);
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));
  }
  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, std::min<int64_t>(1, sparse.size(1))},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_cuda_template(const Tensor& sparse, std::vector<int64_t> dims, ReductionOp rop) {
  if (dims.size() == 1) {
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_cuda_template<scalar_t>(sparse, rop);
    } else {
      TORCH_INTERNAL_ASSERT(dims[0] == 1);
      return reduce_sparse_csr_dim1_cuda_template<scalar_t>(sparse, rop);
    }
  } else if (dims.size() == 2) {
    TORCH_INTERNAL_ASSERT(((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    return reduce_sparse_csr_dim01_cuda_template<scalar_t>(sparse, rop);
  }
  TORCH_INTERNAL_ASSERT(dims.size() == 0);
  // effective after gh-29137 has been resolved
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_cuda_template(const Tensor& sparse, IntArrayRef dims_to_sum, bool keepdim, ReductionOp rop) {
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
  TORCH_INTERNAL_ASSERT(sparse.is_cuda());

  const int64_t input_dim = sparse.dim();
  TORCH_INTERNAL_ASSERT(input_dim == 2);
  auto dims = dims_to_sum.vec();
  maybe_wrap_dims(dims, input_dim);
  if (dims.size() == 0) {
    // after gh-29137 is resolved, delete this if-block
    dims.emplace_back(0);
    dims.emplace_back(1);
  }
  return reduce_sparse_csr_cuda_template<scalar_t>(sparse, dims, rop);
}

template <typename scalar_t>
struct ReductionAddOp {
  __device__ __forceinline__ scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a + b;
  }
  __device__ __forceinline__ scalar_t identity() const { return 0; }
  __forceinline__ scalar_t identity_cpu() const { return 0; }
};

template <typename scalar_t>
struct ReductionMulOp {
  __device__ __forceinline__ scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a * b;
  }
  __device__ __forceinline__ scalar_t identity() const { return 1; }
  __forceinline__ scalar_t identity_cpu() const { return 1; }
};

} // namespace

Tensor _sparse_csr_sum_cuda(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = at::sparse_csr::to_type(input, dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_cuda", [&] {
      // Set `is_cuda` = `true` in acc_type in CPU backend. Because the accumulate type
      // of float should be float in current scenario. In CUDA, float is the accumulate type
      // of float, while in CPU, double is the accumulate type of float.
      using acc_t = at::acc_type<scalar_t, true>;
        result = reduce_sparse_csr_cuda_template<scalar_t>(
            input_, dims_to_sum, keepdim, ReductionAddOp<acc_t>());
      });
  return result;
}

Tensor _sparse_csr_prod_cuda(const Tensor& input, IntArrayRef dims_to_reduce, bool keepdim, std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_cuda",
    [&] {
      result = reduce_sparse_csr_cuda_template<scalar_t>(input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
    });
  return result;
}

} // namespace at::native
