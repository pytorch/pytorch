#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <algorithm>

#include <cuda_runtime.h>
#include <type_traits>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>
#include <ATen/native/sparse/cuda/SparseCUDATensorMath.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

namespace at {
namespace native {

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

Tensor& addmm_out_sparse_csr_dense_cuda(
  const Tensor& self,
  const SparseCsrTensor& sparse,
  const Tensor& dense,
  const Scalar& beta,
  const Scalar& alpha,
  Tensor& r)
{

  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  Tensor t = *expand_size(self, {sparse.size(0), dense.size(1)}, "addmm_out_sparse_csr");

  TORCH_INTERNAL_ASSERT(t.device().type() == kCUDA);
  TORCH_CHECK(
      r.device().type() == kCUDA,
      "addmm: expected 'out' to be CUDA tensor, but got CUDA tensor");
  TORCH_CHECK(
      sparse.device().type() == kCUDA,
      "addmm: expected 'mat1' to be a CUDA tensor, but got a CUDA tensor");
  TORCH_CHECK(
      dense.device().type() == kCUDA,
      "addmm: expected 'mat2' to be a CUDA tensor, but got a CUDA tensor");

  TORCH_CHECK(
      sparse.dim() == 2,
      "addmm: 2-D matrices expected, got ",
      sparse.dim(),
      "D tensor");
  TORCH_CHECK(
      dense.dim() == 2,
      "addmm: 2-D matrices expected, got ",
      dense.dim(),
      "D tensor");

  TORCH_CHECK(
      r.is_contiguous(),
      "out argument must be contiguous, but got: ",
      r.suggest_memory_format());

  // mxk * kxn = mxn
  int64_t m = sparse.size(0);
  int64_t k = sparse.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(
      dense.size(0) == k,
      "addmm: Expected dense matrix (dense) size(0)=",
      k,
      ", got ",
      dense.size(0));
  TORCH_CHECK(
      sparse.size(1) == k,
      "addmm: Expected sparse matrix (sparse) size(1)=",
      k,
      ", got ",
      sparse.size(1));
  resize_output(r, {m, n});

  if (sparse.crow_indices().scalar_type() != kInt) {
    TORCH_WARN(
        "Pytorch is compiled with MKL LP64 and will convert crow_indices to int32.");
  }
  if (sparse.col_indices().scalar_type() != kInt) {
    TORCH_WARN(
        "Pytorch is compiled with MKL LP64 and will convert col_indices to int32.");
  }
  auto col_indices = sparse.col_indices().to(at::kInt);
  auto crow_indices = sparse.crow_indices().to(at::kInt);

  int64_t nnz = sparse._nnz();
  auto values = sparse.values();

  s_addmm_out_csr_sparse_dense_cuda_worker(nnz, m, n, k, r, beta, t, alpha, crow_indices, col_indices, values, dense);
  return r;
}

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
  AT_DISPATCH_ALL_TYPES(
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
                auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
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
                        int32_t start_index = crow_indices_accessor[irow];
                        int32_t end_index = crow_indices_accessor[irow + 1];

                        for (int i = start_index; i < end_index; ++i) {
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
    return add_out_dense_sparse_csr_cuda(out, self, other, alpha);
  } else {
    TORCH_CHECK(
        false,
        "NotImplementedError: Addition of sparse CSR tensors is not yet implemented.")
  }
  return out;
}

} // namespace native
} // namespace at
