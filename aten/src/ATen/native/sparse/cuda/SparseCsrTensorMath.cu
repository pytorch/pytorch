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
  const SparseCsrTensor& op1,
  const Tensor& op2,
  const Scalar& beta,
  const Scalar& alpha,
  Tensor& out)
{

  TORCH_INTERNAL_ASSERT(op1.is_sparse_csr());
  Tensor expand_self = *expand_size(self, {op1.size(0), op2.size(1)}, "addmm_out_sparse_csr");

  TORCH_INTERNAL_ASSERT(expand_self.device().type() == kCUDA);
  TORCH_CHECK(
      out.device().type() == kCUDA,
      "addmm: expected 'out' to be CUDA tensor, but got CUDA tensor");
  TORCH_CHECK(
      op1.device().type() == kCUDA,
      "addmm: expected 'mat1' to be a CUDA tensor, but got a CUDA tensor");
  TORCH_CHECK(
      op2.device().type() == kCUDA,
      "addmm: expected 'mat2' to be a CUDA tensor, but got a CUDA tensor");

  TORCH_CHECK(
      op1.dim() == 2,
      "addmm: 2-D matrices expected, got ",
      op1.dim(),
      "D tensor");
  TORCH_CHECK(
      op2.dim() == 2,
      "addmm: 2-D matrices expected, got ",
      op2.dim(),
      "D tensor");

  TORCH_CHECK(
      out.is_contiguous(),
      "out argument must be contiguous, but got: ",
      out.suggest_memory_format());

  // ixk * kxj = ixj
  int64_t dim_i = op1.size(0);
  int64_t dim_j = op2.size(1);
  int64_t dim_k = op1.size(1);

  TORCH_CHECK(
      op2.size(0) == dim_k,
      "addmm: Expected dense matrix (op2) size(0)=",
      dim_k,
      ", got ",
      op2.size(0));
  TORCH_CHECK(
      op1.size(1) == dim_k,
      "addmm: Expected sparse matrix (op1) size(1)=",
      dim_k,
      ", got ",
      op1.size(1));
  resize_output(out, {dim_i, dim_j});

  auto values = op1.values();

  AT_DISPATCH_FLOATING_TYPES(
    values.scalar_type(), "addmm_sparse_csr_dense", [&] {
      scalar_t cast_beta = beta.to<scalar_t>();
      if (!is_same_tensor(out, expand_self)) {
        out.copy_(expand_self);
      }
      if (cast_beta == 0) {
        out.zero_();
      } else {
        at::mul_out(out, expand_self, scalar_to_tensor(beta));
      }
    });

  if (op1.crow_indices().scalar_type() != kInt) {
    TORCH_WARN(
        "Pytorch is compiled with MKL LP64 and will convert crow_indices to int32.");
  }
  if (op1.col_indices().scalar_type() != kInt) {
    TORCH_WARN(
        "Pytorch is compiled with MKL LP64 and will convert col_indices to int32.");
  }

  int64_t nnz = op1._nnz();
  auto col_indices = op1.col_indices().to(at::kInt);
  auto crow_indices = op1.crow_indices().to(at::kInt);
  int64_t m = op1.size(0);
  int64_t k = op1.size(1);
  int64_t n = op2.size(1);

  s_addmm_out_csr_sparse_dense_cuda_worker(nnz, m, n, k, out, beta, out, alpha, crow_indices, col_indices, values, op2);
  return out;
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
      " (FYI: op2-sparse addition does not currently support broadcasting)");

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
