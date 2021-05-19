#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkl/SparseCsrLinearAlgebra.h>

#include <algorithm>

namespace at {
namespace native {

using namespace at::sparse_csr;
// certain utiliy functions are usable from sparse COO.
using namespace at::sparse;

static constexpr bool is_msvc() {
#ifdef _MSC_VER
  return true;
#else
  return false;
#endif
}

// Functions for matrix multiplication.
Tensor& addmm_out_sparse_csr_dense_cpu(
    const Tensor& self,
    const SparseCsrTensor& op1,
    const Tensor& op2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  AT_ASSERT(op1.is_sparse_csr());
  Tensor expand_self = *expand_size(self, {op1.size(0), op2.size(1)}, "addmm_out_sparse_csr");

  AT_ASSERT(expand_self.device().type() == kCPU);
  TORCH_CHECK(
      out.device().type() == kCPU,
      "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(
      op1.device().type() == kCPU,
      "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(
      op2.device().type() == kCPU,
      "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

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

  auto col_indices = op1.col_indices();
  auto crow_indices = op1.crow_indices();
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

  // Do not use MKL for Windows due to linking issues with sparse MKL routines.
  if (at::hasMKL() && !is_msvc()) {
    _sparse_mm_mkl_(out, op1, op2, expand_self, alpha, beta);
  } else {
    int64_t dense_stride0 = op1.stride(0);
    int64_t dense_stride1 = op1.stride(1);
    int64_t out_stride0 = out.stride(0);
    int64_t out_stride1 = out.stride(1);

    AT_DISPATCH_FLOATING_TYPES(
        values.scalar_type(),
        "sparse_csr_mm_cpu",
        [&alpha,
         &beta,
         &op1,
         &out,
         &values,
         &crow_indices,
         &col_indices,
         &dense_stride0,
         &dense_stride1,
         &out_stride0,
         &out_stride1,
         &dim_k]() {
          AT_DISPATCH_INDEX_TYPES(
              crow_indices.scalar_type(),
              "csr_mm_crow_indices",
              [&alpha,
               &beta,
               &op1,
               &out,
               &values,
               &crow_indices,
               &col_indices,
               &dense_stride0,
               &dense_stride1,
               &out_stride0,
               &out_stride1,
               &dim_k]() {
                scalar_t cast_alpha = alpha.to<scalar_t>();
                // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
                scalar_t cast_beta = beta.to<scalar_t>();
                scalar_t* dense_ptr = op1.data_ptr<scalar_t>();
                scalar_t* out_ptr = out.data_ptr<scalar_t>();

                auto col_indices_accessor = col_indices.accessor<index_t, 1>();
                auto crow_indices_accessor =
                    crow_indices.accessor<index_t, 1>();
                auto values_accessor = values.accessor<scalar_t, 1>();

                at::parallel_for(
                    0,
                    crow_indices.size(0) - 1,
                    internal::GRAIN_SIZE,
                    [&](int64_t irow_start, int64_t irow_end) {
                      for (int irow = irow_start; irow < irow_end; ++irow) {
                        int start_index = crow_indices_accessor[irow];
                        int end_index = crow_indices_accessor[irow + 1];

                        for (int i = start_index; i < end_index; ++i) {
                          auto val = values_accessor[i];
                          auto icol = col_indices_accessor[i];

                          at::native::cpublas::axpy<scalar_t>(
                              dim_k,
                              cast_alpha * val,
                              dense_ptr + icol * dense_stride0,
                              dense_stride1,
                              out_ptr + irow * out_stride0,
                              out_stride1);
                        }
                      }
                    });
              });
        });
  }
  return out;
}

Tensor addmm_sparse_csr_dense(
    const Tensor& self,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor r = at::empty({0}, self.options());
  at::addmm_out(r, self, sparse, dense, beta, alpha);
  return r;
}

SparseCsrTensor& _sparse_csr_mm_out(
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    SparseCsrTensor& result) {
  Tensor t = at::zeros({}, dense.options());
  return at::addmm_out(result, t, sparse, dense, 0.0, 1.0); // redispatch!
}

Tensor _sparse_csr_addmm(
    const Tensor& t,
    const SparseCsrTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha) {
  // _sparse_addmm forward is functionally equivalent to addmm; it's
  // just the backward that is different.  This technically does an
  // unnecessary redispatch, I was too lazy to make it not do that
  return at::addmm(t, sparse, dense, beta, alpha);
}

// Functions for element-wise addition.
Tensor add_sparse_csr(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha); // redispatch!
}

Tensor& add_sparse_csr_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::add_out(self, self, other, alpha); // redispatch!
}

Tensor& add_out_dense_sparse_csr_cpu(
    Tensor& out,
    const Tensor& dense,
    const SparseCsrTensor& src,
    const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(dense.layout() == kStrided);
  TORCH_INTERNAL_ASSERT(src.is_sparse_csr());
  TORCH_INTERNAL_ASSERT(dense.device() == kCPU);

  TORCH_CHECK(
      out.is_contiguous(),
      "out argument must be contiguous, but got: ",
      out.suggest_memory_format());
  TORCH_CHECK(
      out.device() == kCPU,
      "add: expected 'out' to be CPU tensor, but got tensor on device: ",
      out.device());
  TORCH_CHECK(
      src.device() == kCPU,
      "add: expected 'other' to be a CPU tensor, but got tensor on device: ",
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
      canCast(commonDtype, out.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      out.scalar_type(),
      " in add operation");

  auto src_values = src.values();
  auto src_crow_indices = src.crow_indices();
  auto src_col_indices = src.col_indices();

  resize_output(out, dense.sizes());

  Tensor resultBuffer = out;
  Tensor valuesBuffer = src_values.to(commonDtype);

  if (out.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(out, dense)) {
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
              auto values_accessor = valuesBuffer.accessor<scalar_t, 1>();
              scalar_t* out_ptr = resultBuffer.data_ptr<scalar_t>();
              scalar_t cast_value = alpha.to<scalar_t>();

              auto crow_indices_accessor =
                  src_crow_indices.accessor<index_t, 1>();
              auto col_indices_accessor =
                  src_col_indices.accessor<index_t, 1>();
              auto out_strides0 = resultBuffer.strides()[0];
              auto out_strides1 = resultBuffer.strides()[1];

              for (int32_t irow = 0; irow < src_crow_indices.size(0) - 1;
                   ++irow) {
                int32_t start_index = crow_indices_accessor[irow];
                int32_t end_index = crow_indices_accessor[irow + 1];

                for (int i = start_index; i < end_index; ++i) {
                  auto icol = col_indices_accessor[i];
                  auto index = resultBuffer.storage_offset() + irow * out_strides0 +
                      icol * out_strides1;
                  out_ptr[index] += cast_value * values_accessor[i];
                }
              }
            });
      });
  if (out.scalar_type() != commonDtype) {
    out.copy_(resultBuffer);
  }
  return out;
}

Tensor& add_out_sparse_csr_cpu(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    return add_out_dense_sparse_csr_cpu(out, self, other, alpha);
  } else {
    TORCH_CHECK(
        false,
        "NotImplementedError: Addition of sparse CSR tensors is not yet implemented.")
  }
  return out;
}

} // namespace native
} // namespace at
