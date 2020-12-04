#include <ATen/native/sparse/SparseGCSTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <TH/THBlasUtils.h>

#include <algorithm>
#include <iostream>

namespace at { namespace native {

// Functions for matrix multiplication.
using namespace at::sparse;

Tensor& addmm_out_sparse_gcs_dense_cpu(
    Tensor& out,
    const Tensor& self,
    const SparseTensor& op1,
    const Tensor& op2,
    Scalar beta,
    Scalar alpha) {
  Tensor expand_self;
  std::tie(expand_self) = expand_size(self, {op1.size(0), op2.size(1)}, "addmm_out_sparse_gcs");

  AT_ASSERT(expand_self.device().type() == kCPU);
  TORCH_CHECK(out.device().type() == kCPU, "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(op1.device().type() == kCPU, "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(op2.device().type() == kCPU, "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(op1.dim() == 2, "addmm: 2-D matrices expected, got ", op1.dim(), "D tensor");
  TORCH_CHECK(op2.dim() == 2, "addmm: 2-D matrices expected, got ", op2.dim(), "D tensor");

  // ixj * jxk = ixk
  int64_t dim_i = op1.size(0);
  int64_t dim_j = op1.size(1);
  int64_t dim_k = op2.size(1);

  TORCH_CHECK(op2.size(0) == dim_j,
              "addmm: Argument #3 (op2): Expected dim 0 size ", dim_j, ", got ", op2.size(0));
  TORCH_CHECK(expand_self.size(0) == dim_i,
              "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", expand_self.size(0));
  TORCH_CHECK(expand_self.size(1) == dim_k,
              "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", expand_self.size(1));
  out.resize_({dim_i, dim_k});

  // TODO: why does that nnz == 0 condition exist in the COO code?

  at::_sparse_gcs_mm(out, op1, expand_self, op2, alpha, beta);

  return out;
}

Tensor addmm_sparse_gcs_dense_cpu(
    const Tensor& self,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha) {
   Tensor r = at::empty({0}, self.options());
   at::addmm_out(r, self, sparse, dense, beta, alpha);
   return r;
}

SparseTensor& _sparse_gcs_mm_out(
  SparseTensor& result,
  const SparseTensor& sparse,
  const Tensor& dense
) {
  Tensor t = at::zeros({}, dense.options());
  return at::addmm_out(result, t, sparse, dense, 0.0, 1.0);  // redispatch!
}

Tensor _sparse_gcs_addmm(
  const Tensor& t,
  const SparseTensor& sparse,
  const Tensor& dense,
  Scalar beta,
  Scalar alpha
) {
  // _sparse_addmm forward is functionally equivalent to addmm; it's
  // just the backward that is different.  This technically does an
  // unnecessary redispatch, I was too lazy to make it not do that
  return at::addmm(t, sparse, dense, beta, alpha);
}

// Functions for element-wise addition.
Tensor add_sparse_gcs(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha);  // redispatch!
}

Tensor& add_sparse_gcs_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

int32_t gcs_to_dense_convert(int32_t iptr, int32_t icol, Tensor& out,
                             const SparseTensor& src) {
  int32_t drow, dcol;
  int32_t index = out.storage_offset();
  auto src_impl = get_sparse_impl<SparseGCSTensorImpl>(src);
  
  auto strides0 = src_impl->strides0();
  auto strides1 = src_impl->strides1();
  auto dims0 = src_impl->dims0();
  auto dims1 = src_impl->dims1();
  
  for (int i = 0; i < dims0.size(); ++i) {
    index += out.stride(i) * (int(iptr/strides0[i]) % src.size(i));
  }

  for (int i = 0; i < dims1.size(); ++i) {
    index += out.stride(src_impl->rsplit_dim() + i) *
      (int(icol/strides1[i]) % src.size(src_impl->rsplit_dim() + i));
  }

  return index;
}

Tensor& add_out_dense_sparse_gcs_cpu(Tensor& out, const Tensor& dense,
                                     const SparseTensor& src, Scalar alpha) {
  AT_ASSERT(!out.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(src.is_sparse());

  AT_ASSERT(!dense.is_cuda());
  TORCH_CHECK(!out.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", src.sizes(), " (FYI: op2-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());

  TORCH_CHECK(canCast(commonDtype, out.scalar_type()), "Can't convert result type ",
              commonDtype, " to output ", out.scalar_type(), " in add operation");

  auto src_values = src.values().to(commonDtype);
  auto src_pointers = src.pointers();
  auto src_indices = src.indices();

  out.resize_as_(dense);
  Tensor resultBuffer = out;
  Tensor valuesBuffer = src_values.to(commonDtype);
  
  if (out.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(out, dense)) {
    resultBuffer.copy_(dense);
  }

  AT_DISPATCH_ALL_TYPES(commonDtype, "add_out_op2_sparse_gcs", [&] {
    auto values_accessor = src_values.accessor<scalar_t, 1>();
    auto pointers_accessor = src_pointers.accessor<int32_t, 1>();
    auto indices_accessor = src_indices.accessor<int32_t, 1>();

    scalar_t *out_ptr = out.data_ptr<scalar_t>();
    scalar_t cast_value = alpha.to<scalar_t>();

    for (int32_t iptr = 0; iptr < src_pointers.size(0)-1; ++iptr) {
      int32_t start_index = pointers_accessor[iptr];
      int32_t end_index = pointers_accessor[iptr + 1];
      int32_t nindices = end_index - start_index;

      for (int i = start_index; i < end_index; ++i) {
        auto icol = indices_accessor[i];
        auto index = gcs_to_dense_convert(iptr, icol, out, src);
        out_ptr[index] += cast_value * values_accessor[i];
      }
    }
  });
  return out;
}


SparseTensor& add_out_sparse_gcs_cpu(SparseTensor& out, const SparseTensor& self, const SparseTensor& src, Scalar alpha) {
  if (!self.is_sparse()) {
    return add_out_dense_sparse_gcs_cpu(out, self, src, alpha);
  }
  else {
    TORCH_CHECK(false, "NotImplementedError: Addition of sparse GCS tensors is not yet implemented.")
  }  
  return out;
}

}} // namespace at::sparse
