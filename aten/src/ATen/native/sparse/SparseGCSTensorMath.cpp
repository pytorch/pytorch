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

namespace at { namespace native {

using namespace at::sparse;
Tensor& s_addmm_out_sparse_gcs_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha) {
  // TODO: This error message seems awfully opaque
  AT_ASSERT(!t.is_cuda());
  TORCH_CHECK(!r.is_cuda(), "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");

  // ixj * jxk = ixk
  int64_t dim_i = sparse_.size(0);
  int64_t dim_j = sparse_.size(1);
  int64_t dim_k = dense.size(1);

  TORCH_CHECK(dense.size(0) == dim_j,
              "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == dim_i,
              "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == dim_k,
              "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

  r.resize_({dim_i, dim_k});

  // TODO: why does that nnz == 0 condition exist in the COO code?

  at::sparse_gcs_mm(r, sparse_, t, dense, alpha, beta);


  return r;
}
    
Tensor& addmm_out_sparse_gcs_dense_cpu(
    Tensor& result,
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_gcs_dense_cpu(result, b_self, mat1, mat2, beta, alpha);
}

 Tensor addmm_sparse_gcs_dense_cpu(
    const Tensor& self,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha) {
   Tensor r = at::empty({0}, self.options());
   s_addmm_out_sparse_gcs_dense_cpu(r, self, sparse, dense, beta, alpha);
   return r;
}

SparseTensor& _sparse_gcs_mm_out(
  SparseTensor& result,
  const SparseTensor& sparse,
  const Tensor& dense
) {
  Tensor t = at::zeros({}, dense.options());
  return at::addmm_out(result, t, sparse, dense, 0, 1);  // redispatch!
}

Tensor add_sparse_gcs(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha);
  Tensor result = at::empty({0}, self.options().dtype(commonDtype));
  return at::add_out(result, self, other, alpha);  // redispatch!
}

Tensor& add_sparse_gcs_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

int64_t gcs_to_dense_convert(int64_t iptr, int64_t icol, Tensor& out,
                             const SparseTensor& src) {
  int64_t drow, dcol;
  std::vector<int64_t> dense_indices;
  int64_t index = out.storage_offset();
  auto src_impl = get_sparse_impl<SparseGCSTensorImpl>(src);
  
  auto strides0 = src_impl->strides0();
  auto strides1 = src_impl->strides1();
  auto dims0 = src_impl->dims0();
  auto dims1 = src_impl->dims1();
  
  for (int i = 0; i < dims0.size(); ++i) {
    index += out.stride(i) * (int(iptr/strides0[i]) % src.size(i));
  }

  for (int i = 0; i < dims1.size(); ++i) {
    index += out.stride(src_impl->rsplit_dim() + i) * (int(icol/strides1[i]) % src.size(src_impl->rsplit_dim() + i));
  }

  return index;
}

Tensor& add_out_dense_sparse_gcs_cpu(Tensor& out, const Tensor& dense, const SparseTensor& src, Scalar alpha) {
  AT_ASSERT(!out.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(src.is_sparse());

  AT_ASSERT(!dense.is_cuda());
  TORCH_CHECK(!out.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(src.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", src.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), src.scalar_type());

  TORCH_CHECK(canCast(commonDtype, out.scalar_type()), "Can't convert result type ",
              commonDtype, " to output ", out.scalar_type(), " in add operation");

  Tensor src_values = src.values().to(commonDtype);
  Tensor src_pointers = src.pointers();
  Tensor src_indices = src.indices();

  out.resize_as_(dense);
  Tensor resultBuffer = out;
  Tensor valuesBuffer = src_values.to(commonDtype);
  
  if (out.scalar_type() != commonDtype) {
    resultBuffer = dense.to(commonDtype);
  } else if (!is_same_tensor(out, dense)) {
    resultBuffer.copy_(dense);
  }

  AT_DISPATCH_ALL_TYPES(commonDtype, "add_dense_sparse_gcs", [&] {
    auto values_accessor = src_values.accessor<scalar_t, 1>();
    auto pointers_accessor = src_pointers.accessor<int64_t, 1>();
    auto indices_accessor = src_indices.accessor<int64_t, 1>();

    scalar_t *out_ptr = out.data_ptr<scalar_t>();
    scalar_t cast_value = alpha.to<scalar_t>();

    for (int64_t iptr = 0; iptr < src_pointers.size(0)-1; ++iptr) {
      int64_t start_index = pointers_accessor[iptr];
      int64_t end_index = pointers_accessor[iptr + 1];
      int64_t nindices = end_index - start_index;
      int64_t icol;
      int64_t index;

      for (int i = start_index; i < end_index; ++i) {
        icol = indices_accessor[i];
        index = gcs_to_dense_convert(iptr, icol, out, src);
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

  AT_ASSERT(!self.is_cuda());  // the dispatch argument
  TORCH_CHECK(!out.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(self.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ",
              self.sizes(), " != ", src.sizes());

  auto commonDtype = promoteTypes(self.scalar_type(), src.scalar_type());
  TORCH_CHECK(canCast(commonDtype, out.scalar_type()), "Can't convert result type ", commonDtype,
              " to output ", out.scalar_type(), " in add operation");

  out.resize_as_(src);

  int64_t self_nnz = self._nnz(), src_nnz = src._nnz(), max_nnz = self_nnz + src_nnz;
  
  Tensor self_values = self.values().to(commonDtype);
  Tensor src_values = src.values().to(commonDtype);
  Tensor out_values = new_values_with_size_of(self_values, max_nnz).zero_();

  // auto self_indices_accessor = self_indices.accessor<int64_t, 1>();
  // auto out_indices_accessor = out_indices.accessor<int64_t, 1>();
  // auto src_indices_accessor = src_indices.accessor<int64_t, 1>();

  AT_DISPATCH_ALL_TYPES(
    commonDtype, "cadd_sparse_gcs", [&] {
                                      
    }
  );

  if (out.scalar_type() != commonDtype) {
    out_values = out_values.to(out.scalar_type());
  }

  return out;
}

}}
