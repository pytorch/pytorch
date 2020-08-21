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

Tensor& add_sparse_gcs_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

Tensor& add_out_dense_sparse_gcs_cpu(Tensor& out, const Tensor& dense, const SparseTensor& src, Scalar alpha) {
  AT_ASSERT(!out.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(src.is_sparse());

  AT_ASSERT(!dense.is_cuda());
  TORCH_CHECK(!out.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(sparse.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  auto commonDtype = promoteTypes(dense.scalar_type(), sparse.scalar_type());

  TORCH_CHECK(canCast(commonDtype, out.scalar_type()), "Can't convert result type ",
              commonDtype, " to output ", out.scalar_type(), " in add operation");

  out.resize_as_(dense);

  
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
