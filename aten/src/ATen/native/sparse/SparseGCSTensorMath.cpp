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
      int64_t index;
      
      for (int i = start_index; i < end_index; ++i) {
        index = indices_accessor[i];
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
