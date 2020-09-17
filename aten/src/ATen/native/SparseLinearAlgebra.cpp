#include <ATen/native/sparse/SparseGCSTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseGCSTensorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>

namespace at { namespace native {
  using namespace at::sparse;

  Tensor sparse_gcs_mm_cpu(const SparseTensor& sparse_, Tensor& result, const Tensor& t, const Tensor& dense,
                         const Scalar& alpha, const Scalar& beta) {

    LongTensor indices = sparse_._indices();
    LongTensor pointers = sparse_.pointers();
    Tensor values      = sparse_._values();
    int64_t nnz = sparse_._nnz();
    
    auto values_accessor = values.accessor<int64_t, 1>();
    auto pointers_accessor = pointers.accessor<int64_t, 1>();
    auto indices_accessor = indices.accessor<int64_t, 1>();

    AT_DISPATCH_ALL_TYPES(
      values.scalar_type(), "addmm_sparse_gcs_dense", [&] {
        scalar_t cast_alpha = alpha.to<scalar_t>();
        scalar_t cast_beta = beta.to<scalar_t>();

        if (cast_beta == 0) {
          result.zero_();
        } else if (cast_beta == 1) {
          if (!is_same_tensor(result, t)) {
            result.copy_(t);
          }
        } else {
          at::mul_out(result, t, scalar_to_tensor(beta));
        }

        scalar_t * dense_ptr = dense.data_ptr<scalar_t>();
        scalar_t * result_ptr = result.data_ptr<scalar_t>();

    });

    return result;
  }

}}                              // namespace at::native 
