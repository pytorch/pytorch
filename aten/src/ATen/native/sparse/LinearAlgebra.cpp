#include <ATen/native/sparse/SparseGCSTensorMath.h>

#include <ATen/SparseTensorImpl.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/core/grad_mode.h>
#include <functional>
#include <numeric>
#include <vector>
#include <limits>
#include <ATen/NamedTensorUtils.h>


namespace at { namespace native {
  using namespace at::sparse;

  Tensor& sparse_gcs_mm_cpu(Tensor& res, const SparseTensor& sparse_, const Tensor& t, const Tensor& dense,
                         Scalar alpha, Scalar beta) {

    LongTensor indices = sparse_.indices();
    LongTensor pointers = sparse_.pointers();
    Tensor values      = sparse_.values();
    int64_t nnz = sparse_._nnz();

    std::cout << "pre dispatch: " << values.scalar_type() << std::endl;;
    AT_DISPATCH_FLOATING_TYPES(
      values.scalar_type(), "addmm_sparse_gcs_dense", [&] {
        scalar_t cast_alpha = alpha.to<scalar_t>();
        scalar_t cast_beta = beta.to<scalar_t>();

        if (cast_beta == 0) {
          res.zero_();
        } else if (cast_beta == 1) {
          if (!is_same_tensor(res, t)) {
            res.copy_(t);
          }
        } else {
          at::mul_out(res, t, scalar_to_tensor(beta));
        }
    });

    if (at::hasMKL() && (at::native::is_floating_point(res) ||
                         at::native::is_complex(res)) &&
        res.is_contiguous()) {

      at::native::sparse_mm_mkl(res, sparse_, dense, t, alpha, beta);
    }
    else {
          
    }


    return res;
  }

}}                              // namespace at::native 
