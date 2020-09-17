#include <ATen/native/sparse/SparseGCSTensorMath.h>
#include <ATen/SparseTensorUtils.h>

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
#include <ATen/NamedTensorUtils.h>

#include <functional>
#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#if !AT_MKL_ENABLED()

namespace at { namespace native {
    using namespace at::sparse;
    Tensor& sparse_mm_mkl(Tensor& res, const SparseTensor& sparse_, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {

      abort();
    }
}}

#else  // AT_MKL_ENABLED

#include <mkl.h>
#include <mkl_spblas.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>

namespace at { namespace native {
    using namespace at::sparse;
    
    static inline void sparse_mm_mkl_impl(double * res, long * indicies, long * pointers, double * values,
                                          double * dense, double * t, double alpha, double beta, int64_t nrows,
                                          int64_t ncols) {
      sparse_matrix_t A;
      // mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, );
    }

    template <typename scalar_t>
    static inline void sparse_mm_mkl_template(Tensor& res, const LongTensor& indices, const LongTensor& pointers,
      const Tensor& values, const Tensor& dense, const Tensor& t, Scalar alpha, Scalar beta, IntArrayRef size) {
      
      sparse_mm_mkl_impl(res.data(), indices.data(), pointers.data(), values.data(), dense.data(),
                         t.data(), alpha, beta, size[0], size[1]);
    }

  Tensor& sparse_mm_mkl(Tensor& res, const SparseTensor& sparse_, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {

    AT_DISPATCH_FLOATING_TYPES(
      values.scalar_type(), "addmm_sparse_gcs_dense", [&] {
        sparse_mm_mkl_template<scalar_t>(res, sparse_._indices(),
         sparse_.pointers(), sparse_._values(), dense, t, alpha, beta, sparse_.size());
    });
    return res;
  }
}}
#endif
