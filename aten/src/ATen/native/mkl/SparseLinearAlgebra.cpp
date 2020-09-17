#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKL_ENABLED()

namespace at { namespace native {
  Tensor& sparse_mm_mkl(Tensor& res, const Tensor& indices, const Tensor& pointers,
                                        const Tensor& values, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {
    abort();
  }
}}

#else  // AT_MKL_ENABLED

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <mkl.h>
#include <mkl_spblas.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>

namespace at { namespace native {

    static inline void sparse_mm_mkl_impl(double * res, long * indicies, long * pointers, double * values,
                                          double * dense, double * t, double alpha, double beta) {
      
    }

    template <typename scalar_t>
    static inline void sparse_mm_mkl_template(Tensor& res, const LongTensor& indices, const LongTensor& pointers,
      const Tensor& values, const Tensor& dense, const Tensor& t, Scalar alpha, Scalar beta) {
      
      sparse_mm_mkl_impl(res.data(), indices.data(), pointers.data(), values.data(), dense.data(),
                         t.data(), alpha, beta);
    }

  Tensor& sparse_mm_mkl(Tensor& res, const LongTensor& indices, const LongTensor& pointers,
                                        const Tensor& values, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {

    AT_DISPATCH_FLOATING_TYPES(
      values.scalar_type(), "addmm_sparse_gcs_dense", [&] {
                                                        sparse_mm_mkl_template<scalar_t>(res, indices, pointers,
                                                                                         values, dense, t, alpha, beta);
    });
    return res;
  }
}}
#endif
