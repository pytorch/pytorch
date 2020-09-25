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
      AT_ERROR("sparse_mm_mkl: ATen not compiled with MKL support");
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
    
    static inline void sparse_mm_mkl_impl(float * res, int * indices, int * pointers, float * values,
                                          float * dense, float * t, float alpha, float beta, int64_t nrows,
                                          int64_t ncols, int64_t dense_ncols) {
      sparse_matrix_t A;
      matrix_descr desc;
      desc.type = SPARSE_MATRIX_TYPE_GENERAL;
      mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrows, ncols, pointers, pointers+1, indices, values);
      mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, desc,
                      SPARSE_LAYOUT_ROW_MAJOR, dense, dense_ncols, dense_ncols, beta, res, dense_ncols);
    }

    // TODO: The types here are long int but int = LongTensor only when using lp64 for MKL. SHould we resort
    // to using IntTensor for the indices and pointers?
    static inline void sparse_mm_mkl_impl(double * res, int * indices, int * pointers, double * values,
                                          double * dense, double * t, double alpha, double beta, int64_t nrows,
                                          int64_t ncols, int64_t dense_ncols) {
      sparse_matrix_t A;
      matrix_descr desc;
      desc.type = SPARSE_MATRIX_TYPE_GENERAL;
      mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrows, ncols, pointers, pointers+1, indices, values);
      mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, desc,
                      SPARSE_LAYOUT_ROW_MAJOR, dense, dense_ncols, dense_ncols, beta, res, dense_ncols);
    }

    template <typename scalar_t>
    static inline void sparse_mm_mkl_template(Tensor& res, const IntTensor& indices, const IntTensor& pointers,
      const Tensor& values, const Tensor& dense, const Tensor& t, Scalar alpha, Scalar beta, IntArrayRef size,
      IntArrayRef dense_size) {
      
      sparse_mm_mkl_impl(res.data_ptr<scalar_t>(),
                         indices.data_ptr<int>(),
                         pointers.data_ptr<int>(),
                         values.data_ptr<scalar_t>(),
                         dense.data_ptr<scalar_t>(),
                         t.data_ptr<scalar_t>(),
                         alpha.to<scalar_t>(),
                         beta.to<scalar_t>(), size[0], size[1], dense_size[1]);
    }

  Tensor& sparse_mm_mkl(Tensor& res, const SparseTensor& sparse_, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {
    AT_DISPATCH_FLOATING_TYPES(
      dense.scalar_type(), "addmm_sparse_gcs_dense", [&] {
        sparse_mm_mkl_template<scalar_t>(res, sparse_.indices(),
                                         sparse_.pointers(), sparse_.values(), dense, t,
                                         alpha, beta, sparse_.sizes(), dense.sizes());
    });
    return res;
  }
}}
#endif
