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
  Tensor& sparse_mm_mkl(Tensor& res, const Tensor& indices, const Tensor& pointers,
                                        const Tensor& values, const Tensor& dense, const Tensor& t,
                                        Scalar alpha, Scalar beta) {
    return res;
  }
}}
#endif
