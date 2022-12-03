/*
The following source file implements a sparse linear operator using cusparseLt
*/

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>


namespace at {
namespace native {

Tensor _cusparselt_linear(const Tensor& sparse, const Tensor& dense) {
  std::cout << "Calling _cusparselt_linear" << std::endl;
  return sparse.mm(dense);
}

}
}
