#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAHalf.cuh"
#include "ATen/cuda/CUDAHalfMath.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

namespace {
template <typename scalar_t>
void where_cuda(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  // Yes this name is repetitive, but the CPU version is called
  // CPU_tensor_apply4 and we don't have a CPU namespace or directory.
  at::cuda::CUDA_tensor_apply4<scalar_t, uint8_t, scalar_t, scalar_t>(
      ret,
      condition,
      self,
      other,
      [] __device__(
          scalar_t & ret_val,
          const uint8_t& cond_val,
          const scalar_t& self_val,
          const scalar_t& other_val) {
        ret_val = cond_val ? self_val : other_val;
      });
}

template<template<typename T> class Comparator, typename scalar>
struct CmpOpCUDA {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();
    at::cuda::CUDA_tensor_apply2<uint8_t, scalar>(ret, self,
        [other_val] __device__ (uint8_t& ret_val, const scalar& self_val) {
          ret_val = Comparator<scalar>()(self_val, other_val);
      }
    );
  }
};

template<template<typename T> class Comparator>
struct CmpOpCUDA<Comparator, half> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = at::convert<half>(other.to<double>());
    at::cuda::CUDA_tensor_apply2<uint8_t, half>(ret, self,
        [other_val] __device__ (uint8_t& ret_val, const half& self_val) {
          ret_val = Comparator<half>()(self_val, other_val);
      }
    );
  }
};

template<typename scalar>
using LeOpCUDA = CmpOpCUDA<std::less_equal, scalar>;
template<typename scalar>
using GeOpCUDA = CmpOpCUDA<std::greater_equal, scalar>;
template<typename scalar>
using EqOpCUDA = CmpOpCUDA<std::equal_to, scalar>;
template<typename scalar>
using NeOpCUDA = CmpOpCUDA<std::not_equal_to, scalar>;

template<template<typename T> class Comparator>
struct CmpOpFloatingCUDA {
  static void apply(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
    auto other_val = at::convert<half>(other.to<double>());
    at::cuda::CUDA_tensor_apply2<uint8_t, half>(result, self.toType(at::kHalf),
        [other_val] __device__ (uint8_t& result_val, const half& self_val) {
          result_val = Comparator<half>()(self_val, other_val);
      }
    );
  }
};

using LeOpFloatingCUDA = CmpOpFloatingCUDA<std::less_equal>;
using GeOpFloatingCUDA = CmpOpFloatingCUDA<std::greater_equal>;
using EqOpFloatingCUDA = CmpOpFloatingCUDA<std::equal_to>;
using NeOpFloatingCUDA = CmpOpFloatingCUDA<std::not_equal_to>;
} // namespace

namespace at { namespace native {
Tensor _s_where_cuda(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "where", [&] {
    where_cuda<cuda::type<scalar_t>>(ret, condition, self, other);
  });
  return ret;
}

Tensor& le_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    LeOpFloatingCUDA::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_MATH_TYPES(self.type(), "le", [&]() {
      LeOpCUDA<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor& ge_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    GeOpFloatingCUDA::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_MATH_TYPES(self.type(), "ge", [&]() {
      GeOpCUDA<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor& eq_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    EqOpFloatingCUDA::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_MATH_TYPES(self.type(), "eq", [&]() {
      EqOpCUDA<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor& ne_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    NeOpFloatingCUDA::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_MATH_TYPES(self.type(), "ne", [&]() {
      NeOpCUDA<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}
}} // namespace at::native
