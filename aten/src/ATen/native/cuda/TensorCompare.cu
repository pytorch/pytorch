#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAHalf.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

#include <THC/THCTensorTypeUtils.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

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
struct CmpOpTensorCUDA {
  static void apply(at::Tensor& ret, const at::Tensor& self, const at::Tensor& other) {
    at::cuda::CUDA_tensor_apply3<uint8_t, scalar, scalar>(ret, self, other,
        [] __device__ (uint8_t& ret_val, const scalar& self_val, const scalar& other_val) {
          ret_val = Comparator<scalar>()(self_val, other_val);
      }
    );
  }
};

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

template<template<typename T> class Comparator>
at::Tensor& cmp_out_cuda(at::Tensor& result, const at::Tensor& self, at::Scalar other, const char* op_name) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    CmpOpFloatingCUDA<Comparator>::apply(result, self, other);
  } else {
    AT_DISPATCH_ALL_MATH_TYPES(self.type(), op_name, [&]() {
      CmpOpCUDA<Comparator, scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

template<template<typename T> class Comparator>
at::Tensor& cmp_out_cuda(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_out_cuda<Comparator>(result, self, other.pImpl->localScalar(), op_name);
  }
  result.resize_(self.sizes());
  AT_DISPATCH_ALL_MATH_TYPES(self.type(), op_name, [&]() {
      CmpOpTensorCUDA<Comparator, scalar_t>::apply(result, self, other);
  });
  return result;
}

template<template<typename T> class Comparator>
at::Tensor cmp_cuda(const at::Tensor& self, at::Scalar other, const char* op_name) {
  at::Tensor result = self.type().toScalarType(at::kByte).tensor(self.sizes());
  return cmp_out_cuda<Comparator>(result, self, other, op_name);
}

template<template<typename T> class Comparator>
at::Tensor cmp_cuda(const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_cuda<Comparator>(self, other.pImpl->localScalar(), op_name);
  }
  at::Tensor result = self.type().toScalarType(at::kByte).tensor(self.sizes());
  return cmp_out_cuda<Comparator>(result, self, other, op_name);
}
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

Tensor lt_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::less>(self, other, "lt");
}

Tensor gt_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::greater>(self, other, "gt");
}

Tensor le_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::less_equal>(self, other, "le");
}

Tensor ge_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::greater_equal>(self, other, "ge");
}

Tensor eq_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::equal_to>(self, other, "eq");
}

Tensor ne_cuda(const Tensor& self, Scalar other) {
  return cmp_cuda<std::not_equal_to>(self, other, "ne");
}

Tensor& lt_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::less>(result, self, other, "lt");
}

Tensor& gt_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::greater>(result, self, other, "gt");
}

Tensor& le_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::less_equal>(result, self, other, "le");
}

Tensor& ge_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::greater_equal>(result, self, other, "ge");
}

Tensor& eq_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::equal_to>(result, self, other, "eq");
}

Tensor& ne_out_cuda(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cuda<std::not_equal_to>(result, self, other, "ne");
}

Tensor lt_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::less>(self, other, "lt");
}

Tensor gt_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::greater>(self, other, "gt");
}

Tensor le_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::less_equal>(self, other, "le");
}

Tensor ge_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::greater_equal>(self, other, "le");
}

Tensor eq_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::equal_to>(self, other, "eq");
}

Tensor ne_cuda(const Tensor& self, const Tensor& other) {
  return cmp_cuda<std::not_equal_to>(self, other, "ne");
}

Tensor& lt_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::less>(result, self, other, "lt");
}

Tensor& gt_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::greater>(result, self, other, "gt");
}

Tensor& le_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::less_equal>(result, self, other, "le");
}

Tensor& ge_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::greater_equal>(result, self, other, "ge");
}

Tensor& eq_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::equal_to>(result, self, other, "eq");
}

Tensor& ne_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cuda<std::not_equal_to>(result, self, other, "ne");
}
}} // namespace at::native
