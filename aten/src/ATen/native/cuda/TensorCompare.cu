#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAHalf.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include "ATen/TensorCompare.h"

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

template<template<typename T> class Comparator, typename scalar_out, typename scalar>
struct CmpOpCUDA {
  static void apply(at::Tensor& ret, const at::Tensor& self, const at::Tensor& other) {
    at::cuda::CUDA_tensor_apply3<scalar_out, scalar, scalar>(ret, self, other,
        [] __device__ (scalar_out& ret_val, const scalar& self_val, const scalar& other_val) {
          ret_val = ScalarConvert<uint8_t, scalar_out>::to(Comparator<scalar>()(self_val, other_val));
        }
    );
  }

  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = at::convert<scalar>(other.to<double>());
    at::cuda::CUDA_tensor_apply2<scalar_out, scalar>(ret, self,
        [other_val] __device__ (scalar_out& ret_val, const scalar& self_val) {
          ret_val = ScalarConvert<uint8_t, scalar_out>::to(Comparator<scalar>()(self_val, other_val));
        }
    );
  }
};

template<template<typename> class Comparator>
at::Tensor& cmp_out_cuda(at::Tensor& result, const at::Tensor& self, at::Scalar other, const char* op_name) {
  result.resize_(self.sizes());
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), op_name, [&]() {
    using cuda_scalar_t = at::cuda::type<scalar_t>;
    CmpOpScalar<Comparator, uint8_t, cuda_scalar_t, CmpOpCUDA>::apply(result, self, other);
  });
  return result;
}

template<template<typename> class Comparator>
at::Tensor& cmp_out_cuda(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_out_cuda<Comparator>(result, self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor b_self, b_other;
  std::tie(b_self, b_other) = at::expand_outplace(self, other, op_name);
  result.resize_(b_self.sizes());
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), op_name, [&]() {
    using cuda_scalar_t = at::cuda::type<scalar_t>;
    CmpOpCUDA<Comparator, uint8_t, cuda_scalar_t>::apply(result, b_self, b_other);
  });
  return result;
}

template<template<typename T> class Comparator>
at::Tensor cmp_cuda(const at::Tensor& self, at::Scalar other, const char* op_name) {
  at::Tensor result = self.type().toScalarType(at::kByte).tensor();
  return cmp_out_cuda<Comparator>(result, self, other, op_name);
}

template<template<typename T> class Comparator>
at::Tensor cmp_cuda(const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_cuda<Comparator>(self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor result = self.type().toScalarType(at::kByte).tensor();
  return cmp_out_cuda<Comparator>(result, self, other, op_name);
}

template<template<typename T> class Comparator>
at::Tensor& cmp_inplace_cuda(at::Tensor& self, at::Scalar other, const char* op_name) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), op_name, [&]() {
    using cuda_scalar_t = at::cuda::type<scalar_t>;
    CmpOpScalar<Comparator, cuda_scalar_t, cuda_scalar_t, CmpOpCUDA>::apply(self, self, other);
  });
  return self;
}

template<template<typename T> class Comparator>
at::Tensor& cmp_inplace_cuda(at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_inplace_cuda<Comparator>(self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor b_other;
  std::tie(b_other) = at::expand_inplace(self, other, op_name);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), op_name, [&]() {
    using cuda_scalar_t = at::cuda::type<scalar_t>;
    CmpOpCUDA<Comparator, cuda_scalar_t, cuda_scalar_t>::apply(self, self, b_other);
  });
  return self;
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


#define TENSOR_IMPLEMENT_COMPARATOR_CUDA(NAME, COMP)                                 \
  Tensor NAME##_cuda(const Tensor& self, Scalar other) {                             \
    return cmp_cuda<COMP>(self, other, #NAME);                                       \
  }                                                                                  \
  Tensor& NAME##_out_cuda(Tensor& result, const Tensor& self, Scalar other) {        \
    return cmp_out_cuda<COMP>(result, self, other, #NAME);                           \
  }                                                                                  \
  Tensor NAME##_cuda(const Tensor& self, const Tensor& other) {                      \
    return cmp_cuda<COMP>(self, other, #NAME);                                       \
  }                                                                                  \
  Tensor& NAME##_out_cuda(Tensor& result, const Tensor& self, const Tensor& other) { \
    return cmp_out_cuda<COMP>(result, self, other, #NAME);                           \
  }                                                                                  \
  Tensor& NAME##_inplace_cuda(Tensor& self, Scalar other) {                          \
    return cmp_inplace_cuda<COMP>(self, other, #NAME);                               \
  }                                                                                  \
  Tensor& NAME##_inplace_cuda(Tensor& self, const Tensor& other) {                   \
    return cmp_inplace_cuda<COMP>(self, other,  #NAME);                              \
  }                                                                                  \


TENSOR_IMPLEMENT_COMPARATOR_CUDA(lt, std::less)
TENSOR_IMPLEMENT_COMPARATOR_CUDA(gt, std::greater)
TENSOR_IMPLEMENT_COMPARATOR_CUDA(le, std::less_equal)
TENSOR_IMPLEMENT_COMPARATOR_CUDA(ge, std::greater_equal)
TENSOR_IMPLEMENT_COMPARATOR_CUDA(eq, std::equal_to)
TENSOR_IMPLEMENT_COMPARATOR_CUDA(ne, std::not_equal_to)
}} // namespace at::native
