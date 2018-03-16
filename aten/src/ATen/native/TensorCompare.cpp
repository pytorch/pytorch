#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorCompare.h"

namespace {
template <typename scalar_t>
void where_cpu(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  at::CPU_tensor_apply4<scalar_t, uint8_t, scalar_t, scalar_t>(
      ret,
      condition,
      self,
      other,
      [](scalar_t& ret_val,
         const uint8_t& cond_val,
         const scalar_t& self_val,
         const scalar_t& other_val) {
        ret_val = cond_val ? self_val : other_val;
      });
}

template<template<typename T> class Comparator, typename scalar_out, typename scalar>
struct CmpOp {
  static void apply(at::Tensor& ret, const at::Tensor& self, const at::Tensor& other) {
    at::CPU_tensor_apply3<scalar_out, scalar, scalar>(ret, self, other,
        [](scalar_out& ret_val, const scalar& self_val, const scalar& other_val) {
          ret_val = at::convert<scalar_out>(Comparator<scalar>()(self_val, other_val));
        }
    );
  }

  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();
    at::CPU_tensor_apply2<scalar_out, scalar>(ret, self,
        [other_val](scalar_out& ret_val, const scalar& self_val) {
          ret_val = at::convert<scalar_out>(Comparator<scalar>()(self_val, other_val));
      }
    );
  }
};

template<template<typename> class Comparator>
at::Tensor& cmp_out_cpu(at::Tensor& result, const at::Tensor& self, at::Scalar other, const char* op_name) {
  result.resize_(self.sizes());
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOpScalar<Comparator, uint8_t, scalar_t, CmpOp>::apply(result, self, other);
  });
  return result;
}

template<template<typename> class Comparator>
at::Tensor& cmp_out_cpu(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_out_cpu<Comparator>(result, self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor b_self, b_other;
  std::tie(b_self, b_other) = at::expand_outplace(self, other, op_name);
  result.resize_(b_self.sizes());
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOp<Comparator, uint8_t, scalar_t>::apply(result, b_self, b_other);
  });
  return result;
}

template<template<typename T> class Comparator>
at::Tensor cmp_cpu(const at::Tensor& self, at::Scalar other, const char* op_name) {
  at::Tensor result = self.type().toScalarType(at::kByte).tensor();
  return cmp_out_cpu<Comparator>(result, self, other, op_name);
}

template<template<typename T> class Comparator>
at::Tensor cmp_cpu(const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_cpu<Comparator>(self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor result = self.type().toScalarType(at::kByte).tensor();
  return cmp_out_cpu<Comparator>(result, self, other, op_name);
}

template<template<typename T> class Comparator>
at::Tensor& cmp_inplace_cpu(at::Tensor& self, at::Scalar other, const char* op_name) {
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOpScalar<Comparator, scalar_t, scalar_t, CmpOp>::apply(self, self, other);
  });
  return self;
}

template<template<typename T> class Comparator>
at::Tensor& cmp_inplace_cpu(at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_inplace_cpu<Comparator>(self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor b_other;
  std::tie(b_other) = at::expand_inplace(self, other, op_name);
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOp<Comparator, scalar_t, scalar_t>::apply(self, self, b_other);
  });
  return self;
}
} // namespace

namespace at { namespace native {

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol) {
  if (!self.sub(other).abs().le(other.abs().mul(rtol).add(atol)).all().toCByte()) {
    return false;
  }

  return true;
}

bool is_nonzero(const Tensor& self) {
  if (self.numel() != 1) {
    runtime_error("bool value of Tensor with more than one value is ambiguous");
  }
  Scalar localScalar = self.pImpl->localScalar();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isIntegral()){
    return localScalar.to<int64_t>() != 0;
  }
  runtime_error("expected non-Tensor backed scalar");
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  if (condition.type().scalarType() != ScalarType::Byte) {
    runtime_error("Expected condition to have ScalarType Byte, but got ScalarType %s",
                  toString(condition.type().scalarType()));
  }
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

Tensor _s_where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_ALL_TYPES(ret.type(), "where", [&] {
    where_cpu<scalar_t>(ret, condition, self, other);
  });
  return ret;
}

#define TENSOR_IMPLEMENT_COMPARATOR(NAME, COMP)                                     \
  Tensor NAME##_cpu(const Tensor& self, Scalar other) {                             \
    return cmp_cpu<COMP>(self, other, #NAME);                                       \
  }                                                                                 \
  Tensor& NAME##_out_cpu(Tensor& result, const Tensor& self, Scalar other) {        \
    return cmp_out_cpu<COMP>(result, self, other, #NAME);                           \
  }                                                                                 \
  Tensor NAME##_cpu(const Tensor& self, const Tensor& other) {                      \
    return cmp_cpu<COMP>(self, other, #NAME);                                       \
  }                                                                                 \
  Tensor& NAME##_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) { \
    return cmp_out_cpu<COMP>(result, self, other, #NAME);                           \
  }                                                                                 \
  Tensor& NAME##_inplace_cpu(Tensor& self, Scalar other) {                          \
    return cmp_inplace_cpu<COMP>(self, other, #NAME);                               \
  }                                                                                 \
  Tensor& NAME##_inplace_cpu(Tensor& self, const Tensor& other) {                   \
    return cmp_inplace_cpu<COMP>(self, other, #NAME);                               \
  }                                                                                 \


TENSOR_IMPLEMENT_COMPARATOR(lt, std::less)
TENSOR_IMPLEMENT_COMPARATOR(gt, std::greater)
TENSOR_IMPLEMENT_COMPARATOR(le, std::less_equal)
TENSOR_IMPLEMENT_COMPARATOR(ge, std::greater_equal)
TENSOR_IMPLEMENT_COMPARATOR(eq, std::equal_to)
TENSOR_IMPLEMENT_COMPARATOR(ne, std::not_equal_to)
}} // namespace at::native
