#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

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

template<template<typename T> class Comparator, typename scalar>
struct CmpOp {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();
    at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
        [other_val](uint8_t& ret_val, scalar self_val) {
          ret_val = Comparator<scalar>()(self_val, other_val);
      }
    );
  }
};

template<typename scalar>
using LeOp = CmpOp<std::less_equal, scalar>;
template<typename scalar>
using GeOp = CmpOp<std::greater_equal, scalar>;
template<typename scalar>
using EqOp = CmpOp<std::equal_to, scalar>;
template<typename scalar>
using NeOp = CmpOp<std::not_equal_to, scalar>;

template<template<typename T> class Comparator>
struct CmpOpFloating {
  static void apply(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<double>();
    at::CPU_tensor_apply2<uint8_t, double>(result, self.toType(at::kDouble),
        [other_val](uint8_t& result_val, double self_val) {
          result_val = Comparator<double>()(self_val, other_val);
      }
    );
  }
};

using LeOpFloating = CmpOpFloating<std::less_equal>;
using GeOpFloating = CmpOpFloating<std::greater_equal>;
using EqOpFloating = CmpOpFloating<std::equal_to>;
using NeOpFloating = CmpOpFloating<std::not_equal_to>;
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

Tensor le(const Tensor& self, Scalar other) {
  Tensor result = self.type().toScalarType(kByte).tensor(self.sizes());
  return at::le_out(result, self, other);
}

Tensor ge(const Tensor& self, Scalar other) {
  Tensor result = self.type().toScalarType(kByte).tensor(self.sizes());
  return at::ge_out(result, self, other);
}

Tensor eq(const Tensor& self, Scalar other) {
  Tensor result = self.type().toScalarType(kByte).tensor(self.sizes());
  return at::eq_out(result, self, other);
}

Tensor ne(const Tensor& self, Scalar other) {
  Tensor result = self.type().toScalarType(kByte).tensor(self.sizes());
  return at::ne_out(result, self, other);
}

Tensor & le_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    LeOpFloating::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_TYPES(self.type(), "le", [&]() {
      LeOp<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor & ge_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    GeOpFloating::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_TYPES(self.type(), "ge", [&]() {
      GeOp<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor & eq_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    EqOpFloating::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_TYPES(self.type(), "eq", [&]() {
      EqOp<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}

Tensor & ne_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  result.resize_(self.sizes());
  if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
    NeOpFloating::apply(result, self, other);
  }
  else {
    AT_DISPATCH_ALL_TYPES(self.type(), "ne", [&]() {
      NeOp<scalar_t>::apply(result, self, other);
    });
  }
  return result;
}
}} // namespace at::native
