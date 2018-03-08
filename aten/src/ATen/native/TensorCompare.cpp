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
struct CmpOpTensor {
  static void apply(at::Tensor& ret, const at::Tensor& self, const at::Tensor& other) {
    at::CPU_tensor_apply3<uint8_t, scalar, scalar>(ret, self, other,
        [](uint8_t& ret_val, const scalar& self_val, const scalar& other_val) {
          ret_val = Comparator<scalar>()(self_val, other_val);
        }
    );
  }
};

template<template<typename T> class Comparator, typename scalar>
struct CmpOp {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();
    at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
        [other_val](uint8_t& ret_val, const scalar& self_val) {
          ret_val = Comparator<scalar>()(self_val, other_val);
      }
    );
  }
};

// <=, >=, ==, != have special cases for integral tensors and floating scalars due to
// the floating scalars being automatically cast to integral types during the comparison
template<typename scalar>
struct CmpOp<std::less_equal, scalar> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();

    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<long>();
      if (other_double != other_long) {
        other_val = at::convert<scalar>(floor(other_double));
      }
    }

    at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
        [other_val](uint8_t& ret_val, const scalar& self_val) {
          ret_val = std::less_equal<scalar>()(self_val, other_val);
      }
    );
  }
};

template<typename scalar>
struct CmpOp<std::greater_equal, scalar> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();

    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<long>();
      if (other_double != other_long) {
        other_val = at::convert<scalar>(ceil(other_double));
      }
    }

    at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
        [other_val](uint8_t& ret_val, const scalar& self_val) {
          ret_val = std::greater_equal<scalar>()(self_val, other_val);
      }
    );
  }
};

template<typename scalar>
struct CmpOp<std::equal_to, scalar> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();

    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<long>();
      if (other_double != other_long) {
        ret.fill_(0);
      }
    } else {
      at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
          [other_val](uint8_t& ret_val, const scalar& self_val) {
            ret_val = std::equal_to<scalar>()(self_val, other_val);
        }
      );
    }
  }
};

template<typename scalar>
struct CmpOp<std::not_equal_to, scalar> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    auto other_val = other.to<scalar>();

    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<long>();
      if (other_double != other_long) {
        ret.fill_(1);
      }
    } else {
      at::CPU_tensor_apply2<uint8_t, scalar>(ret, self,
          [other_val](uint8_t& ret_val, const scalar& self_val) {
            ret_val = std::not_equal_to<scalar>()(self_val, other_val);
        }
      );
    }
  }
};

template<template<typename T> class Comparator>
at::Tensor& cmp_out_cpu(at::Tensor& result, const at::Tensor& self, at::Scalar other, const char* op_name) {
  result.resize_(self.sizes());
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOp<Comparator, scalar_t>::apply(result, self, other);
  });
  return result;
}

template<template<typename T> class Comparator>
at::Tensor& cmp_out_cpu(at::Tensor& result, const at::Tensor& self, const at::Tensor& other, const char* op_name) {
  if (other.dim() == 0) {
    return cmp_out_cpu<Comparator>(result, self, other.pImpl->localScalar(), op_name);
  }

  at::Tensor b_self, b_other;
  std::tie(b_self, b_other) = at::expand_outplace(self, other, op_name);
  result.resize_(b_self.sizes());
  AT_DISPATCH_ALL_TYPES(self.type(), op_name, [&]() {
    CmpOpTensor<Comparator, scalar_t>::apply(result, b_self, b_other);
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

Tensor lt_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::less>(self, other, "lt");
}

Tensor gt_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::greater>(self, other, "gt");
}

Tensor le_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::less_equal>(self, other, "le");
}

Tensor ge_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::greater_equal>(self, other, "le");
}

Tensor eq_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::equal_to>(self, other, "eq");
}

Tensor ne_cpu(const Tensor& self, Scalar other) {
  return cmp_cpu<std::not_equal_to>(self, other, "ne");
}

Tensor& lt_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::less>(result, self, other, "lt");
}

Tensor& gt_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::greater>(result, self, other, "gt");
}

Tensor& le_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::less_equal>(result, self, other, "le");
}

Tensor& ge_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::greater_equal>(result, self, other, "ge");
}

Tensor& eq_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::equal_to>(result, self, other, "eq");
}

Tensor& ne_out_cpu(Tensor& result, const Tensor& self, Scalar other) {
  return cmp_out_cpu<std::not_equal_to>(result, self, other, "ne");
}

Tensor lt_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::less>(self, other, "lt");
}

Tensor gt_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::greater>(self, other, "gt");
}

Tensor le_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::less_equal>(self, other, "le");
}

Tensor ge_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::greater_equal>(self, other, "le");
}

Tensor eq_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::equal_to>(self, other, "eq");
}

Tensor ne_cpu(const Tensor& self, const Tensor& other) {
  return cmp_cpu<std::not_equal_to>(self, other, "ne");
}

Tensor& lt_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::less>(result, self, other, "lt");
}

Tensor& gt_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::greater>(result, self, other, "gt");
}

Tensor& le_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::less_equal>(result, self, other, "le");
}

Tensor& ge_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::greater_equal>(result, self, other, "ge");
}

Tensor& eq_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::equal_to>(result, self, other, "eq");
}

Tensor& ne_out_cpu(Tensor& result, const Tensor& self, const Tensor& other) {
  return cmp_out_cpu<std::not_equal_to>(result, self, other, "ne");
}
}} // namespace at::native
