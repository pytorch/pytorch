#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"


template <template<typename> class, typename, typename, template<template<typename> class, typename, typename> class>
struct CmpOpScalar {};

// Comparators have special cases for integral tensors and floating scalars due to
// the floating scalars being automatically cast to integral types during the comparison
template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::less, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        other = at::Scalar(ceil(other_double));
      }
    }

    CmpOpImpl<std::less, scalar_out, scalar>::apply(ret, self, other);
  }
};

template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::greater, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        other = at::Scalar(floor(other_double));
      }
    }

    CmpOpImpl<std::greater, scalar_out, scalar>::apply(ret, self, other);
  }
};

template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::less_equal, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        other = at::Scalar(floor(other_double));
      }
    }

    CmpOpImpl<std::less_equal, scalar_out, scalar>::apply(ret, self, other);
  }
};

template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::greater_equal, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        other = at::Scalar(ceil(other_double));
      }
    }

    CmpOpImpl<std::greater_equal, scalar_out, scalar>::apply(ret, self, other);
  }
};

template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::equal_to, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        ret.fill_(0);
      }
    } else {
      CmpOpImpl<std::equal_to, scalar_out, scalar>::apply(ret, self, other);
    }
  }
};

template<typename scalar_out, typename scalar, template<template<typename> class, typename, typename> class CmpOpImpl>
struct CmpOpScalar<std::not_equal_to, scalar_out, scalar, CmpOpImpl> {
  static void apply(at::Tensor& ret, const at::Tensor& self, at::Scalar other) {
    if (isIntegralType(self.type().scalarType()) && other.isFloatingPoint()) {
      auto other_double = other.to<double>();
      auto other_long = other.to<int64_t>();
      if (other_double != other_long) {
        ret.fill_(1);
      }
    } else {
      CmpOpImpl<std::not_equal_to, scalar_out, scalar>::apply(ret, self, other);
    }
  }
};
