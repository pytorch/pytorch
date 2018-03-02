#include <THC/THCTensorTypeUtils.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

namespace std {
  template <>
  struct less_equal<half> {
    bool operator()(const half& lhs, const half& rhs) {
      return THCNumerics<half>::le(lhs, rhs);
    }
  };

  template <>
  struct greater_equal<half> {
    bool operator()(const half& lhs, const half& rhs) {
      return THCNumerics<half>::ge(lhs, rhs);
    }
  };

  template <>
  struct equal_to<half> {
    bool operator()(const half& lhs, const half& rhs) {
      return THCNumerics<half>::eq(lhs, rhs);
    }
  };

  template <>
  struct not_equal_to<half> {
    bool operator()(const half& lhs, const half& rhs) {
      return THCNumerics<half>::ne(lhs, rhs);
    }
  };
} // namespace std
