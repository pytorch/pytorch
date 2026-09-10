#pragma once

#include <string>

namespace at {
class Tensor;
} // namespace at

namespace c10 {
struct IValue;
namespace detail {
// Determine the return type of `IValue::to() const &`. It's a const
// reference when possible and a copy otherwise. It is in this
// separate header so that List can use it as well.
template<typename T>
struct ivalue_to_const_ref_overload_return {
  using type = T;
};

template<>
struct ivalue_to_const_ref_overload_return<at::Tensor> {
  using type = const at::Tensor&;
};

template<>
struct ivalue_to_const_ref_overload_return<std::string> {
  using type = const std::string&;
};

template<>
struct ivalue_to_const_ref_overload_return<IValue> {
  using type = const IValue&;
};

} // namespace detail
} // namespace c10
