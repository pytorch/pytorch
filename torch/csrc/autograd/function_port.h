#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace torch {
namespace autograd {

class Function;
class Variable;

/// Represents an input or "port" of a function.
struct FunctionPort {
  explicit FunctionPort(
      const std::shared_ptr<Function>& function_ = nullptr,
      uint32_t port_ = 0);

  /// Constructs a `FunctionPort` for the gradient function of a variable.
  static FunctionPort for_gradient(const Variable& variable);

  // See https://stackoverflow.com/questions/13414652/forward-declaration-with-unique-ptr
  // for why this destructor is needed.
  ~FunctionPort();

  // Required for use in associative containers.
  bool operator==(const FunctionPort& other) const noexcept;
  bool operator!=(const FunctionPort& other) const noexcept;

  /// The function this `FunctionPort` points to.
  std::shared_ptr<Function> function;

  /// The identifier of a particular input to the function.
  uint32_t port;
};
} // namespace autograd
} // namespace torch

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.
namespace std {
template <>
struct hash<torch::autograd::FunctionPort> {
  // These type aliases are required by the standard.
  using argument_type = torch::autograd::FunctionPort;
  using return_type = size_t;

  return_type operator()(const argument_type& function_port) const noexcept {
    const auto first = hash_value(function_port.function);
    const auto second = hash_value(function_port.port);
    // See http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html.
    return first ^ (second + 0x9e3779b9 + (first << 6) + (first >> 2));
  }

  // Helper function to call std::hash on a value of some type.
  template <typename T>
  return_type hash_value(const T& value) const noexcept {
    return hash<T>{}(value);
  }
};
} // namespace std
