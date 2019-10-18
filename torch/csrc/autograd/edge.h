#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <ATen/Tensor.h>
#include <torch/csrc/utils/hash.h>

// TODO: this is orphaned right now and should go to where the actual
// definition of Edge is

// The idiomatic way of enabling use of a custom type as the key of hash
// containers in C++11. This method removes the requirement of having to pass
// a custom hasher to std::unordered_{map, set}.
// See http://en.cppreference.com/w/cpp/utility/hash for more information.
namespace std {
template <>
struct hash<torch::autograd::Edge> {
  // These type aliases are required by the standard.
  using argument_type = torch::autograd::Edge;
  using return_type = size_t;
  return_type operator()(const argument_type& edge) const noexcept {
    return torch::get_hash(edge.function, edge.input_nr);
  }
};
} // namespace std
