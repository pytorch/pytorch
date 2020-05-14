#pragma once

#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/core/DimVector.h>

// This file contains some helpers for writing meta (shape) functions (which
// are registered under the Meta dispatch key).  This allows you to write a
// meta function once, and then generate the functional, inplace and out
// variants from it (using meta_wrapper).  Example usage:
//
//  TensorMeta foo_meta(const Tensor& self, const Tensor& other) {
//    ...
//    return TensorMeta(output_size, self.options());
//  }
//
//  Tensor& foo_out(Tensor &result, const Tensor& self, const Tensor& vec2) {
//    return meta_wrapper_out<decltype(meta::ger_meta), meta::ger_meta>(result, self, vec2);
//  }
//
//  Tensor foo_(Tensor& self, const Tensor& vec2) {
//    return meta_wrapper_<decltype(meta::ger_meta), meta::ger_meta>(self, vec2);
//  }
//
//  Tensor foo(const Tensor& self, const Tensor& vec2) {
//    return meta_wrapper<decltype(meta::ger_meta), meta::ger_meta>(self, vec2);
//  }
//
// If you are an internal library developer, the boilerplate above can be
// generated automatically for you.

namespace at {

// Representation of all aspects of a tensor that a meta function infers
class TensorMeta {
  DimVector sizes_;
  TensorOptions options_;
public:
  TensorMeta(IntArrayRef sizes, TensorOptions options)
    : sizes_(sizes), options_(options) {}
  IntArrayRef sizes() const { return sizes_; }
  TensorOptions options() const { return options_; }
};

// TODO: Generalize to multiargument return

template <typename F, F* Func, typename... T>
std::enable_if_t<std::is_same<typename c10::guts::function_traits<F>::return_type, TensorMeta>::value, Tensor>
meta_wrapper(T&&... args) {
  auto meta = Func(std::forward<T>(args)...);
  return at::empty(meta.sizes(), meta.options());
}

template <typename F, F* Func, typename... T>
std::enable_if_t<std::is_same<typename c10::guts::function_traits<F>::return_type, TensorMeta>::value, Tensor&>
meta_wrapper_(Tensor& self, T&&... args) {
  auto meta = Func(self, std::forward<T>(args)...);
  TORCH_INTERNAL_ASSERT(self.options().type_equal(meta.options()));
  TORCH_CHECK(self.sizes() == meta.sizes(),
    "Cannot resize self in an in-place operation (was self broadcasted with the other arguments?)");
  return self;
}

template <typename F, F* Func, typename... T>
std::enable_if_t<std::is_same<typename c10::guts::function_traits<F>::return_type, TensorMeta>::value, Tensor&>
meta_wrapper_out(Tensor& result, T&&... args) {
  auto meta = Func(std::forward<T>(args)...);
  TORCH_CHECK(result.options().type_equal(meta.options()),
    "Passed in output tensor did not have the correct dtype/device.  Expected ", meta.options(),
    " but got ", result.options());
  result.resize_(meta.sizes());
  return result;
}

} // namespace at
