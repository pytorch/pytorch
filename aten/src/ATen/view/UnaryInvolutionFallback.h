#pragma once

#include <ATen/core/stack.h>
#include <ATen/view/TransformFallback.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

namespace at { class Tensor; }
namespace c10 { class OperatorHandle; }
namespace torch { class Library; }

namespace at::view {

// Register a dispatch for a "simple" dispatch based transformation
// with the following requirements:
//  * is an involution (i.e. its own inverse)
//  * is unary
//  * requires no state other than whether or not the transformation
//    is applied to the tensor
//  * the transformation is implemented within copy_
template <c10::DispatchKey key>
auto register_unary_involution_fallback(torch::Library& library);

namespace detail {

class UnaryInvolutionFallback final : public TransformFallback {
 public:
  explicit UnaryInvolutionFallback(c10::DispatchKey key) : TransformFallback(key) {}
  ~UnaryInvolutionFallback() final;

 private:
  auto transform(Tensor const& tensor) const -> Tensor final;
  auto untransform(Tensor& output, Tensor const& result) const -> void final;
};

template <c10::DispatchKey key>
auto unary_involution_fallback_trampoline(c10::OperatorHandle const& op, c10::DispatchKeySet dispatch_keys,
                                          c10::Stack* stack) -> void {
  UnaryInvolutionFallback{key}(op, dispatch_keys, stack);
}

} // namespace detail

template <c10::DispatchKey key>
auto register_unary_involution_fallback(torch::Library& library) {
  TransformFallback::register_fallback<&detail::unary_involution_fallback_trampoline<key>>(library);
}

} // namespace at::view
