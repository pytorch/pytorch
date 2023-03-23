#include <ATen/core/Tensor.h>
#include <ATen/native/MathBitFallThroughLists.h>
#include <ATen/view/TransformFallback.h>
#include <ATen/view/copy_into_view.h>
#include <ATen/view/materialize.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>

namespace at::view {
namespace {

class CompositeViewFallback final : public TransformFallback {
 public:
  CompositeViewFallback() : TransformFallback(c10::DispatchKey::CompositeView) {}
  ~CompositeViewFallback() final = default;

 private:
  auto transform(Tensor const& tensor) const -> Tensor final;
  auto untransform(Tensor& output, Tensor const& result) const -> void final;
};

auto composite_view_fallback(c10::OperatorHandle const& op, c10::DispatchKeySet dispatch_keys, c10::Stack* stack) -> void {
  CompositeViewFallback()(op, dispatch_keys, stack);
}

auto CompositeViewFallback::transform(Tensor const& tensor) const -> Tensor {
  TORCH_INTERNAL_ASSERT(view::has_composite_view(tensor));
  return at::view::materialize(tensor);
}

auto CompositeViewFallback::untransform(Tensor& output, Tensor const& result) const -> void {
  TORCH_INTERNAL_ASSERT(view::has_composite_view(output));
  at::view::copy_into_view(output, result);
}

TORCH_LIBRARY_IMPL(_, CompositeView, m) {
  TransformFallback::register_fallback<composite_view_fallback>(m);
}

TORCH_LIBRARY_IMPL(aten, CompositeView, m) {
  for (auto func : {
      "clone",
      "copy_",
    }) {
    m.impl(func, torch::CppFunction::makeFallthrough());
  }

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)
}

} // namespace
} // namespace at::native
