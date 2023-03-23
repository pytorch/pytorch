#include <ATen/view/copy_into_view.h>

#include <ATen/Tensor.h>
#include <c10/core/impl/ExtraMeta.h>

#include <cassert>

namespace at::view {

auto copy_into_view(Tensor& view_tensor, Tensor const& source) -> void {
  TensorImpl const& view_tensor_impl = *view_tensor.unsafeGetTensorImpl();
  assert(view_tensor_impl.extra_meta_ != nullptr);
  c10::impl::ExtraMeta const& extra_meta = *view_tensor_impl.extra_meta_;
  assert(!extra_meta.composite_views.empty());

  // Reverse the materialize operations. First apply the strides, then
  // reshape it back to the input shape.
  Tensor reshaped = source.as_strided(extra_meta.composite_views.back().physical.sizes_arrayref(),
                                      extra_meta.composite_views.back().physical.strides_arrayref(),
                                      0).reshape(view_tensor.sizes());
  view_tensor.copy_(reshaped);
}

} // namespace at::view
