#include <ATen/view/has_composite_view.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/impl/ExtraMeta.h>
#include <c10/util/Exception.h>

namespace at::view {

auto has_composite_view(TensorBase const& tensor) -> bool {
  bool has_key = tensor.key_set().has(c10::DispatchKey::CompositeView);
  c10::impl::ExtraMeta* extra_meta = tensor.unsafeGetTensorImpl()->extra_meta_.get();
  bool has_views = extra_meta != nullptr && !extra_meta->composite_views.empty();
  TORCH_INTERNAL_ASSERT(has_key == has_views);
  return has_key;
}

} // namespace at::view
