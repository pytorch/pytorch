#include <ATen/view/materialize.h>

#include <ATen/Tensor.h>
#include <c10/core/impl/ExtraMeta.h>

#include <cassert>

namespace at::view {

auto materialize(Tensor const& tensor) -> Tensor {
  c10::TensorImpl& tensor_impl = *tensor.unsafeGetTensorImpl();
  TORCH_CHECK(!tensor_impl.has_symbolic_sizes_strides_);
  TORCH_CHECK(tensor_impl.extra_meta_ != nullptr);
  TORCH_CHECK(!tensor_impl.extra_meta_->composite_views.empty());
  DimVector sizes(tensor_impl.sizes().begin(), tensor_impl.sizes().end());
  DimVector strides(tensor_impl.strides().begin(), tensor_impl.strides().end());
  std::int64_t storage_offset = tensor_impl.storage_offset();

  c10::impl::ExtraMeta::CompositeViews composite_views = std::move(tensor_impl.extra_meta_->composite_views);
  tensor_impl.set_sizes_and_strides(composite_views.back().physical.sizes_arrayref(),
                                    composite_views.back().physical.strides_arrayref(),
                                    composite_views.back().physical_storage_offset);
  tensor_impl.key_set_ = tensor_impl.key_set_.remove(DispatchKey::CompositeView);
  Tensor result = tensor.reshape(composite_views.back().virtual_sizes);
  c10::TensorImpl& result_tensor_impl = *result.unsafeGetTensorImpl();
  result_tensor_impl.set_sizes_and_strides(sizes, strides, storage_offset);

  // Restore tensor_impl to its original state.
  tensor_impl.set_sizes_and_strides(sizes, strides, storage_offset);
  tensor_impl.extra_meta().composite_views = std::move(composite_views);
  tensor_impl.key_set_ = tensor_impl.key_set().add(DispatchKey::CompositeView);
  return result;
}

} // namespace at::view
