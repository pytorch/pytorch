#include <ATen/FunctionalTensorImpl.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

namespace at {

FunctionalTensorImpl::FunctionalTensorImpl(Tensor value, int64_t level)
  : FunctionalTensorImplBase(value.dtype(), value.device()),
    value_(value),
    level_(level)
{
  TORCH_INTERNAL_ASSERT(value_.defined());
}

void FunctionalTensorImpl::replace_(const Tensor& other) {
    auto self_impl = value_.unsafeGetTensorImpl();
    auto other_impl = other.unsafeGetTensorImpl();
    if (typeid(*self_impl) == typeid(*other_impl)) {
        // It is valid to swap out the metadata on the tensorImpl
        // but we can only do that if the two tensor's we're swapping have the same type.
        // This allows us to ensure that programs that mutate their inputs
        // preserve their semantics under a functionalization pass.
        self_impl->replace_(other_impl);
    } else {
        value_ = other;
    }
}

void FunctionalTensorImpl::set_size(int64_t dim, int64_t new_size) {
    value_.unsafeGetTensorImpl()->set_size(dim, new_size);
}
void FunctionalTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
    value_.unsafeGetTensorImpl()->set_stride(dim, new_stride);
}
void FunctionalTensorImpl::set_storage_offset(int64_t storage_offset) {
    value_.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
}
bool FunctionalTensorImpl::has_storage() const {
    return value_.unsafeGetTensorImpl()->has_storage();
}
IntArrayRef FunctionalTensorImpl::sizes() const {
    return value_.unsafeGetTensorImpl()->sizes();
}
int64_t FunctionalTensorImpl::dim() const {
    return value_.unsafeGetTensorImpl()->dim();
}
const Storage& FunctionalTensorImpl::storage() const {
    return value_.unsafeGetTensorImpl()->storage();
}
int64_t FunctionalTensorImpl::numel() const {
    return value_.unsafeGetTensorImpl()->numel();
}
bool FunctionalTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
    return value_.unsafeGetTensorImpl()->is_contiguous(memory_format);
}
int64_t FunctionalTensorImpl::storage_offset() const {
    return value_.unsafeGetTensorImpl()->storage_offset();
}
int64_t FunctionalTensorImpl::size(int64_t d) const {
    return value_.unsafeGetTensorImpl()->size(d);
}
int64_t FunctionalTensorImpl::stride(int64_t d) const {
    return value_.unsafeGetTensorImpl()->stride(d);
}
c10::intrusive_ptr<TensorImpl> FunctionalTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
    // TODO: maybe just don't allow this
    return value_.unsafeGetTensorImpl()->shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
}
c10::intrusive_ptr<TensorImpl> FunctionalTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
    // TODO: maybe just don't allow this
    return value_.unsafeGetTensorImpl()->shallow_copy_and_detach(version_counter, allow_tensor_metadata_change);
}
void FunctionalTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
    // TODO: maybe just don't allow this
    value_.unsafeGetTensorImpl()->shallow_copy_from(impl);
}
const char* FunctionalTensorImpl::tensorimpl_type_name() const {
    return "FunctionalTensorImpl";
}

namespace functionalization {
namespace impl {

void maybe_add_update(Tensor& self) {
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(self.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  functional_impl->maybe_add_update(self);
}

void set_view_meta(at::Tensor& self, const at::Tensor& other, ViewMeta meta) {
    auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(self.unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
    functional_impl->set_view_meta(other, meta);
}

Tensor makeFunctional(const Tensor& tensor, int64_t level) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!dynamic_cast<FunctionalTensorImpl*>(tensor.unsafeGetTensorImpl()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!tensor.key_set().has(c10::DispatchKey::Functionalize));
  return at::detail::make_tensor<FunctionalTensorImpl>(tensor, level);
}

c10::optional<Tensor> makeFunctional(const c10::optional<Tensor>& tensor, int64_t level) {
  if (tensor.has_value()) {
    return makeFunctional(*tensor, level);
  }
  return c10::nullopt;
}

c10::List<Tensor> makeFunctional(const c10::List<Tensor>& t_list, int64_t level) {
  std::vector<at::Tensor> functional_tensors;
  for (auto& t: t_list.vec()) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return c10::List<at::Tensor>(functional_tensors);
}

std::vector<Tensor> makeFunctional(const at::TensorList t_list, int64_t level) {
  std::vector<at::Tensor> functional_tensors;
  for (auto& t: t_list) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return functional_tensors;
}

c10::List<c10::optional<Tensor>> makeFunctional(const c10::List<c10::optional<Tensor>>& t_list, int64_t level) {
  std::vector<c10::optional<at::Tensor>> functional_tensors;
  for (auto& t: t_list.vec()) {
	functional_tensors.push_back(makeFunctional(t, level));
  }
  return c10::List<c10::optional<at::Tensor>>(functional_tensors);
}

void maybe_sync(const Tensor& t) {
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(t.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  if (functional_impl->is_view() && !functional_impl->is_up_to_date()) {
    functional_impl->sync_();
  }
}
void maybe_sync(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    maybe_sync(*t);
  }
}
void maybe_sync(const c10::List<Tensor>& t_list) {
  for (auto& t: t_list.vec()) {
    maybe_sync(t);
  }
}
void maybe_sync(const at::TensorList t_list) {
  for (auto& t: t_list) {
    maybe_sync(t);
  }
}
void maybe_sync(const c10::List<c10::optional<Tensor>>& t_list) {
  for (auto& t: t_list.vec()) {
    maybe_sync(t);
  }
}

const Tensor& maybeUnwrapFunctional(const Tensor& tensor) {
  FunctionalTensorImpl* maybe_impl = dynamic_cast<FunctionalTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (maybe_impl == nullptr) {
    return tensor;
  }
  return maybe_impl->value();
}

const c10::optional<Tensor> maybeUnwrapFunctional(const c10::optional<Tensor>& tensor) {
  if (tensor.has_value()) {
    return maybeUnwrapFunctional(*tensor);
  }
  return c10::nullopt;
}
const c10::List<Tensor> maybeUnwrapFunctional(const c10::List<Tensor>& t_list) {
  std::vector<at::Tensor> unwrapped_tensors;
  for (auto& t: t_list.vec()) {
	unwrapped_tensors.push_back(maybeUnwrapFunctional(t));
  }
  return c10::List<at::Tensor>(unwrapped_tensors);
}
const std::vector<at::Tensor> maybeUnwrapFunctional(const at::TensorList t_list) {
  std::vector<at::Tensor> unwrapped_tensors;
  for (auto& t: t_list) {
	unwrapped_tensors.push_back(maybeUnwrapFunctional(t));
  }
  return unwrapped_tensors;
}
const c10::List<c10::optional<Tensor>> maybeUnwrapFunctional(const c10::List<c10::optional<Tensor>>& t_list) {
  std::vector<c10::optional<at::Tensor>> unwrapped_tensors;
  for (auto& t: t_list.vec()) {
	unwrapped_tensors.push_back(maybeUnwrapFunctional(t));
  }
  return c10::List<c10::optional<at::Tensor>>(unwrapped_tensors);
}

} // namespace impl
} // namespace functionalization
} // namespace at
