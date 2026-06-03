#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>

namespace at::native {

static at::Tensor& shallow_copy_data_(at::Tensor& self, const at::Tensor& source) {
  if (self.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    self.unsafeGetTensorImpl()->shallow_copy_from(source.getIntrusivePtr());
  }
  return self;
}

static at::Tensor& shallow_copy_data_functionalize(
    at::Tensor& self,
    const at::Tensor& src) {
  TORCH_CHECK(
      at::functionalization::impl::isFunctionalTensor(self) ||
          !at::functionalization::impl::isFunctionalTensor(src),
      "shallow_copy_data_: cannot mutate a non-functional tensor with a functional tensor");

  // Non-functional path: redispatch through the op schema so
  // ProxyTorchDispatch can record an FX node during make_fx tracing.
  if (!at::functionalization::impl::isFunctionalTensor(self) &&
      !at::functionalization::impl::isFunctionalTensor(src)) {
    at::AutoDispatchSkipFunctionalize guard;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("aten::shallow_copy_data_", "")
                         .typed<at::Tensor&(at::Tensor&, const at::Tensor&)>();
    return op.call(self, src);
  }

  TORCH_CHECK(
      at::functionalization::impl::isFunctionalTensor(src),
      "shallow_copy_data_: source must be a FunctionalTensor");

  auto self_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  auto src_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(src);
  TORCH_CHECK(
      !self_impl->was_inductor_storage_resized(),
      "storage_resize_() followed by shallow_copy_data_() is not supported");
  self_impl->set__impl(src_impl);
  return self;
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def("shallow_copy_data_(Tensor(a!) self, Tensor source) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("shallow_copy_data_", shallow_copy_data_);
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("shallow_copy_data_", shallow_copy_data_);
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("shallow_copy_data_", shallow_copy_data_);
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("shallow_copy_data_", shallow_copy_data_);
}

TORCH_LIBRARY_IMPL(aten, Functionalize, m) {
  m.impl("shallow_copy_data_", shallow_copy_data_functionalize);
}

} // namespace at::native
