#include <torch/library.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/native/Resize.h>

#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#endif

namespace torch {
namespace inductor {
using namespace at;

static void resize_storage_bytes_(const Tensor& variable, SymInt new_size) {
  // similar to THPStorage_resize_ in StorageMethods.cpp, but is traceable
  if (variable.storage().device_type() == at::kCUDA) {
    // rocm build has undefined reference to resize_bytes_cuda
#if defined(USE_CUDA) && !defined(USE_ROCM)
    at::native::resize_bytes_cuda(
        variable.storage().unsafeGetStorageImpl(), new_size.expect_int());
#else
    TORCH_CHECK(false, "built without cuda");
#endif
  } else {
    at::native::resize_bytes_nocuda(variable.storage(), new_size);
  }
}

static void resize_storage_bytes__functionalize(
    const Tensor& variable,
    SymInt new_size) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("inductor::resize_storage_bytes_", "")
                       .typed<void(const Tensor&, SymInt)>();
  if (!at::functionalization::impl::isFunctionalTensor(variable)) {
    // Functionalization not active: nop
    at::AutoDispatchSkipFunctionalize guard;
    op.call(variable, new_size);
    return;
  }
  // Don't functionalize, call the mutable op on the inner tensor.
  auto functional_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(variable);
  {
    at::AutoDispatchSkipFunctionalize guard;
    op.call(functional_impl->value(), new_size);
    return;
  }
}

TORCH_LIBRARY_FRAGMENT(inductor, m) {
  m.def(
      "resize_storage_bytes_(Tensor variable, SymInt new_size) -> ()",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, resize_storage_bytes_),
      {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(inductor, Functionalize, m) {
  m.impl(
      "resize_storage_bytes_", TORCH_FN(resize_storage_bytes__functionalize));
}

} // namespace inductor
} // namespace torch
