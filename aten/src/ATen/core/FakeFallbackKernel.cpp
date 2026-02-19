#include <ATen/EmptyTensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at::impl {

TORCH_API at::Tensor to_fake(const at::Tensor& tensor) {
  auto* allocator = c10::GetAllocator(c10::DeviceType::Meta);
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  auto fake = at::detail::empty_strided_generic(
      tensor.sizes(),
      tensor.strides(),
      allocator,
      ks,
      tensor.scalar_type());
  fake.unsafeGetTensorImpl()->_change_backend_component_keys(tensor.device());
  fake.unsafeGetTensorImpl()->_set_fake(true, tensor.device());
  return fake;
}

namespace {

// Converts a fake tensor to meta in-place (removes Fake key, swaps
// backend to Meta, sets device to meta).
void fake_to_meta_inplace(at::Tensor& tensor) {
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->_set_fake(false, c10::Device(c10::DeviceType::Meta));
  impl->_change_backend_component_keys(c10::Device(c10::DeviceType::Meta));
}

// Converts a meta tensor to a fake tensor in-place (adds Fake key,
// swaps backend to the target device, sets device).
void meta_to_fake_inplace(at::Tensor& tensor, c10::Device device) {
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->_change_backend_component_keys(device);
  impl->_set_fake(true, device);
}

void fake_fallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();

  // Single pass: validate all tensor operands are fake, record the
  // device, convert fake->meta in-place, and remember which TensorImpls
  // we converted so we can restore them after the call.
  c10::Device device(c10::DeviceType::CPU);
  bool found_device = false;
  c10::SmallVector<c10::TensorImpl*, 8> converted_impls;

  auto args = torch::jit::last(*stack, num_arguments);
  for (const auto i : c10::irange(num_arguments)) {
    if (args[i].isTensor()) {
      auto& t = const_cast<at::Tensor&>(args[i].toTensor());
      if (t.defined()) {
        TORCH_CHECK(
            t.key_set().has(c10::DispatchKey::Fake),
            "FakeTensorMode: all tensor operands must be fake tensors, but "
            "argument ", i, " is not a fake tensor");
        if (!found_device) {
          device = t.device();
          found_device = true;
        }
        converted_impls.push_back(t.unsafeGetTensorImpl());
        fake_to_meta_inplace(t);
      }
    } else if (args[i].isTensorList()) {
      auto tl = args[i].toTensorList();
      for (const auto j : c10::irange(tl.size())) {
        at::Tensor t = tl.get(j);
        if (t.defined()) {
          TORCH_CHECK(
              t.key_set().has(c10::DispatchKey::Fake),
              "FakeTensorMode: all tensor operands must be fake tensors, but "
              "a tensor in a TensorList argument is not a fake tensor");
          if (!found_device) {
            device = t.device();
            found_device = true;
          }
          converted_impls.push_back(t.unsafeGetTensorImpl());
          fake_to_meta_inplace(t);
          tl.set(j, std::move(t));
        }
      }
    }
  }

  // Exclude Fake to prevent re-entry, then dispatch to Meta kernels.
  {
    c10::impl::ExcludeDispatchKeyGuard exclude_guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake));
    op.callBoxed(stack);
  }

  // Restore input TensorImpls from meta back to fake in-place.
  // The caller still holds references to these TensorImpls so they
  // remain alive even after callBoxed pops them from the stack.
  for (auto* impl : converted_impls) {
    impl->_change_backend_component_keys(device);
    impl->_set_fake(true, device);
  }

  // Convert meta tensor outputs back to fake tensors in-place.
  auto rets = torch::jit::last(*stack, num_returns);
  for (const auto i : c10::irange(num_returns)) {
    if (rets[i].isTensor()) {
      auto& t = const_cast<at::Tensor&>(rets[i].toTensor());
      if (t.defined() && t.device().type() == c10::DeviceType::Meta) {
        meta_to_fake_inplace(t, device);
      }
    }
  }
}

// Allocates a fake tensor with the given sizes, contiguous strides,
// scalar type, and device.  Uses the meta allocator so no real memory
// is allocated.
at::Tensor make_fake(
    c10::IntArrayRef sizes,
    c10::ScalarType dtype,
    c10::Device device) {
  auto* allocator = c10::GetAllocator(c10::DeviceType::Meta);
  c10::DispatchKeySet ks(c10::DispatchKey::Dense);
  auto ndim = sizes.size();
  std::vector<int64_t> strides(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = 1;
    for (int64_t i = static_cast<int64_t>(ndim) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }
  auto result = at::detail::empty_strided_generic(
      sizes, strides, allocator, ks, dtype);
  auto* impl = result.unsafeGetTensorImpl();
  impl->_change_backend_component_keys(device);
  impl->_set_fake(true, device);
  return at::Tensor(std::move(result));
}

at::Tensor fake_mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be 2D, got ", self.dim(), "D");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be 2D, got ", mat2.dim(), "D");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "mm: self.size(1) must match mat2.size(0), got ",
      self.size(1), " and ", mat2.size(0));
  return make_fake(
      {self.size(0), mat2.size(1)}, self.scalar_type(), self.device());
}

// Meta kernel for mm — computes output shape only, no real computation.
// This lets us measure the boxed fallback overhead in isolation since
// the fallback will dispatch to this instead of the default Meta kernel.
at::Tensor meta_mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2, "mm: self must be 2D, got ", self.dim(), "D");
  TORCH_CHECK(mat2.dim() == 2, "mm: mat2 must be 2D, got ", mat2.dim(), "D");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "mm: self.size(1) must match mat2.size(0), got ",
      self.size(1), " and ", mat2.size(0));
  auto* allocator = c10::GetAllocator(c10::DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  std::array<int64_t, 2> sizes = {self.size(0), mat2.size(1)};
  std::array<int64_t, 2> strides = {mat2.size(1), 1};
  return at::Tensor(at::detail::empty_strided_generic(
      sizes, strides, allocator, meta_dks, self.scalar_type()));
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fake_fallback>());
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("mm", meta_mm);
}

} // anonymous namespace
} // namespace at::impl
