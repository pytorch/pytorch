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

at::Tensor to_meta(const at::Tensor& tensor) {
  auto* allocator = c10::GetAllocator(c10::DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_generic(
      tensor.sizes(),
      tensor.strides(),
      allocator,
      meta_dks,
      tensor.scalar_type());
}

// Converts a meta tensor to a fake tensor in-place by mutating its
// TensorImpl (backend keys, Fake key, and device) rather than
// allocating a new tensor.
void meta_to_fake_inplace(at::Tensor& meta_tensor, c10::Device device) {
  auto* impl = meta_tensor.unsafeGetTensorImpl();
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

  // Determine the device from input tensors, and validate that all
  // tensor operands are fake.
  c10::Device device(c10::DeviceType::CPU);
  bool found_device = false;
  auto args = torch::jit::last(*stack, num_arguments);
  for (const auto i : c10::irange(num_arguments)) {
    if (args[i].isTensor() && args[i].toTensor().defined()) {
      TORCH_CHECK(
          args[i].toTensor().key_set().has(c10::DispatchKey::Fake),
          "FakeTensorMode: all tensor operands must be fake tensors, but "
          "argument ",
          i,
          " is not a fake tensor");
      if (!found_device) {
        device = args[i].toTensor().device();
        found_device = true;
      }
    } else if (args[i].isTensorList()) {
      auto tl = args[i].toTensorList();
      for (const auto j : c10::irange(tl.size())) {
        const at::Tensor& t = tl.get(j);
        if (t.defined()) {
          TORCH_CHECK(
              t.key_set().has(c10::DispatchKey::Fake),
              "FakeTensorMode: all tensor operands must be fake tensors, but "
              "a tensor in a TensorList argument is not a fake tensor");
          if (!found_device) {
            device = t.device();
            found_device = true;
          }
        }
      }
    }
  }

  // Replace fake tensor inputs with meta tensors.
  auto popped = torch::jit::pop(*stack, num_arguments);
  for (const auto i : c10::irange(num_arguments)) {
    if (popped[i].isTensor()) {
      auto& t = popped[i].toTensor();
      if (t.defined() && t.key_set().has(c10::DispatchKey::Fake)) {
        torch::jit::push(*stack, to_meta(t));
      } else {
        torch::jit::push(*stack, std::move(popped[i]));
      }
    } else if (popped[i].isTensorList()) {
      auto tl = popped[i].toTensorList();
      for (const auto j : c10::irange(tl.size())) {
        if (tl.get(j).defined() &&
            tl.get(j).key_set().has(c10::DispatchKey::Fake)) {
          tl.set(j, to_meta(tl.get(j)));
        }
      }
      torch::jit::push(*stack, std::move(tl));
    } else {
      torch::jit::push(*stack, std::move(popped[i]));
    }
  }

  // Exclude Fake to prevent re-entry, then dispatch to Meta kernels.
  c10::impl::ExcludeDispatchKeyGuard exclude_guard(
      c10::DispatchKeySet(c10::DispatchKey::Fake));
  op.callBoxed(stack);

  // Convert meta tensor outputs back to fake tensors in-place.
  auto rets = torch::jit::pop(*stack, num_returns);
  for (const auto i : c10::irange(num_returns)) {
    if (rets[i].isTensor()) {
      auto t = std::move(rets[i]).toTensor();
      if (t.defined() && t.device().type() == c10::DeviceType::Meta) {
        meta_to_fake_inplace(t, device);
        torch::jit::push(*stack, std::move(t));
      } else {
        torch::jit::push(*stack, std::move(t));
      }
    } else {
      torch::jit::push(*stack, std::move(rets[i]));
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
  // Compute contiguous strides.
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
  return make_fake({self.size(0), mat2.size(1)}, self.scalar_type(), self.device());
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fake_fallback>());
}

TORCH_LIBRARY_IMPL(aten, Fake, m) {
  m.impl("mm", fake_mm);
}

} // anonymous namespace
} // namespace at::impl
