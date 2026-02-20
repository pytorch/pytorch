#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <c10/util/irange.h>

namespace {

// Get the fake device from the first fake tensor input, or nullopt for factory ops.
static std::optional<c10::Device> get_common_device(
    torch::jit::Stack* stack,
    size_t num_arguments) {
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (t.defined() && t.is_fake()) return t.device();
    } else if (ivalue.isTensorList()) {
      for (const auto& elem : ivalue.toTensorList()) {
        at::Tensor t = elem;
        if (t.defined() && t.is_fake()) return t.device();
      }
    } else if (ivalue.isOptionalTensorList()) {
      for (const auto& elem : ivalue.toOptionalTensorList()) {
        std::optional<at::Tensor> ot = elem;
        if (ot.has_value() && ot->defined() && ot->is_fake())
          return ot->device();
      }
    }
  }
  return std::nullopt;
}

// For factory ops: find Device args in the stack, rewrite to meta, return original.
static std::optional<c10::Device> rewrite_device_args_to_meta(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments) {
  std::optional<c10::Device> original_device;
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isDevice()) {
      auto dev = ivalue.toDevice();
      TORCH_CHECK(dev.type() != c10::DeviceType::Meta,
          "FakeTensor does not support meta device inputs");
      if (!original_device.has_value()) original_device = dev;
      (*stack)[arguments_begin + idx] =
          c10::IValue(c10::Device(c10::DeviceType::Meta));
    }
  }
  return original_device;
}

static void transmute_to_fake(const at::Tensor& t, c10::Device fake_device) {
  t.unsafeGetTensorImpl()->set_fake_device(fake_device);
}

static void wrap_outputs(
    torch::jit::Stack* stack,
    size_t returns_begin,
    size_t num_returns,
    c10::Device fake_device) {
  auto returns = torch::jit::last(*stack, num_returns);
  for (size_t idx = 0; idx < num_returns; ++idx) {
    const auto& ivalue = returns[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (t.defined() && !t.is_fake()) {
        transmute_to_fake(t, fake_device);
      }
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      for (const auto i : c10::irange(tensors.size())) {
        at::Tensor t = tensors[i];
        if (t.defined() && !t.is_fake()) {
          transmute_to_fake(t, fake_device);
        }
      }
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      for (const auto i : c10::irange(opt_tensors.size())) {
        std::optional<at::Tensor> ot = opt_tensors[i];
        if (ot.has_value() && ot->defined() && !ot->is_fake()) {
          transmute_to_fake(*ot, fake_device);
        }
      }
    }
  }
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  // 1. Determine fake device from inputs
  auto fake_device = get_common_device(stack, num_arguments);

  // 2. For factory ops (no fake tensor inputs), rewrite device args to meta
  if (!fake_device.has_value()) {
    fake_device = rewrite_device_args_to_meta(
        stack, arguments_begin, num_arguments);
    if (!fake_device.has_value()) {
      fake_device = c10::Device(c10::DeviceType::CPU);
    }
  }

  // 3. Redispatch with Fake excluded
  {
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::Fake);
    op.callBoxed(stack);
  }

  // 4. Wrap outputs as fake tensors
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  wrap_outputs(stack, returns_begin, num_returns, *fake_device);
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
