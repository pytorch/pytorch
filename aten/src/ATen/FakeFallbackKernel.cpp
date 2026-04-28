#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace {

// Iterate over tensor IValues in a stack range and apply a transform.
// The callback receives a tensor and returns an optional replacement.
// If the callback returns a value, the tensor is replaced on the stack.
// If it returns nullopt, the tensor is left unchanged (useful for in-place
// mutations like transmute_to_fake).
template <typename Fn>
static void for_each_tensor(
    torch::jit::Stack* stack,
    size_t begin,
    size_t count,
    const Fn& fn) {
  for (const auto idx : c10::irange(count)) {
    auto& ivalue = (*stack)[begin + idx];
    if (ivalue.isTensor()) {
      auto result = fn(ivalue.toTensor());
      if (result.has_value()) {
        (*stack)[begin + idx] = std::move(*result);
      }
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      for (const auto i : c10::irange(tensors.size())) {
        auto result = fn(tensors[i]);
        if (result.has_value()) {
          tensors[i] = std::move(*result);
        }
      }
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      for (const auto i : c10::irange(opt_tensors.size())) {
        std::optional<at::Tensor> ot = opt_tensors[i];
        if (ot.has_value()) {
          auto result = fn(*ot);
          if (result.has_value()) {
            opt_tensors[i] = std::move(*result);
          }
        }
      }
    }
  }
}

static std::optional<c10::Device> get_common_device(
    torch::jit::Stack* stack,
    size_t num_arguments) {
  std::optional<c10::Device> common_device;
  bool is_cpu_zero_dim = false;

  auto merge = [&](const at::Tensor& t) {
    if (!t.defined() || !t.is_fake())
      return;
    bool t_is_cpu_zero_dim = t.device().is_cpu() && t.dim() == 0;
    if (!common_device.has_value()) {
      common_device = t.device();
      is_cpu_zero_dim = t_is_cpu_zero_dim;
      return;
    }
    if (t.device() == *common_device) {
      if (is_cpu_zero_dim)
        is_cpu_zero_dim = t_is_cpu_zero_dim;
      return;
    }
    if (t_is_cpu_zero_dim)
      return;
    TORCH_CHECK(
        is_cpu_zero_dim,
        "Unhandled FakeTensor device propagation: ",
        *common_device,
        " vs ",
        t.device());
    common_device = t.device();
    is_cpu_zero_dim = false;
  };

  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      merge(ivalue.toTensor());
    } else if (ivalue.isTensorList()) {
      for (const auto& elem : ivalue.toTensorList())
        merge(elem);
    } else if (ivalue.isOptionalTensorList()) {
      for (const auto& elem : ivalue.toOptionalTensorList()) {
        std::optional<at::Tensor> ot = elem;
        if (ot.has_value())
          merge(*ot);
      }
    }
  }
  return common_device;
}

static std::shared_ptr<c10::FakeTensorMode> get_fake_tensor_mode(
    torch::jit::Stack* stack,
    size_t num_arguments) {
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (t.defined() && t.is_fake()) {
        auto mode = t.unsafeGetTensorImpl()->fake_tensor_mode();
        if (mode)
          return mode;
      }
    }
  }
  return nullptr;
}

static bool is_device_type_arg(const c10::Argument& arg) {
  const auto& type = arg.type();
  if (type->kind() == c10::TypeKind::DeviceObjType)
    return true;
  if (type->kind() == c10::TypeKind::OptionalType) {
    auto elem = type->castRaw<c10::OptionalType>()->getElementType();
    return elem->kind() == c10::TypeKind::DeviceObjType;
  }
  return false;
}

static std::optional<c10::Device> rewrite_device_args_to_meta(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const c10::FunctionSchema& schema) {
  std::optional<c10::Device> original_device;
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isDevice()) {
      auto dev = ivalue.toDevice();
      TORCH_CHECK(
          dev.type() != c10::DeviceType::Meta,
          "FakeTensor does not support meta device inputs");
      if (!original_device.has_value())
        original_device = dev;
      (*stack)[arguments_begin + idx] =
          c10::IValue(c10::Device(c10::DeviceType::Meta));
    } else if (ivalue.isNone() && is_device_type_arg(schema.arguments()[idx])) {
      if (!original_device.has_value()) {
        original_device = c10::Device(c10::DeviceType::CPU);
      }
      (*stack)[arguments_begin + idx] =
          c10::IValue(c10::Device(c10::DeviceType::Meta));
    }
  }
  return original_device;
}

static void transmute_to_fake(
    const at::Tensor& t,
    c10::Device fake_device,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  t.unsafeGetTensorImpl()->set_and_normalize_fake_device(fake_device);
  if (mode) {
    t.unsafeGetTensorImpl()->set_fake_tensor_mode(mode);
  }
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  auto fake_device = get_common_device(stack, num_arguments);
  auto mode = get_fake_tensor_mode(stack, num_arguments);

  // Always rewrite device kwargs to meta so composite kernels create meta
  // tensors internally (e.g. rand_like(x, device='cpu') must not create real
  // CPU tensors inside the CompositeExplicitAutograd kernel).
  auto device_from_args = rewrite_device_args_to_meta(
      stack, arguments_begin, num_arguments, schema);

  if (!fake_device.has_value()) {
    fake_device = device_from_args;
    if (!fake_device.has_value()) {
      fake_device = c10::Device(c10::DeviceType::CPU);
    }
  }

  {
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake) |
        c10::DispatchKeySet(c10::DispatchKey::Python) |
        c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
    c10::impl::IncludeDispatchKeyGuard meta_guard(c10::DispatchKey::Meta);
    // auto ks = dispatchKeySet.remove(c10::DispatchKey::Fake) |
    //     c10::DispatchKeySet(c10::DispatchKey::Meta);
    // op.redispatchBoxed(ks, stack);
    op.callBoxed(ks, stack);
  }

  // Stamp meta tensor outputs with the fake device.
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for_each_tensor(
      stack,
      returns_begin,
      num_returns,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && (!t.is_fake() || t.device().is_meta()))
          transmute_to_fake(t, *fake_device, mode);
        return std::nullopt;
      });
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
