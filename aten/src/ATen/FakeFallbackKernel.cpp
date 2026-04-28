#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace {

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

// Takes a real tensor and creates a corresponding fake (meta) tensor
// stamped with the original device.
static at::Tensor real_tensor_to_fake(
    const at::Tensor& t,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  auto original_device = t.device();
  at::Tensor meta_t;
  {
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake));
    meta_t = at::empty_strided(
        t.sizes(), t.strides(), t.options().device(c10::DeviceType::Meta));
  }
  if (t.requires_grad()) {
    meta_t.set_requires_grad(true);
  }
  transmute_to_fake(meta_t, original_device, mode);
  return meta_t;
}

static bool can_run_unsafe_fallback(const c10::FunctionSchema& schema) {
  const auto& name = schema.name();
  // Match Python FakeTensorMode._can_run_unsafe_fallback_allowed_namespaces
  return name.rfind("aten::", 0) == 0 || name.rfind("prims::", 0) == 0 ||
      name.rfind("quantized::", 0) == 0;
}

// Creates a zero-filled real tensor on the fake tensor's original device.
// Temporarily exits FakeTensorMode TLS so the created tensor is genuinely
// real (not stamped with the Fake dispatch key).
static at::Tensor to_real_tensor(const at::Tensor& t) {
  auto device = t.device(); // returns fake device (e.g. CPU)
  auto saved_mode = c10::impl::FakeTensorModeTLS::get_state();
  c10::impl::FakeTensorModeTLS::reset_state();
  auto real = at::empty_strided(
                  t.sizes(),
                  t.strides(),
                  t.options().device(device))
                  .zero_();
  c10::impl::FakeTensorModeTLS::set_state(saved_mode);
  return real;
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  auto mode = c10::impl::FakeTensorModeTLS::get_state();

  // Convert real (non-fake) tensor inputs to fake tensors.
  for_each_tensor(
      stack,
      arguments_begin,
      num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && !t.is_fake())
          return real_tensor_to_fake(t, mode);
        return std::nullopt;
      });

  auto fake_device = get_common_device(stack, num_arguments);

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

  // Try the Meta kernel. If it raises NotImplementedError (no working meta
  // implementation), fall back to running the real kernel with zero-filled
  // inputs to discover output metadata. We must save the arguments first
  // because redispatchBoxed consumes them from the stack.
  torch::jit::Stack saved_args;
  if (can_run_unsafe_fallback(schema)) {
    auto arguments = torch::jit::last(*stack, num_arguments);
    saved_args.insert(saved_args.end(), arguments.begin(), arguments.end());
  }

  try {
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake) |
        c10::DispatchKeySet(c10::DispatchKey::Python) |
        c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
    c10::impl::IncludeDispatchKeyGuard meta_guard(c10::DispatchKey::Meta);
    op.callBoxed(stack);

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
  } catch (c10::NotImplementedError&) {
    // Meta kernel failed — restore the stack and run the real kernel
    // with zero-filled inputs to discover output metadata.
    stack->resize(arguments_begin);
    for (auto& arg : saved_args) {
      stack->push_back(std::move(arg));
    }

    // Convert fake inputs to zero-filled real tensors.
    for_each_tensor(
        stack,
        arguments_begin,
        num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && t.is_fake())
            return to_real_tensor(t);
          return std::nullopt;
        });

    // Run the real kernel.
    {
      c10::impl::ExcludeDispatchKeyGuard guard(
          c10::DispatchKeySet(c10::DispatchKey::Fake) |
          c10::DispatchKeySet(c10::DispatchKey::Python) |
          c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
      op.callBoxed(stack);
    }

    // Convert real outputs to fake tensors.
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    for_each_tensor(
        stack,
        returns_begin,
        num_returns,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && !t.is_fake())
            return real_tensor_to_fake(t, mode);
          return std::nullopt;
        });
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
