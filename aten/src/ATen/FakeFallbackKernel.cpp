#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <string>
#include <unordered_set>

namespace {

static bool cpp_meta_supports_symint(const c10::OperatorHandle& op) {
  static const std::unordered_set<std::string> allowlist = {
      "aten::empty.memory_format",
      "aten::empty_strided",
      "aten::as_strided",
      "aten::as_strided_",
      "aten::zeros",
      "aten::detach",
      "aten::view_as_real",
      "aten::view_as_complex",
      "aten::set_.source_Storage_storage_offset",
      "aten::_sparse_coo_tensor_with_dims_and_tensors",
  };
  const auto& name = op.operator_name();
  auto full_name = name.name;
  if (!name.overload_name.empty()) {
    full_name += "." + name.overload_name;
  }
  if (allowlist.count(full_name)) {
    return true;
  }
  // view_copy ops also support SymInt
  return full_name.find("view_copy") != std::string::npos;
}

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

static bool has_symbolic_sizes(
    torch::jit::Stack* stack,
    size_t begin,
    size_t num_arguments) {
  bool found = false;
  for_each_tensor(
      stack, begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() &&
            t.unsafeGetTensorImpl()->has_symbolic_sizes_strides())
          found = true;
        return std::nullopt;
      });
  if (found)
    return true;
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isSymInt() || ivalue.isSymFloat() || ivalue.isSymIntList())
      return true;
  }
  return false;
}

static std::optional<c10::Device> _find_common_device(
    torch::jit::Stack* stack,
    size_t begin,
    size_t num_arguments) {
  std::optional<c10::Device> common_device;
  bool is_cpu_zero_dim = false;

  for_each_tensor(
      stack, begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (!t.defined() || !t.is_fake())
          return std::nullopt;
        bool t_is_cpu_zero_dim = t.device().is_cpu() && t.dim() == 0;
        if (!common_device.has_value()) {
          common_device = t.device();
          is_cpu_zero_dim = t_is_cpu_zero_dim;
          return std::nullopt;
        }
        if (t.device() == *common_device) {
          if (is_cpu_zero_dim)
            is_cpu_zero_dim = t_is_cpu_zero_dim;
          return std::nullopt;
        }
        if (t_is_cpu_zero_dim)
          return std::nullopt;
        TORCH_CHECK(
            is_cpu_zero_dim,
            "Unhandled FakeTensor device propagation: ",
            *common_device,
            " vs ",
            t.device());
        common_device = t.device();
        is_cpu_zero_dim = false;
        return std::nullopt;
      });
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

static bool is_our_fake(const at::Tensor& t, const std::shared_ptr<c10::FakeTensorMode>& mode) {
  return t.defined() && t.is_fake() &&
          t.unsafeGetTensorImpl()->fake_tensor_mode() == mode;
}

// static void wrap(
//   const at::Tensor& t,
//   c10::Device fake_device,
//   const std::shared_ptr<c10::FakeTensorMode>& mode) {
//     if (is_our_fake(t)) {
//       // check t's fake device = common device?
//       // not calling converter.from_meta_and_device
//       // (at least for now) because
//       // that makes another FakeTensor
//       // and we wanna change this in place
//       transmute_to_fake(t, fake_device, mode);
//     }
//     // else if converter is not None:
//       // do things
// }

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
  auto real =
      at::empty_strided(t.sizes(), t.strides(), t.options().device(device))
          .zero_();
  c10::impl::FakeTensorModeTLS::set_state(saved_mode);
  return real;
}

static void validate_and_convert_non_fake_tensors(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {

  auto validate = [&](const at::Tensor& t) -> std::optional<at::Tensor> {
    if (t.defined() && !is_our_fake(t, mode)) {
      // TODO: check if hasattr(func, "tags") and torch.Tag.inplace_view in func.tags
      // TODO: allow non fake inputs
      // TODO: if not allow non fake inputs checks

      // TODO: change to call converter
      // return converter.from_real_tensor(t);
      return real_tensor_to_fake(t, mode);
    } else if (is_our_fake(t, mode)) {
      return std::nullopt; // already fake, leave as is
    }
    return std::nullopt;
  };
  for_each_tensor(
      stack, arguments_begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        return validate(t);
      });
}

static bool is_lift_func(const c10::OperatorHandle& op) {
  const auto& name = op.operator_name();
  return (name.name == "aten::lift_fresh" || name.name == "aten::lift_fresh_copy");
}

static void maybe_run_unsafe_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  // check if can generate trivial fake imp
  //
  // Convert fake inputs to zero-filled real tensors.
  for_each_tensor(
      stack, arguments_begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && t.is_fake())
          return to_real_tensor(t); // should call back to python
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
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for_each_tensor(
      stack, returns_begin, num_returns,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && !t.is_fake())
          return real_tensor_to_fake(t, mode);
        return std::nullopt;
      });
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  auto mode = c10::impl::FakeTensorModeTLS::get_state();

  // in python fake tensor we check has unrecognized types here
  // and throw an error if so to pass control to the next torch dispatch
  // but since C++ fake sits below python i dont think we need that here

  bool has_symints = has_symbolic_sizes(stack, arguments_begin, num_arguments);

  std::vector<at::Tensor> flat_arg_fake_tensors;
  for_each_tensor(
      stack, arguments_begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (is_our_fake(t, mode))
          flat_arg_fake_tensors.push_back(t);
        return std::nullopt;
      });

  // Skip constant prop for _to_copy when the input is already on meta device
  // TODO: implement avoiding_device_init (requires avoid_device_init on C++ FakeTensorMode)
  auto arguments = torch::jit::last(*stack, num_arguments);

  bool device_conversion_skip_const_prop =
      op.operator_name().name == "aten::_to_copy" &&
      (num_arguments > 0 && arguments[0].isTensor()) &&
      arguments[0].toTensor().device().is_meta(); // or avoiding_device_init

  // TODO: constant propagation requires torch::should_allow_numbers_as_tensors
  // which lives in the Python layer and is unavailable here.
  if ((is_lift_func(op) && flat_arg_fake_tensors.empty()) ||
      (false /* TODO: should_allow_numbers_as_tensors */ &&
       !has_symints &&
       flat_arg_fake_tensors.empty() &&
       !device_conversion_skip_const_prop)) {
    // TODO: implement constant propagation
    TORCH_CHECK(false, "constant propagation not implemented in C++ faketensor");

  }

  if (is_lift_func(op)) {
    // i think ? this is just converter.from_real_tensor ??
    // but need to revisit
    TORCH_CHECK(false, "lift_fresh not implemented in C++ faketensor");
  }

  validate_and_convert_non_fake_tensors(stack, arguments_begin, num_arguments, mode);

  // TODO: CONSTANT PROP LOGIC

  // HOPs
  // this is already taken care of by adding @py_impl(DispatchKey.Fake) to all HOPs
  // when the dispatcher for HOPs is hit (which happens before fake fallback),
  // it will route to the proper fake kernel
  // THIS BEHAVIOUR IS DIFFERENT THAN TODAY'S PYTHON IMPL

  // TODO: invalidate_written_to_constants

  // TODO: propagate_real_tensors
  /*if propagate_real_tensors {
      TORCH_CHECK(false, "propagate_real_tensors not implemented in C++ faketensor");
  } */

  auto common_device = _find_common_device(stack, arguments_begin, num_arguments);

  // doing this in place (talked about this at the beginning, richard thinks this is ok)
  auto wrap_meta_outputs_with_default_device_logic = [&]() {
    if (!common_device.has_value()) {
      common_device = c10::Device(c10::DeviceType::CPU);
    }
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    for_each_tensor(
        stack,
        returns_begin,
        num_returns,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && (!t.is_fake() || t.device().is_meta()))
            transmute_to_fake(t, *common_device, mode);
          return std::nullopt;
        });
  };

  // For ops with symbolic sizes, try decompositions before the Meta kernel.
  // Python FakeTensorMode (fake_tensor.py:2826) checks `meta_table` (a Python
  // dict), NOT hasKernelForDispatchKey(Meta). Many ops (e.g. aten::where,
  // aten::expand) have Meta kernels registered via @register_decomposition
  // which registers as BOTH decomposition AND Meta kernel. We must try the
  // decomp table for these ops because the bottom path runs Meta kernels with
  // Fake excluded from TLS, so sub-ops can't handle SymInt. The decomp path
  // keeps Fake in TLS, allowing sub-ops to re-enter fakeFallback.
  if (has_symints && !cpp_meta_supports_symint(op) && mode) {
    // 1. Try Python decomposition table
    if (mode->decomp_fn_ && mode->decomp_fn_(&op, stack)) {
      wrap_meta_outputs_with_default_device_logic();
      return;
    }

    // 2. Try CompositeImplicitAutograd decomposition. Remove Fake from the
    //    redispatch keyset so this op doesn't re-enter fakeFallback (which
    //    would infinite-loop). Sub-ops compute fresh keysets from TLS where
    //    Fake is still active, so they re-enter fakeFallback correctly.
    if (!op.hasKernelForDispatchKey(c10::DispatchKey::Meta) &&
        op.hasKernelForDispatchKey(
            c10::DispatchKey::CompositeImplicitAutograd)) {
      auto ks = dispatchKeySet;
      ks = ks.remove(c10::DispatchKey::Fake);
      ks = ks.remove(c10::DispatchKey::Python);
      ks = ks.remove(c10::DispatchKey::PythonTLSSnapshot);
      op.redispatchBoxed(ks, stack);
      wrap_meta_outputs_with_default_device_logic();
      return;
    }
  }

  // Prims ops with Meta kernels: redispatch to the Meta kernel with Fake
  // removed from the keyset (so this op doesn't re-enter fakeFallback) but
  // Fake stays in TLS (so sub-ops like torch.empty_strided re-enter
  // fakeFallback and get properly fakified). This matches Python
  // FakeTensorMode's `with self: func.prim_meta_impl(...)` pattern.
  // We must add Meta to the keyset because C++ fake tensors report
  // device=cpu (their fake device), so Meta isn't in the tensor's keyset.
  // BackendSelect must be removed because it has higher priority than Meta
  // and some prims (e.g. iota) have BackendSelect kernels that call back
  // to aten ops (torch.arange), which would re-enter the decomp table and
  // infinite-loop.
  if (schema.name().rfind("prims::", 0) == 0 &&
      op.hasKernelForDispatchKey(c10::DispatchKey::Meta)) {
    // Python calls prim_meta_impl directly so scalar args stay as Python
    // floats/ints. In C++, the dispatcher wraps them as tensors with default
    // dtypes (float64 for floats, int64 for ints) before we get here, causing
    // dtype mismatches. Get the target dtype from non-0-dim non-bool tensors.
    // If all tensors are 0-dim, use the first whose dtype isn't a default
    // wrapping dtype (float64/int64) since those are likely wrapped scalars.
    std::optional<c10::ScalarType> target_dtype;
    for_each_tensor(
        stack, arguments_begin, num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && t.dim() > 0 &&
              t.scalar_type() != c10::ScalarType::Bool &&
              !target_dtype.has_value()) {
            target_dtype = t.scalar_type();
          }
          return std::nullopt;
        });
    if (!target_dtype.has_value()) {
      for_each_tensor(
          stack, arguments_begin, num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (t.defined() &&
                t.scalar_type() != c10::ScalarType::Bool &&
                t.scalar_type() != c10::ScalarType::Double &&
                t.scalar_type() != c10::ScalarType::Long &&
                !target_dtype.has_value()) {
              target_dtype = t.scalar_type();
            }
            return std::nullopt;
          });
    }
    if (target_dtype.has_value()) {
      for_each_tensor(
          stack, arguments_begin, num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (t.defined() &&
                t.scalar_type() != c10::ScalarType::Bool &&
                t.scalar_type() != *target_dtype) {
              return t.to(*target_dtype);
            }
            return std::nullopt;
          });
    }

    auto ks = dispatchKeySet;
    ks = ks.remove(c10::DispatchKey::Fake);
    ks = ks.remove(c10::DispatchKey::Python);
    ks = ks.remove(c10::DispatchKey::PythonTLSSnapshot);
    ks = ks.remove(c10::DispatchKey::BackendSelect);
    ks = ks | c10::DispatchKeySet(c10::DispatchKey::Meta);
    op.redispatchBoxed(ks, stack);
    wrap_meta_outputs_with_default_device_logic();
    return;
  }

  // TODO: profiles

  // TODO: infer fake kernel

  // TODO: user registered
  // structured kernels?

  // special handling for funcs registered through `register_op_impl`


  auto device_from_args = rewrite_device_args_to_meta(
      stack, arguments_begin, num_arguments, schema);
  if (!common_device.has_value()) {
    common_device = device_from_args;
  }
  if (!common_device.has_value()) {
    common_device = c10::Device(c10::DeviceType::CPU);
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
    wrap_meta_outputs_with_default_device_logic();
  } catch (c10::NotImplementedError&) {
    // Meta kernel failed. With symbolic sizes we cannot run the unsafe
    // fallback (it materializes real tensors from symbolic shapes).
    // Match Python FakeTensorMode: raise UnsupportedOperatorException.
    TORCH_CHECK(
        !has_symints,
        "Unsupported operator for C++ FakeTensor with symbolic sizes: ",
        op.operator_name());

    // Restore the stack and run the real kernel with zero-filled inputs
    // to discover output metadata.
    stack->resize(arguments_begin);
    for (auto& arg : saved_args) {
      stack->push_back(std::move(arg));
    }
    maybe_run_unsafe_fallback(op, stack, arguments_begin, num_arguments, mode);
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
