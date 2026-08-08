#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_set>
#include <utility>

namespace {

// copied from fake_tensor.py _cpp_meta_supports_symint
bool cpp_meta_supports_symint(const c10::OperatorHandle& op) {
  static const std::unordered_set<c10::OperatorHandle> allowlist = {
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::empty", "memory_format"),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty_strided", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::as_strided_scatter", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::as_strided", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::as_strided_", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::zeros", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::detach", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::view_as_real", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::view_as_complex", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::set_", "source_Storage_storage_offset"),
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::_sparse_coo_tensor_with_dims_and_tensors", ""),
  };
  if (allowlist.contains(op)) {
    return true;
  }
  return op.hasTag(at::Tag::view_copy);
}

// copied from fake_tensor.py _unbacked_special_fake_handling_ops.
const std::unordered_set<c10::OperatorHandle>&
_unbacked_special_fake_handling_ops() {
  static const std::unordered_set<c10::OperatorHandle> ops = {
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::view", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::_unsafe_view", ""),
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::slice", "Tensor"),
  };
  return ops;
}

template <typename Fn>
void for_each_tensor(
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

bool has_symbolic_sizes(
    torch::jit::Stack const* const stack,
    size_t begin,
    size_t num_arguments) {
  bool found = false;
  for (const auto idx : c10::irange(num_arguments)) {
    (*stack)[begin + idx].visit([&](const c10::IValue& ivalue) -> bool {
      if (ivalue.isTensor()) {
        const auto& t = ivalue.toTensor();
        if (t.defined() &&
            t.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
          found = true;
          return true;
        }
      } else if (
          ivalue.isSymInt() || ivalue.isSymFloat() || ivalue.isSymIntList()) {
        found = true;
        return true;
      }
      return false;
    });
    if (found)
      return true;
  }
  return false;
}

bool bypass_zero_dim_cpu_tensor_check(const c10::OperatorHandle& op) {
  static const c10::OperatorHandle nextafter =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::nextafter", "");
  return op == nextafter;
}

// list of ops which can have args(tensor/tensorList) in mixed device
bool mixed_device_fns(const c10::OperatorHandle& op) {
  static const c10::OperatorHandle foreach_copy =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::_foreach_copy", "");
  return op == foreach_copy;
}

// These in-place ops keep the destination tensor's device even if the
// rhs was explicitly constructed on meta.
bool meta_rhs_mixed_device_fns(const c10::OperatorHandle& op) {
  static const c10::OperatorHandle add_ =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::add_", "Tensor");
  return op == add_;
}

std::optional<c10::Device> _find_common_device(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    size_t begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    std::optional<std::pair<c10::Device, c10::Device>>& mismatch_out) {
  std::optional<c10::Device> common_device;
  bool is_cpu_zero_dim = false;
  const bool is_bypass_zero_dim_cpu_tensor_check_op =
      bypass_zero_dim_cpu_tensor_check(op);
  const bool mixed_device = mixed_device_fns(op);
  const bool meta_rhs_mixed_device = meta_rhs_mixed_device_fns(op);
  const std::optional<c10::DeviceType> prefer_device_type =
      mode ? mode->prefer_device_type : std::nullopt;

  for_each_tensor(
      stack,
      begin,
      num_arguments,
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
        // mismatching devices !
        // if current tensor is cpu 0 dim, defer to existing device
        if (t_is_cpu_zero_dim && !is_bypass_zero_dim_cpu_tensor_check_op)
          return std::nullopt;
        // current device is from cpu 0 dim tensor, overwrite
        if (is_cpu_zero_dim && !is_bypass_zero_dim_cpu_tensor_check_op) {
          common_device = t.device();
          is_cpu_zero_dim = false;
          return std::nullopt;
        }

        if (mixed_device &&
            (common_device->is_cpu() || t.device().is_cpu())) {
          return std::nullopt;
        }

        if (meta_rhs_mixed_device &&
            (common_device->type() == c10::DeviceType::Meta ||
             t.device().type() == c10::DeviceType::Meta)) {
          return std::nullopt;
        }
        // if prefer_device_type is set, prefer that device type over others
        if (prefer_device_type.has_value()) {
          bool common_has_preferred =
              common_device->type() == *prefer_device_type;
          bool t_has_preferred = t.device().type() == *prefer_device_type;
          if (!common_has_preferred && t_has_preferred) {
            // Switch to the preferred device type
            common_device = t.device();
            is_cpu_zero_dim = t_is_cpu_zero_dim;
            return std::nullopt;
          }
          if (common_has_preferred && !t_has_preferred) {
            // Keep the existing preferred device type
            return std::nullopt;
          }
        }
        // genuine mismatch of non-zero-dim tensors: record (don't raise)
        if (!mismatch_out.has_value())
          mismatch_out = std::make_pair(*common_device, t.device());
        return std::nullopt;
      });
  return common_device;
}

bool is_device_type_arg(const c10::Argument& arg) {
  const auto& type = arg.type();
  if (type->kind() == c10::TypeKind::DeviceObjType)
    return true;
  if (type->kind() == c10::TypeKind::OptionalType) {
    auto elem = type->castRaw<c10::OptionalType>()->getElementType();
    return elem->kind() == c10::TypeKind::DeviceObjType;
  }
  return false;
}

std::optional<c10::Device> find_and_rewrite_device_args(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const c10::FunctionSchema& schema,
    bool rewrite_to_meta) {
  std::optional<c10::Device> original_device;
  for (const auto idx : c10::irange(num_arguments)) {
    auto& ivalue = (*stack)[arguments_begin + idx];
    if (ivalue.isDevice()) {
      auto dev = ivalue.toDevice();
      if (rewrite_to_meta) {
        if (dev.type() == c10::DeviceType::Meta) {
          auto mode = c10::impl::FakeTensorModeTLS::get_state();
          TORCH_CHECK(
              mode == nullptr || mode->allow_meta_,
              "device.type must not be 'meta' when allow_meta is False");
        }
        ivalue = c10::IValue(c10::Device(c10::DeviceType::Meta));
      }
      if (!original_device.has_value())
        original_device = dev;
    } else if (ivalue.isNone() && is_device_type_arg(schema.arguments()[idx])) {
      if (rewrite_to_meta)
        ivalue = c10::IValue(c10::Device(c10::DeviceType::Meta));
      if (!original_device.has_value())
        original_device = c10::Device(c10::DeviceType::CPU);
    }
  }
  return original_device;
}

bool is_our_fake(
    const at::Tensor& t,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  return t.defined() && t.is_fake() &&
      t.unsafeGetTensorImpl()->fake_tensor_mode() == mode;
}

void transmute_to_fake(
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
at::Tensor from_real_tensor(
    const at::Tensor& t,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  // Delegate to the mode's Python converter (FakeTensorConverter.from_real_tensor,
  // via the PyInterpreter to_meta_tensor hook) so storage memoization aliases the
  // meta storage for real tensors that share storage (e.g. a constant tensor and
  // its views).
  (void)mode;
  auto* interp = c10::impl::getGlobalPyInterpreter();
  return at::Tensor((*interp)->to_meta_tensor(t.getIntrusivePtr()));
}

bool can_generate_trivial_fake_impl(const c10::FunctionSchema& schema) {
  auto is_builtin = [&]() {
    auto ns = schema.operator_name().getNamespace();
    return ns.has_value() && (*ns == "aten" || *ns == "prim" || *ns == "prims");
  };
  return !is_builtin() && schema.is_mutable() && schema.returns().empty();
}

bool can_run_unsafe_fallback(const c10::FunctionSchema& schema) {
  auto ns = schema.operator_name().getNamespace();
  return ns.has_value() &&
      (*ns == "aten" || *ns == "prims" || *ns == "quantized");
}

constexpr int64_t CONSTANT_NUMEL_LIMIT = 1;
bool may_turn_const(const at::Tensor& t) {
  return t.numel() <= CONSTANT_NUMEL_LIMIT && !t.is_sparse() && !t.is_fake() &&
      t.device().type() != c10::DeviceType::Meta;
}

bool should_allow_numbers_as_tensors(const c10::OperatorHandle& op) {
  auto& dispatcher = c10::Dispatcher::singleton();
  static const std::unordered_set<c10::OperatorHandle> allowed = {
      dispatcher.findSchemaOrThrow("aten::add", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::add_", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::sub", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::sub_", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::mul", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::mul_", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::div", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::div", "Tensor_mode"),
      dispatcher.findSchemaOrThrow("aten::div_", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::div_", "Tensor_mode"),
      dispatcher.findSchemaOrThrow("aten::divide", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::divide", "Tensor_mode"),
      dispatcher.findSchemaOrThrow("aten::multiply", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::subtract", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::true_divide", "Tensor"),
      dispatcher.findSchemaOrThrow("aten::floor_divide", ""),
      dispatcher.findSchemaOrThrow("aten::_conj", ""),
  };
  return allowed.contains(op);
}

void set_constant_on_mode(
    const at::Tensor& fake_tensor,
    c10::intrusive_ptr<c10::TensorImpl> constant,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (!mode || !constant)
    return;
  c10::StorageImpl* storage = nullptr;
  if (constant->has_storage())
    storage = constant->storage().unsafeGetStorageImpl();
  mode->set_constant(
      fake_tensor.getIntrusivePtr(), std::move(constant), storage);
}

void invalidate_written_to_constants(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::vector<at::Tensor>& flat_arg_fake_tensors,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (!mode)
    return;
  const auto& schema = op.schema();
  bool any_constant = std::any_of(
      flat_arg_fake_tensors.begin(),
      flat_arg_fake_tensors.end(),
      [&](const at::Tensor& t) {
        return mode->get_constant(t.unsafeGetTensorImpl()) != nullptr;
      });
  if (!any_constant || !schema.is_mutable())
    return;
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = (*stack)[arguments_begin + idx];
    if (!ivalue.isTensor())
      continue;
    const auto& t = ivalue.toTensor();
    if (!is_our_fake(t, mode))
      continue;
    auto constant = mode->get_constant(t.unsafeGetTensorImpl());
    if (!constant)
      continue;
    if (!schema.is_mutable({c10::SchemaArgType::input, idx}))
      continue;
    if (constant->has_storage())
      mode->invalidate_constant_aliases(
          constant->storage().unsafeGetStorageImpl());
  }
}

// creates a zero-filled real tensor on the fake tensor's original device
// we need to temporarily exit FakeTensorMode TLS so the created tensor is
// actually real
// matches Python FakeTensor behaviour (with no_dispatch())
at::Tensor to_real_tensor(const at::Tensor& t) {
  auto device = t.device(); // returns fake device (e.g. CPU)
  auto saved_mode = c10::impl::FakeTensorModeTLS::get_state();
  c10::impl::FakeTensorModeTLS::reset_state();
  auto real =
      at::empty_strided(t.sizes(), t.strides(), t.options().device(device))
          .zero_();
  c10::impl::FakeTensorModeTLS::set_state(saved_mode);
  return real;
}

std::vector<at::Tensor> validate_and_convert_non_fake_tensors(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  std::vector<at::Tensor> flat_arg_fake_tensors;

  for_each_tensor(
      stack,
      arguments_begin,
      num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && !is_our_fake(t, mode)) {
          // TODO: check if hasattr(func, "tags") and torch.Tag.inplace_view in
          // func.tags

          if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
            return std::nullopt;
          }
          auto* interp = c10::impl::getGlobalPyInterpreter();
          if (interp && !(*interp)->allow_non_fake_inputs()) {
            // Match Python FakeTensorMode.validate: reject non-fake inputs
            // unless allow_non_fake_inputs. A fake from a different mode is a
            // distinct (mixing-modes) error.
            TORCH_CHECK(!t.is_fake(), "Mixing fake modes NYI");
            TORCH_CHECK(
                false,
                "Please convert all Tensors to FakeTensors first or "
                "instantiate FakeTensorMode with 'allow_non_fake_inputs'.");
          }
          auto out = from_real_tensor(t, mode);
          flat_arg_fake_tensors.push_back(out);
          return out;
        }
        if (is_our_fake(t, mode)) {
          flat_arg_fake_tensors.push_back(t);
        }
        return std::nullopt;
      });

  return flat_arg_fake_tensors;
}

bool is_lift_func(const c10::OperatorHandle& op) {
  static const c10::OperatorHandle lift_fresh =
      c10::Dispatcher::singleton().findSchemaOrThrow("aten::lift_fresh", "");
  static const c10::OperatorHandle lift_fresh_copy =
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "aten::lift_fresh_copy", "");
  return op == lift_fresh || op == lift_fresh_copy;
}

void maybe_run_unsafe_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    bool has_symints,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  const auto& schema = op.schema();

  if (can_generate_trivial_fake_impl(schema)) {
    stack->resize(arguments_begin);
    return;
  }

  TORCH_CHECK(
      !has_symints && can_run_unsafe_fallback(schema),
      "Unsupported operator for C++ FakeTensor: ",
      op.operator_name());

  for_each_tensor(
      stack,
      arguments_begin,
      num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && t.is_fake())
          return to_real_tensor(t);
        return std::nullopt;
      });
  {
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake) |
        c10::DispatchKeySet(c10::DispatchKey::Python) |
        c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
    op.callBoxed(stack);
  }

  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for_each_tensor(
      stack,
      returns_begin,
      num_returns,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && !t.is_fake())
          return from_real_tensor(t, mode);
        return std::nullopt;
      });
}

bool contains_tensor_types(const c10::TypePtr& type) {
  if (type->isSubtypeOf(*c10::TensorType::get())) {
    return true;
  }
  for (const auto& contained : type->containedTypes()) {
    if (contains_tensor_types(contained)) {
      return true;
    }
  }
  return false;
}

bool _is_tensor_constructor(const c10::FunctionSchema& schema) {
  for (const auto& arg : schema.arguments()) {
    if (contains_tensor_types(arg.type())) {
      return false;
    }
  }
  return schema.returns().size() == 1 &&
      schema.returns()[0].type()->kind() == c10::TypeKind::TensorType;
}

bool may_have_op_impl(
    const c10::OperatorHandle& op,
    const c10::FunctionSchema& schema) {
  if (at::impl::fakeDispatchTableContains(
          at::impl::FakeDispatchCategory::OpImpl, op.operator_name())) {
    return true;
  }
  if (op.hasTag(at::Tag::dynamic_output_shape) ||
      op.hasTag(at::Tag::data_dependent_output)) {
    return true;
  }
  if (_is_tensor_constructor(schema)) {
    return true;
  }
  const auto& name = op.operator_name().name;
  return name.rfind("aten::_foreach_", 0) == 0 &&
      op.hasKernelForDispatchKey(c10::DispatchKey::Meta);
}

struct RestoreInactiveFakeMode {
  ~RestoreInactiveFakeMode() {
    c10::impl::FakeTensorModeTLS::reset_state();
  }
};

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet /*dispatchKeySet*/,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  auto mode = c10::impl::FakeTensorModeTLS::get_state();

  // same as python FakeTensor dispatch re-entering FakeTensorMode dispatches
  std::optional<RestoreInactiveFakeMode> restore_fake_mode;
  if (mode == nullptr) {
    for_each_tensor(
        stack,
        arguments_begin,
        num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (mode == nullptr && t.defined() && t.is_fake()) {
            mode = t.unsafeGetTensorImpl()->fake_tensor_mode();
          }
          return std::nullopt;
        });
    if (mode != nullptr) {
      restore_fake_mode.emplace();
      c10::impl::FakeTensorModeTLS::set_state(mode);
    }
  }

  bool has_symints = has_symbolic_sizes(stack, arguments_begin, num_arguments);

  std::vector<at::Tensor> flat_arg_fake_tensors;
  for_each_tensor(
      stack,
      arguments_begin,
      num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (is_our_fake(t, mode))
          flat_arg_fake_tensors.push_back(t);
        return std::nullopt;
      });

  // skip constant prop for _to_copy when the input is already on meta device
  // TODO: implement avoiding_device_init (requires avoid_device_init on C++
  // FakeTensorMode)

  auto const_prop_arguments = torch::jit::last(*stack, num_arguments);
  bool device_conversion_skip_const_prop =
      op.operator_name().name == "aten::_to_copy" &&
      !const_prop_arguments.empty() && const_prop_arguments[0].isTensor() &&
      const_prop_arguments[0].toTensor().device().is_meta();
  if ((is_lift_func(op) && flat_arg_fake_tensors.empty()) ||
      (should_allow_numbers_as_tensors(op) && !has_symints &&
       flat_arg_fake_tensors.empty() && !device_conversion_skip_const_prop)) {
    {
      c10::impl::ExcludeDispatchKeyGuard guard(
          c10::DispatchKeySet(c10::DispatchKey::Fake) |
          c10::DispatchKeySet(c10::DispatchKey::Python) |
          c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
      op.callBoxed(stack);
    }
    const auto num_returns = schema.returns().size();
    const auto returns_begin = stack->size() - num_returns;
    for_each_tensor(
        stack,
        returns_begin,
        num_returns,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (!t.defined() || t.is_fake())
            return std::nullopt;
          auto fake = from_real_tensor(t, mode);
          if (may_turn_const(t)) {
            set_constant_on_mode(fake, t.getIntrusivePtr(), mode);
          }
          return fake;
        });
    return;
  }

  // lift_fresh with fake inputs: convert any non-fake inputs to fake.
  // lift_fresh is identity so the stack already holds the return value.
  if (is_lift_func(op)) {
    for_each_tensor(
        stack,
        arguments_begin,
        num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && !t.is_fake())
            return from_real_tensor(t, mode);
          return std::nullopt;
        });
    return;
  }

  // TODO: constant propagation for should_allow_numbers_as_tensors
  // (requires access to torch::should_allow_numbers_as_tensors from Python
  // layer)

  flat_arg_fake_tensors = validate_and_convert_non_fake_tensors(
      stack, arguments_begin, num_arguments, mode);

  // constant prop, if every fake-tensor argument carries a backing
  // constant, run the real op on those constants
  {
    bool all_constant = !flat_arg_fake_tensors.empty() &&
        std::all_of(
            flat_arg_fake_tensors.begin(),
            flat_arg_fake_tensors.end(),
            [&](const at::Tensor& t) {
              return mode &&
                  mode->get_constant(t.unsafeGetTensorImpl()) != nullptr;
            });

    // isinstance(func, torch._ops.OpOverload) - always true in C++ fallback
    if (!op.hasTag(at::Tag::nondeterministic_seeded) &&
        (!op.hasTag(at::Tag::inplace_view) ||
         schema.name() == "aten::detach_") &&
        all_constant && !flat_arg_fake_tensors.empty() && !has_symints &&
        // TODO: avoiding_device_init
        schema.name() != "aten::_nested_tensor_from_tensor_list") {
      // save the original arguments so we can restore the stack if the
      // outputs are too large to keep as constants.
      auto orig_arguments = torch::jit::last(*stack, num_arguments).vec();

      // build memo from constant tensorimpl to original fake tensor
      // for in-place ops the output real tensor is the same object as the
      // input constant, so we must return the original fake tensor (with an
      // updated constant) instead of creating a new one
      std::unordered_map<c10::TensorImpl*, at::Tensor> tensor_memo;
      for_each_tensor(
          stack,
          arguments_begin,
          num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (is_our_fake(t, mode)) {
              auto constant = mode->get_constant(t.unsafeGetTensorImpl());
              if (constant) {
                tensor_memo[constant.get()] = t;
                return at::Tensor(constant);
              }
            }
            return std::nullopt;
          });

      // run real op
      {
        c10::impl::ExcludeDispatchKeyGuard guard(
            c10::DispatchKeySet(c10::DispatchKey::Fake) |
            c10::DispatchKeySet(c10::DispatchKey::Python) |
            c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
        op.callBoxed(stack);
      }

      // check if all output tensors can be turned into constants
      const auto num_returns = schema.returns().size();
      const auto returns_begin = stack->size() - num_returns;
      bool all_outputs_const = true;
      for_each_tensor(
          stack,
          returns_begin,
          num_returns,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (!may_turn_const(t))
              all_outputs_const = false;
            return std::nullopt;
          });

      if (all_outputs_const) {
        for_each_tensor(
            stack,
            returns_begin,
            num_returns,
            [&](const at::Tensor& t) -> std::optional<at::Tensor> {
              if (!may_turn_const(t))
                return std::nullopt;
              auto constant = t.getIntrusivePtr();
              auto memo_it = tensor_memo.find(t.unsafeGetTensorImpl());
              if (memo_it != tensor_memo.end()) {
                auto& orig_fake = memo_it->second;
                set_constant_on_mode(orig_fake, std::move(constant), mode);
                return orig_fake;
              }
              auto fake = from_real_tensor(t, mode);
              set_constant_on_mode(fake, std::move(constant), mode);
              return fake;
            });
        return;
      }

      // outputs too large to keep as constants
      // invalidate all constants that might alias the output tensors
      for_each_tensor(
          stack,
          returns_begin,
          num_returns,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (t.defined() && !t.is_fake() && t.has_storage())
              mode->invalidate_constant_aliases(
                  t.storage().unsafeGetStorageImpl());
            return std::nullopt;
          });

      // restore the original arguments to re-run through meta dispatch
      stack->resize(arguments_begin);
      for (auto& arg : orig_arguments) {
        stack->push_back(std::move(arg));
      }
    }
  }

  // HOPs
  // this is already taken care of by adding @register_fake
  invalidate_written_to_constants(
      op, stack, arguments_begin, num_arguments, flat_arg_fake_tensors, mode);

  // propagate_real_tensors is handled by PropagateRealTensorsGuard below, which
  // runs after the fake outputs are produced.

  std::optional<std::pair<c10::Device, c10::Device>> device_mismatch;
  auto common_device = _find_common_device(
      op, stack, arguments_begin, num_arguments, mode, device_mismatch);

  auto wrap_meta_outputs_with_default_device_logic = [&]() {
    if (device_mismatch.has_value()) {
      TORCH_CHECK(
          false,
          "Expected all tensors to be on the same device, but found at least "
          "two devices, ",
          device_mismatch->first,
          " and ",
          device_mismatch->second,
          "!");
    }
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

  auto* interp = c10::impl::getGlobalPyInterpreter();

  // propagate_real_tensors: run the op on the real tensors shadowing the fake
  // inputs and stamp the reals / hint unbacked symbols onto the fake outputs.
  // Snapshot the fake inputs now (before dispatch consumes the stack); the
  // guard propagates after the fake outputs are produced, on any normal exit of
  // the dispatch below (skipping exception paths, where outputs are invalid).
  const bool do_propagate_real_tensors =
      mode && mode->propagate_real_tensors_ && interp != nullptr;
  struct PropagateRealTensorsGuard {
    const c10::OperatorHandle& op;
    torch::jit::Stack* stack;
    c10::impl::PyInterpreter* interp;
    torch::jit::Stack fake_args;
    int entry_uncaught;
    bool enabled;
    ~PropagateRealTensorsGuard() {
      if (!enabled || std::uncaught_exceptions() > entry_uncaught) {
        return;
      }
      try {
        (*interp)->propagate_real_tensors(op, fake_args, stack);
      } catch (...) {
        // best-effort: a propagation failure must not break the trace
      }
    }
  } propagate_guard{
      op,
      stack,
      interp,
      do_propagate_real_tensors ? torch::jit::last(*stack, num_arguments).vec()
                                : torch::jit::Stack{},
      static_cast<int>(std::uncaught_exceptions()),
      do_propagate_real_tensors};

  if (has_symints && mode && interp) {
    if ((*interp)->fake_try_fast_op_impls(
            op,
            stack,
            common_device.value_or(c10::Device(c10::DeviceType::CPU)))) {
      return;
    }
  }

  // for ops with symbolic sizes, try decompositions before the meta kernel.
  // Skip ops with a Python meta registration; those match on the meta kernel,
  // matching Python FakeTensorMode's `op not in meta_table` decomp gating.
  if (has_symints && !cpp_meta_supports_symint(op) &&
      !_unbacked_special_fake_handling_ops().contains(op) &&
      !at::impl::fakeDispatchTableContains(
          at::impl::FakeDispatchCategory::Meta, op.operator_name()) &&
      mode) {
    if (interp &&
        at::impl::fakeDispatchTableContains(
            at::impl::FakeDispatchCategory::Decomp, op.operator_name())) {
      if ((*interp)->fake_try_decomp(op, stack)) {
        wrap_meta_outputs_with_default_device_logic();
        return;
      }
    }

    // Run the CIA decomposition by calling its kernel directly
    // same as python in torch/_ops.py
    if (!op.hasKernelForDispatchKey(c10::DispatchKey::Meta) &&
        op.hasKernelForDispatchKey(
            c10::DispatchKey::CompositeImplicitAutograd)) {
      op.callBoxedForDispatchKey(
          c10::DispatchKey::CompositeImplicitAutograd, *stack);
      wrap_meta_outputs_with_default_device_logic();
      return;
    }
  }

  // Prims: call prim_meta_impl directly via Python callback, matching
  // Python FakeTensorMode's `with self: func.prim_meta_impl(*args, **kwargs)`.
  // Sub-ops (e.g. torch.empty inside _iota_meta) still enter fakeFallback
  // because Fake remains in TLS.
  auto op_ns = op.operator_name().getNamespace();
  if (op_ns.has_value() && *op_ns == "prims" && mode && interp &&
      at::impl::fakeDispatchTableContains(
          at::impl::FakeDispatchCategory::PrimMeta, op.operator_name())) {
    // In Python, scalar args stay as Python floats/ints. In C++, the
    // dispatcher wraps them as tensors with default dtypes (float64 for
    // floats, int64 for ints), causing dtype mismatches in prim_meta_impl.
    // Fix up by casting all tensors to a common dtype before calling.
    std::optional<c10::ScalarType> target_dtype;
    for_each_tensor(
        stack,
        arguments_begin,
        num_arguments,
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
          stack,
          arguments_begin,
          num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (t.defined() && t.scalar_type() != c10::ScalarType::Bool &&
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
          stack,
          arguments_begin,
          num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            // Skip wrapped-number (scalar) tensors: they are weak-typed in
            // promotion, so prim_meta_impl handles their dtype without a cast,
            // and casting a symbolic scalar dispatches _to_copy on a SymFloat,
            // which its decomposition rejects.
            if (t.defined() && t.scalar_type() != c10::ScalarType::Bool &&
                !t.unsafeGetTensorImpl()->is_wrapped_number() &&
                t.scalar_type() != *target_dtype) {
              return t.to(*target_dtype);
            }
            return std::nullopt;
          });
    }

    if ((*interp)->fake_try_prim_meta(op, stack)) {
      wrap_meta_outputs_with_default_device_logic();
      return;
    }
  }

  // TODO: profiles

  // TODO: infer fake kernel

  // User-registered fake implementations (torch.library.register_fake) are
  // handled by dispatch, not here: register_fake also registers a DispatchKey
  // Fake kernel (see torch/_library/fake_impl.py), so those ops route to their
  // own Fake kernel before reaching this fallback.

  if (!common_device.has_value()) {
    common_device = find_and_rewrite_device_args(
        stack, arguments_begin, num_arguments, schema, /*rewrite_to_meta=*/false);
  }

  if (mode && interp && may_have_op_impl(op, schema)) {
    bool op_impl_handled = (*interp)->fake_try_op_impl(
        op, stack, common_device.value_or(c10::Device(c10::DeviceType::CPU)));
    if (op_impl_handled) {
      return;
    }
  }

  auto device_from_args = find_and_rewrite_device_args(
      stack, arguments_begin, num_arguments, schema, /*rewrite_to_meta=*/true);
  if (device_from_args.has_value()) {
    common_device = device_from_args;
  }
  if (!common_device.has_value()) {
    common_device = c10::Device(c10::DeviceType::CPU);
  }

  // Try the Meta kernel. If it raises, fall back to:
  //   1. Python op_impl handlers (for ops like _local_scalar_dense whose
  //      Meta kernel raises but have a Python fake impl), or
  //   2. The unsafe fallback with zero-filled inputs.
  // Save arguments first because callBoxed consumes them from the stack.
  torch::jit::Stack saved_args;
  {
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
  } catch (...) {
    auto eptr = std::current_exception();

    // For NotImplementedError, try the unsafe fallback.
    // For other errors, rethrow.
    try {
      std::rethrow_exception(eptr);
    } catch (c10::NotImplementedError&) {
      stack->resize(arguments_begin);
      for (auto& arg : saved_args) {
        stack->push_back(std::move(arg));
      }
      maybe_run_unsafe_fallback(
          op, stack, arguments_begin, num_arguments, has_symints, mode);
    }
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
