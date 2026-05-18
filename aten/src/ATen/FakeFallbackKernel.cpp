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

// copied from fake_tensor.py _cpp_meta_supports_symint
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
    c10::impl::ExcludeDispatchKeyGuard guard{
        c10::DispatchKeySet(c10::DispatchKey::Fake)};
    meta_t = at::empty_strided(
        t.sizes(), t.strides(), t.options().device(c10::DeviceType::Meta));
  }
  if (t.requires_grad()) {
    meta_t.set_requires_grad(true);
  }
  transmute_to_fake(meta_t, original_device, mode);
  return meta_t;
}

static bool can_generate_trivial_fake_impl(
    const c10::FunctionSchema& schema) {
  auto is_builtin = [&]() {
    auto ns = schema.operator_name().getNamespace();
    return ns.has_value() &&
        (*ns == "aten" || *ns == "prim" || *ns == "prims");
  };
  return !is_builtin() && schema.is_mutable() && schema.returns().empty();
}

static bool can_run_unsafe_fallback(const c10::FunctionSchema& schema) {
  const auto& name = schema.name();
  return name.rfind("aten::", 0) == 0 || name.rfind("prims::", 0) == 0 ||
      name.rfind("quantized::", 0) == 0;
}

static constexpr int64_t CONSTANT_NUMEL_LIMIT = 1;
static bool may_turn_const(const at::Tensor& t) {
  return t.numel() <= CONSTANT_NUMEL_LIMIT &&
      !t.is_sparse() &&
      !t.is_fake() &&
      t.device().type() != c10::DeviceType::Meta;
}

// Registers a fake tensor's constant in the mode's storage mapping
static void add_constant_storage_mapping(
    const at::Tensor& fake_tensor,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (!mode)
    return;
  auto constant = fake_tensor.unsafeGetTensorImpl()->constant();
  if (!constant || !constant->has_storage())
    return;
  auto* storage_impl = constant->storage().unsafeGetStorageImpl();
  // weak reference to not keep the fake tensor alive, math
  auto weak_impl = c10::weak_intrusive_ptr<c10::TensorImpl>(
      c10::intrusive_ptr<c10::TensorImpl>::unsafe_reclaim_from_nonowning(
          fake_tensor.unsafeGetTensorImpl()));
  mode->constant_storage_mapping_[storage_impl].push_back(std::move(weak_impl));
}

// Given a real tensor, finds all fake tensors whose constant shares the same
// underlying storage and clears their constants
static void invalidate_constant_aliases(
    const at::Tensor& real_tensor,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (!mode || !real_tensor.has_storage())
    return;
  auto* storage_impl = real_tensor.storage().unsafeGetStorageImpl();
  auto it = mode->constant_storage_mapping_.find(storage_impl);
  if (it == mode->constant_storage_mapping_.end())
    return;
  for (auto& weak_ref : it->second) {
    // try to promote to strong intrusive_ptr
    // if faketensor is dead, impl will be nullptr
    auto impl = weak_ref.lock();
    if (impl) {
      impl->set_constant(nullptr); // clear constant
    }
  }
  mode->constant_storage_mapping_.erase(it);
}

// Before falling through to meta dispatch, checks if any mutable argument
// carries a constant and invalidates it (and all its storage aliases)
static void invalidate_written_to_constants(
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
      [](const at::Tensor& t) {
        return t.unsafeGetTensorImpl()->constant() != nullptr;
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
    auto constant = t.unsafeGetTensorImpl()->constant();
    if (!constant)
      continue;
    if (!schema.is_mutable({c10::SchemaArgType::input, idx}))
      continue;
    invalidate_constant_aliases(*constant, mode);
  }
}

// creates a zero-filled real tensor on the fake tensor's original device.
// we need to temporarily exit FakeTensorMode TLS so the created tensor is
// actually real
// matches Python FakeTensor behaviour (with no_dispatch())
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

static std::vector<at::Tensor> validate_and_convert_non_fake_tensors(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {

  std::vector<at::Tensor> flat_arg_fake_tensors;

  for_each_tensor(
      stack, arguments_begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (t.defined() && !is_our_fake(t, mode)) {
          // TODO: check if hasattr(func, "tags") and torch.Tag.inplace_view in func.tags
          // TODO: allow non fake inputs
          // TODO: if not allow non fake inputs checks

          // TODO: change to call converter
          // return converter.from_real_tensor(t);
          auto out = real_tensor_to_fake(t, mode);
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

static bool is_lift_func(const c10::OperatorHandle& op) {
  const auto& name = op.operator_name();
  return (name.name == "aten::lift_fresh" || name.name == "aten::lift_fresh_copy");
}

static void maybe_run_unsafe_fallback(
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
      stack, arguments_begin, num_arguments,
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

  bool has_symints = has_symbolic_sizes(stack, arguments_begin, num_arguments);

  std::vector<at::Tensor> flat_arg_fake_tensors;
  for_each_tensor(
      stack, arguments_begin, num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (is_our_fake(t, mode))
          flat_arg_fake_tensors.push_back(t);
        return std::nullopt;
      });

  // skip constant prop for _to_copy when the input is already on meta device
  // TODO: implement avoiding_device_init (requires avoid_device_init on C++ FakeTensorMode)
  // auto arguments = torch::jit::last(*stack, num_arguments);

  // bool device_conversion_skip_const_prop =
  //     op.operator_name().name == "aten::_to_copy" &&
  //     (num_arguments > 0 && arguments[0].isTensor()) &&
  //     arguments[0].toTensor().device().is_meta(); // or avoiding_device_init

  // matches python FakeTensorMode
  // lift_fresh with no fake inputs (e.g. torch.tensor(())): run the real op
  // to produce a constant tensor, then convert to fake.
  if (is_lift_func(op) && flat_arg_fake_tensors.empty()) {
    /*
      or (TODO: should_allow_numbers_as_tensors &&
       !has_symints &&
       flat_arg_fake_tensors.empty() &&
       !device_conversion_skip_const_prop)) {
    */
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
        stack, returns_begin, num_returns,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (!t.defined() || t.is_fake())
            return std::nullopt;
          auto fake = real_tensor_to_fake(t, mode);
          if (may_turn_const(t)) {
            fake.unsafeGetTensorImpl()->set_constant(
                std::make_shared<at::Tensor>(t.clone()));
            add_constant_storage_mapping(fake, mode);
          }
          return fake;
        });
    return;
  }

  // lift_fresh with fake inputs: convert any non-fake inputs to fake.
  // lift_fresh is identity so the stack already holds the return value.
  if (is_lift_func(op)) {
    for_each_tensor(
        stack, arguments_begin, num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && !t.is_fake())
            return real_tensor_to_fake(t, mode);
          return std::nullopt;
        });
    return;
  }

  // TODO: constant propagation for should_allow_numbers_as_tensors
  // (requires access to torch::should_allow_numbers_as_tensors from Python layer)

  flat_arg_fake_tensors =
      validate_and_convert_non_fake_tensors(stack, arguments_begin, num_arguments, mode);

  // constant prop, if every fake-tensor argument carries a backing
  // constant, run the real op on those constants
  {
    bool all_constant = !flat_arg_fake_tensors.empty() &&
        std::all_of(
            flat_arg_fake_tensors.begin(),
            flat_arg_fake_tensors.end(),
            [](const at::Tensor& t) {
              return t.unsafeGetTensorImpl()->constant() != nullptr;
            });

    // isinstance(func, torch._ops.OpOverload) — always true in C++ fallback
    if (!op.hasTag(at::Tag::nondeterministic_seeded) &&
        (!op.hasTag(at::Tag::inplace_view) ||
          schema.name() == "aten::detach_") &&
        all_constant &&
        !flat_arg_fake_tensors.empty() &&
        !has_symints &&
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
          stack, arguments_begin, num_arguments,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (is_our_fake(t, mode)) {
              auto constant = t.unsafeGetTensorImpl()->constant();
              if (constant) {
                tensor_memo[constant->unsafeGetTensorImpl()] = t;
                return *constant;
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
          stack, returns_begin, num_returns,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (!may_turn_const(t))
              all_outputs_const = false;
            return std::nullopt;
          });

      if (all_outputs_const) {
        for_each_tensor(
            stack, returns_begin, num_returns,
            [&](const at::Tensor& t) -> std::optional<at::Tensor> {
              if (!may_turn_const(t))
                return std::nullopt;
              auto cloned = std::make_shared<at::Tensor>(t.clone());

              auto memo_it = tensor_memo.find(t.unsafeGetTensorImpl());
              if (memo_it != tensor_memo.end()) {
                auto& orig_fake = memo_it->second;
                orig_fake.unsafeGetTensorImpl()->set_constant(
                    std::move(cloned));
                add_constant_storage_mapping(orig_fake, mode);
                return orig_fake;
              }
              auto fake = real_tensor_to_fake(t, mode);
              fake.unsafeGetTensorImpl()->set_constant(std::move(cloned));
              add_constant_storage_mapping(fake, mode);
              return fake;
            });
        return;
      }

      // outputs too large to keep as constants
      // invalidate all constants that might alias the output tensors
      for_each_tensor(
          stack, returns_begin, num_returns,
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (t.defined() && !t.is_fake())
              invalidate_constant_aliases(t, mode);
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
  // this is already taken care of by adding @py_impl(DispatchKey.Fake) to all HOPs
  // when the dispatcher for HOPs is hit (which happens before fake fallback),
  // it will route to the proper fake kernel
  // THIS BEHAVIOUR IS DIFFERENT THAN TODAY'S PYTHON IMPL

  invalidate_written_to_constants(
      op, stack, arguments_begin, num_arguments, flat_arg_fake_tensors, mode);

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

  // for ops with symbolic sizes, try decompositions before the meta kernel
  // THIS IS CALLING BACK TO PYTHON
  // we need to re-entry only for the specific op being decomposed
  // since @register_decomposition can register the same fn as both decomp
  // and meta kernel, so re-entering the decomp for the same op will infinite loop

  // but sub-ops are different ops so they need to be able to use their own decomps
  bool is_same_op_reentry = mode && mode->decomposing_op_ == &op;
  if (has_symints && !cpp_meta_supports_symint(op) && mode &&
      !is_same_op_reentry) {
    if (mode->decomp_fn_) {
      auto prev_decomposing = mode->decomposing_op_;
      mode->decomposing_op_ = &op; // keep track of current op we are decomping
      bool found = false;
      try {
        found = mode->decomp_fn_(&op, stack);
      } catch (...) {
        mode->decomposing_op_ = prev_decomposing;
        throw;
      }
      mode->decomposing_op_ = prev_decomposing;
      if (found) {
        wrap_meta_outputs_with_default_device_logic();
        return;
      }
    }

    // try CIA decomposition. remove Fake from the redispatch keyset so this op
    // doesn't re-enter fakeFallback (which would infinite-loop)
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

  // prim ops with Meta kernels: redispatch to the Meta kernel with Fake
  // removed from the keyset (so this op doesn't re-enter fakeFallback)
  // prims can already handle Fake so we just need to re dispatch
  // sub ops need to enter though so we keep Fake in TLS
  // this matches Python's `with self: func.prim_meta_impl(...)` pattern
  if (schema.name().rfind("prims::", 0) == 0 &&
      op.hasKernelForDispatchKey(c10::DispatchKey::Meta)) {
    // Python calls prim_meta_impl directly so scalar args stay as Python
    // floats/ints. In C++, the dispatcher wraps them as tensors with default
    // dtypes (float64 for floats, int64 for ints) before we get here, causing
    // dtype mismatches. Get the target dtype from non-0-dim non-bool tensors.
    // If all tensors are 0-dim, use the first whose dtype isn't a default
    // wrapping dtype (float64/int64) since those are likely wrapped scalars.

    // ed: there's like a correct way to do the IValue conversion here
    // fix this later
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
    ks = ks | c10::DispatchKeySet(c10::DispatchKey::Meta);
    op.redispatchBoxed(ks, stack);
    wrap_meta_outputs_with_default_device_logic();
    return;
  }

  // TODO: profiles

  // TODO: infer fake kernel

  // TODO: user-registered fake implementations (torch.library.register_fake)

  // Handlers registered via register_op_impl
  // idk if this is right because im using a try catch pattern for this
  // and i also made a wrapper to be able to pass in "FakeTensorMode" to the op impls
  // this seemed like the least intrusive way to do it for now but idk
  // probalby should revist
  bool has_meta_kernel = op.hasKernelForDispatchKey(c10::DispatchKey::Meta);
  if (!has_meta_kernel && mode && mode->op_impl_fn_) {
    if (mode->op_impl_fn_(&op, stack)) {
      return;
    }
  }

  auto device_from_args = rewrite_device_args_to_meta(
      stack, arguments_begin, num_arguments, schema);
  if (!common_device.has_value()) {
    common_device = device_from_args;
  }
  if (!common_device.has_value()) {
    common_device = c10::Device(c10::DeviceType::CPU);
  }

  if (can_generate_trivial_fake_impl(schema)) {
    // do nothing for mutable ops that only modify out tensor
    stack->resize(arguments_begin);
    return;
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

    // Meta kernel failed. Try the Python op_impl handler as a fallback
    // (e.g. _local_scalar_dense has a Meta kernel that raises but a Python
    // handler that creates unbacked symbolic values).
    stack->resize(arguments_begin);
    for (const auto& arg : saved_args) {
      stack->push_back(arg);
    }

    if (mode && mode->op_impl_fn_) {
      if (mode->op_impl_fn_(&op, stack)) {
        return;
      }
    }

    // Python handler didn't handle it either. For NotImplementedError,
    // try the unsafe fallback. For other errors, rethrow.
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
