#include <ATen/Context.h>
#include <ATen/FakeTensor.h>
#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <algorithm>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <utility>

// Note [C++ fake fallback]
// The C++ counterpart of FakeTensorMode.__torch_dispatch__ for C++ fakes. It
// runs an op whose tensor inputs are fakes of one mode, or a
// constructor under a mode.
// Decompositions, CompositeImplicitAutograd kernels and prim metas run with the
// mode entered and return outputs as is. Registered custom fake kernels then
// run through run_fake_impl, which enters the mode itself. Otherwise,
// op_implementations handlers run with the mode uninstalled. If none handles
// the op, it
// runs on the inputs' meta storage through its Meta kernel (its sparse Meta
// kernel for sparse inputs), and the new outputs become fakes of that mode on
// the device argument or the inputs' common device, as Python's
// wrap_meta_outputs_with_default_device_logic does.
// Non-fake inputs are converted through the mode's converter when allowed;
// metadata-mutating operations always require inputs owned by the mode.
// Lifts and numbers-as-tensors calls first try Python's constant-creation path.
// Deterministic ops on constant inputs run on their real constants, and writes
// that cannot be evaluated invalidate constants sharing the written storage.
// Symbolic inputs use Python's fast implementations before decomposition;
// these implementations enter the mode only for their slow reference paths.
// Python Meta registrations
// that torch._meta_registrations installs as Meta kernels run here like C++
// ones.
//
// The mode is uninstalled while this fallback runs, except for decomposition
// and prim metas.
// Other callbacks that need an active mode must run outside that scope.
//
// So that ops do not give results Python's FakeTensorMode would not, it raises
// NotImplementedError where Python runs something other than these,
// apart from the known divergences in the TODOs below:
//  - handlers that allocate unbacked sizes;
//  - ops run under the Python dispatcher, whose Python-dispatcher-only Meta
//    registrations do not run here;
//  - tensors nested deeper than Tensor[] (Tensor[][], dicts of tensors), or Any;
//  - sparse inputs to ops without a sparse Meta kernel; unlike Meta, the sparse
//    Meta keys have no backend fallback;
//  - ops whose Meta kernel raises c10::NotImplementedError (as the Meta
//    backend fallback does for an op without one), which Python then runs
//    through maybe_run_unsafe_fallback (a Python Meta kernel's
//    NotImplementedError is not caught here, so it propagates as is, without
//    the message callers match below);
//  - and the input checks below.
// Callers that need to tell these apart from other NotImplementedErrors match
// the message unsupported() raises.
//
// TODO: an op in torch._dynamo.config._custom_ops_profile runs its Meta kernel
// here, where Python runs the profile's generic fake kernel.
//
// TODO: in-place output conversion changes raw Meta tensors retained by a
// kernel and bypasses the converter's memo. On later calls they are existing
// fakes, so a changed common device or a prior device override can cause a
// mismatch where Python reuses a memoized wrapper.
//
// TODO: new outputs other than aliases do not require grad here, as they are
// made below autograd, while Python's FakeTensor keeps the requires_grad a Meta
// kernel set on its meta output.
//
// TODO: Python CIA kernels written directly into py_kernels for ops without
// native CIA kernels are not recorded in the tables. In-tree direct writers
// (torch.export) override ops with native CIA kernels, which we always probe.
//
// rngprims are not in PrimMeta and use their registered fake implementations.

namespace {

template <typename Fn>
void for_each_tensor(
    torch::jit::Stack* stack,
    size_t begin,
    size_t count,
    const Fn& fn) {
  for (const auto idx : c10::irange(count)) {
    const auto& ivalue = (*stack)[begin + idx];
    if (ivalue.isTensor()) {
      fn(ivalue.toTensor());
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      for (const c10::IValue& element : ivalue.toListRef()) {
        if (element.isTensor()) {
          fn(element.toTensor());
        }
      }
    }
  }
}

// Copy tensor lists before replacing elements so caller-owned lists stay intact.
template <typename Fn>
void replace_each_tensor(
    torch::jit::Stack* stack,
    size_t begin,
    size_t count,
    const Fn& fn) {
  for (const auto idx : c10::irange(count)) {
    auto& ivalue = (*stack)[begin + idx];
    if (ivalue.isTensor()) {
      if (auto out = fn(ivalue.toTensor())) {
        ivalue = std::move(*out);
      }
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      std::optional<c10::List<c10::IValue>> copy;
      const auto elements = ivalue.toListRef();
      for (const auto i : c10::irange(elements.size())) {
        if (!elements[i].isTensor()) {
          continue;
        }
        if (auto out = fn(elements[i].toTensor())) {
          if (!copy.has_value()) {
            copy = ivalue.toList().copy();
          }
          copy->set(i, std::move(*out));
        }
      }
      if (copy.has_value()) {
        ivalue = std::move(*copy);
      }
    }
  }
}

[[noreturn]] void unsupported(
    const c10::OperatorHandle& op,
    std::string_view what,
    std::string_view cause = {}) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "The C++ fake tensor fallback does not support ",
      what,
      " yet, found in ",
      op.operator_name(),
      cause.empty() ? "" : ": ",
      cause);
}

// Uninstalls this thread's C++ FakeTensorMode, if any, for its lifetime
class UnsetFakeTensorModeGuard {
 public:
  UnsetFakeTensorModeGuard()
      : mode_(c10::impl::FakeTensorModeTLS::get_state()),
        fake_included_(
            c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::Fake)) {
    if (mode_ != nullptr) {
      c10::impl::FakeTensorModeTLS::reset_state();
    }
  }
  ~UnsetFakeTensorModeGuard() {
    if (mode_ != nullptr) {
      // Releasing a displaced mode can run callbacks that must see restored TLS.
      auto prev_mode =
          c10::impl::FakeTensorModeTLS::create_state(std::move(mode_));
      c10::impl::tls_set_dispatch_key_included(
          c10::DispatchKey::Fake, fake_included_);
    }
  }
  UnsetFakeTensorModeGuard(const UnsetFakeTensorModeGuard&) = delete;
  UnsetFakeTensorModeGuard& operator=(const UnsetFakeTensorModeGuard&) = delete;
  UnsetFakeTensorModeGuard(UnsetFakeTensorModeGuard&&) = delete;
  UnsetFakeTensorModeGuard& operator=(UnsetFakeTensorModeGuard&&) = delete;

 private:
  std::shared_ptr<c10::FakeTensorMode> mode_;
  bool fake_included_;
};

// FakeTensorMode's symbolic-dispatch predicate counts SymInt, not SymFloat
// or SymBool, including when a SymInt was bound to a Tensor argument.
bool has_symbolic_sizes(
    const c10::IValue& ivalue,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  if (ivalue.isSymInt()) {
    return true;
  }
  if (ivalue.isTensor() && ivalue.toTensor().defined()) {
    const auto& t = ivalue.toTensor();
    const auto* impl = t.unsafeGetTensorImpl();
    return (t.is_fake() && impl->fake_tensor_mode() == mode &&
            impl->has_symbolic_sizes_strides()) ||
        (impl->is_symbolic_wrapped_number() && t.scalar_type() == at::kLong);
  }
  if (ivalue.isList()) {
    const auto elements = ivalue.toListRef();
    return std::any_of(elements.begin(), elements.end(), [&](const auto& value) {
      return has_symbolic_sizes(value, mode);
    });
  }
  return false;
}

bool contains_tensor_types(const c10::TypePtr& type) {
  if (type->isSubtypeOf(*c10::TensorType::get())) {
    return true;
  }
  const auto contained = type->containedTypes();
  return std::any_of(contained.begin(), contained.end(), contains_tensor_types);
}

// Whether type holds tensors other than as Tensor, Tensor?, Tensor[] or
// Tensor?[], the forms for_each_tensor and the output wrapping visit. Any can
// hide tensors, including inside otherwise tensor-free containers.
bool has_nested_tensors(const c10::TypePtr& type) {
  auto unwrap = [](const c10::TypePtr& t) {
    const auto optional = t->cast<c10::OptionalType>();
    return optional ? optional->getElementType() : t;
  };
  auto t = unwrap(type);
  if (const auto list = t->cast<c10::ListType>()) {
    t = unwrap(list->getElementType());
  }
  if (t->isSubtypeOf(*c10::TensorType::get())) {
    return false;
  }
  const auto contained = t->containedTypes();
  return t->kind() == c10::TypeKind::AnyType || contains_tensor_types(t) ||
      std::any_of(contained.begin(), contained.end(), has_nested_tensors);
}

// Same as Python's _is_tensor_constructor in fake_impls.py
bool _is_tensor_constructor(const c10::FunctionSchema& schema) {
  const auto& arguments = schema.arguments();
  return std::none_of(
             arguments.begin(),
             arguments.end(),
             [](const c10::Argument& a) {
               return contains_tensor_types(a.type());
             }) &&
      schema.returns().size() == 1 &&
      schema.returns()[0].type() == c10::TensorType::get();
}

bool has_op_impl_predicate(const c10::OperatorHandle& op) {
  return _is_tensor_constructor(op.schema()) ||
      (op.operator_name().name.starts_with("aten::_foreach_") &&
       op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta));
}

bool cpp_meta_supports_symint(
    const c10::OperatorHandle& op,
    uint32_t categories) {
  return op.hasTag(at::Tag::view_copy) ||
      at::impl::fakeDispatchCategoriesHave(
          categories, at::impl::FakeDispatchCategory::CppMetaSupportsSymint);
}

bool is_plain_tensor(const at::Tensor& t) {
  return t.defined() && !t.is_fake() &&
      !t.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python) &&
      (*c10::impl::getGlobalPyInterpreter())
          ->fake_has_plain_tensor_type(t.unsafeGetTensorImpl());
}

// FakeTensorMode.may_turn_const with CONSTANT_NUMEL_LIMIT == 1.
bool may_turn_const(const at::Tensor& t) {
  return t.numel() <= 1 && !t.is_fake() && !t.is_sparse() &&
      !t.unsafeGetTensorImpl()->is_sparse_compressed() && !t.device().is_meta();
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;
  const bool metadata_copy = schema.overload_name().empty() &&
      (schema.name() == "aten::detach" ||
       schema.name() == "aten::shallow_copy_data_");
  if (c10::impl::PythonDispatcherTLS::get_state() != nullptr) {
    unsupported(op, "the Python dispatcher");
  }

  // The active mode, or, as Python's FakeTensor dispatch enters a fake's
  // mode, the mode of the first fake input
  auto mode = c10::impl::FakeTensorModeTLS::get_state();
  if (mode == nullptr) {
    for_each_tensor(
        stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
          if (mode == nullptr && t.defined() && t.is_fake()) {
            mode = t.unsafeGetTensorImpl()->fake_tensor_mode();
          }
        });
  }
  if (mode == nullptr) {
    unsupported(op, "fake tensors without a mode");
  }
  if (mode->propagate_real_tensors_) {
    unsupported(op, "propagate_real_tensors");
  }
  // Python's FakeTensorMode is out of the FAKE dispatch mode slot while its
  // __torch_dispatch__ runs, apart from what it runs under `with self:`
  // (decompositions, say), so a Python Meta kernel calling active_fake_mode()
  // sees no fake mode; so here
  UnsetFakeTensorModeGuard unset_mode;
  // Python dispatch disables autocast, autograd and view bookkeeping before
  // running fake implementations.
  c10::impl::ExcludeDispatchKeyGuard exclude_autocast_and_autograd(
      c10::autocast_dispatch_keyset |
      c10::autograd_dispatch_keyset_with_ADInplaceOrView);

  // Python's _DISPATCH_HANDLE_DIRECTLY bypasses input conversion.
  if ((schema.overload_name().empty() &&
       (schema.name() == "aten::is_coalesced" ||
        schema.name() == "aten::dense_dim" ||
        schema.name() == "aten::sparse_dim")) ||
      (schema.name() == "profiler::_record_function_exit" &&
       schema.overload_name() == "_RecordFunction")) {
    c10::impl::ExcludeDispatchKeyGuard exclude_fake(c10::DispatchKey::Fake);
    c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
    c10::impl::InKernelInvocationGuard in_kernel;
    op.redispatchBoxed(
        dispatchKeySet &
            c10::DispatchKeySet(
                c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Fake),
        stack);
    return;
  }

  for (const auto* arguments : {&schema.arguments(), &schema.returns()}) {
    for (const auto& argument : *arguments) {
      if (has_nested_tensors(argument.type())) {
        unsupported(op, "nested containers of tensors or Any");
      }
    }
  }

  // Subclasses without a Python dispatch key still reach the native fallback.
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined()) {
          return;
        }
        const auto* impl = t.unsafeGetTensorImpl();
        if (!impl->is_wrapped_number() &&
            !(t.is_fake() && impl->fake_tensor_mode() == mode) &&
            (*c10::impl::getGlobalPyInterpreter())
                ->_check_for_subclass_arg(impl)) {
          unsupported(op, "tensor subclass inputs");
        }
      });
  bool has_mode_fakes = false;
  const bool symbolic_sizes = std::any_of(
      stack->end() - num_arguments,
      stack->end(),
      [&](const auto& value) { return has_symbolic_sizes(value, mode); });
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined()) {
          return;
        }
        const auto* impl = t.unsafeGetTensorImpl();
        has_mode_fakes = has_mode_fakes ||
            (t.is_fake() && impl->fake_tensor_mode() == mode);
      });
  // Python's whitelist uses the base name without namespace or overload.
  const auto should_allow_numbers_as_tensors = [&op] {
    std::string_view name = op.operator_name().name;
    if (const auto end = name.rfind("::"); end != std::string_view::npos) {
      name.remove_prefix(end + 2);
    }
    return at::should_allow_numbers_as_tensors(name);
  };
  const bool is_lift = schema.overload_name().empty() &&
      (schema.name() == "aten::lift_fresh" ||
       schema.name() == "aten::lift_fresh_copy");
  const bool is_to_copy =
      schema.name() == "aten::_to_copy" && schema.overload_name().empty();
  const auto avoiding_device_init = [&] {
    if (is_to_copy) {
      const auto& device =
          (*stack)[arguments_begin + *schema.argumentIndexWithName("device")];
      return device.isDevice() && !device.toDevice().is_cpu() &&
          (*c10::impl::getGlobalPyInterpreter())->fake_avoid_device_init(mode);
    }
    return schema.name() == "prims::device_put" &&
        schema.overload_name().empty() &&
        (*c10::impl::getGlobalPyInterpreter())->fake_avoid_device_init(mode);
  };
  const bool makes_constant = !has_mode_fakes &&
      (is_lift ||
       (!symbolic_sizes && should_allow_numbers_as_tensors() &&
        !(is_to_copy &&
          (*stack)[arguments_begin].toTensor().device().is_meta()) &&
        !avoiding_device_init()));
  if (makes_constant) {
    torch::jit::Stack outputs(stack->end() - num_arguments, stack->end());
    {
      // The selected mode is uninstalled, but a foreign fake still dispatches
      // through its owning mode, as it does in Python's constant attempt.
      auto tls = c10::impl::tls_local_dispatch_key_set();
      tls.excluded_ = tls.excluded_.remove(c10::DispatchKey::Fake);
      c10::impl::ForceDispatchKeyGuard guard(tls);
      op.callBoxed(outputs);
    }
    if (outputs.size() == 1 && outputs[0].isTensor() &&
        is_plain_tensor(outputs[0].toTensor()) &&
        may_turn_const(outputs[0].toTensor())) {
      at::Tensor constant;
      {
        c10::impl::ExcludeDispatchKeyGuard no_dispatch(c10::DispatchKeySet(
            {c10::DispatchKey::Python,
             c10::DispatchKey::PreDispatch,
             c10::DispatchKey::Fake,
             c10::DispatchKey::PythonTLSSnapshot}));
        constant = outputs[0].toTensor().clone();
      }
      at::Tensor fake(
          (*c10::impl::getGlobalPyInterpreter())
              ->from_real_tensor(
                  mode, constant.getIntrusivePtr(), /*make_constant=*/true));
      torch::jit::drop(*stack, num_arguments);
      torch::jit::push(*stack, std::move(fake));
      return;
    }
  }
  // Larger plain lifts use the converter's existing alias and memo behavior.
  if (is_lift && is_plain_tensor((*stack)[arguments_begin].toTensor())) {
    at::Tensor fake(
        (*c10::impl::getGlobalPyInterpreter())
            ->from_real_tensor(
                mode, (*stack)[arguments_begin].toTensor().getIntrusivePtr()));
    torch::jit::drop(*stack, num_arguments);
    torch::jit::push(*stack, std::move(fake));
    return;
  }
  replace_each_tensor(
      stack,
      arguments_begin,
      num_arguments,
      [&](const at::Tensor& t) -> std::optional<at::Tensor> {
        if (!t.defined()) {
          return std::nullopt;
        }
        auto* impl = t.unsafeGetTensorImpl();
        if (impl->is_wrapped_number()) {
          return std::nullopt;
        }
        const auto input_mode = impl->fake_tensor_mode();
        if (t.is_fake()) {
          if (input_mode == nullptr) {
            unsupported(op, "fake tensors without a mode");
          }
          if (input_mode == mode) {
            return std::nullopt;
          }
        }
        const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
        const auto raise = [&](std::string_view what) {
          interpreter->fake_raise_python_error(
              c10::impl::FakePythonError::AssertionError,
              c10::str(what, ". Found in ", op.operator_name()));
        };
        if (op.hasTag(at::Tag::inplace_view)) {
          raise("Can't call metadata mutating ops on non-Fake Tensor inputs");
        }
        if (!interpreter->allow_non_fake_inputs(mode)) {
          raise(
              t.is_fake() ? "Mixing fake modes NYI"
                          : "Please convert all Tensors to FakeTensors first or "
                            "instantiate FakeTensorMode with "
                            "'allow_non_fake_inputs'");
        }
        return at::Tensor(
            interpreter->from_real_tensor(mode, t.getIntrusivePtr()));
      });

  bool all_constant = true;
  bool any_constant = false;
  c10::SmallVector<std::pair<const c10::TensorImpl*, at::Tensor>, 4> constants;
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined() || !t.is_fake() ||
            t.unsafeGetTensorImpl()->fake_tensor_mode() != mode) {
          return;
        }
        auto constant = mode->get_constant(t.unsafeGetTensorImpl());
        all_constant = all_constant && constant != nullptr;
        any_constant = any_constant || constant != nullptr;
        if (constant) {
          constants.emplace_back(t.unsafeGetTensorImpl(), at::Tensor(std::move(constant)));
        }
      });
  if (any_constant && all_constant && !symbolic_sizes &&
      !op.hasTag(at::Tag::nondeterministic_seeded) &&
      (!op.hasTag(at::Tag::inplace_view) ||
       (schema.name() == "aten::detach_" && schema.overload_name().empty())) &&
      !avoiding_device_init() &&
      schema.name() != "aten::_nested_tensor_from_tensor_list") {
    torch::jit::Stack outputs(stack->end() - num_arguments, stack->end());
    replace_each_tensor(
        &outputs,
        0,
        num_arguments,
        [&](const at::Tensor& t) -> std::optional<at::Tensor> {
          if (t.defined() && t.is_fake() &&
              t.unsafeGetTensorImpl()->fake_tensor_mode() == mode) {
            const auto it = std::find_if(
                constants.begin(), constants.end(), [&](const auto& entry) {
                  return entry.first == t.unsafeGetTensorImpl();
                });
            TORCH_INTERNAL_ASSERT(it != constants.end());
            return it->second;
          }
          return std::nullopt;
        });
    {
      c10::impl::ExcludeDispatchKeyGuard no_dispatch(c10::DispatchKeySet(
          {c10::DispatchKey::Python,
           c10::DispatchKey::PreDispatch,
           c10::DispatchKey::Fake,
           c10::DispatchKey::PythonTLSSnapshot}));
      op.callBoxed(outputs);
    }
    bool all_outputs_constant = true;
    for_each_tensor(&outputs, 0, outputs.size(), [&](const at::Tensor& t) {
      if (t.defined() && !may_turn_const(t)) {
        all_outputs_constant = false;
      }
    });
    if (all_outputs_constant) {
      replace_each_tensor(
          &outputs,
          0,
          outputs.size(),
          [&](const at::Tensor& t) -> std::optional<at::Tensor> {
            if (!t.defined()) {
              return std::nullopt;
            }
            return at::Tensor(
                (*c10::impl::getGlobalPyInterpreter())
                    ->from_real_tensor(
                        mode, t.getIntrusivePtr(), /*make_constant=*/true));
          });
      torch::jit::drop(*stack, num_arguments);
      stack->insert(
          stack->end(),
          std::make_move_iterator(outputs.begin()),
          std::make_move_iterator(outputs.end()));
      return;
    }
    for_each_tensor(&outputs, 0, outputs.size(), [&](const at::Tensor& t) {
      if (t.defined()) {
        mode->invalidate_constant_aliases(t.storage().unsafeGetStorageImpl());
      }
    });
  }
  if (any_constant) {
    (*c10::impl::getGlobalPyInterpreter())
        ->invalidate_written_to_constants(op, *stack, mode);
  }

  // Input devices in argument order, paired with whether each is a CPU scalar.
  c10::SmallVector<std::pair<c10::Device, bool>, 4> input_devices;
  // The sparse inputs' Meta keys; Meta is added before redispatch.
  c10::DispatchKeySet meta_keyset;
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined()) {
          return;
        }
        auto* impl = t.unsafeGetTensorImpl();
        // A Python number passed for a Tensor argument, which the Meta kernel
        // takes as is, as Python's FakeTensorMode does
        if (impl->is_wrapped_number()) {
          return;
        }
        // Their Meta kernels would make tensors of the wrong TensorImpl type
        if (impl->is_quantized() || impl->is_nested() || impl->is_mkldnn()) {
          unsupported(op, "quantized, nested or mkldnn inputs");
        }
        const auto input_mode = impl->fake_tensor_mode();
        if (input_mode == nullptr) {
          unsupported(op, "fake tensors without a mode");
        }
        if (impl->real_tensor() != nullptr) {
          unsupported(op, "propagate_real_tensors");
        }
        const auto fake_device = impl->fake_device();
        TORCH_INTERNAL_ASSERT(fake_device.has_value());
        input_devices.emplace_back(
            *fake_device, fake_device->is_cpu() && impl->dim() == 0);
        if (impl->is_sparse()) {
          meta_keyset = meta_keyset.add(c10::DispatchKey::SparseMeta);
        } else if (impl->is_sparse_compressed()) {
          meta_keyset = meta_keyset.add(c10::DispatchKey::SparseCsrMeta);
        }
      });
  const bool has_sparse_inputs =
      meta_keyset.has(c10::DispatchKey::SparseMeta) ||
      meta_keyset.has(c10::DispatchKey::SparseCsrMeta);
  if (metadata_copy) {
    // These kernels only copy metadata. Run them on the fakes directly so
    // shallow_copy_data_ never replaces an existing fake inside a Meta kernel.
    c10::impl::ExcludeDispatchKeyGuard no_dispatch(c10::DispatchKeySet(
        {c10::DispatchKey::Fake,
         c10::DispatchKey::Python,
         c10::DispatchKey::PreDispatch,
         c10::DispatchKey::PythonTLSSnapshot}));
    auto self = (*stack)[arguments_begin].toTensor();
    auto result = schema.name() == "aten::detach"
        ? at::impl::fake_detach(self)
        : at::impl::fake_shallow_copy_data_(
              self, (*stack)[arguments_begin + 1].toTensor());
    torch::jit::drop(*stack, num_arguments);
    torch::jit::push(*stack, std::move(result));
    return;
  }
  // Mirror FakeTensor._find_common_device. Probe without resolving conflicts
  // for op_implementations, which may choose devices themselves. Resolve only
  // when wrapping a Meta output, so kernel errors take precedence.
  const bool cpu_zero_dim_defers = schema.name() != "aten::nextafter" ||
      !schema.overload_name().empty();
  const bool allows_cpu = schema.name() == "aten::_foreach_copy" &&
      schema.overload_name().empty();
  const bool allows_meta =
      schema.name() == "aten::add_" && schema.overload_name() == "Tensor";
  const auto _find_common_device =
      [&](bool resolve) -> std::optional<c10::Device> {
    std::optional<c10::Device> common_device;
    bool common_is_cpu_zero_dim = false;
    for (const auto& [input_device, is_cpu_zero_dim] : input_devices) {
      if (!common_device.has_value()) {
        common_device = input_device;
        common_is_cpu_zero_dim = is_cpu_zero_dim;
        continue;
      }
      if (input_device == *common_device) {
        common_is_cpu_zero_dim = common_is_cpu_zero_dim && is_cpu_zero_dim;
        continue;
      }
      if (cpu_zero_dim_defers && is_cpu_zero_dim) {
        continue;
      }
      if (cpu_zero_dim_defers && common_is_cpu_zero_dim) {
        common_device = input_device;
        common_is_cpu_zero_dim = is_cpu_zero_dim;
        continue;
      }
      if ((allows_cpu && (common_device->is_cpu() || input_device.is_cpu())) ||
          (allows_meta &&
           (common_device->is_meta() || input_device.is_meta()))) {
        continue;
      }
      if (!resolve) {
        return std::nullopt;
      }
      const auto resolved =
          (*c10::impl::getGlobalPyInterpreter())
              ->_resolve_device_conflict(op, *common_device, input_device);
      if (resolved == input_device) {
        common_device = input_device;
        common_is_cpu_zero_dim = is_cpu_zero_dim;
      }
    }
    return common_device;
  };
  using at::impl::FakeDispatchCategory;
  const auto categories =
      at::impl::fakeDispatchTableCategories(op.operator_name());
  const auto has = [&](FakeDispatchCategory category) {
    return at::impl::fakeDispatchCategoriesHave(categories, category);
  };
  if (symbolic_sizes &&
      (*c10::impl::getGlobalPyInterpreter())
          ->fake_try_fast_impl(op, stack, mode)) {
    return;
  }
  // FakeTensorMode._unbacked_special_fake_handling_ops, applied to all
  // symbolic inputs, including backed symbols.
  const bool unbacked_special =
      (schema.overload_name().empty() &&
       (schema.name() == "aten::view" || schema.name() == "aten::_unsafe_view")) ||
      (schema.name() == "aten::slice" && schema.overload_name() == "Tensor");
  if (!has(FakeDispatchCategory::Meta) &&
      !cpp_meta_supports_symint(op, categories) &&
      !(symbolic_sizes && unbacked_special)) {
    const bool has_cia =
        op.hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd);
    // Probe every native CIA op too: export writes py_kernels directly,
    // bypassing the registration hook that sets the PythonCIA bit.
    const bool try_decomp =
        (symbolic_sizes && has(FakeDispatchCategory::Decomp)) ||
        (has(FakeDispatchCategory::StaticShapeDecomp) && !has_sparse_inputs) ||
        has(FakeDispatchCategory::PythonCIA) || has_cia;
    if (try_decomp) {
      const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
      auto decompose = [&] {
        bool has_python_cia = false;
        if (interpreter->fake_try_decomp(
                op, stack, symbolic_sizes, &has_python_cia)) {
          return true;
        }
        // OpOverload.decompose does not try native CIA after a Python CIA
        // kernel returns NotImplemented.
        if (has_python_cia || !has_cia) {
          return false;
        }
        op.callBoxedForDispatchKey(
            c10::DispatchKey::CompositeImplicitAutograd, *stack);
        return true;
      };
      if (interpreter->fake_run_with_mode_entered(mode, decompose)) {
        return;
      }
    }
  }
  if (has(FakeDispatchCategory::PrimMeta)) {
    const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
    if (interpreter->fake_run_with_mode_entered(
            mode, [&] { return interpreter->prim_meta_impl(op, stack); })) {
      return;
    }
  }
  if (has(FakeDispatchCategory::CustomOpImpl) &&
      (*c10::impl::getGlobalPyInterpreter())
          ->run_fake_impl(op, stack, mode, /*real=*/nullptr)) {
    return;
  }
  // Integer indexing works without unbacked sizes; bool indexing calls nonzero.
  const bool is_index_tensor =
      schema.name() == "aten::index" && schema.overload_name() == "Tensor";
  const bool has_repeat_interleave_output_size =
      schema.name() == "aten::repeat_interleave" &&
      schema.overload_name() == "Tensor" &&
      ((*stack)[arguments_begin + 1].isInt() ||
       (*stack)[arguments_begin + 1].isSymInt());
  const bool allocates_unbacked = schema.name() == "aten::_spdiags" ||
      schema.name() == "aten::_pack_padded_sequence" ||
      (schema.name() == "aten::_padded_dense_to_jagged_forward" &&
       (*stack)[arguments_begin + 2].isNone());
  if (allocates_unbacked ||
      (!is_index_tensor && !has_repeat_interleave_output_size &&
       (op.hasTag(at::Tag::dynamic_output_shape) ||
        op.hasTag(at::Tag::data_dependent_output)))) {
    unsupported(op, "ops with data-dependent outputs");
  }
  // Constructors finalize their own output devices.
  const auto common_device = input_devices.empty()
      ? std::optional(c10::Device(c10::kCPU))
      : _find_common_device(/*resolve=*/false);
  if (has(FakeDispatchCategory::OpImpl) || has_op_impl_predicate(op)) {
    if ((*c10::impl::getGlobalPyInterpreter())
            ->fake_try_op_impl(op, stack, mode, common_device)) {
      return;
    }
  }
  if (has_sparse_inputs &&
      !op.hasComputedKernelForDispatchKey(
          meta_keyset.highestPriorityTypeId())) {
    // Give a clearer error without constructing the dispatcher's kernel table.
    unsupported(op, "sparse inputs to ops without a sparse Meta kernel");
  }
  meta_keyset = meta_keyset.add(c10::DispatchKey::Meta);

  // As in Python, alias keeps its input's recorded dispatch keys (see Note
  // [C++ fake dispatch keys]), and other new Meta outputs have none.
  std::optional<c10::DispatchKeySet> output_dispatch_keys;
  if (schema.overload_name().empty() && schema.name() == "aten::alias") {
    const auto& self = (*stack)[arguments_begin].toTensor();
    output_dispatch_keys = self.unsafeGetTensorImpl()->fake_dispatch_keys();
  }
  // __torch_dispatch__ receives non-default kwarg-only arguments as kwargs.
  std::optional<c10::Device> device_argument;
  if (const auto idx = schema.argumentIndexWithName("device")) {
    const auto& argument = schema.arguments()[*idx];
    const auto& ivalue = (*stack)[arguments_begin + *idx];
    if (argument.kwarg_only() && ivalue.isDevice() &&
        argument.default_value() != ivalue) {
      device_argument = ivalue.toDevice();
    }
  }

  // Run the (sparse) Meta kernel with the inputs reporting the meta device (see
  // Note [C++ in_kernel_invocation]), as Python's in_kernel_invocation_manager
  // does. Match _DisableTorchDispatch so nested ops bypass dispatch modes.
  {
    c10::impl::ExcludeDispatchKeyGuard exclude_dispatch(c10::DispatchKeySet(
        {c10::DispatchKey::Fake,
         c10::DispatchKey::Python,
         c10::DispatchKey::PreDispatch,
         c10::DispatchKey::PythonTLSSnapshot}));
    c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
    c10::impl::InKernelInvocationGuard in_kernel;
    // Match Python's global suppression of checks that read sparse index data.
    const bool suppress_invariants = has_sparse_inputs &&
        at::globalContext().checkSparseTensorInvariants().value_or(false);
    auto restore_invariants = c10::make_scope_exit([suppress_invariants] {
      if (suppress_invariants) {
        at::globalContext().setCheckSparseTensorInvariants(true);
      }
    });
    if (suppress_invariants) {
      at::globalContext().setCheckSparseTensorInvariants(false);
    }
    try {
      op.redispatchBoxed(meta_keyset, stack);
    } catch (const c10::NotImplementedError& e) {
      // Python runs these through maybe_run_unsafe_fallback
      unsupported(
          op, "ops without a working Meta kernel", e.what_without_backtrace());
    }
  }

  // Mark the original outputs fake, including tensors retained by a Meta
  // kernel. Unlike Python's wrapping, this also changes those retained tensors.
  std::optional<c10::Device> output_common_device;
  std::unordered_set<const c10::TensorImpl*> converted_outputs;
  auto wrap = [&](const at::Tensor& t, bool is_alias) -> std::optional<at::Tensor> {
    if (!t.defined()) {
      return std::nullopt;
    }
    if (input_devices.empty()) {
      const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
      if (!should_allow_numbers_as_tensors()) {
        auto name = schema.name();
        name.replace(name.find("::"), 2, ".");
        interpreter->fake_raise_python_error(
            c10::impl::FakePythonError::AssertionError,
            c10::str(
                "Could not find common device for ",
                name,
                ".",
                schema.overload_name().empty() ? "default"
                                               : schema.overload_name()));
      }
      return at::Tensor(interpreter->from_real_tensor(mode, t.getIntrusivePtr()));
    }
    if (!output_common_device.has_value()) {
      output_common_device = _find_common_device(/*resolve=*/true);
    }
    auto* impl = t.unsafeGetTensorImpl();
    // The same Meta tensor can occur several times in the return values.
    if (converted_outputs.contains(impl)) {
      return std::nullopt;
    }
    if (t.is_fake() && impl->fake_tensor_mode() == mode) {
      const auto fake_device = impl->fake_device();
      TORCH_CHECK(
          fake_device == output_common_device,
          "FakeTensor is wrapped to wrong device, found ",
          *fake_device,
          ", expected ",
          *output_common_device);
      if (output_dispatch_keys.has_value()) {
        impl->set_fake_dispatch_keys(output_dispatch_keys);
      }
      return std::nullopt;
    }
    TORCH_CHECK(
        !t.is_fake(),
        "Meta kernel returned a fake tensor from another mode");
    if (!is_alias && t.requires_grad()) {
      // Backend kernels run below autograd, but a Meta kernel can copy
      // requires_grad from an input
      t.set_requires_grad(false);
    }
    // New views and shallow copies inside the kernel are plain Meta tensors.
    at::init_cpp_fake(
        impl, device_argument.value_or(*output_common_device), mode);
    impl->set_real_tensor(nullptr);
    impl->set_fake_dispatch_keys(output_dispatch_keys);
    if (device_argument.has_value() &&
        device_argument != output_common_device) {
      converted_outputs.insert(impl);
    }
    return std::nullopt;
  };
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    const bool is_alias = schema.returns()[i].alias_info() != nullptr;
    replace_each_tensor(
        stack, returns_begin + i, 1, [&](const at::Tensor& t) {
          return wrap(t, is_alias);
        });
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // namespace
