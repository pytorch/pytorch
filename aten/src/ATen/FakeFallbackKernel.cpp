#include <ATen/Context.h>
#include <ATen/FakeTensor.h>
#include <ATen/FakeTensorDispatchTables.h>
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
#include <string>
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
// Python Meta registrations
// that torch._meta_registrations installs as Meta kernels run here like C++
// ones.
//
// Sparse invariant checks read index data that Meta tensors do not have. Like
// Python, this fallback disables the process-wide Context setting around sparse
// redispatch when it was enabled, restoring it on normal or exceptional exit.
// This keeps the existing global policy without introducing a Context override.
// The setting is a non-atomic optional<bool>, and this path can run without the
// GIL. Concurrent reads/writes can race: A disables checks, B observes false
// and skips suppression, then A restores true while B's kernel still runs.
// Real sparse constructors on other threads may also see checks disabled.
// The scope guard provides restoration for serial calls, not thread isolation.
// A thread-safe policy would need a thread-local override integrated with the
// Context getters/setters, or synchronization covering all readers and writers.
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
//  - symbolic inputs (sizes, strides or scalars), for which Python decomposes
//    most ops that lack a C++ Meta kernel supporting them;
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
// TODO: native common-device resolution snapshots fake_tensor_prefer_device_type
// at mode construction to avoid GIL/Python lookups on device conflicts. Python
// op_impls, including foreach and sparse compressed constructors, still read
// live config. Changing None to "cuda" after construction can make CPU/CUDA add
// raise while _foreach_add chooses CUDA; changing "cuda" to None reverses that
// behavior. We assume the preference stays fixed for a mode's lifetime and have
// not found real use cases requiring changes during that lifetime. We accept
// this divergence rather than pay for live lookups or propagate the snapshot
// through Python handlers.
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
// TODO: rngprims are not in PrimMeta. Philox also needs the unsupported
// lift_fresh path to construct its offset tensor.

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

// Whether ivalue is or holds a symbolic number
bool is_symbolic(const c10::IValue& ivalue) {
  if (ivalue.isSymInt() || ivalue.isSymFloat() || ivalue.isSymBool()) {
    return true;
  }
  if (ivalue.isList()) {
    const auto elements = ivalue.toListRef();
    return std::any_of(elements.begin(), elements.end(), is_symbolic);
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
bool is_tensor_constructor(const c10::FunctionSchema& schema) {
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
  return is_tensor_constructor(op.schema()) ||
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
        (dispatchKeySet &
         c10::DispatchKeySet(
             c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Fake))
            .add(c10::DispatchKey::Meta),
        stack);
    return;
  }

  for (const auto idx : c10::irange(num_arguments)) {
    const auto& ivalue = (*stack)[arguments_begin + idx];
    if (is_symbolic(ivalue)) {
      unsupported(op, "symbolic shapes");
    }
  }
  for (const auto* arguments : {&schema.arguments(), &schema.returns()}) {
    for (const auto& argument : *arguments) {
      if (has_nested_tensors(argument.type())) {
        unsupported(op, "nested containers of tensors or Any");
      }
    }
  }

  // Lifts and scalar-only calls are handled before validation in Python;
  // those paths are added separately.
  bool has_mode_fakes = false;
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        has_mode_fakes = has_mode_fakes ||
            (t.defined() && t.is_fake() &&
             t.unsafeGetTensorImpl()->fake_tensor_mode() == mode);
      });
  if (!has_mode_fakes && !is_tensor_constructor(schema)) {
    unsupported(op, "ops without fake tensor inputs");
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
        if (t.is_fake() && (input_mode == mode || input_mode == nullptr)) {
          return std::nullopt;
        }
        const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
        const auto raise = [&](std::string_view what) {
          interpreter->raise_python_error(
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
          if (impl->is_symbolic_wrapped_number()) {
            unsupported(op, "symbolic shapes");
          }
          return;
        }
        // Their Meta kernels would make tensors of the wrong TensorImpl type
        if (impl->is_quantized() || impl->is_nested() || impl->is_mkldnn()) {
          unsupported(op, "quantized, nested or mkldnn inputs");
        }
        if (impl->has_symbolic_sizes_strides()) {
          unsupported(op, "symbolic shapes");
        }
        const auto input_mode = impl->fake_tensor_mode();
        if (input_mode == nullptr) {
          unsupported(op, "fake tensors without a mode");
        }
        if (impl->real_tensor() != nullptr) {
          unsupported(op, "propagate_real_tensors");
        }
        if (mode->get_constant(impl) != nullptr) {
          unsupported(op, "constant inputs");
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
  // Mirror FakeTensor._find_common_device, with the device preference captured
  // at mode construction. Probe without resolving conflicts for
  // op_implementations, which may choose devices themselves. Resolve only when
  // wrapping a Meta output, so kernel errors take precedence.
  const bool cpu_zero_dim_defers = schema.name() != "aten::nextafter" ||
      !schema.overload_name().empty();
  const bool allows_cpu = schema.name() == "aten::_foreach_copy" &&
      schema.overload_name().empty();
  const bool allows_meta =
      schema.name() == "aten::add_" && schema.overload_name() == "Tensor";
  const auto find_common_device =
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
      if (mode->prefer_device_type_.has_value()) {
        const auto& preferred = *mode->prefer_device_type_;
        const bool common_has_preferred =
            c10::DeviceTypeName(common_device->type(), /*lower_case=*/true)
                .find(preferred) != std::string::npos;
        const bool input_has_preferred =
            c10::DeviceTypeName(input_device.type(), /*lower_case=*/true)
                .find(preferred) != std::string::npos;
        if (common_has_preferred != input_has_preferred) {
          if (input_has_preferred) {
            common_device = input_device;
            common_is_cpu_zero_dim = is_cpu_zero_dim;
          }
          continue;
        }
      }
      (*c10::impl::getGlobalPyInterpreter())
          ->fake_raise_device_mismatch(op, *common_device, input_device);
    }
    return common_device;
  };
  using at::impl::FakeDispatchCategory;
  const auto categories =
      at::impl::fakeDispatchTableCategories(op.operator_name());
  const auto has = [&](FakeDispatchCategory category) {
    return at::impl::fakeDispatchCategoriesHave(categories, category);
  };
  if (!has(FakeDispatchCategory::Meta) &&
      !cpp_meta_supports_symint(op, categories)) {
    const bool has_cia =
        op.hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd);
    // Probe every native CIA op too: export writes py_kernels directly,
    // bypassing the registration hook that sets the PythonCIA bit.
    const bool try_decomp =
        (has(FakeDispatchCategory::StaticShapeDecomp) && !has_sparse_inputs) ||
        has(FakeDispatchCategory::PythonCIA) || has_cia;
    if (try_decomp) {
      const auto& interpreter = *c10::impl::getGlobalPyInterpreter();
      auto decompose = [&] {
        bool has_python_cia = false;
        if (interpreter->fake_try_decomp(
                op, stack, /*has_symbolic_sizes=*/false, &has_python_cia)) {
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
            mode, [&] { return interpreter->fake_try_prim_meta(op, stack); })) {
      return;
    }
  }
  if (has(FakeDispatchCategory::CustomOpImpl) &&
      (*c10::impl::getGlobalPyInterpreter())
          ->fake_try_custom_op_impl(op, stack, mode, /*real=*/nullptr)) {
    return;
  }
  // Integer indexing works without unbacked sizes; bool indexing calls nonzero.
  const bool is_index_tensor =
      schema.name() == "aten::index" && schema.overload_name() == "Tensor";
  const bool is_static_repeat_interleave =
      schema.name() == "aten::repeat_interleave" &&
      schema.overload_name() == "Tensor" &&
      (*stack)[arguments_begin + 1].isInt();
  const bool allocates_unbacked = schema.name() == "aten::_spdiags" ||
      schema.name() == "aten::_pack_padded_sequence" ||
      (schema.name() == "aten::_padded_dense_to_jagged_forward" &&
       (*stack)[arguments_begin + 2].isNone());
  if (allocates_unbacked ||
      (!is_index_tensor && !is_static_repeat_interleave &&
       (op.hasTag(at::Tag::dynamic_output_shape) ||
        op.hasTag(at::Tag::data_dependent_output)))) {
    unsupported(op, "ops with data-dependent outputs");
  }
  // Constructors finalize their own output devices.
  const auto common_device = input_devices.empty()
      ? std::optional(c10::Device(c10::kCPU))
      : find_common_device(/*resolve=*/false);
  if (has(FakeDispatchCategory::OpImpl) || has_op_impl_predicate(op)) {
    if ((*c10::impl::getGlobalPyInterpreter())
            ->fake_try_op_impl(op, stack, mode, common_device)) {
      return;
    }
  }
  if (input_devices.empty()) {
    unsupported(op, "ops without fake tensor inputs");
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
  c10::impl::ExcludeDispatchKeyGuard exclude_dispatch(c10::DispatchKeySet(
      {c10::DispatchKey::Fake,
       c10::DispatchKey::Python,
       c10::DispatchKey::PreDispatch,
       c10::DispatchKey::PythonTLSSnapshot}));
  c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
  c10::impl::InKernelInvocationGuard in_kernel;
  {
    // Uses global state; see Note [C++ fake fallback] for the concurrency limits.
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

  // Note [In-place conversion of Meta outputs]
  // The ownership argument for normal dense kernels is that an output is
  // either an existing fake, handled below without reinitializing it, or a
  // TensorImpl created by this invocation. A new view has its own TensorImpl
  // even though it shares storage with an input. Concurrent calls on the same
  // fake input therefore do not initialize the same output TensorImpl. Extra
  // references from Python or return-value boxing do not imply that another
  // invocation can return that object; releasing the GIL does not change this
  // ownership argument.
  //
  // Shared storage is a separate concern: a view's storage was already marked
  // for fake use when its input was initialized. Reapplying an established
  // storage marking must be a no-op, rather than another write. Copying the
  // output TensorImpl would not isolate its shared StorageImpl anyway.
  //
  // A custom Meta kernel could instead cache a plain Meta tensor and return
  // that same TensorImpl across calls. In-place conversion changes that cached
  // object, so cross-mode reuse can fail and concurrent first conversions would
  // need synchronization. This is outside the dense ownership argument above;
  // no such retained-output pattern was established in the supported in-tree
  // dense kernels.
  //
  // COO _values/_indices are in-tree examples of retained outputs.
  // _from_meta_and_device copies the outer sparse TensorImpl but shares its
  // component TensorImpls with the caller's plain Meta tensor. These accessors
  // return the stored components directly, so init_cpp_fake below also marks
  // the caller's component fake. ADInplaceOrView calls Fake before as_view
  // copies the result; that later copy cannot undo the shared mutation.
  // The outer Meta tensor remains non-fake, so converting it again can succeed,
  // but accessing the marked component through another mode or device fails
  // the mode/device checks below. Concurrent first accesses to the same
  // component also need synchronization. CSR accessors return .alias(), giving
  // the fallback a separate TensorImpl and avoiding this component mutation.
  //
  // We keep the original TensorImpl instead of allocating a second wrapper.
  // Shallow-copying retained outputs would protect their original TensorImpls
  // without copying data. A per-call copy can address that isolation alone,
  // but preserving Python's converter identity across calls would also need
  // a per-mode memo, synchronized lookup/creation, and weak-reference lifetime
  // handling, including TensorImpl address reuse. Ordinary dense outputs do
  // not require that machinery. The current allocation-saving choice also
  // converts retained custom and COO outputs in place, accepting the mutation,
  // cross-mode/device failures and concurrent-conversion limitation above.
  // It does not provide Python's wrapping/memoization semantics.
  std::optional<c10::Device> output_common_device;
  std::unordered_set<const c10::TensorImpl*> converted_outputs;
  auto wrap = [&](const at::Tensor& t, bool is_alias) {
    if (!t.defined()) {
      return;
    }
    if (!output_common_device.has_value()) {
      output_common_device = find_common_device(/*resolve=*/true);
    }
    auto* impl = t.unsafeGetTensorImpl();
    // The same Meta tensor can occur several times in the return values.
    if (converted_outputs.contains(impl)) {
      return;
    }
    if (t.is_fake() && impl->fake_tensor_mode() == mode) {
      // device() reports meta inside the kernel invocation scope.
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
      return;
    }
    TORCH_CHECK(
        !t.is_fake(),
        "Meta kernel returned a fake tensor from another mode");
    // New views and shallow copies inside the kernel are plain Meta tensors.
    at::init_cpp_fake(
        impl, device_argument.value_or(*output_common_device), mode);
    if (!is_alias && t.requires_grad()) {
      // Backend kernels run below autograd, but a Meta kernel can copy
      // requires_grad from an input. Existing fakes returned above, and
      // init_cpp_fake checked this output's eligibility before we change its
      // gradient state. Validation is per output, not a rollback of earlier
      // outputs if a later one is rejected.
      t.set_requires_grad(false);
    }
    impl->set_real_tensor(nullptr);
    impl->set_fake_dispatch_keys(output_dispatch_keys);
    if (device_argument.has_value() &&
        device_argument != output_common_device) {
      converted_outputs.insert(impl);
    }
  };
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    const bool is_alias = schema.returns()[i].alias_info() != nullptr;
    for_each_tensor(stack, returns_begin + i, 1, [&](const at::Tensor& t) {
      wrap(t, is_alias);
    });
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // namespace
