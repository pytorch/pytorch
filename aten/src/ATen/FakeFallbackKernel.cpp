#include <ATen/FakeTensor.h>
#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <algorithm>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

// Note [C++ fake fallback]
// The C++ counterpart of FakeTensorMode.__torch_dispatch__ for C++ fakes. It
// runs an op whose tensor inputs are fakes of one mode on one device: the op
// runs on the inputs' meta storage through its Meta kernel, and the new
// outputs become fakes of that mode on that device, as Python's
// wrap_meta_outputs_with_default_device_logic does. Python Meta registrations
// that torch._meta_registrations installs as Meta kernels run here like C++
// ones.
//
// So that ops do not give results Python's FakeTensorMode would not, it raises
// NotImplementedError where Python runs something other than the Meta kernel,
// apart from the known divergences in the TODOs below:
//  - ops Python runs through a Python implementation (op_implementations,
//    custom op fake kernels, prim metas, Python CompositeImplicitAutograd
//    kernels; the fake_try_* callbacks in c10/core/impl/PyInterpreter.h);
//  - ops Python decomposes under static shapes with a torch._decomp
//    decomposition (FakeDispatchCategory::StaticShapeDecomp), which can
//    disagree with the Meta kernel (log_sigmoid_forward's checks the device,
//    which reads meta here);
//  - CompositeImplicitAutograd ops, which reach this only below autograd:
//    Python decomposes them with the mode active, so their sub-ops are
//    fake-dispatched, and here they would run on the Meta key instead;
//  - ops run under the Python dispatcher, whose Python-dispatcher-only Meta
//    registrations do not run here;
//  - symbolic inputs (sizes, strides or scalars), for which Python decomposes
//    most ops that lack a C++ Meta kernel supporting them;
//  - tensors nested deeper than Tensor[] (Tensor[][], dicts of tensors);
//  - sparse inputs;
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
// TODO: new outputs other than aliases do not require grad here, as they are
// made below autograd, while Python's FakeTensor keeps the requires_grad a Meta
// kernel set on its meta output.

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
// Tensor?[], the forms for_each_tensor and the output wrapping visit
bool has_nested_tensors(const c10::TypePtr& type) {
  auto unwrap = [](const c10::TypePtr& t) {
    const auto optional = t->cast<c10::OptionalType>();
    return optional ? optional->getElementType() : t;
  };
  auto t = unwrap(type);
  if (const auto list = t->cast<c10::ListType>()) {
    t = unwrap(list->getElementType());
  }
  return !t->isSubtypeOf(*c10::TensorType::get()) && contains_tensor_types(t);
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
      schema.returns()[0].type()->kind() == c10::TypeKind::TensorType;
}

// See fakeHasPythonFakeImpl; categories is the op's fakeDispatchTableCategories
bool has_python_fake_impl(const c10::OperatorHandle& op, uint32_t categories) {
  using at::impl::FakeDispatchCategory;
  for (const auto category :
       {FakeDispatchCategory::OpImpl,
        FakeDispatchCategory::CustomOpImpl,
        FakeDispatchCategory::PrimMeta,
        FakeDispatchCategory::PythonCIA}) {
    if (at::impl::fakeDispatchCategoriesHave(categories, category)) {
      return true;
    }
  }
  return op.hasTag(at::Tag::dynamic_output_shape) ||
      op.hasTag(at::Tag::data_dependent_output) ||
      is_tensor_constructor(op.schema()) ||
      (op.operator_name().name.starts_with("aten::_foreach_") &&
       op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta));
}

} // namespace

namespace at::impl {

bool fakeHasPythonFakeImpl(const c10::OperatorHandle& op) {
  return has_python_fake_impl(op, fakeDispatchTableCategories(op.operator_name()));
}

} // namespace at::impl

namespace {

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet /*dispatchKeySet*/,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;
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
  // Python's FakeTensorMode is out of the FAKE dispatch mode slot while its
  // __torch_dispatch__ runs, so the code the op runs (a Python Meta kernel
  // calling active_fake_mode(), say) sees no fake mode; so here
  UnsetFakeTensorModeGuard unset_mode;

  for (const auto idx : c10::irange(num_arguments)) {
    const auto& ivalue = (*stack)[arguments_begin + idx];
    if (ivalue.isDevice()) {
      unsupported(op, "device arguments");
    }
    if (is_symbolic(ivalue)) {
      unsupported(op, "symbolic shapes");
    }
  }
  using at::impl::FakeDispatchCategory;
  const auto categories =
      at::impl::fakeDispatchTableCategories(op.operator_name());
  if (has_python_fake_impl(op, categories)) {
    unsupported(op, "ops with a Python fake implementation");
  }
  if (op.hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd)) {
    unsupported(op, "CompositeImplicitAutograd ops below autograd");
  }
  for (const auto* arguments : {&schema.arguments(), &schema.returns()}) {
    for (const auto& argument : *arguments) {
      if (has_nested_tensors(argument.type())) {
        unsupported(op, "nested containers of tensors");
      }
    }
  }

  // The op's tensor inputs, so output wrapping can tell an input returned as
  // an output (by in-place and out= ops) from a new tensor
  std::unordered_set<const c10::TensorImpl*> input_impls;
  std::optional<c10::Device> device;
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
        if (!t.is_fake()) {
          unsupported(op, "non-fake tensor inputs");
        }
        // Their Meta kernels would make tensors of the wrong TensorImpl type
        if (impl->is_quantized() || impl->is_nested() || impl->is_mkldnn()) {
          unsupported(op, "quantized, nested or mkldnn inputs");
        }
        if (impl->is_sparse() || impl->is_sparse_compressed()) {
          unsupported(op, "sparse inputs");
        }
        if (impl->has_symbolic_sizes_strides()) {
          unsupported(op, "symbolic shapes");
        }
        const auto input_mode = impl->fake_tensor_mode();
        if (input_mode == nullptr) {
          unsupported(op, "fake tensors without a mode");
        }
        if (input_mode != mode) {
          unsupported(op, "fake tensors of another mode");
        }
        if (impl->real_tensor() != nullptr) {
          unsupported(op, "propagate_real_tensors");
        }
        if (mode->get_constant(impl) != nullptr) {
          unsupported(op, "constant inputs");
        }
        input_impls.insert(impl);
        const auto fake_device = impl->fake_device();
        TORCH_INTERNAL_ASSERT(fake_device.has_value());
        if (!device.has_value()) {
          device = fake_device;
        } else if (*device != *fake_device) {
          // Including a zero-dim CPU tensor with others, which Python allows
          unsupported(
              op,
              c10::str(
                  "inputs on different devices (",
                  *device,
                  " and ",
                  *fake_device,
                  ")"));
        }
      });
  if (!device.has_value()) {
    unsupported(op, "ops without fake tensor inputs");
  }
  if (at::impl::fakeDispatchCategoriesHave(
          categories, FakeDispatchCategory::StaticShapeDecomp) &&
      !at::impl::fakeDispatchCategoriesHave(
          categories, FakeDispatchCategory::Meta)) {
    unsupported(op, "ops Python decomposes");
  }

  // As in Python, the outputs of alias and detach of a fake keep its recorded
  // dispatch keys (see Note [C++ fake dispatch keys]), and other new outputs
  // have none
  std::optional<c10::DispatchKeySet> output_dispatch_keys;
  if (schema.overload_name().empty() &&
      (schema.name() == "aten::alias" || schema.name() == "aten::detach")) {
    const auto& self = (*stack)[arguments_begin].toTensor();
    output_dispatch_keys = self.unsafeGetTensorImpl()->fake_dispatch_keys();
  }

  // Run the Meta kernel with the inputs reporting the meta device (see Note
  // [C++ in_kernel_invocation]), as Python's in_kernel_invocation_manager
  // does, and with the ops it runs, on meta tensors, kept out of fake dispatch
  c10::impl::ExcludeDispatchKeyGuard exclude_fake(c10::DispatchKey::Fake);
  c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
  c10::impl::InKernelInvocationGuard in_kernel;
  try {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta), stack);
  } catch (const c10::NotImplementedError& e) {
    // Python runs these through maybe_run_unsafe_fallback
    unsupported(
        op, "ops without a working Meta kernel", e.what_without_backtrace());
  }

  // The fakes that replaced outputs, so that a tensor returned twice gets the
  // same fake, as Python's memo of the fakes it makes of meta outputs gives
  std::unordered_map<const c10::TensorImpl*, at::Tensor> replaced;
  // Returns the fake that replaces t on the stack, if t is not itself made one
  auto wrap_output = [&](const at::Tensor& t,
                         bool is_alias) -> std::optional<at::Tensor> {
    if (!t.defined()) {
      return std::nullopt;
    }
    auto* impl = t.unsafeGetTensorImpl();
    if (const auto it = replaced.find(impl); it != replaced.end()) {
      return it->second;
    }
    // An input returned as an output (by in-place and out= ops, or an op that
    // returns its input without an alias annotation) is kept as is, as Python
    // does. Its stored fake device is read, as device() reports meta here.
    if (input_impls.contains(impl)) {
      TORCH_INTERNAL_ASSERT(impl->fake_device() == device);
      return std::nullopt;
    }
    std::optional<at::Tensor> out;
    if (c10::raw::intrusive_ptr::use_count(impl) > 1) {
      // A tensor held elsewhere: a Python Meta kernel's output, which its
      // Python object refers to, is new on each call; it is not changed, and a
      // shallow copy becomes the fake
      out = at::Tensor(impl->shallow_copy_and_detach(
          impl->version_counter(), impl->allow_tensor_metadata_change()));
      replaced.emplace(impl, *out);
      impl = out->unsafeGetTensorImpl();
    } else if (!is_alias && t.requires_grad()) {
      // Backend kernels run below autograd, but a Meta kernel can copy
      // requires_grad from an input
      t.set_requires_grad(false);
    }
    // A copy a Meta kernel made of a fake input inherited the input's fake
    // state (see init_cpp_fake), where Python would see a plain meta tensor
    at::init_cpp_fake(impl, *device, mode);
    impl->set_real_tensor(nullptr);
    impl->set_fake_dispatch_keys(output_dispatch_keys);
    return out;
  };
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    const bool is_alias = schema.returns()[i].alias_info() != nullptr;
    auto& ivalue = (*stack)[returns_begin + i];
    if (ivalue.isTensor()) {
      if (auto out = wrap_output(ivalue.toTensor(), is_alias)) {
        ivalue = std::move(*out);
      }
    } else if (ivalue.isList()) {
      auto list = ivalue.toList();
      // Read through toListRef, as reading the list copies its elements
      const auto elements = ivalue.toListRef();
      for (const auto j : c10::irange(elements.size())) {
        if (!elements[j].isTensor()) {
          continue;
        }
        if (auto out = wrap_output(elements[j].toTensor(), is_alias)) {
          list.set(j, std::move(*out));
        }
      }
    }
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // namespace
