#include <ATen/Context.h>
#include <ATen/FakeTensor.h>
#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/SmallVector.h>
#include <c10/util/irange.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/FakeTensorMode.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/schema_info.h>
#include <torch/library.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
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
// Lifts and numbers-as-tensors calls first try Python's constant-creation path.
// Deterministic ops on constant inputs run on their real constants, and writes
// that cannot be evaluated invalidate constants sharing the written storage.
// Symbolic inputs use Python's fast implementations before decomposition;
// these implementations enter the mode only for their slow reference paths.
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
//  - ops run under the Python dispatcher, whose Python-dispatcher-only Meta
//    registrations do not run here;
//  - tensors nested deeper than Tensor[] (Tensor[][], dicts of tensors), or
//  Any;
//  - and the input checks below.
// Callers that need to tell these apart from other NotImplementedErrors match
// the message unsupported() raises.
//
// TODO: native common-device resolution snapshots
// fake_tensor_prefer_device_type at mode construction to avoid GIL/Python
// lookups on device conflicts. Python op_impls, including foreach and sparse
// compressed constructors, still read live config. Changing None to "cuda"
// after construction can make CPU/CUDA add raise while _foreach_add chooses
// CUDA; changing "cuda" to None reverses that behavior. We assume the
// preference stays fixed for a mode's lifetime and have not found real use
// cases requiring changes during that lifetime. We accept this divergence
// rather than pay for live lookups or propagate the snapshot through Python
// handlers.
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

namespace torch::detail {
namespace {

using at::impl::may_turn_const;

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

// Copy tensor lists before replacing elements so caller-owned lists stay
// intact.
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
      // Releasing a displaced mode can run callbacks that must see restored
      // TLS.
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
    return std::any_of(
        elements.begin(), elements.end(), [&](const auto& value) {
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
  return op.hasTag(at::Tag::dynamic_output_shape) ||
      op.hasTag(at::Tag::data_dependent_output) ||
      _is_tensor_constructor(op.schema()) ||
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

enum class FakeTensorConversion {
  Default,
  Constant,
  PlainTensor,
  ClonedConstant,
};

// use this pattern to avoid decref after pyinterpreter finalization
// this is non owning. Mirrors DEFINE_CACHING_PYTHON_IMPORT_GETTER in
// torch/csrc/autograd/python_variable.cpp.
#if IS_PYBIND_2_13_PLUS
#define DEFINE_CACHED_PYTHON_IMPORT(name, import_expr)                     \
  py::handle name() {                                                      \
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> \
        storage;                                                           \
    return storage                                                         \
        .call_once_and_store_result(                                       \
            []() -> py::object { return import_expr; })                    \
        .get_stored();                                                     \
  }
#else
#define DEFINE_CACHED_PYTHON_IMPORT(name, import_expr)             \
  py::handle name() {                                              \
    static py::handle storage = py::object(import_expr).release(); \
    return storage;                                                \
  }
#endif

DEFINE_CACHED_PYTHON_IMPORT(
    get_meta_table,
    py::module::import("torch._decomp").attr("meta_table"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_decomposition_table,
    py::module::import("torch._decomp").attr("decomposition_table"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_schema_info,
    py::module::import("torch._subclasses.fake_tensor").attr("get_schema_info"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_check_for_subclass_arg,
    py::module::import("torch._subclasses.fake_tensor")
        .attr("_check_for_subclass_arg"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_fast_op_impls,
    py::module::import("torch._subclasses.fake_impls")
        .attr("get_fast_op_impls"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_op_implementations_checks,
    py::module::import("torch._subclasses.fake_impls")
        .attr("op_implementations_checks"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_torch_decomp_decompositions,
    py::module::import("torch._subclasses.fake_tensor")
        .attr("torch_decomp_decompositions"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_run_fake_impl,
    py::module::import("torch._library.fake_impl").attr("run_fake_impl"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_has_fake_kernel,
    py::module::import("torch._library.utils").attr("has_fake_kernel"))
// fake_tensor_tls is a module-global threading.local, so caching the object is
// safe: attribute reads on it still resolve to the calling thread's value.
DEFINE_CACHED_PYTHON_IMPORT(
    get_fake_tensor_tls,
    py::module::import("torch._subclasses.fake_tensor").attr("fake_tensor_tls"))
DEFINE_CACHED_PYTHON_IMPORT(
    get_fake_tensor_device_mismatch_error,
    py::module::import("torch._subclasses.fake_tensor")
        .attr("FakeTensorDeviceMismatchError"))

#undef DEFINE_CACHED_PYTHON_IMPORT

// Note [C++ fake op_impl outputs]
// Under a C++ FakeTensorMode, Python op_implementations handlers may return raw
// Meta tensors and leave it to try_op_impl to finalize them:
//  - A returned Meta tensor that is not yet fake is wrapped through the mode's
//    converter on the op's common_device, or is an error without one.
//  - An already-fake output is kept as is, including its device.
//  - A handler that needs a different device per output must return finalized
//    fake tensors.
//  - Only a returned tensor or the immediate elements of a returned tuple/list
//    are converted, not arbitrary nested structures.
// Any other tensor, e.g. a real CPU tensor, is an error.
std::function<py::object(py::object)> make_fake_device_stamp(
    const c10::OperatorHandle& op,
    std::optional<c10::Device> common_device,
    py::object py_mode) {
  return [&op, common_device, py_mode = std::move(py_mode)](
             py::object obj) -> py::object {
    if (!THPVariable_Check(obj.ptr())) {
      return obj;
    }
    at::Tensor t = THPVariable_Unpack(obj.ptr());
    if (!t.defined() || t.is_fake()) {
      return obj;
    }
    TORCH_CHECK(
        t.is_meta(),
        "op_impl handler for ",
        op.operator_name(),
        " returned a non-fake ",
        t.device(),
        " tensor; handlers must return Meta or fake tensors, see "
        "Note [C++ fake op_impl outputs]");
    TORCH_CHECK(
        common_device.has_value(),
        "op_impl handler for ",
        op.operator_name(),
        " returned a Meta tensor for inputs on devices with no common device; "
        "handlers must return fake tensors for them, see "
        "Note [C++ fake op_impl outputs]");
    return py_mode.attr("fake_tensor_converter")
        .attr("from_meta_and_device")(py_mode, obj, py::cast(*common_device));
  };
}

// Apply `convert` to each returned tensor (tuple/list/single). Lists are
// mutated in place (e.g. the _foreach_* ops hand back raw meta tensors to
// stamp).
py::object apply_output_convert(
    py::object result,
    const std::function<py::object(py::object)>& convert) {
  if (!convert) {
    return result;
  }
  if (py::isinstance<py::tuple>(result)) {
    py::tuple tup = result.cast<py::tuple>();
    py::tuple converted(tup.size());
    for (const auto i : c10::irange(tup.size())) {
      converted[i] = convert(py::reinterpret_borrow<py::object>(tup[i]));
    }
    return converted;
  }
  if (py::isinstance<py::list>(result)) {
    py::list lst = result.cast<py::list>();
    // NOLINTNEXTLINE(modernize-loop-convert)
    for (const auto i : c10::irange(lst.size())) {
      lst[i] = convert(py::reinterpret_borrow<py::object>(lst[i]));
    }
    return result;
  }
  return convert(std::move(result));
}

// pushPyOutToStack can throw after pushing some of the outputs, so box them
// into a temporary stack and append it to the real one only on success.
torch::jit::Stack box_py_outputs(
    const c10::OperatorHandle& op,
    py::object out,
    const char* label) {
  torch::jit::Stack outputs;
  pushPyOutToStack(op, &outputs, std::move(out), label);
  return outputs;
}

// Runs call over the op's arguments on top of stack. If it returns
// NotImplemented, returns false. Otherwise replaces the arguments with its
// converted result as the op's outputs and returns true. On false or an
// exception (from call, convert or boxing), the stack is unchanged.
bool run_python_callback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::function<py::object(py::object, py::dict)>& call,
    const std::function<py::object(py::object)>& convert,
    const char* label) {
  const auto& schema = op.schema();
  auto arguments = torch::jit::pop(*stack, schema.arguments().size());
  bool committed = false;
  auto restore_args = c10::make_scope_exit([&]() {
    if (!committed) {
      for (auto& arg : arguments) {
        stack->push_back(std::move(arg));
      }
    }
  });
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  py::object result = call(args_kwargs.first, args_kwargs.second);
  if (result.is(py::handle(Py_NotImplemented))) {
    return false;
  }
  result = apply_output_convert(std::move(result), convert);
  auto outputs = box_py_outputs(op, std::move(result), label);
  stack->insert(
      stack->end(),
      std::make_move_iterator(outputs.begin()),
      std::make_move_iterator(outputs.end()));
  committed = true;
  return true;
}

// The current TLS with Fake no longer excluded, so ops run by a Python fake
// implementation dispatch back to Fake.
c10::impl::LocalDispatchKeySet fake_reenabled_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  tls.excluded_ = tls.excluded_.remove(c10::DispatchKey::Fake);
  return tls;
}

bool run_fake_python_callback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::function<py::object(py::object, py::dict)>& call,
    const std::function<py::object(py::object)>& convert,
    const char* label) {
  c10::impl::ForceDispatchKeyGuard guard(fake_reenabled_tls());
  return run_python_callback(op, stack, call, convert, label);
}

bool fake_run_with_mode_entered(
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    c10::function_ref<bool()> step) {
  // Keep threading.local data alive from entry to exit on native threads too.
  py::gil_scoped_acquire gil;
  py::object py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  py_mode.attr("__enter__")();
  bool done = false;
  try {
    py::gil_scoped_release no_gil;
    done = step();
  } catch (py::error_already_set& e) {
    py_mode.attr("__exit__")(e.type(), e.value(), e.trace());
    throw;
  } catch (...) {
    // Native exceptions have no Python exception object; __exit__ only
    // restores mode state.
    py_mode.attr("__exit__")(
        py::handle(PyExc_BaseException), py::none(), py::none());
    throw;
  }
  py_mode.attr("__exit__")(py::none(), py::none(), py::none());
  return done;
}

void fake_raise_device_mismatch(
    const c10::OperatorHandle& op,
    c10::Device common_device,
    c10::Device device) {
  py::gil_scoped_acquire gil;
  auto error_type = get_fake_tensor_device_mismatch_error();
  auto error = error_type(getTorchApiFunction(op), common_device, device);
  PyErr_SetObject(error_type.ptr(), error.ptr());
  throw py::error_already_set();
}

[[noreturn]] void raise_assertion_error(const std::string& msg) {
  py::gil_scoped_acquire gil;
  PyErr_SetString(PyExc_AssertionError, msg.c_str());
  throw py::error_already_set();
}

void invalidate_written_to_constants(
    const c10::OperatorHandle& op,
    const torch::jit::Stack& stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  py::gil_scoped_acquire gil;
  py::object schema_info_obj = get_schema_info()(getTorchApiFunction(op));
  auto& schema_info = schema_info_obj.cast<torch::utils::SchemaInfo&>();
  const auto arguments =
      torch::jit::last(stack, op.schema().arguments().size());
  for (const auto i : c10::irange(arguments.size())) {
    if (!arguments[i].isTensor()) {
      continue;
    }
    const auto& t = arguments[i].toTensor();
    if (!t.defined() || !t.is_fake() ||
        t.unsafeGetTensorImpl()->fake_tensor_mode() != mode) {
      continue;
    }
    auto constant = mode->get_constant(t.unsafeGetTensorImpl());
    if (constant && schema_info.is_mutable({c10::SchemaArgType::input, i})) {
      mode->invalidate_constant_aliases(
          constant->storage().unsafeGetStorageImpl());
    }
  }
}

bool try_fast_impl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  py::object fast_impl = get_fast_op_impls()().attr("get")(py_op);
  if (fast_impl.is_none()) {
    return false;
  }
  py::object py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  return run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        py::object out = fast_impl(py_mode, *args, **kwargs);
        TORCH_CHECK(
            !out.is(py::handle(Py_NotImplemented)),
            "fast implementation for ",
            op.operator_name(),
            " returned NotImplemented");
        return out;
      },
      /*convert=*/{},
      "fast implementation");
}

// Try a registered Python decomposition for op.
//
// This mirrors the decomposition-selection block in
// FakeTensorMode._dispatch_impl (torch/_subclasses/fake_tensor.py) and must be
// kept in sync with it, as must _register_static_shape_decomps in
// torch/_meta_registrations.py for its static shape case. This callback handles
// the meta_table exclusion, the decomposition_table lookup with its
// symbolic/static-shape and sparse eligibility, and the Python
// CompositeImplicitAutograd lookup. A caller must handle
// the cpp_meta_supports_symint and _unbacked_special_fake_handling_ops
// exclusions, and afterwards the native CompositeImplicitAutograd fallback.
bool try_decomp(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool has_symbolic_sizes,
    bool* has_python_cia) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  py::dict py_kernels = py_op.attr("py_kernels");
  py::object cia_key = py::cast(c10::DispatchKey::CompositeImplicitAutograd);
  *has_python_cia = py_kernels.contains(cia_key);

  if (get_meta_table().contains(py_op)) {
    return false;
  }
  py::object decomp_fn = py::none();
  py::handle decomp_table = get_decomposition_table();
  if (decomp_table.contains(py_op)) {
    bool use_decomp = has_symbolic_sizes;
    // Match Python FakeTensorMode: under static shapes, only torch._decomp
    // decompositions run here and only for non-sparse inputs; other registered
    // decompositions run exclusively under symbolic sizes.
    if (!use_decomp && get_torch_decomp_decompositions()(py_op).cast<bool>()) {
      use_decomp = true;
      auto is_sparse = [](const at::Tensor& t) {
        switch (t.layout()) {
          case c10::kSparse:
          case c10::kSparseCsr:
          case c10::kSparseCsc:
          case c10::kSparseBsr:
          case c10::kSparseBsc:
            return true;
          default:
            return false;
        }
      };
      auto args = torch::jit::last(*stack, op.schema().arguments().size());
      auto is_sparse_arg = [&](const c10::IValue& v) {
        return v.isTensor() && v.toTensor().defined() &&
            is_sparse(v.toTensor());
      };
      // isList() covers both Tensor[] and Tensor?[].
      for (const auto& arg : args) {
        if (is_sparse_arg(arg) ||
            (arg.isList() &&
             std::any_of(
                 arg.toListRef().begin(),
                 arg.toListRef().end(),
                 is_sparse_arg))) {
          use_decomp = false;
          break;
        }
      }
    }
    if (use_decomp) {
      decomp_fn = decomp_table[py_op];
    }
  }

  bool is_cia = false;
  if (decomp_fn.is_none()) {
    if (!*has_python_cia) {
      return false;
    }
    decomp_fn = py_kernels[cia_key];
    is_cia = true;
  }

  return run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        py::object out = decomp_fn(*args, **kwargs);
        // As in Python, a CIA kernel may decline with NotImplemented (see
        // OpOverload.decompose), but a decomposition_table result is the op's
        // result. Python would return that NotImplemented as the op's output;
        // raise instead of silently trying another implementation.
        TORCH_CHECK(
            is_cia || !out.is(py::handle(Py_NotImplemented)),
            "decomposition for ",
            op.operator_name(),
            " returned NotImplemented");
        return out;
      },
      /*convert=*/{},
      "decomposition");
}

bool run_fake_impl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    PyObject* real) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  py::object py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  py::handle py_real = real ? py::handle(real) : py::none();
  return run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        return get_run_fake_impl()(py_mode, py_op, args, kwargs, py_real);
      },
      /*convert=*/{},
      "fake_impl");
}

// Try the Python op_implementations handlers for op. Like FakeTensorMode, this
// walks op_implementations_checks, which covers both the exact-op dict lookup
// (dispatch_to_op_implementations_dict) and the predicate-based checks
// (constructors, like-ops, etc.); a handler returning NotImplemented passes to
// the next one. Outputs are finalized per Note [C++ fake op_impl outputs].
bool try_op_impl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    std::optional<c10::Device> common_device) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  py::object py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  return run_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) -> py::object {
        for (auto item : get_op_implementations_checks()) {
          py::tuple check_impl = item.cast<py::tuple>();
          py::object run_impl_check = check_impl[0];
          py::object op_impl = check_impl[1];
          if (!run_impl_check(py_op).cast<bool>()) {
            continue;
          }
          py::object result;
          {
            c10::impl::ForceDispatchKeyGuard guard(fake_reenabled_tls());
            result = op_impl(py_mode, py_op, *args, **kwargs);
          }
          if (!result.is(py::handle(Py_NotImplemented))) {
            return result;
          }
        }
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
      },
      make_fake_device_stamp(op, common_device, py_mode),
      "op_impl");
}

// Try op's prim_meta_impl (prims meta rule) if it defines one.
bool prim_meta_impl(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  if (!py::hasattr(py_op, "prim_meta_impl")) {
    return false;
  }
  py::object prim_meta_impl = py_op.attr("prim_meta_impl");

  return run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        py::object out = prim_meta_impl(*args, **kwargs);
        TORCH_CHECK(
            !out.is(py::handle(Py_NotImplemented)),
            "prim_meta_impl for ",
            op.operator_name(),
            " returned NotImplemented");
        return out;
      },
      /*convert=*/{},
      "prim_meta_impl");
}

bool fake_infer_from_real_out(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    PyObject* real) {
  py::gil_scoped_acquire gil;
  py::handle py_op = getTorchApiFunction(op);
  auto py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  if (get_has_fake_kernel()(py_op).cast<bool>()) {
    return false;
  }
  return run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        return py::module::import("torch._subclasses.fake_tensor")
            .attr("infer_fake_from_real_out")(py_mode, py_op, py::handle(real));
      },
      /*convert=*/{},
      "infer_fake_from_real_out");
}

void maybe_run_unsafe_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    bool has_symbolic_sizes,
    const std::exception_ptr& error) {
  py::gil_scoped_acquire gil;
  py::object py_error = py::none();
  if (error) {
    torch::translate_exception_to_python(error);
    py::error_already_set translated;
    if (!translated.matches(PyExc_NotImplementedError)) {
      // Translation has consumed the original Python exception's error state.
      translated.restore();
      torch::throw_python_error();
    }
    py_error = translated.value();
  }
  auto py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  run_fake_python_callback(
      op,
      stack,
      [&](const py::object& args, const py::dict& kwargs) {
        return py_mode.attr("maybe_run_unsafe_fallback")(
            getTorchApiFunction(op),
            args,
            kwargs,
            has_symbolic_sizes,
            py_error);
      },
      /*convert=*/{},
      "maybe_run_unsafe_fallback");
}

c10::intrusive_ptr<c10::TensorImpl> from_real_tensor(
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    const c10::intrusive_ptr<c10::TensorImpl>& real_impl,
    FakeTensorConversion conversion = FakeTensorConversion::Default) {
  // The initial constant attempt already holds the GIL. A nested guard keeps
  // it held; standalone conversion calls acquire it here.
  py::gil_scoped_acquire gil;
  const bool only_plain_tensor =
      conversion == FakeTensorConversion::PlainTensor ||
      conversion == FakeTensorConversion::ClonedConstant;
  // Subclasses with custom dispatch run at Python before Fake.
  if (auto* obj = real_impl->pyobj_slot()->load_pyobj()) {
    auto* type = Py_TYPE(obj);
    if (only_plain_tensor &&
        type != reinterpret_cast<PyTypeObject*>(THPVariableClass)) {
      return nullptr;
    }
    TORCH_CHECK(
        type == reinterpret_cast<PyTypeObject*>(THPVariableClass) ||
            PyType_IsSubtype(
                type, reinterpret_cast<PyTypeObject*>(ParameterClass)),
        "C++ FakeTensorMode does not support converting tensor subclass ",
        type->tp_name);
  }
  at::Tensor real(real_impl);
  if (conversion == FakeTensorConversion::ClonedConstant) {
    if (!at::impl::may_turn_const(real)) {
      return nullptr;
    }
    // Check before cloning: a clone can erase the original Python type.
    c10::impl::ExcludeDispatchKeyGuard no_dispatch(c10::DispatchKeySet(
        {c10::DispatchKey::Python,
         c10::DispatchKey::PreDispatch,
         c10::DispatchKey::Fake,
         c10::DispatchKey::PythonTLSSnapshot}));
    py::gil_scoped_release no_gil;
    real = real.clone();
  }
  const bool make_constant = conversion == FakeTensorConversion::Constant ||
      conversion == FakeTensorConversion::ClonedConstant;
  py::object py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  auto converter = py::reinterpret_borrow<py::object>(
      mode->fake_tensor_converter_->ptr(getPyInterpreter()));

  at::Tensor fake_tensor;
  {
    // Exclude Fake so from_real_tensor's meta conversion doesn't re-enter the
    // fake fallback (the meta factory ops it runs are otherwise plain). With
    // allow_non_fake_inputs this also re-fakeifies fakes from another mode; the
    // describer then reads the source's fake device, which Fake exclusion no
    // longer hides (see Note [C++ in_kernel_invocation]). from_real_tensor
    // must only query the source's metadata, build outputs with an explicit
    // meta device, and run ops on the fakes it makes only under
    // in_kernel_invocation_manager (as its set_ for a shared storage does).
    c10::impl::ExcludeDispatchKeyGuard exclude_fake(
        {c10::DispatchKeySet(c10::DispatchKey::Fake)});
    auto obj = converter.attr("from_real_tensor")(
        py_mode, real, py::arg("make_constant") = make_constant);
    fake_tensor = py::cast<at::Tensor>(std::move(obj));
  }
  return fake_tensor.unsafeReleaseIntrusivePtr();
}

bool try_initial_constant(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode,
    bool makes_constant,
    bool is_lift) {
  py::gil_scoped_acquire gil;
  const auto num_arguments = op.schema().arguments().size();
  const auto arguments = torch::jit::last(*stack, num_arguments);
  const auto check_input = [&](const at::Tensor& t) {
    if (!t.defined() || t.is_fake()) {
      return;
    }
    auto* obj = t.unsafeGetTensorImpl()->pyobj_slot()->load_pyobj();
    if (obj &&
        Py_TYPE(obj) != reinterpret_cast<PyTypeObject*>(THPVariableClass)) {
      // Share Python's Parameter/subclass rules. An output-only check is too
      // late: real execution may mutate an input or erase its subclass type.
      TORCH_CHECK_NOT_IMPLEMENTED(
          !get_check_for_subclass_arg()(py::handle(obj)).cast<bool>(),
          "The C++ fake tensor fallback does not support tensor subclass inputs "
          "yet, found in ",
          op.operator_name());
    }
  };
  for (const auto& argument : arguments) {
    if (argument.isTensor()) {
      check_input(argument.toTensor());
    } else if (argument.isTensorList() || argument.isOptionalTensorList()) {
      for (const auto& element : argument.toListRef()) {
        if (element.isTensor()) {
          check_input(element.toTensor());
        }
      }
    }
  }
  const auto may_be_plain_tensor = [](const at::Tensor& t) {
    return t.defined() && !t.is_fake() &&
        !t.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python);
  };
  c10::intrusive_ptr<c10::TensorImpl> fake;
  if (makes_constant) {
    torch::jit::Stack outputs(arguments.begin(), arguments.end());
    {
      // The selected mode is uninstalled, but a foreign fake still dispatches
      // through its owning mode, as it does in Python's constant attempt.
      auto tls = c10::impl::tls_local_dispatch_key_set();
      tls.excluded_ = tls.excluded_.remove(c10::DispatchKey::Fake);
      c10::impl::ForceDispatchKeyGuard guard(tls);
      py::gil_scoped_release no_gil;
      op.callBoxed(outputs);
    }
    if (outputs.size() == 1 && outputs[0].isTensor() &&
        may_be_plain_tensor(outputs[0].toTensor())) {
      fake = from_real_tensor(
          mode,
          outputs[0].toTensor().getIntrusivePtr(),
          FakeTensorConversion::ClonedConstant);
    }
  }
  // Larger plain lifts reuse the converter's existing alias and memo behavior.
  if (!fake && is_lift && may_be_plain_tensor(arguments[0].toTensor())) {
    fake = from_real_tensor(
        mode,
        arguments[0].toTensor().getIntrusivePtr(),
        FakeTensorConversion::PlainTensor);
  }
  if (!fake) {
    return false;
  }
  torch::jit::drop(*stack, num_arguments);
  torch::jit::push(*stack, at::Tensor(std::move(fake)));
  return true;
}

bool allow_non_fake_inputs(const std::shared_ptr<c10::FakeTensorMode>& mode) {
  py::gil_scoped_acquire gil;
  py::object override_val =
      get_fake_tensor_tls().attr("allow_non_fake_inputs_override");
  if (!override_val.is_none()) {
    return override_val.cast<bool>();
  }
  return mode->allow_non_fake_inputs_;
}

PyObject* fake_run_real_op(
    const c10::OperatorHandle& op,
    const torch::jit::Stack& fake_args,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  py::gil_scoped_acquire gil;
  auto py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  auto args_kwargs = parseIValuesToPyArgsKwargs(
      op, torch::jit::last(fake_args, op.schema().arguments().size()).vec());
  py::object result = py::module::import("torch._subclasses.fake_tensor")
                          .attr("run_real_op")(
                              py_mode,
                              getTorchApiFunction(op),
                              args_kwargs.first,
                              args_kwargs.second);
  return result.is_none() ? nullptr : result.release().ptr();
}

void propagate_real_tensors(
    const c10::OperatorHandle& op,
    PyObject* real,
    torch::jit::Stack* stack,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  py::gil_scoped_acquire gil;
  auto py_mode = fake_tensor::getCppFakeTensorModePyObj(mode);
  auto num_returns = op.schema().returns().size();
  // Pass the outputs the way a Python op returns them: None, the single value,
  // or a tuple.
  py::object py_fake_out = torch::jit::createPyObjectForStack(
      torch::jit::last(*stack, num_returns).vec());
  py::object result =
      py::module::import("torch._subclasses.fake_tensor")
          .attr("propagate_real_tensors")(
              py_mode, getTorchApiFunction(op), py::handle(real), py_fake_out);
  auto outputs =
      box_py_outputs(op, std::move(result), "propagate_real_tensors");
  torch::jit::drop(stack, num_returns);
  stack->insert(
      stack->end(),
      std::make_move_iterator(outputs.begin()),
      std::make_move_iterator(outputs.end()));
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

  for (const auto* arguments : {&schema.arguments(), &schema.returns()}) {
    for (const auto& argument : *arguments) {
      if (has_nested_tensors(argument.type())) {
        unsupported(op, "nested containers of tensors or Any");
      }
    }
  }

  bool has_mode_fakes = false;
  bool has_python_inputs = false;
  const bool symbolic_sizes = std::any_of(
      stack->end() - num_arguments, stack->end(), [&](const auto& value) {
        return has_symbolic_sizes(value, mode);
      });
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined()) {
          return;
        }
        const auto* impl = t.unsafeGetTensorImpl();
        has_mode_fakes =
            has_mode_fakes || (t.is_fake() && impl->fake_tensor_mode() == mode);
        has_python_inputs = has_python_inputs ||
            (!t.is_fake() && impl->pyobj_slot()->load_pyobj() != nullptr);
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
          mode->avoid_device_init_;
    }
    return schema.name() == "prims::device_put" &&
        schema.overload_name().empty() && mode->avoid_device_init_;
  };
  const bool makes_constant = !has_mode_fakes &&
      (is_lift ||
       (!symbolic_sizes && should_allow_numbers_as_tensors() &&
        !(is_to_copy &&
          (*stack)[arguments_begin].toTensor().device().is_meta()) &&
        !avoiding_device_init()));
  if (has_python_inputs || makes_constant) {
    // Validate input subclasses before any real execution. Share the GIL scope
    // with the initial constant attempt, output checks, and lift conversion.
    if (try_initial_constant(op, stack, mode, makes_constant, is_lift)) {
      return;
    }
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
        const auto raise = [&](std::string_view what) {
          raise_assertion_error(
              c10::str(what, ". Found in ", op.operator_name()));
        };
        if (op.hasTag(at::Tag::inplace_view)) {
          raise("Can't call metadata mutating ops on non-Fake Tensor inputs");
        }
        if (!allow_non_fake_inputs(mode)) {
          raise(
              t.is_fake()
                  ? "Mixing fake modes NYI"
                  : "Please convert all Tensors to FakeTensors first or "
                    "instantiate FakeTensorMode with "
                    "'allow_non_fake_inputs'");
        }
        return at::Tensor(from_real_tensor(mode, t.getIntrusivePtr()));
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
          constants.emplace_back(
              t.unsafeGetTensorImpl(), at::Tensor(std::move(constant)));
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
            return at::Tensor(from_real_tensor(
                mode, t.getIntrusivePtr(), FakeTensorConversion::Constant));
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
    invalidate_written_to_constants(op, *stack, mode);
  }

  auto* interpreter = getPyInterpreter();
  c10::SafePyObject real(
      mode->propagate_real_tensors_ ? fake_run_real_op(op, *stack, mode)
                                    : nullptr,
      interpreter);
  const auto maybe_propagate_real_tensors = [&] {
    if (real.ptr(interpreter)) {
      propagate_real_tensors(op, real.ptr(interpreter), stack, mode);
    }
  };

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
    {
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
    }
    maybe_propagate_real_tensors();
    return;
  }
  // Mirror FakeTensor._find_common_device, with the device preference captured
  // at mode construction. Probe without resolving conflicts for
  // op_implementations, which may choose devices themselves. Resolve only when
  // wrapping a Meta output, so kernel errors take precedence.
  const bool cpu_zero_dim_defers =
      schema.name() != "aten::nextafter" || !schema.overload_name().empty();
  const bool allows_cpu =
      schema.name() == "aten::_foreach_copy" && schema.overload_name().empty();
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
      fake_raise_device_mismatch(op, *common_device, input_device);
    }
    return common_device;
  };
  using at::impl::FakeDispatchCategory;
  const auto categories =
      at::impl::fakeDispatchTableCategories(op.operator_name());
  const auto has = [&](FakeDispatchCategory category) {
    return at::impl::fakeDispatchCategoriesHave(categories, category);
  };
  if (symbolic_sizes && try_fast_impl(op, stack, mode)) {
    maybe_propagate_real_tensors();
    return;
  }
  // FakeTensorMode._unbacked_special_fake_handling_ops, applied to all
  // symbolic inputs, including backed symbols.
  const bool unbacked_special = (schema.overload_name().empty() &&
                                 (schema.name() == "aten::view" ||
                                  schema.name() == "aten::_unsafe_view")) ||
      (schema.name() == "aten::slice" && schema.overload_name() == "Tensor");
  if (!has(FakeDispatchCategory::Meta) &&
      !cpp_meta_supports_symint(op, categories) &&
      !(symbolic_sizes && unbacked_special)) {
    const bool has_cia =
        op.hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd);
    // Probe every native CIA op too: export writes py_kernels directly,
    // bypassing the registration hook that sets the PythonCIA bit.
    const bool should_try_decomp =
        (symbolic_sizes && has(FakeDispatchCategory::Decomp)) ||
        (has(FakeDispatchCategory::StaticShapeDecomp) && !has_sparse_inputs) ||
        has(FakeDispatchCategory::PythonCIA) || has_cia;
    if (should_try_decomp) {
      auto decompose = [&] {
        bool has_python_cia = false;
        if (try_decomp(op, stack, symbolic_sizes, &has_python_cia)) {
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
      if (fake_run_with_mode_entered(mode, decompose)) {
        maybe_propagate_real_tensors();
        return;
      }
    }
  }
  if (has(FakeDispatchCategory::PrimMeta)) {
    if (fake_run_with_mode_entered(
            mode, [&] { return prim_meta_impl(op, stack); })) {
      maybe_propagate_real_tensors();
      return;
    }
  }
  // Match library_utils.is_builtin before probing custom kernel registrations.
  const auto op_namespace = op.operator_name().getNamespace();
  const bool is_builtin = op_namespace &&
      (*op_namespace == "aten" || *op_namespace == "prim" ||
       *op_namespace == "prims");
  if (real.ptr(interpreter) && mode->shape_env_ && !is_builtin &&
      fake_infer_from_real_out(op, stack, mode, real.ptr(interpreter))) {
    maybe_propagate_real_tensors();
    return;
  }
  if (has(FakeDispatchCategory::CustomOpImpl) &&
      run_fake_impl(op, stack, mode, real.ptr(interpreter))) {
    maybe_propagate_real_tensors();
    return;
  }
  // Constructors finalize their own output devices.
  const auto common_device = input_devices.empty()
      ? std::optional(c10::Device(c10::kCPU))
      : _find_common_device(/*resolve=*/false);
  if (has(FakeDispatchCategory::OpImpl) || has_op_impl_predicate(op)) {
    if (try_op_impl(op, stack, mode, common_device)) {
      maybe_propagate_real_tensors();
      return;
    }
  }
  if (!op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta)) {
    maybe_run_unsafe_fallback(op, stack, mode, symbolic_sizes, nullptr);
    maybe_propagate_real_tensors();
    return;
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

  // Boxed kernels may consume their arguments even when they throw.
  auto arguments = torch::jit::last(*stack, num_arguments);
  c10::SmallVector<c10::IValue, 8> saved_args(
      arguments.begin(), arguments.end());
  // Run the (sparse) Meta kernel with the inputs reporting the meta device (see
  // Note [C++ in_kernel_invocation]), as Python's in_kernel_invocation_manager
  // does. Match _DisableTorchDispatch so nested ops bypass dispatch modes.
  try {
    c10::impl::ExcludeDispatchKeyGuard exclude_dispatch(c10::DispatchKeySet(
        {c10::DispatchKey::Fake,
         c10::DispatchKey::Python,
         c10::DispatchKey::PreDispatch,
         c10::DispatchKey::PythonTLSSnapshot}));
    c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
    c10::impl::InKernelInvocationGuard in_kernel;
    // Uses global state; see Note [C++ fake fallback] for the concurrency
    // limits.
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
    op.redispatchBoxed(meta_keyset, stack);
  } catch (...) {
    auto error = std::current_exception();
    stack->resize(arguments_begin);
    stack->insert(stack->end(), saved_args.begin(), saved_args.end());
    maybe_run_unsafe_fallback(op, stack, mode, symbolic_sizes, error);
    // Python's NotImplementedError path returns the fallback directly.
    return;
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
  auto wrap = [&](const at::Tensor& t,
                  bool is_alias) -> std::optional<at::Tensor> {
    if (!t.defined()) {
      return std::nullopt;
    }
    if (input_devices.empty()) {
      if (!should_allow_numbers_as_tensors()) {
        auto name = schema.name();
        name.replace(name.find("::"), 2, ".");
        raise_assertion_error(c10::str(
            "Could not find common device for ",
            name,
            ".",
            schema.overload_name().empty() ? "default"
                                           : schema.overload_name()));
      }
      return at::Tensor(from_real_tensor(mode, t.getIntrusivePtr()));
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
        !t.is_fake(), "Meta kernel returned a fake tensor from another mode");
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
    return std::nullopt;
  };
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    const bool is_alias = schema.returns()[i].alias_info() != nullptr;
    replace_each_tensor(stack, returns_begin + i, 1, [&](const at::Tensor& t) {
      return wrap(t, is_alias);
    });
  }
  maybe_propagate_real_tensors();
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // namespace
} // namespace torch::detail
