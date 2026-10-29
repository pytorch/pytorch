#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

#include <ATen/DTensorState.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/NestedIntSymNodeImpl.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <ATen/functorch/BatchedTensorImpl.h>
#include <torch/library.h>

#include <c10/core/SafePyObject.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/tensor_new.h>

#include <c10/util/Synchronized.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/inductor/aoti_eager/kernel_holder.h>
#include <torch/csrc/utils/python_raii.h>

#include <c10/util/irange.h>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>

namespace py = pybind11;

namespace torch::impl::dispatch {

// Global storage for leaked Python filenames to ensure they remain valid
// for the lifetime of Library objects. We use unique_ptr<string> rather than
// plain string so that c_str() pointers handed to Library objects remain valid
// when the vector reallocates.
static c10::Synchronized<std::vector<std::unique_ptr<std::string>>>
    leaked_python_filenames_;

// NB: I'd like to index this on OperatorHandle, but I can't, as I can't
// guarantee that the main interpreter has finish doing all registrations before
// the other interpreters start banging on it
static c10::Synchronized<ska::flat_hash_map<
    c10::OperatorName,
    ska::flat_hash_map<c10::DispatchKey, std::shared_ptr<c10::SafePyObject>>>>
    python_registrations_;

static torch::Library::Kind parseKind(const std::string& k) {
  static std::unordered_map<std::string, torch::Library::Kind> kind_map = {
      {"DEF", torch::Library::DEF},
      {"IMPL", torch::Library::IMPL},
      {"FRAGMENT", torch::Library::FRAGMENT},
  };
  auto it = kind_map.find(k);
  TORCH_CHECK(it != kind_map.end(), "could not parse ", k);
  return it->second;
}
static c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
  static std::unordered_map<std::string, c10::AliasAnalysisKind> key_map = {
      {"CONSERVATIVE", c10::AliasAnalysisKind::CONSERVATIVE},
      {"FROM_SCHEMA", c10::AliasAnalysisKind::FROM_SCHEMA},
      {"PURE_FUNCTION", c10::AliasAnalysisKind::PURE_FUNCTION},
      {"", c10::AliasAnalysisKind::FROM_SCHEMA}, // default
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  return it->second;
}

template <typename Func>
inline static torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  if (key[0] != '\0') {
    return torch::dispatch(
        c10::parseDispatchKey(key), std::forward<Func>(raw_f));
  } else {
    torch::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}

// Whether autograd needs to see a call with these arguments: grad mode is on
// and a Tensor among them (or in a list among them) requires grad.
static bool any_requires_grad(c10::ArrayRef<c10::IValue> arguments) {
  if (!at::GradMode::is_enabled()) {
    return false;
  }
  for (const auto& ivalue : arguments) {
    if (ivalue.isTensor()) {
      if (ivalue.toTensor().requires_grad()) {
        return true;
      }
    } else if (ivalue.isList()) {
      for (const auto& elem : ivalue.toListRef()) {
        if (elem.isTensor() && elem.toTensor().requires_grad()) {
          return true;
        }
      }
    }
  }
  return false;
}

class PythonKernelHolder : public c10::OperatorKernel {
  c10::SafePyObject func_;
  c10::DispatchKey dispatch_key_;
  // If "with_keyset", then we expect a keyset as the first arg.
  bool with_keyset_;
  // If "with_op", then we expect the op as first arg (or second if keyset)
  bool with_op_;
  // An Autograd kernel that only redispatches below Autograd when grad mode
  // is off or no input requires grad, so the dispatcher can do that itself
  // without calling into Python (torch.library.custom_op's).
  bool skip_if_no_grad_;

 public:
  PythonKernelHolder(
      py::object func,
      c10::DispatchKey dispatch_key,
      bool with_keyset = false,
      bool with_op = false,
      bool skip_if_no_grad = false)
      : func_(func.release().ptr(), getPyInterpreter()),
        dispatch_key_(dispatch_key),
        with_keyset_(with_keyset),
        with_op_(with_op),
        skip_if_no_grad_(skip_if_no_grad) {}

  PyObject* func(c10::impl::PyInterpreter* interpreter) const {
    return func_.ptr(interpreter);
  }

  bool with_keyset() const {
    return with_keyset_;
  }

  bool with_op() const {
    return with_op_;
  }

  bool skip_if_no_grad() const {
    return skip_if_no_grad_;
  }

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack) {
    if (skip_if_no_grad_ &&
        !any_requires_grad(
            torch::jit::last(*stack, op.schema().arguments().size()))) {
      at::AutoDispatchBelowAutograd guard;
      op.redispatchBoxed(keyset & c10::after_autograd_keyset, stack);
      return;
    }

    // Figure out if we can handle it hermetically, or if we have
    // to double dispatch

    // If Torch Dispatch Mode is active, use its PyInterpreter for dispatch
    const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
    if (mode_stack_len > 0) {
      const auto& cur_torch_dispatch_mode_state =
          c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
      cur_torch_dispatch_mode_state->pyinterpreter()
          ->python_op_registration_trampoline(
              op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
      return;
    }

    const auto& schema = op.schema();
    const auto num_arguments = schema.arguments().size();

    // Otherwise, find a PyInterpreter on a Tensor if it has Python key (which
    // means it's a nontrivial tensor subclass)
    for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
      if (ivalue.isTensor()) {
        auto* impl = ivalue.unsafeToTensorImpl();
        if (impl->pyobj_slot()->load_pyobj() &&
            impl->key_set().has(at::DispatchKey::Python)) {
          (*c10::impl::getGlobalPyInterpreter())
              ->python_op_registration_trampoline(
                  op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
          return;
        }
      } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
        // NB: use toListRef as it doesn't induce refcount bumps
        // (toTensorListRef is not a thing)
        for (const auto& nv : ivalue.toListRef()) {
          if (nv.isNone()) {
            continue;
          }
          auto* impl = nv.unsafeToTensorImpl();
          if (impl->pyobj_slot()->load_pyobj() &&
              impl->key_set().has(at::DispatchKey::Python)) {
            (*c10::impl::getGlobalPyInterpreter())
                ->python_op_registration_trampoline(
                    op, dispatch_key_, keyset, stack, with_keyset_, with_op_);
            return;
          }
        }
      }
    }

    // Nothing requires the operator to be homed to a specific interpreter, so
    // run it on the current interpreter

    auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
    py::gil_scoped_acquire g;
    auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
    auto func =
        py::reinterpret_borrow<py::object>(func_.ptr(getPyInterpreter()));
    auto obj = with_op_ ? with_keyset_
            ? func(
                  keyset,
                  torch::detail::getTorchApiFunction(op),
                  *args_kwargs.first,
                  **args_kwargs.second)
            : func(
                  torch::detail::getTorchApiFunction(op),
                  *args_kwargs.first,
                  **args_kwargs.second)
        : with_keyset_ ? func(keyset, *args_kwargs.first, **args_kwargs.second)
                        : func(*args_kwargs.first, **args_kwargs.second);
    TORCH_CHECK_PYTHON(obj);
    pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
  }
};

// @todo sahanp: Afait only register is used in the codebase. This can be
// removed / simplified
static torch::_RegisterOrVerify register_or_verify() {
  return torch::_RegisterOrVerify::REGISTER;
}

static py::object ophandle_call_boxed(
    const c10::OperatorHandle& handle,
    const py::args& args,
    const py::kwargs& kwargs) {
  HANDLE_TH_ERRORS
  auto stack = torch::jit::createStackForSchema(
      handle.schema(),
      args,
      kwargs,
      /*self=*/std::nullopt);
  {
    pybind11::gil_scoped_release no_gil_guard;
    handle.callBoxed(stack);
  }
  return torch::jit::createPyObjectForStack(std::move(stack));
  END_HANDLE_TH_ERRORS_PYBIND
}

struct PyObjectDispatchSchema;

// Function that performs PyObject dispatch
struct PyObjectDispatchFunc {
  PyObject_HEAD
  PyObject* cpp_dispatch_fn;
  PyObject* cpp_redispatch_fn;
  // PyObject_New does not run C++ constructors, so keep non-trivial C++
  // state behind pointers and delete it in tp_dealloc.
  c10::OperatorHandle* handle;
  PyObjectDispatchSchema* schema_info;
  c10::impl::PyInterpreter* interpreter;
  vectorcallfunc vectorcall;
};

// Function that performs PyObject dispatch
struct PyObjectRedispatchFunc {
  PyObject_HEAD
  PyObject* cpp_redispatch_fn;
  // PyObject_New does not run C++ constructors, so keep non-trivial C++
  // state behind pointers and delete it in tp_dealloc.
  c10::OperatorHandle* handle;
  PyObjectDispatchSchema* schema_info;
  c10::impl::PyInterpreter* interpreter;
  vectorcallfunc vectorcall;
  // Redispatch below Autograd: mask the key set with after_autograd_keyset
  // and exclude the Autograd keys for the call.
  bool below_autograd;
};

// ORs the dispatch keys from `obj` into key_set
static void pyobject_dispatch_collect_keys(PyObject* obj, uint64_t& key_set) {
  if (C10_LIKELY(THPVariable_CheckExact(obj))) {
    key_set |=
        THPVariable_Unpack(obj).unsafeGetTensorImpl()->key_set().raw_repr();
    return;
  }
  if (PyList_CheckExact(obj)) {
    Py_ssize_t size = PyList_GET_SIZE(obj);
    for (const auto i : c10::irange(size)) {
      pyobject_dispatch_collect_keys(PyList_GET_ITEM(obj, i), key_set);
    }
    return;
  }
  if (THPVariable_Check(obj)) {
    key_set |=
        THPVariable_Unpack(obj).unsafeGetTensorImpl()->key_set().raw_repr();
  }
}

static c10::DispatchKeySet pyobject_dispatch_compute_keyset(
    const c10::DispatchKeyExtractor& extractor,
    PyObject* const* args,
    Py_ssize_t nargs) {
  uint64_t key_set = 0;
  c10::utils::for_each_set_bit(
      extractor.dispatchArgIndicesReverse(), [&](size_t reverse_arg_index) {
        pyobject_dispatch_collect_keys(
            args[nargs - 1 - static_cast<Py_ssize_t>(reverse_arg_index)],
            key_set);
      });
  return extractor.getDispatchKeySetFromRawDispatchKeySet(
      c10::DispatchKeySet::from_raw_repr(key_set));
}

static bool pyobject_dispatch_arg_has_torch_function(PyObject* arg) {
  if (THPVariable_CheckExact(arg)) {
    return false;
  }
  if (PyList_CheckExact(arg) || PyTuple_CheckExact(arg)) {
    const Py_ssize_t len = PySequence_Fast_GET_SIZE(arg);
    PyObject** items = PySequence_Fast_ITEMS(arg);
    for (const auto i : c10::irange(len)) {
      if (pyobject_dispatch_arg_has_torch_function(items[i])) {
        return true;
      }
    }
    return false;
  }
  return torch::check_has_torch_function(arg, /*ignore_mode=*/true);
}

static bool pyobject_dispatch_has_torch_function(
    PyObject* const* args,
    Py_ssize_t nargs,
    Py_ssize_t nkwargs,
    Py_ssize_t op_args_start) {
  if (!torch::torch_function_enabled()) {
    return false;
  }
  // Keep this a peek: callers consume the one-hop skip flag after deciding not
  // to hand off to the regular C++ Python op entry.
  if (torch::peek_should_skip_torch_function()) {
    return false;
  }
  if (at::impl::torch_function_mode_enabled()) {
    return true;
  }
  for (const auto i : c10::irange(op_args_start, nargs + nkwargs)) {
    if (pyobject_dispatch_arg_has_torch_function(args[i])) {
      return true;
    }
  }
  return false;
}

static PyObject* pyobject_dispatch_call_redispatch_cpp(
    PyObject* cpp_redispatch_fn,
    c10::DispatchKeySet key_set,
    PyObject* const* args,
    Py_ssize_t nargs,
    Py_ssize_t nkwargs,
    PyObject* kwnames,
    Py_ssize_t op_args_start) {
  py::object py_key_set = py::cast(key_set);
  c10::SmallVector<PyObject*, 64> call_args;
  call_args.reserve(1 + nargs - op_args_start + nkwargs);
  call_args.push_back(py_key_set.ptr());
  for (const auto i : c10::irange(op_args_start, nargs + nkwargs)) {
    call_args.push_back(args[i]);
  }
  return PyObject_Vectorcall(
      cpp_redispatch_fn,
      call_args.data(),
      static_cast<size_t>(1 + nargs - op_args_start),
      kwnames);
}

static PyObject* pyobject_dispatch_call_python(
    PyObject* kernel,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames,
    Py_ssize_t op_args_start) {
  return PyObject_Vectorcall(
      kernel,
      args + op_args_start,
      static_cast<size_t>(nargs - op_args_start),
      kwnames);
}

static PyObject* pyobject_dispatch_call_python_with_keyset(
    PyObject* kernel,
    c10::DispatchKeySet key_set,
    PyObject* const* args,
    Py_ssize_t nargs,
    Py_ssize_t nkwargs,
    PyObject* kwnames,
    Py_ssize_t op_args_start) {
  py::object py_key_set = py::cast(key_set);
  c10::SmallVector<PyObject*, 64> call_args;
  call_args.reserve(1 + nargs - op_args_start + nkwargs);
  call_args.push_back(py_key_set.ptr());
  for (const auto i : c10::irange(op_args_start, nargs + nkwargs)) {
    call_args.push_back(args[i]);
  }
  return PyObject_Vectorcall(
      kernel,
      call_args.data(),
      static_cast<size_t>(1 + nargs - op_args_start),
      kwnames);
}

// The Python types PyObject dispatch can check a value against cheaply.
struct PyObjectDispatchType {
  enum class Kind : uint8_t {
    Tensor,
    TensorList,
    Int,
    Float,
    Bool,
    Str,
    Other,
  };
  Kind kind = Kind::Other;
  bool optional = false;
};

static PyObjectDispatchType pyobject_dispatch_type(c10::TypePtr type) {
  using Kind = PyObjectDispatchType::Kind;
  PyObjectDispatchType result;
  if (const auto* opt = type->castRaw<c10::OptionalType>()) {
    result.optional = true;
    type = opt->getElementType();
  }
  switch (type->kind()) {
    case c10::TypeKind::TensorType:
      result.kind = Kind::Tensor;
      break;
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType:
      result.kind = Kind::Int;
      break;
    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType:
      result.kind = Kind::Float;
      break;
    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType:
      result.kind = Kind::Bool;
      break;
    case c10::TypeKind::StringType:
      result.kind = Kind::Str;
      break;
    case c10::TypeKind::ListType:
      if (type->castRaw<c10::ListType>()->getElementType()->kind() ==
          c10::TypeKind::TensorType) {
        result.kind = Kind::TensorList;
      }
      break;
    default:
      break;
  }
  return result;
}

// NOTE [PyObject dispatch argument normalization]
// Before calling a Python kernel, the C++ Dispatcher converts (args, kwargs)
// to IValues and back (see parseIValuesToPyArgsKwargs), which normalizes them:
// - every argument is filled in, omitted ones from their defaults;
// - the kernel gets the positional arguments up to the last one that differs
//   from its default, and the kwarg-only arguments that differ from their
//   defaults, as kwargs;
// - values are converted to the schema's types, e.g. "cuda" for a Device
//   becomes a torch.device and a tuple for an int[] becomes a list.
// PyObject dispatch does the same without IValues when every argument already
// has the Python type that the round trip would produce (an exact int for an
// int, a list or tuple of Tensors for a Tensor[], ...). Otherwise
// pyobject_dispatch_try_normalize_args returns false and the caller falls back
// to the C++ Dispatcher.
struct PyObjectDispatchArg {
  enum class Kind : uint8_t {
    Tensor,
    TensorList,
    OptionalTensorList,
    Int,
    Float,
    Bool,
    Str,
    ScalarType,
    Layout,
    Device,
    Scalar,
    IntList,
    FloatList,
    Unsupported,
  };
  Kind kind = Kind::Unsupported;
  bool optional = false;
  bool has_default = false;
  // The default as a Python object, or null if we can't produce it cheaply.
  py::object default_value;
  py::object name;
};

struct PyObjectDispatchSchema {
  std::vector<PyObjectDispatchArg> args;
  std::vector<PyObjectDispatchType> returns;
  // The common case of a single, non-optional Tensor return.
  bool returns_one_tensor = false;
  // Arguments before this index are positional, the rest kwarg-only.
  size_t num_positional = 0;
  // Every argument is a positional Tensor or Tensor? without a default, so a
  // call passing them all positionally needs no normalization.
  bool only_tensors = true;
};

static PyObjectDispatchArg::Kind pyobject_dispatch_kind(const c10::TypePtr& t) {
  using Kind = PyObjectDispatchArg::Kind;
  switch (t->kind()) {
    case c10::TypeKind::TensorType:
      return Kind::Tensor;
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType:
      return Kind::Int;
    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType:
      return Kind::Float;
    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType:
      return Kind::Bool;
    case c10::TypeKind::StringType:
      return Kind::Str;
    case c10::TypeKind::ScalarTypeType:
      return Kind::ScalarType;
    case c10::TypeKind::LayoutType:
      return Kind::Layout;
    case c10::TypeKind::DeviceObjType:
      return Kind::Device;
    case c10::TypeKind::NumberType:
      return Kind::Scalar;
    case c10::TypeKind::ListType: {
      const auto& elem = t->castRaw<c10::ListType>()->getElementType();
      switch (elem->kind()) {
        case c10::TypeKind::TensorType:
          return Kind::TensorList;
        case c10::TypeKind::IntType:
        case c10::TypeKind::SymIntType:
          return Kind::IntList;
        case c10::TypeKind::FloatType:
          return Kind::FloatList;
        case c10::TypeKind::OptionalType:
          return elem->castRaw<c10::OptionalType>()->getElementType()->kind() ==
                  c10::TypeKind::TensorType
              ? Kind::OptionalTensorList
              : Kind::Unsupported;
        default:
          return Kind::Unsupported;
      }
    }
    default:
      return Kind::Unsupported;
  }
}

static py::object pyobject_dispatch_default(
    PyObjectDispatchArg::Kind kind,
    const c10::IValue& value) {
  using Kind = PyObjectDispatchArg::Kind;
  if (value.isNone()) {
    return py::none();
  }
  switch (kind) {
    case Kind::Int:
      return value.isInt() ? torch::jit::toPyObject(value) : py::object();
    case Kind::Float:
      return value.isDouble() ? torch::jit::toPyObject(value) : py::object();
    case Kind::Bool:
      return value.isBool() ? torch::jit::toPyObject(value) : py::object();
    case Kind::Str:
      return value.isString() ? torch::jit::toPyObject(value) : py::object();
    case Kind::Scalar:
      return value.isInt() || value.isDouble() || value.isBool()
          ? torch::jit::toPyObject(value)
          : py::object();
    case Kind::IntList:
    case Kind::FloatList:
      return value.isList() ? torch::jit::toPyObject(value) : py::object();
    case Kind::ScalarType:
      return value.isInt()
          ? py::reinterpret_borrow<py::object>(
                reinterpret_cast<PyObject*>(torch::getTHPDtype(
                    static_cast<c10::ScalarType>(value.toInt()))))
          : py::object();
    case Kind::Layout:
      return value.isInt()
          ? py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
                torch::getTHPLayout(static_cast<c10::Layout>(value.toInt()))))
          : py::object();
    default:
      return py::object();
  }
}

static bool pyobject_dispatch_is_int64(PyObject* obj) {
  if (!PyLong_CheckExact(obj)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongLongAndOverflow(obj, &overflow);
  return overflow == 0;
}

static std::unique_ptr<PyObjectDispatchSchema> make_pyobject_dispatch_schema(
    const c10::FunctionSchema& schema) {
  auto info = std::make_unique<PyObjectDispatchSchema>();
  for (const auto& arg : schema.arguments()) {
    PyObjectDispatchArg a;
    c10::TypePtr type = arg.real_type();
    if (const auto* opt = type->castRaw<c10::OptionalType>()) {
      a.optional = true;
      type = opt->getElementType();
    }
    a.kind = pyobject_dispatch_kind(type);
    if (arg.default_value().has_value()) {
      a.has_default = true;
      a.default_value = pyobject_dispatch_default(a.kind, *arg.default_value());
    }
    info->only_tensors = info->only_tensors &&
        a.kind == PyObjectDispatchArg::Kind::Tensor && !a.has_default &&
        !arg.kwarg_only();
    a.name = py::reinterpret_steal<py::object>(
        PyUnicode_InternFromString(arg.name().c_str()));
    if (!arg.kwarg_only()) {
      info->num_positional++;
    }
    info->args.push_back(std::move(a));
  }
  for (const auto& ret : schema.returns()) {
    info->returns.push_back(pyobject_dispatch_type(ret.real_type()));
  }
  info->returns_one_tensor = info->returns.size() == 1 &&
      info->returns[0].kind == PyObjectDispatchType::Kind::Tensor &&
      !info->returns[0].optional;
  return info;
}

// Returns whether `obj` is exactly what the IValue round trip would give the
// kernel for `arg`. Lists are replaced by a new list, as the round trip does.
static bool pyobject_dispatch_check_value(
    const PyObjectDispatchArg& arg,
    PyObject*& obj,
    std::vector<py::object>& owned_args) {
  using Kind = PyObjectDispatchArg::Kind;
  if (obj == Py_None) {
    return arg.optional;
  }
  switch (arg.kind) {
    case Kind::Tensor:
      return THPVariable_Check(obj);
    case Kind::Int:
      return pyobject_dispatch_is_int64(obj);
    case Kind::Float:
      return PyFloat_CheckExact(obj);
    case Kind::Bool:
      return PyBool_Check(obj);
    case Kind::Str:
      return PyUnicode_CheckExact(obj);
    case Kind::ScalarType:
      return THPDtype_Check(obj);
    case Kind::Layout:
      return THPLayout_Check(obj);
    case Kind::Device:
      return THPDevice_Check(obj);
    case Kind::Scalar:
      return PyBool_Check(obj) || PyFloat_CheckExact(obj) ||
          pyobject_dispatch_is_int64(obj);
    case Kind::TensorList:
    case Kind::OptionalTensorList:
    case Kind::IntList:
    case Kind::FloatList: {
      if (!PyList_CheckExact(obj) && !PyTuple_CheckExact(obj)) {
        return false;
      }
      PyObject* list = PySequence_List(obj);
      if (list == nullptr) {
        PyErr_Clear();
        return false;
      }
      owned_args.emplace_back(py::reinterpret_steal<py::object>(list));
      for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i) {
        PyObject* item = PyList_GET_ITEM(list, i);
        bool ok = false;
        switch (arg.kind) {
          case Kind::TensorList:
            ok = THPVariable_Check(item);
            break;
          case Kind::OptionalTensorList:
            ok = item == Py_None || THPVariable_Check(item);
            break;
          case Kind::IntList:
            ok = pyobject_dispatch_is_int64(item);
            break;
          default:
            ok = PyFloat_CheckExact(item);
            break;
        }
        if (!ok) {
          return false;
        }
      }
      obj = list;
      return true;
    }
    case Kind::Unsupported:
      return false;
  }
  return false;
}

// Whether `obj` is equal to arg's default, compared the way the C++
// Dispatcher compares IValues. Arguments whose default we don't have as a
// Python object never get here (see pyobject_dispatch_try_normalize_args).
static bool pyobject_dispatch_is_default(
    const PyObjectDispatchArg& arg,
    PyObject* obj) {
  PyObject* default_value = arg.default_value.ptr();
  if (default_value == nullptr) {
    return false;
  }
  if (obj == default_value) {
    return true;
  }
  if (obj == Py_None || default_value == Py_None ||
      Py_TYPE(obj) != Py_TYPE(default_value)) {
    return false;
  }
  const int result = PyObject_RichCompareBool(obj, default_value, Py_EQ);
  if (result < 0) {
    // Can't happen for the exact types that get here; treat as not equal.
    PyErr_Clear();
    return false;
  }
  return result == 1;
}

// A fast-path call's arguments after pyobject_dispatch_try_normalize_args.
struct PyObjectDispatchArgs {
  // Every argument in schema order, for the dispatch key set and C++ kernels.
  PyObject* const* all = nullptr;
  // The kernel's vectorcall arguments: the positional ones, then the values
  // for `kwnames`.
  PyObject* const* call = nullptr;
  Py_ssize_t call_nargs = 0;
  PyObject* kwnames = nullptr;
  // Backing storage, unless `all` and `call` are the caller's arguments.
  c10::SmallVector<PyObject*, 64> all_storage;
  c10::SmallVector<PyObject*, 64> call_storage;
  py::object kwnames_storage;
  std::vector<py::object> owned;
};

// The common case, which needs no normalization: the operator only takes
// Tensors, and they are all passed positionally.
static bool pyobject_dispatch_only_tensors(
    const PyObjectDispatchSchema& info,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames) {
  if (!info.only_tensors || kwnames != nullptr ||
      static_cast<size_t>(nargs) != info.args.size()) {
    return false;
  }
  for (const auto i : c10::irange(nargs)) {
    PyObject* obj = args[i];
    if (!THPVariable_Check(obj) && !(obj == Py_None && info.args[i].optional)) {
      return false;
    }
  }
  return true;
}

// See NOTE [PyObject dispatch argument normalization].
static bool pyobject_dispatch_try_normalize_args(
    const c10::FunctionSchema& schema,
    const PyObjectDispatchSchema& info,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames,
    PyObjectDispatchArgs& out) {
  const size_t num_args = info.args.size();
  if (static_cast<size_t>(nargs) > info.num_positional) {
    return false;
  }
  auto& all_args = out.all_storage;
  all_args.assign(num_args, nullptr);
  std::copy(args, args + nargs, all_args.begin());
  const Py_ssize_t nkwargs = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
  for (Py_ssize_t k = 0; k < nkwargs; ++k) {
    PyObject* key = PyTuple_GET_ITEM(kwnames, k);
    size_t idx = num_args;
    for (size_t i = 0; i < num_args && idx == num_args; ++i) {
      if (info.args[i].name.ptr() == key ||
          PyUnicode_Compare(info.args[i].name.ptr(), key) == 0) {
        idx = i;
      }
    }
    if (idx == num_args || all_args[idx] != nullptr) {
      return false;
    }
    all_args[idx] = args[nargs + k];
  }
  for (const auto i : c10::irange(num_args)) {
    const auto& arg = info.args[i];
    if (all_args[i] == nullptr) {
      if (!arg.default_value) {
        return false;
      }
      all_args[i] = arg.default_value.ptr();
    } else if (
        (arg.has_default && !arg.default_value) ||
        !pyobject_dispatch_check_value(arg, all_args[i], out.owned)) {
      return false;
    }
  }

  const auto split = splitPythonKernelArgs(schema, [&](size_t i) {
    return pyobject_dispatch_is_default(info.args[i], all_args[i]);
  });
  auto& call_args = out.call_storage;
  call_args.assign(all_args.begin(), all_args.begin() + split.num_positional);
  // An omitted list argument was filled in with the operator's default list.
  // Give the kernel its own copy, as the IValue round trip does, so mutating it
  // can't change the default.
  for (const auto i : c10::irange(split.num_positional)) {
    if (call_args[i] == info.args[i].default_value.ptr() &&
        PyList_CheckExact(call_args[i])) {
      PyObject* copy =
          PyList_GetSlice(call_args[i], 0, PyList_GET_SIZE(call_args[i]));
      if (copy == nullptr) {
        PyErr_Clear();
        return false;
      }
      out.owned.emplace_back(py::reinterpret_steal<py::object>(copy));
      call_args[i] = copy;
    }
  }
  out.call_nargs = static_cast<Py_ssize_t>(split.num_positional);
  c10::SmallVector<size_t, 8> kwarg_indices;
  for (size_t i = split.kwarg_only_start; i < num_args; ++i) {
    if (!pyobject_dispatch_is_default(info.args[i], all_args[i])) {
      kwarg_indices.push_back(i);
    }
  }
  if (!kwarg_indices.empty()) {
    PyObject* names =
        PyTuple_New(static_cast<Py_ssize_t>(kwarg_indices.size()));
    if (names == nullptr) {
      PyErr_Clear();
      return false;
    }
    out.kwnames_storage = py::reinterpret_steal<py::object>(names);
    out.kwnames = names;
    for (const auto j : c10::irange(kwarg_indices.size())) {
      PyObject* name = info.args[kwarg_indices[j]].name.ptr();
      Py_INCREF(name);
      PyTuple_SET_ITEM(names, static_cast<Py_ssize_t>(j), name);
      call_args.push_back(all_args[kwarg_indices[j]]);
    }
  }
  out.all = all_args.data();
  out.call = call_args.data();
  return true;
}

// Whether a kernel's return value is already what the C++ Dispatcher's round
// trip through IValues would give the caller. A Tensor[] still needs copying
// into a new list.
static bool pyobject_dispatch_result_ok(
    const PyObjectDispatchType& type,
    PyObject* obj) {
  using Kind = PyObjectDispatchType::Kind;
  if (obj == Py_None) {
    return type.optional;
  }
  switch (type.kind) {
    case Kind::Tensor:
      return THPVariable_Check(obj);
    case Kind::Int:
      return pyobject_dispatch_is_int64(obj);
    case Kind::Float:
      return PyFloat_CheckExact(obj);
    case Kind::Bool:
      return PyBool_Check(obj);
    case Kind::Str:
      return PyUnicode_CheckExact(obj);
    case Kind::TensorList:
      if (!PyList_CheckExact(obj)) {
        return false;
      }
      for (Py_ssize_t i = 0; i < PyList_GET_SIZE(obj); ++i) {
        if (!THPVariable_Check(PyList_GET_ITEM(obj, i))) {
          return false;
        }
      }
      return true;
    case Kind::Other:
      return false;
  }
  return false;
}

// The C++ Dispatcher converts a Python kernel's return value to IValues
// (pushPyOutToStack) and back (createPyObjectForStack). Return `result` (a
// new reference) if it already is what that would produce; otherwise do
// exactly that round trip, including its errors.
static PyObject* pyobject_dispatch_normalize_result(
    const c10::OperatorHandle& op,
    const PyObjectDispatchSchema& info,
    PyObject* result) {
  if (C10_LIKELY(info.returns_one_tensor && THPVariable_Check(result))) {
    return result;
  }
  auto owned = py::reinterpret_steal<py::object>(result);
  const auto& returns = info.returns;
  if (returns.empty()) {
    if (result == Py_None) {
      return owned.release().ptr();
    }
  } else if (returns.size() == 1) {
    if (pyobject_dispatch_result_ok(returns[0], result)) {
      if (returns[0].kind == PyObjectDispatchType::Kind::TensorList &&
          result != Py_None) {
        return PyList_GetSlice(result, 0, PyList_GET_SIZE(result));
      }
      return owned.release().ptr();
    }
  } else if (
      PyTuple_CheckExact(result) &&
      static_cast<size_t>(PyTuple_GET_SIZE(result)) == returns.size()) {
    bool ok = true;
    for (size_t i = 0; i < returns.size() && ok; ++i) {
      ok =
          returns[i].kind != PyObjectDispatchType::Kind::TensorList &&
          pyobject_dispatch_result_ok(
              returns[i], PyTuple_GET_ITEM(result, static_cast<Py_ssize_t>(i)));
    }
    if (ok) {
      return owned.release().ptr();
    }
  }
  torch::jit::Stack stack;
  pushPyOutToStack(op, &stack, std::move(owned), "PythonKernelHolder");
  return torch::jit::createPyObjectForStack(std::move(stack)).release().ptr();
}

// Whether autograd needs to see a call with these arguments; like
// any_requires_grad, but for the PyObject arguments of a fast-path call.
static bool pyobject_dispatch_any_requires_grad(
    PyObject* const* args,
    Py_ssize_t n) {
  if (!at::GradMode::is_enabled()) {
    return false;
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* obj = args[i];
    if (THPVariable_Check(obj)) {
      if (THPVariable_Unpack(obj).requires_grad()) {
        return true;
      }
    } else if (PyList_CheckExact(obj) || PyTuple_CheckExact(obj)) {
      PyObject** items = PySequence_Fast_ITEMS(obj);
      for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(obj); ++j) {
        if (THPVariable_Check(items[j]) &&
            THPVariable_Unpack(items[j]).requires_grad()) {
          return true;
        }
      }
    }
  }
  return false;
}

// The kernel for the dispatch key set is a C++ kernel, which needs IValues
// anyway: convert the arguments here and call the Dispatcher, instead of going
// back through the Python entry point. `args` are all of the operator's
// arguments, in schema order.
static PyObject* pyobject_dispatch_call_cpp(
    const c10::OperatorHandle& op,
    PyObject* const* args,
    Py_ssize_t nargs) {
  const auto& schema = op.schema();
  torch::jit::Stack stack;
  stack.reserve(nargs);
  for (const auto i : c10::irange(nargs)) {
    stack.push_back(torch::jit::argumentToIValue(schema, i, args[i]));
  }
  {
    pybind11::gil_scoped_release no_gil;
    op.callBoxed(&stack);
  }
  return torch::jit::createPyObjectForStack(std::move(stack)).release().ptr();
}

template <typename Func>
static PyObject* pyobject_dispatch_with_keyset(
    Func* self,
    c10::DispatchKeySet key_set,
    bool skip_torch_function,
    PyObject* const* args,
    Py_ssize_t nargs,
    PyObject* kwnames,
    Py_ssize_t op_args_start,
    // For a top-level dispatch: every argument in schema order.
    PyObject* const* all_args = nullptr) {
  Py_ssize_t nkwargs = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
  // TODO(#187974): Figure out why we need a special case to send the Python
  // key back to the C++ Dispatcher.
  if (C10_UNLIKELY(key_set.has(c10::DispatchKey::Python))) {
    return pyobject_dispatch_call_redispatch_cpp(
        self->cpp_redispatch_fn,
        key_set,
        args,
        nargs,
        nkwargs,
        kwnames,
        op_args_start);
  }
  const auto& kernel_function = self->handle->lookup(key_set);
  // If there's no Python kernel, call back to the C++ Dispatcher
  const auto* holder =
      kernel_function.template boxedKernelFunctor<PythonKernelHolder>();
  if (holder == nullptr && all_args != nullptr && !skip_torch_function) {
    return pyobject_dispatch_call_cpp(
        *self->handle,
        all_args,
        static_cast<Py_ssize_t>(self->schema_info->args.size()));
  }
  if (C10_UNLIKELY(holder == nullptr || holder->with_op())) {
    return pyobject_dispatch_call_redispatch_cpp(
        self->cpp_redispatch_fn,
        key_set,
        args,
        nargs,
        nkwargs,
        kwnames,
        op_args_start);
  }
  if (C10_UNLIKELY(holder->skip_if_no_grad()) && !skip_torch_function &&
      !pyobject_dispatch_any_requires_grad(
          args + op_args_start, nargs - op_args_start + nkwargs)) {
    // See PythonKernelHolder::operator().
    at::AutoDispatchBelowAutograd guard;
    return pyobject_dispatch_with_keyset(
        self,
        key_set & c10::after_autograd_keyset,
        skip_torch_function,
        args,
        nargs,
        kwnames,
        op_args_start,
        all_args);
  }
  // Otherwise, just directly invoke the Python kernel.
  auto* kernel = holder->func(self->interpreter);
  TORCH_INTERNAL_ASSERT(kernel != nullptr);
  if (C10_UNLIKELY(skip_torch_function)) {
    torch::consume_should_skip_torch_function();
  }
  PyObject* result = nullptr;
  if (C10_UNLIKELY(holder->with_keyset())) {
    result = pyobject_dispatch_call_python_with_keyset(
        kernel, key_set, args, nargs, nkwargs, kwnames, op_args_start);
  } else {
    result = pyobject_dispatch_call_python(
        kernel, args, nargs, kwnames, op_args_start);
  }
  if (result == nullptr) {
    return nullptr;
  }
  return pyobject_dispatch_normalize_result(
      *self->handle, *self->schema_info, result);
}

static PyObject* pyobject_dispatch_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<PyObjectDispatchFunc*>(callable);
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  Py_ssize_t nkwargs = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
  bool skip_torch_function = torch::peek_should_skip_torch_function();
  if (C10_UNLIKELY(
          !skip_torch_function &&
          pyobject_dispatch_has_torch_function(args, nargs, nkwargs, 0))) {
    return PyObject_Vectorcall(
        self->cpp_dispatch_fn, args, static_cast<size_t>(nargs), kwnames);
  }
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  // The C++ Dispatcher records each call for the profiler (RecordFunction in
  // Dispatcher::callBoxed); let it handle the call while anything listens.
  // This check doesn't sample callbacks: callBoxed does, and sampling here too
  // would record a call with probability p^2 instead of p.
  if (C10_UNLIKELY(at::hasCallbacks())) {
    return PyObject_Vectorcall(
        self->cpp_dispatch_fn, args, static_cast<size_t>(nargs), kwnames);
  }
#endif
  const auto& info = *self->schema_info;
  const auto num_args = static_cast<Py_ssize_t>(info.args.size());
  auto dispatch = [&](PyObject* const* all_args,
                      PyObject* const* call_args,
                      Py_ssize_t call_nargs,
                      PyObject* call_kwnames) {
    auto key_set = pyobject_dispatch_compute_keyset(
        self->handle->dispatchKeyExtractor(), all_args, num_args);
    if (C10_UNLIKELY(skip_torch_function)) {
      key_set = key_set.remove(c10::DispatchKey::Python);
    }
    return pyobject_dispatch_with_keyset(
        self,
        key_set,
        skip_torch_function,
        call_args,
        call_nargs,
        call_kwnames,
        0,
        all_args);
  };
  if (C10_LIKELY(pyobject_dispatch_only_tensors(info, args, nargs, kwnames))) {
    return dispatch(args, args, nargs, nullptr);
  }
  PyObjectDispatchArgs normalized;
  if (!pyobject_dispatch_try_normalize_args(
          self->handle->schema(), info, args, nargs, kwnames, normalized)) {
    return PyObject_Vectorcall(
        self->cpp_dispatch_fn, args, static_cast<size_t>(nargs), kwnames);
  }
  return dispatch(
      normalized.all,
      normalized.call,
      normalized.call_nargs,
      normalized.kwnames);
  END_HANDLE_TH_ERRORS
}

static PyObject* pyobject_redispatch_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<PyObjectRedispatchFunc*>(callable);
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  Py_ssize_t nkwargs = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
  if (nargs == 0) {
    PyErr_SetString(
        PyExc_TypeError,
        "redispatch() expected redispatch(keyset, *args, **kwargs)");
    return nullptr;
  }
  c10::DispatchKeySet key_set;
  try {
    key_set = py::handle(args[0]).cast<c10::DispatchKeySet>();
  } catch (const py::cast_error&) {
    PyErr_SetString(
        PyExc_TypeError,
        "redispatch() expected a DispatchKeySet as its first argument");
    return nullptr;
  }
  std::optional<at::AutoDispatchBelowAutograd> below_autograd_guard;
  if (self->below_autograd) {
    below_autograd_guard.emplace();
    key_set = key_set & c10::after_autograd_keyset;
  }
  bool skip_torch_function = torch::peek_should_skip_torch_function();
  if (C10_UNLIKELY(
          !skip_torch_function &&
          pyobject_dispatch_has_torch_function(args, nargs, nkwargs, 1))) {
    return pyobject_dispatch_call_redispatch_cpp(
        self->cpp_redispatch_fn, key_set, args, nargs, nkwargs, kwnames, 1);
  }
  if (C10_UNLIKELY(skip_torch_function)) {
    key_set = key_set.remove(c10::DispatchKey::Python);
  }
  return pyobject_dispatch_with_keyset(
      self, key_set, skip_torch_function, args, nargs, kwnames, 1);
  END_HANDLE_TH_ERRORS
}

static void pyobject_dispatch_dealloc(PyObjectDispatchFunc* self) {
  Py_XDECREF(self->cpp_dispatch_fn);
  Py_XDECREF(self->cpp_redispatch_fn);
  delete self->handle;
  delete self->schema_info;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static void pyobject_redispatch_dealloc(PyObjectRedispatchFunc* self) {
  Py_XDECREF(self->cpp_redispatch_fn);
  delete self->handle;
  delete self->schema_info;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyTypeObject PyObjectDispatchFuncType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyTypeObject PyObjectRedispatchFuncType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

static PyObject* make_pyobject_dispatch_func(
    const c10::OperatorHandle& handle,
    PyObject* cpp_dispatch_fn,
    PyObject* cpp_redispatch_fn,
    vectorcallfunc vectorcall) {
  auto owned_handle = std::make_unique<c10::OperatorHandle>(handle);
  auto schema_info = make_pyobject_dispatch_schema(handle.schema());
  auto* result = PyObject_New(PyObjectDispatchFunc, &PyObjectDispatchFuncType);
  TORCH_CHECK_PYTHON(result != nullptr);
  Py_INCREF(cpp_dispatch_fn);
  Py_INCREF(cpp_redispatch_fn);
  result->cpp_dispatch_fn = cpp_dispatch_fn;
  result->cpp_redispatch_fn = cpp_redispatch_fn;
  result->handle = owned_handle.release();
  result->schema_info = schema_info.release();
  result->interpreter = getPyInterpreter();
  result->vectorcall = vectorcall;
  return reinterpret_cast<PyObject*>(result);
}

static PyObject* make_pyobject_redispatch_func(
    const c10::OperatorHandle& handle,
    PyObject* cpp_redispatch_fn,
    vectorcallfunc vectorcall,
    bool below_autograd) {
  auto owned_handle = std::make_unique<c10::OperatorHandle>(handle);
  auto schema_info = make_pyobject_dispatch_schema(handle.schema());
  auto* result =
      PyObject_New(PyObjectRedispatchFunc, &PyObjectRedispatchFuncType);
  TORCH_CHECK_PYTHON(result != nullptr);
  Py_INCREF(cpp_redispatch_fn);
  result->cpp_redispatch_fn = cpp_redispatch_fn;
  result->handle = owned_handle.release();
  result->schema_info = schema_info.release();
  result->interpreter = getPyInterpreter();
  result->vectorcall = vectorcall;
  result->below_autograd = below_autograd;
  return reinterpret_cast<PyObject*>(result);
}

static py::tuple make_pyobj_dispatch_fns(
    const py::object& py_handle,
    const py::object& cpp_dispatch_fn,
    const py::object& cpp_redispatch_fn) {
  TORCH_CHECK(
      PyCallable_Check(cpp_dispatch_fn.ptr()),
      "cpp_dispatch_fn must be callable");
  TORCH_CHECK(
      PyCallable_Check(cpp_redispatch_fn.ptr()),
      "cpp_redispatch_fn must be callable");
  const auto& handle = py_handle.cast<c10::OperatorHandle&>();
  auto dispatch = py::reinterpret_steal<py::object>(make_pyobject_dispatch_func(
      handle,
      cpp_dispatch_fn.ptr(),
      cpp_redispatch_fn.ptr(),
      pyobject_dispatch_vectorcall));
  auto make_redispatch = [&](bool below_autograd) {
    return py::reinterpret_steal<py::object>(make_pyobject_redispatch_func(
        handle,
        cpp_redispatch_fn.ptr(),
        pyobject_redispatch_vectorcall,
        below_autograd));
  };
  return py::make_tuple(
      std::move(dispatch), make_redispatch(false), make_redispatch(true));
}

// A small RAII guard that lets you explicitly *remove* a key from the TLS
// exclude set.
class SetExcludeDispatchKeyGuard {
 public:
  SetExcludeDispatchKeyGuard(at::DispatchKey k, bool set_excluded)
      : k(k), old(c10::impl::tls_is_dispatch_key_excluded(k)) {
    c10::impl::tls_set_dispatch_key_excluded(k, set_excluded);
  }
  ~SetExcludeDispatchKeyGuard() {
    c10::impl::tls_set_dispatch_key_excluded(k, old);
  }
  SetExcludeDispatchKeyGuard(const SetExcludeDispatchKeyGuard&) = delete;
  SetExcludeDispatchKeyGuard operator=(const SetExcludeDispatchKeyGuard&) =
      delete;
  SetExcludeDispatchKeyGuard(SetExcludeDispatchKeyGuard&&) = delete;
  SetExcludeDispatchKeyGuard operator=(SetExcludeDispatchKeyGuard&&) = delete;

 private:
  at::DispatchKey k;
  bool old;
};

// Thread-local stack used to save/restore the local dispatch key set around a
// torch.compile'd call (see compile_wrapper in eval_frame.py). Keeping the
// saved keyset in C++ avoids materializing it as a (registered) pybind11
// instance on every compiled call, and a stack supports nested compiled calls.
static std::vector<c10::impl::LocalDispatchKeySet>& dynamoDispatchKeySetStack() {
  static thread_local std::vector<c10::impl::LocalDispatchKeySet> stack;
  return stack;
}

void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  PyObjectDispatchFuncType.tp_name = "torch._C._PyObjectDispatchFunc";
  PyObjectDispatchFuncType.tp_basicsize = sizeof(PyObjectDispatchFunc);
  PyObjectDispatchFuncType.tp_dealloc =
      reinterpret_cast<destructor>(pyobject_dispatch_dealloc);
  PyObjectDispatchFuncType.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
  PyObjectDispatchFuncType.tp_call = PyVectorcall_Call;
  PyObjectDispatchFuncType.tp_vectorcall_offset =
      offsetof(PyObjectDispatchFunc, vectorcall);
  TORCH_CHECK_PYTHON(PyType_Ready(&PyObjectDispatchFuncType) >= 0);

  PyObjectRedispatchFuncType.tp_name = "torch._C._PyObjectRedispatchFunc";
  PyObjectRedispatchFuncType.tp_basicsize = sizeof(PyObjectRedispatchFunc);
  PyObjectRedispatchFuncType.tp_dealloc =
      reinterpret_cast<destructor>(pyobject_redispatch_dealloc);
  PyObjectRedispatchFuncType.tp_flags =
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL;
  PyObjectRedispatchFuncType.tp_call = PyVectorcall_Call;
  PyObjectRedispatchFuncType.tp_vectorcall_offset =
      offsetof(PyObjectRedispatchFunc, vectorcall);
  TORCH_CHECK_PYTHON(PyType_Ready(&PyObjectRedispatchFuncType) >= 0);

  py::class_<c10::OperatorHandle>(m, "_DispatchOperatorHandle")
      .def("schema", &c10::OperatorHandle::schema)
      .def("debug", &c10::OperatorHandle::debug)
      .def(
          "redispatch_boxed",
          [](const py::object& self,
             c10::DispatchKeySet keyset,
             const py::args& args,
             const py::kwargs& kwargs) {
            HANDLE_TH_ERRORS
            auto& handle = self.cast<c10::OperatorHandle&>();
            auto stack = torch::jit::createStackForSchema(
                handle.schema(),
                args,
                kwargs,
                /*self=*/std::nullopt);
            {
              pybind11::gil_scoped_release no_gil_guard;
              handle.redispatchBoxed(keyset, &stack);
            }
            return torch::jit::createPyObjectForStack(std::move(stack));
            END_HANDLE_TH_ERRORS_PYBIND
          });

  m.def("_dispatch_call_boxed", &ophandle_call_boxed);
  m.def("_dispatch_make_pyobj_dispatch_fns", &make_pyobj_dispatch_fns);

  // TODO: figure out how to do chaining
  py::class_<torch::Library>(m, "_DispatchModule")
      .def(
          "reset",
          [](const py::object& self) {
            self.cast<torch::Library&>().reset();
            return;
          },
          "")
      // Some of these APIs are only for testing and do not work in
      // multipy environment  // codespell:ignore multipy
      .def(
          "def_",
          [](py::object self, const char* schema, const char* alias) {
            self.cast<torch::Library&>().def(
                torch::schema(schema, parseAliasAnalysisKind(alias)));
            return self;
          },
          "",
          py::arg("schema"),
          py::arg("alias") = "")
      // Simulated "legacy" def where alias analysis kind is not set.
      // Ordinarily this can only be exercised from RegisterOperators() API
      // but I am not going to bind that here
      .def(
          "def_legacy",
          [](py::object self, const char* schema) {
            self.cast<torch::Library&>().def(torch::jit::parseSchema(schema));
            return self;
          },
          "",
          py::arg("schema"))
      // We can't conveniently turn Python functions into valid functions
      // in the dispatcher.  So instead we provide a bunch of precanned
      // functions for testing purposes.  You're NOT intended to actually
      // call these functions; they're just here so we can actually register
      // something
      //
      // Mangling scheme: args_rets.  One character per.
      //  t = Tensor
      .def(
          "def_name_t_t",
          [](py::object self,
             const char* name,
             const char* dispatch,
             const char* debug) {
            self.cast<torch::Library&>().def(
                name, dispatch_str(dispatch, [](const at::Tensor& a) {
                        return a;
                      }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("debug") = "default_def_name_t_t")
      .def(
          "def_schema_t_t",
          [](py::object self,
             const char* schema,
             const char* dispatch,
             const char* alias,
             const char* debug) {
            self.cast<torch::Library&>().def(
                torch::schema(schema, parseAliasAnalysisKind(alias)),
                dispatch_str(dispatch, [](const at::Tensor& a) {
                  return a;
                }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("alias") = "",
          py::arg("debug") = "default_def_schema_t_t")
      // TODO: maybe consider deduplicating the definitions here, it's getting
      // pretty long
      .def(
          "impl_t_t",
          [](py::object self,
             const char* name,
             const char* dispatch,
             const char* debug) {
            self.cast<torch::Library&>().impl(
                name, dispatch_str(dispatch, [](const at::Tensor& a) {
                        return a;
                      }).debug(debug));
            return self;
          },
          "",
          py::arg("name"),
          py::arg("dispatch") = "",
          py::arg("debug") = "impl_t_t")
      .def(
          "impl_with_aoti_compile",
          [](const py::object& self,
             const char* ns,
             const char* op_name_with_overload,
             c10::DispatchKey dispatch) {
            HANDLE_TH_ERRORS
            std::string reg_op_name =
                std::string(ns).append("::").append(op_name_with_overload);

            auto& lib = self.cast<torch::Library&>();
            lib.impl(
                reg_op_name.c_str(),
                torch::dispatch(
                    dispatch,
                    CppFunction::makeFromBoxedFunctor(
                        std::make_unique<
                            torch::inductor::AOTIPythonKernelHolder>(
                            dispatch, ns, op_name_with_overload))),
                register_or_verify());
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("ns"),
          py::arg("op_name_with_overload"),
          py::arg("dispatch"))
      .def(
          "impl",
          [](const py::object& self,
             const char* name,
             // TODO: empty string no longer works
             c10::DispatchKey dispatch,
             py::object func,
             bool with_keyset,
             bool skip_if_no_grad) {
            HANDLE_TH_ERRORS
            TORCH_CHECK(
                !skip_if_no_grad || dispatch == c10::DispatchKey::Autograd ||
                    c10::isIncludedInAlias(
                        dispatch, c10::DispatchKey::Autograd),
                "skip_if_no_grad is only supported for Autograd kernels");
            auto& lib = self.cast<torch::Library&>();
            if (func.is(py::module::import("torch.library")
                            .attr("fallthrough_kernel"))) {
              lib.impl(
                  name,
                  torch::dispatch(dispatch, CppFunction::makeFallthrough()),
                  register_or_verify());
            } else {
              lib.impl(
                  name,
                  torch::dispatch(
                      dispatch,
                      CppFunction::makeFromBoxedFunctor(
                          std::make_unique<PythonKernelHolder>(
                              func,
                              dispatch,
                              with_keyset,
                              /*with_op=*/false,
                              skip_if_no_grad))),
                  register_or_verify());
              python_registrations_.withLock([&](auto& regs) {
                regs[lib._resolve(name)].insert_or_assign(
                    dispatch,
                    std::make_shared<c10::SafePyObject>(
                        func.release().ptr(), getPyInterpreter()));
              });
            }
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("name"),
          py::arg("dispatch"),
          py::arg("func"),
          py::arg("with_keyset") = false,
          py::arg("skip_if_no_grad") = false)
      .def(
          "define",
          [](const py::object& self,
             const char* schema,
             const char* alias_analysis,
             const std::vector<at::Tag>& tags) {
            auto parsed_schema =
                torch::schema(schema, parseAliasAnalysisKind(alias_analysis));
            self.cast<torch::Library&>().def(
                std::move(parsed_schema), tags, register_or_verify());
            // TODO: this is dumb, had to make a second copy
            return torch::schema(schema, parseAliasAnalysisKind(alias_analysis))
                .name();
          },
          "",
          py::arg("schema"),
          py::arg("alias_analysis") = "",
          py::arg("tags") = std::vector<at::Tag>())
      .def(
          "fallback_fallthrough",
          [](py::object self, const char* dispatch) {
            self.cast<torch::Library&>().fallback(
                dispatch_str(dispatch, CppFunction::makeFallthrough()));
            return self;
          },
          "",
          py::arg("dispatch") = "")
      .def(
          "fallback",
          [](const py::object& self,
             c10::DispatchKey dispatch,
             const py::object& func,
             bool with_keyset) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            if (func.is(py::module::import("torch.library")
                            .attr("fallthrough_kernel"))) {
              lib.fallback(
                  torch::dispatch(dispatch, CppFunction::makeFallthrough()));
            } else {
              lib.fallback(torch::dispatch(
                  dispatch,
                  CppFunction::makeFromBoxedFunctor(
                      std::make_unique<PythonKernelHolder>(
                          func, dispatch, with_keyset, /*with_op*/ true))));
            }
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("dispatch"),
          py::arg("func"),
          py::arg("with_keyset") = false)
      .def(
          "register_ad_inplace_or_view_fallback",
          [](const py::object& self, const char* name) {
            HANDLE_TH_ERRORS
            auto& lib = self.cast<torch::Library&>();
            lib.impl(
                name,
                c10::DispatchKey::ADInplaceOrView,
                torch::autograd::autogradNotImplementedInplaceOrViewFallback());
            END_HANDLE_TH_ERRORS_PYBIND
          },
          "",
          py::arg("name"));

  m.def(
      "_dispatch_library",
      [](const char* kind,
         const std::string& name,
         const char* dispatch,
         const char* file,
         uint32_t linenum) {
        HANDLE_TH_ERRORS
        // Store the file string in global storage to ensure it remains valid
        // for the lifetime of the Library object
        const char* leaked_file =
            leaked_python_filenames_.withLock([&](auto& filenames) {
              filenames.push_back(std::make_unique<std::string>(file));
              return filenames.back()->c_str();
            });

        return std::make_unique<torch::Library>(
            parseKind(kind),
            name,
            std::string(dispatch).empty()
                ? std::nullopt
                : std::make_optional(c10::parseDispatchKey(dispatch)),
            leaked_file,
            linenum);
        END_HANDLE_TH_ERRORS_PYBIND
      },
      "",
      py::arg("kind"),
      py::arg("name"),
      py::arg("dispatch"),
      py::arg("file") = "/dev/null",
      py::arg("linenum") = 0);

  m.def(
      "_dispatch_clear_leaked_python_filenames",
      []() { leaked_python_filenames_.withLock([](auto& f) { f.clear(); }); },
      "Clear the global storage of leaked Python filenames. "
      "WARNING: Only call this if you're sure no Library objects are still using the filenames.");

  m.def(
      "_dispatch_find_schema_or_throw",
      [](const char* name, const char* overload_name) -> c10::OperatorHandle {
        return c10::Dispatcher::singleton().findSchemaOrThrow(
            name, overload_name);
      });

  m.def("_dispatch_dump", [](const char* name) -> std::string {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    if (!op) {
      return "";
    } else {
      return op->dumpState();
    }
  });

  m.def("_dispatch_dump_table", [](const char* name) -> std::string {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    if (!op) {
      return "";
    } else {
      return op->dumpComputedTable();
    }
  });

  m.def("_dispatch_check_invariants", [](const char* name) {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    if (op) {
      op->checkInvariants();
    }
  });

  m.def("_dispatch_check_all_invariants", []() {
    c10::Dispatcher::singleton().checkInvariants();
  });

  m.def("_dispatch_has_kernel", [](const char* name) -> bool {
    auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
    return static_cast<bool>(op);
  });

  m.def(
      // Returns whether or not a direct kernel registration exists
      // for this <op_name, dispatch_key> pair.
      "_dispatch_has_kernel_for_dispatch_key",
      [](const char* name, c10::DispatchKey dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasKernelForDispatchKey(dispatch);
      });

  m.def(
      // Returns whether or not the kernel for this dispatch key is a
      // fallthrough kernel
      "_dispatch_kernel_for_dispatch_key_is_fallthrough",
      [](const char* name, c10::DispatchKey dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        return op->isKernelFallthroughKernel(dispatch);
      });

  m.def(
      "_dispatch_has_kernel_for_any_dispatch_key",
      [](const char* name, c10::DispatchKeySet ks) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasKernelForAnyDispatchKey(ks);
      });

  m.def(
      // Returns whether or not there is an entry in the runtime computed
      // dispatch table, for this <op_name, dispatch_key> pair. For example, if
      // "op" has a `CompositeImplicitAutograd` kernel, Then
      // _dispatch_has_computed_kernel_for_dispatch_key(op, backend) will return
      // true for all backends that are part of the alias set for
      // CompositeImplicitAutograd.
      "_dispatch_has_computed_kernel_for_dispatch_key",
      [](const char* name, const char* dispatch) -> bool {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->hasComputedKernelForDispatchKey(
            c10::parseDispatchKey(dispatch));
      });

  // Bind SafeKernelFunction class
  py::class_<c10::SafeKernelFunction>(m, "_SafeKernelFunction")
      .def(
          "call_boxed",
          [](const c10::SafeKernelFunction& self,
             c10::DispatchKeySet keyset,
             const py::args& args,
             const py::kwargs& kwargs) {
            const auto& op = self.opHandle();
            auto stack = torch::jit::createStackForSchema(
                op.schema(),
                args,
                kwargs,
                /*self=*/std::nullopt);
            self.callBoxed(op, keyset, &stack);
            return torch::jit::createPyObjectForStack(std::move(stack));
          })
      .def(
          "__repr__",
          [](const c10::SafeKernelFunction& self) {
            return "SafeKernelFunction(debug='" + self.debug() + "')";
          })
      .def_property_readonly(
          "op_handle", [](const c10::SafeKernelFunction& self) -> py::object {
            return py::cast(self.opHandle());
          });

  m.def(
      "_dispatch_get_computed_kernel_for_dispatch_key",
      [](const char* name,
         c10::DispatchKey dispatch) -> c10::SafeKernelFunction {
        auto op =
            c10::Dispatcher::singleton().findOp(torch::jit::parseName(name));
        TORCH_CHECK(op, "operator ", name, " does not exist");
        return op->getComputedKernelForDispatchKey(dispatch);
      });

  m.def("_dispatch_find_dangling_impls", []() -> std::vector<std::string> {
    auto danglingImpls = c10::Dispatcher::singleton().findDanglingImpls();

    std::vector<std::string> states;
    states.reserve(danglingImpls.size());
    for (auto& danglingImpl : danglingImpls) {
      states.emplace_back(danglingImpl.dumpState());
    }

    return states;
  });

  m.def("_dispatch_get_all_op_names", []() -> std::vector<std::string> {
    auto op_names = c10::Dispatcher::singleton().getAllOpNames();

    std::vector<std::string> names;
    names.reserve(op_names.size());
    for (auto& op : op_names) {
      std::stringstream ss;
      ss << op.name;
      if (!op.overload_name.empty()) {
        ss << '.' << op.overload_name;
      }
      names.emplace_back(std::move(ss).str());
    }

    return names;
  });

  m.def(
      "_dispatch_tls_set_dispatch_key_excluded",
      [](c10::DispatchKey dispatch_key, bool desired_state) {
        c10::impl::tls_set_dispatch_key_excluded(dispatch_key, desired_state);
      });
  m.def(
      "_dispatch_tls_is_dispatch_key_excluded",
      [](c10::DispatchKey dispatch_key) {
        return c10::impl::tls_is_dispatch_key_excluded(dispatch_key);
      });
  m.def(
      "_dispatch_tls_set_dispatch_key_included",
      [](c10::DispatchKey dispatch_key, bool desired_state) {
        c10::impl::tls_set_dispatch_key_included(dispatch_key, desired_state);
      });
  m.def(
      "_dispatch_tls_is_dispatch_key_included",
      [](c10::DispatchKey dispatch_key) {
        return c10::impl::tls_is_dispatch_key_included(dispatch_key);
      });

  m.def("_dispatch_isTensorSubclassLike", [](const at::Tensor& tensor) {
    return at::isTensorSubclassLike(tensor);
  });

  m.def("_dispatch_key_name", [](c10::DispatchKey k) {
    return c10::toString(k);
  });
  m.def("_dispatch_key_parse", [](c10::DispatchKey k) { return k; });
  m.def("_to_functionality_key", [](c10::DispatchKey k) {
    return c10::toFunctionalityKey(k);
  });
  // E.g. given `DispatchKey::AutogradFunctionality`, returns a keyset of:
  //  AutogradCPU
  //  AutogradCUDA
  //  ...
  //  AutogradPrivateUse3
  m.def("_functionality_to_backend_keys", [](c10::DispatchKey key) {
    std::vector<c10::DispatchKey> keys;
    if (c10::isPerBackendFunctionalityKey(key)) {
      auto ks = c10::DispatchKeySet(key) |
          c10::DispatchKeySet(c10::DispatchKeySet::RAW, c10::full_backend_mask);
      for (auto k : ks) {
        keys.push_back(k);
      }
    } else {
      keys.push_back(key);
    }
    return keys;
  });
  m.def("_dispatch_num_backends", []() { return c10::num_backends; });

#define DEF_ONE(n) .value(#n, c10::DispatchKey::n)

  py::enum_<c10::DispatchKey>(m, "DispatchKey")
      // clang-format off
      DEF_ONE(Undefined)
      DEF_ONE(CompositeExplicitAutogradNonFunctional)
      DEF_ONE(CompositeExplicitAutograd)
      DEF_ONE(CompositeImplicitAutogradNestedTensor)
      DEF_ONE(CompositeImplicitAutograd)
      // NestedTensor is not a backend key
      DEF_ONE(AutogradNestedTensor)
      DEF_ONE(AutogradOther)
      DEF_ONE(Autograd)
      DEF_ONE(Conjugate)
      DEF_ONE(ZeroTensor)
      DEF_ONE(Negative)
      DEF_ONE(BackendSelect)
      DEF_ONE(ADInplaceOrView)
      DEF_ONE(PythonTLSSnapshot)
      DEF_ONE(Python)
      DEF_ONE(FuncTorchDynamicLayerFrontMode)
      DEF_ONE(FuncTorchDynamicLayerBackMode)
      DEF_ONE(FuncTorchBatchedDecomposition)
      DEF_ONE(FuncTorchBatched)
      DEF_ONE(FuncTorchVmapMode)
      DEF_ONE(FuncTorchGradWrapper)
      DEF_ONE(PythonDispatcher)
      // Export DispatchKey::Fake to Python so Python guards and dispatch
      // utilities can construct, inspect, include, or exclude that key.
      DEF_ONE(Fake)
      DEF_ONE(PreDispatch)
      DEF_ONE(Functionalize)
      DEF_ONE(AutocastCPU)
      DEF_ONE(AutocastMPS)
      DEF_ONE(AutocastXPU)
      DEF_ONE(AutocastHPU)
      DEF_ONE(AutocastIPU)
      DEF_ONE(AutocastCUDA)
      DEF_ONE(AutocastPrivateUse1)
  // clang-format on

#define DEF_SINGLE(n, prefix) .value(#prefix #n, c10::DispatchKey::prefix##n)
#define DEF_MULTIPLE(fullname, prefix)              \
  DEF_SINGLE(, fullname)                            \
  DEF_SINGLE(, StartOf##fullname##Backends)         \
  C10_FORALL_BACKEND_COMPONENTS(DEF_SINGLE, prefix) \
  DEF_SINGLE(, EndOf##fullname##Backends)

      // clang-format off
  C10_FORALL_FUNCTIONALITY_KEYS(DEF_MULTIPLE)
  // clang-format on

#undef DEF_MULTIPLE
#undef DEF_SINGLE
          ;

  py::class_<c10::DispatchKeySet>(m, "DispatchKeySet")
      .def(py::init<c10::DispatchKey>())
      .def("__or__", &c10::DispatchKeySet::operator|)
      .def("__sub__", &c10::DispatchKeySet::operator-)
      .def("__and__", &c10::DispatchKeySet::operator&)
      .def("raw_repr", &c10::DispatchKeySet::raw_repr)
      .def("highestPriorityTypeId", &c10::DispatchKeySet::highestPriorityTypeId)
      .def(
          "remove",
          [](c10::DispatchKeySet self, c10::DispatchKey k) {
            return self.remove(k);
          })
      .def(
          "add",
          [](c10::DispatchKeySet self, c10::DispatchKey k) {
            return self.add(k);
          })
      .def("has", &c10::DispatchKeySet::has)
      .def("__repr__", [](c10::DispatchKeySet d) { return c10::toString(d); })
      .def(
          "__eq__",
          [](c10::DispatchKeySet self, c10::DispatchKeySet other) {
            return self.raw_repr() == other.raw_repr();
          })
      .def(py::pickle(
          [](const c10::DispatchKeySet&
                 obj) { // __getstate__ : creates tuple of state
            return py::make_tuple(obj.raw_repr());
          },
          [](const py::tuple& t) { // __setstate__ : restores state from tuple
            TORCH_CHECK(
                t.size() == 1, "__setstate__ expected tuple with one element");
            return c10::DispatchKeySet::from_raw_repr(t[0].cast<uint64_t>());
          }))
      .def_static("from_raw_repr", &c10::DispatchKeySet::from_raw_repr);

  m.attr("_dispatch_autogradother_backends") =
      py::cast(c10::autogradother_backends);

  m.attr("_additional_keys_to_prop_for_wrapper_tensors") =
      py::cast(at::functorch::kKeysToPropagateToWrapper);

  m.attr("_after_autograd_keyset") = py::cast(c10::after_autograd_keyset);
  m.attr("_after_ADInplaceOrView_keyset") =
      py::cast(c10::after_ADInplaceOrView_keyset);

  m.def("_dispatch_has_backend_fallback", [](c10::DispatchKey t) {
    return c10::Dispatcher::singleton().hasBackendFallbackForDispatchKey(t);
  });

  m.def("_dispatch_keyset_full_after", [](c10::DispatchKey t) {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, t);
  });

  m.def("_dispatch_keyset_full", []() {
    return c10::DispatchKeySet(c10::DispatchKeySet::FULL);
  });

  m.def("_dispatch_is_alias_key", c10::isAliasDispatchKey);

  m.def("_dispatch_keyset_to_string", [](c10::DispatchKeySet keyset) {
    return c10::toString(keyset);
  });

  m.def("_dispatch_get_backend_keyset_from_autograd", [](c10::DispatchKey k) {
    return c10::getBackendKeySetFromAutograd(k);
  });

  m.def("_dispatch_keys", [](const at::Tensor& tensor) {
    auto* impl = tensor.unsafeGetTensorImpl();
    return impl->key_set();
  });
  // Save the current local dispatch key set onto a C++ thread-local stack, to
  // be restored by _dynamo_restore_local_dispatch_key_set. Used by
  // torch.compile's per-call wrapper to preserve dispatch state across the
  // compiled call without constructing pybind11 DispatchKeySet instances.
  m.def("_dynamo_save_local_dispatch_key_set", []() {
    dynamoDispatchKeySetStack().push_back(
        c10::impl::tls_local_dispatch_key_set());
  });
  m.def("_dynamo_restore_local_dispatch_key_set", []() {
    auto& stack = dynamoDispatchKeySetStack();
    TORCH_CHECK(
        !stack.empty(),
        "_dynamo_restore_local_dispatch_key_set called with an empty stack");
    c10::impl::_force_tls_local_dispatch_key_set(stack.back());
    stack.pop_back();
  });
  m.def("_dispatch_tls_local_include_set", []() {
    return c10::impl::tls_local_dispatch_key_set().included_;
  });
  m.def("_dispatch_tls_local_exclude_set", []() {
    return c10::impl::tls_local_dispatch_key_set().excluded_;
  });
  m.def("_functionalization_reapply_views_tls", []() {
    return at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  });
  m.def(
      "_dispatch_is_included_in_alias",
      [](c10::DispatchKey a, c10::DispatchKey b) {
        return c10::isIncludedInAlias(a, b);
      });

  // DEPRECATED, please don't use this. Instead use
  // torch._C._ExcludeDispatchKeyGuard
  py_context_manager_DEPRECATED<
      c10::impl::ExcludeDispatchKeyGuard,
      c10::DispatchKeySet>(m, "ExcludeDispatchKeyGuard");

  py_context_manager<
      c10::impl::ForceDispatchKeyGuard,
      c10::DispatchKeySet,
      c10::DispatchKeySet>(m, "_ForceDispatchKeyGuard");
  py_context_manager<c10::impl::ForceDispatchKeyGuard>(
      m, "_PreserveDispatchKeyGuard");
  py_context_manager<c10::impl::IncludeDispatchKeyGuard, c10::DispatchKey>(
      m, "_IncludeDispatchKeyGuard");
  py_context_manager<c10::impl::ExcludeDispatchKeyGuard, c10::DispatchKeySet>(
      m, "_ExcludeDispatchKeyGuard");
  py_context_manager<SetExcludeDispatchKeyGuard, c10::DispatchKey, bool>(
      m, "_SetExcludeDispatchKeyGuard");

  py_context_manager_DEPRECATED<at::AutoDispatchBelowAutograd>(
      m, "_AutoDispatchBelowAutograd");
  py_context_manager<at::AutoDispatchBelowADInplaceOrView>(
      m, "_AutoDispatchBelowADInplaceOrView");

  // Prints out the name of every operator that has a kernel registered to the
  // Dispatcher under [dispatch_key]. If no arguments are specified, it'll print
  // out the name of every operator that the Dispatcher knows of. This can be
  // useful to answer questions like "list all operators that do not have a CPU
  // kernel".
  m.def(
      "_dispatch_print_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key).empty()
            ? std::nullopt
            : std::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        for (auto& op : op_names) {
          std::cout << op << '\n';
        }
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));

  m.def(
      "_parse_dispatch_key",
      [](const char* dispatch_key) -> std::optional<c10::DispatchKey> {
        try {
          return c10::parseDispatchKey(dispatch_key);
        } catch (const c10::Error& e) {
          (void)e;
          return std::nullopt;
        }
      });

  m.def(
      "_dispatch_get_registrations_for_dispatch_key",
      [](const char* dispatch_key = "") {
        auto k = std::string(dispatch_key).empty()
            ? std::nullopt
            : std::make_optional(c10::parseDispatchKey(dispatch_key));
        auto op_names =
            c10::Dispatcher::singleton().getRegistrationsForDispatchKey(k);
        std::vector<std::string> names;
        names.reserve(op_names.size());
        for (auto& op : op_names) {
          names.emplace_back(
              op.name +
              (op.overload_name.empty() ? "" : "." + op.overload_name));
        }
        return names;
      },
      py::arg("dispatch_key") = static_cast<const char*>(""));
  m.def(
      "_dispatch_set_report_error_callback",
      [](c10::OperatorHandle& handle, py::object callback) {
        auto obj = callback.release().ptr();
        auto callback_obj =
            std::make_unique<c10::SafePyObject>(obj, getPyInterpreter());
        handle.setReportErrorCallback_(std::move(callback_obj));
      });

  m.def("_dispatch_pystub", [](const char* name, const char* overload) {
    return c10::Dispatcher::singleton().getPyStub(
        c10::OperatorName(name, overload));
  });

  m.def("_replace_", [](const at::Tensor& a, const at::Tensor& b) {
    at::functionalization::impl::replace_(a, b);
  });
  m.def("_propagate_xla_data", [](const at::Tensor& a, const at::Tensor& b) {
    at::functionalization::impl::propagate_xla_data(a, b);
  });
  m.def("_commit_update", [](const at::Tensor& a) {
    at::functionalization::impl::commit_update(a);
  });
  m.def("_unsafe_reset_storage", [](const at::Tensor& a) {
    at::functionalization::impl::unsafe_reset_storage(a);
  });

  m.def("_dispatch_key_for_device", [](const std::string& device_type) {
    auto device = c10::Device(device_type);
    TORCH_CHECK(
        !device.has_index(),
        "Expected device_type string to not have a device index; got ",
        device_type);
    return c10::toString(
        c10::computeDispatchKey(std::nullopt, std::nullopt, device));
  });

  m.def("_are_functorch_transforms_active", []() {
    auto include_set = c10::impl::tls_local_dispatch_key_set().included_;
    return (
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        include_set.has(c10::DispatchKey::FuncTorchDynamicLayerBackMode));
  });

  m.def("_autocast_supported_devices", []() {
    std::vector<std::string> result;
    for (const auto device_type : at::autocast::_AUTOCAST_SUPPORTED_DEVICES) {
      result.emplace_back(
          c10::DeviceTypeName(device_type, /*lower_case*/ true));
    }
    return result;
  });

  m.def("_get_nested_int", [](int64_t data, int64_t coeff) {
    return c10::SymInt(c10::SymNode(
        c10::make_intrusive<c10::NestedIntSymNodeImpl>(data, coeff)));
  });

  m.def("_get_constant_bool_symnode", [](int64_t data) {
    return c10::SymNode(
        c10::make_intrusive<c10::ConstantSymNodeImpl<bool>>(data));
  });

  m.def("_non_sym_sizes", [](const at::Tensor& a) {
    return a.sizes(); // NB: NOT sym_size
  });

  m.def("_set_throw_on_mutable_data_ptr", [](const at::Tensor& t) {
    if (!t.unsafeGetTensorImpl()->has_storage()) {
      // If the Tensor doesn't have a storage, then accessing .data_ptr()
      // will already raise an error.
      return;
    }
    // Otherwise, set (on the StorageImpl) that accessing (mutable) data_ptr
    // will throw.
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_throw_on_mutable_data_ptr();
  });

  // Invariant: you must ONLY call this with FakeTensors.
  m.def("_set_warn_deprecated_on_mutable_data_ptr", [](const at::Tensor& t) {
    if (!t.unsafeGetTensorImpl()->has_storage()) {
      // If the Tensor doesn't have a storage, then accessing .data_ptr()
      // will already raise an error.
      return;
    }
    t.unsafeGetTensorImpl()
        ->storage()
        .unsafeGetStorageImpl()
        ->set_warn_deprecated_on_mutable_data_ptr();
  });

  m.def("_only_lift_cpu_tensors", &torch::utils::only_lift_cpu_tensors);
  m.def("_set_only_lift_cpu_tensors", &torch::utils::set_only_lift_cpu_tensors);

  m.def(
      "_get_dtensor_allow_implicit_replication",
      &at::get_dtensor_allow_implicit_replication);
  m.def(
      "_set_dtensor_allow_implicit_replication",
      &at::set_dtensor_allow_implicit_replication);

  using c10::impl::TorchDispatchModeKey;
  py::enum_<TorchDispatchModeKey>(m, "_TorchDispatchModeKey")
      .value("FUNCTIONAL", TorchDispatchModeKey::FUNCTIONAL)
      .value("PROXY", TorchDispatchModeKey::PROXY)
      .value("FAKE", TorchDispatchModeKey::FAKE);
}

// TODO: dedupe with the kernel
// NOLINTNEXTLINE(misc-use-internal-linkage)
void python_op_registration_trampoline_impl(
    const c10::OperatorHandle& op,
    c10::DispatchKey key,
    c10::DispatchKeySet keyset,
    torch::jit::Stack* stack,
    bool with_keyset,
    bool with_op) {
  auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
  py::gil_scoped_acquire g;
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  auto func = python_registrations_.withLock(
      [&](auto& regs) { return regs[op.operator_name()][key]; });
  TORCH_INTERNAL_ASSERT(func != nullptr);
  auto* pyobj = func->ptr(getPyInterpreter());
  TORCH_INTERNAL_ASSERT(pyobj != nullptr);
  auto callable = py::reinterpret_borrow<py::object>(pyobj);
  auto obj = with_op ? with_keyset ? callable(
                                         keyset,
                                         torch::detail::getTorchApiFunction(op),
                                         *args_kwargs.first,
                                         **args_kwargs.second)
                                   : callable(
                                         torch::detail::getTorchApiFunction(op),
                                         *args_kwargs.first,
                                         **args_kwargs.second)
      : with_keyset ? callable(keyset, *args_kwargs.first, **args_kwargs.second)
                    : callable(*args_kwargs.first, **args_kwargs.second);
  TORCH_CHECK_PYTHON(obj);
  pushPyOutToStack(op, stack, obj, "PythonKernelHolder");
}

} // namespace torch::impl::dispatch
