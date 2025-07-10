#include <torch/csrc/utils/python_arg_parser.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_torch_function_mode.h>
#include <torch/csrc/utils/torch_dispatch_mode.h>

#include <ATen/ATen.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/TracerMode.h>
#include <c10/util/irange.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {

static std::unordered_map<std::string, ParameterType> type_map = {
    {"Tensor", ParameterType::TENSOR},
    {"Scalar", ParameterType::SCALAR},
    {"int64_t", ParameterType::INT64},
    {"SymInt", ParameterType::SYM_INT},
    {"double", ParameterType::DOUBLE},
    {"complex", ParameterType::COMPLEX},
    {"TensorList", ParameterType::TENSOR_LIST},
    {"c10::List<::std::optional<Tensor>>", ParameterType::TENSOR_LIST},
    {"IntArrayRef", ParameterType::INT_LIST},
    {"SymIntArrayRef", ParameterType::SYM_INT_LIST},
    {"ArrayRef<double>", ParameterType::FLOAT_LIST},
    {"Generator", ParameterType::GENERATOR},
    {"bool", ParameterType::BOOL},
    {"Storage", ParameterType::STORAGE},
    {"PyObject*", ParameterType::PYOBJECT},
    {"ScalarType", ParameterType::SCALARTYPE},
    {"Layout", ParameterType::LAYOUT},
    {"MemoryFormat", ParameterType::MEMORY_FORMAT},
    {"QScheme", ParameterType::QSCHEME},
    {"Device", ParameterType::DEVICE},
    {"DeviceIndex", ParameterType::INT64},
    {"Stream", ParameterType::STREAM},
    {"std::string", ParameterType::STRING},
    {"std::string_view", ParameterType::STRING},
    {"::std::string_view", ParameterType::STRING},
    {"Dimname", ParameterType::DIMNAME},
    {"DimnameList", ParameterType::DIMNAME_LIST},
    {"ScalarList", ParameterType::SCALAR_LIST},
    {"DispatchKeySet", ParameterType::DISPATCH_KEY_SET},
};

// Default arg name translations for compatibility with NumPy.
//
// Example:
// ```python
// t = torch.randn(10,10)
// torch.sum(a=t, axis=0, keepdim=True)
// ```
//
// A vector is necessary, because we might need to try multiple values.
// In particular, NumPy sometimes uses "x" and sometimes "a" for the main input
// tensor. Rather than annotate each function separately with whether it should
// take "x" or "a", just try both.
//
// TODO: Allow individual functions to specify non-default translations:
// For example, `torch.pow` should translate "exponent" to "x2".
static const std::unordered_map<std::string, std::vector<std::string>>
    numpy_compatibility_arg_names = {
        {"dim", {"axis"}},
        {"keepdim", {"keepdims"}},
        {"input", {"x", "a", "x1"}},
        {"other", {"x2"}},
};

// TODO: remove this. This is a temporary list of functions that allow Python
// numbers to bind to Tensors. Some binary ops have separate Tensor and Scalar
// overloads and binding to the Tensor overload with a number of a different
// type will trigger a type error.
//
// If you modify this, you will need to adjust the blocklist in
// tools/pyi/gen_pyi.py (and add hardcoded signatures for these
// functions.)
bool should_allow_numbers_as_tensors(const std::string& name) {
  static std::unordered_set<std::string> allowed = {
      "add",
      "add_",
      "add_out",
      "div",
      "div_",
      "div_out",
      "divide",
      "divide_",
      "divide_out", // alias of div
      "mul",
      "mul_",
      "mul_out",
      "multiply",
      "multiply_",
      "multiply_out", // alias of mul
      "sub",
      "sub_",
      "sub_out",
      "subtract",
      "subtract_",
      "subtract_out", // alias of sub
      "true_divide",
      "true_divide_",
      "true_divide_out",
      "to",
      "_to_copy",
      "copy_",
      "copy",
      "floor_divide",
      "floor_divide_",
      "floor_divide_out",
      "_conj"}; // _conj needed because mul.Tensor backward calls it
  return allowed.find(name) != allowed.end();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
FunctionParameter::FunctionParameter(const std::string& fmt, bool keyword_only)
    : optional(false),
      allow_none(false),
      keyword_only(keyword_only),
      size(0),
      default_scalar(0) {
  auto space = fmt.find(' ');
  if (space == std::string::npos) {
    throw std::runtime_error("FunctionParameter(): missing type: " + fmt);
  }

  auto type_str = fmt.substr(0, space);

  auto question = type_str.find('?');
  if (question != std::string::npos) {
    allow_none = true;
    type_str = type_str.substr(0, question);
  }

  // Parse and remove brackets from type_str
  auto bracket = type_str.find('[');
  if (bracket != std::string::npos) {
    auto size_str =
        type_str.substr(bracket + 1, type_str.length() - bracket - 2);
    size = atoi(size_str.c_str());
    type_str = type_str.substr(0, bracket);
  }

  auto name_str = fmt.substr(space + 1);
  auto it = type_map.find(type_str);
  if (it == type_map.end()) {
    throw std::runtime_error(
        "FunctionParameter(): invalid type string: " + type_str);
  }
  type_ = it->second;

  auto eq = name_str.find('=');
  if (eq != std::string::npos) {
    name = name_str.substr(0, eq);
    optional = true;
    set_default_str(name_str.substr(eq + 1));
  } else {
    name = name_str;
  }
  python_name = THPUtils_internString(name);
  auto np_compat_it = numpy_compatibility_arg_names.find(name);
  if (np_compat_it != numpy_compatibility_arg_names.end()) {
    for (const auto& str : np_compat_it->second) {
      numpy_python_names.push_back(THPUtils_internString(str));
    }
  }
}

auto handle_torch_function_getter(
    THPVariable* self,
    const std::string& property_name) -> PyObject* {
  py::object torch_api = PyObject_FastGetAttrString(
      THPVariableClass, (char*)property_name.c_str());
  std::string module_name = "torch.Tensor." + property_name;
  return handle_torch_function(
      (PyObject*)self,
      "__get__",
      nullptr,
      nullptr,
      torch_api.ptr(),
      module_name);
}

auto handle_torch_function_setter(
    THPVariable* self,
    const std::string& property_name,
    PyObject* value) -> int {
  py::object torch_api = PyObject_FastGetAttrString(
      THPVariableClass, (char*)property_name.c_str());
  std::string module_name = "torch.Tensor." + property_name;
  if (value != nullptr) {
    py::tuple args_ = py::make_tuple(py::handle(value));
    handle_torch_function(
        (PyObject*)self,
        "__set__",
        args_.ptr(),
        nullptr,
        torch_api.ptr(),
        module_name);
  } else {
    handle_torch_function(
        (PyObject*)self,
        "__delete__",
        nullptr,
        nullptr,
        torch_api.ptr(),
        module_name);
  }
  return 0;
}

// Combines self and args into one tuple.
static auto combine_self_args(PyObject* self, PyObject* args) -> py::tuple {
  if (args == nullptr) {
    return py::make_tuple(py::handle(self));
  } else if (self == nullptr) {
    return py::reinterpret_borrow<py::tuple>(args);
  }

  auto py_args = py::reinterpret_borrow<py::tuple>(args);
  size_t n = py_args.size();
  auto args_ = py::tuple(n + 1);
  args_[0] = py::handle(self);
  for (const auto i : c10::irange(n)) {
    args_[i + 1] = py_args[i];
  }
  return args_;
}

auto handle_torch_function(
    PyObject* self,
    const std::string& func_name,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const std::string& module_name) -> PyObject* {
  py::object torch_api_function =
      PyObject_FastGetAttrString(torch_api, (char*)func_name.c_str());
  TORCH_INTERNAL_ASSERT(
      torch_api_function.ptr() != nullptr, "torch API function must exist");
  py::tuple args_ = combine_self_args(self, args);
  return handle_torch_function_no_python_arg_parser(
      {self},
      args_.ptr(),
      kwargs,
      func_name.c_str(),
      torch_api_function.ptr(),
      module_name.c_str(),
      TorchFunctionName::TorchFunction);
}

// Note: [Overloaded args]
// An overloaded arg may be one of the following:
// - an instance of an object that has a __torch_function__ method
// - an instance of an object that has a __torch_dispatch__ classmethod
// - a class type that has a __torch_dispatch__ classmethod
//
// This function returns the type of the arg (if the arg is an instance),
// otherwise, it returns the arg.
static PyObject* get_type_of_overloaded_arg(PyObject* obj_or_type) {
  if (PyType_Check(obj_or_type)) {
    return obj_or_type;
  }
  return (PyObject*)Py_TYPE(obj_or_type);
}

static py::object maybe_get_registered_torch_dispatch_rule(
    PyObject* torch_api_function,
    const py::object& torch_dispatch_object) {
  // This is a static object, so we must leak the Python object
  // "release()" is used here to preserve 1 refcount on the
  // object, preventing it from ever being de-allocated by CPython.
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  py::object find_torch_dispatch_rule =
      storage
          .call_once_and_store_result([]() -> py::object {
            return py::module_::import("torch._library.simple_registry")
                .attr("find_torch_dispatch_rule");
          })
          .get_stored();
#else
  static const py::handle find_torch_dispatch_rule =
      py::object(py::module_::import("torch._library.simple_registry")
                     .attr("find_torch_dispatch_rule"))
          .release();
#endif
  auto result = find_torch_dispatch_rule(
      py::reinterpret_borrow<py::object>(torch_api_function),
      py::type::handle_of(torch_dispatch_object));
  return result;
}

static py::object dispatch_on_subclass(
    PyObject* args,
    PyObject* kwargs,
    at::ArrayRef<PyObject*> overloaded_args,
    py::tuple py_types,
    PyObject* torch_api_function,
    bool is_torch_function,
    const char* torch_function_name_str,
    std::optional<c10::impl::TorchDispatchModeKey> maybe_mode_key =
        std::nullopt) {
  py::object ret;
  for (auto& arg : overloaded_args) {
    py::object torch_function =
        PyObject_FastGetAttrString(arg, torch_function_name_str);
    if (!torch_function) {
      TORCH_INTERNAL_ASSERT(0);
    }
    if (torch_function.ptr() == torch::disabled_torch_dispatch_impl()) {
      // During __torch_dispatch__, don't dispatch on args with a disabled
      // torch_dispatch. This code runs before infra modes, so we need to make
      // sure that infra modes can run first. (In theory, maybe we can rearrange
      // things so that infra modes are *always* attempted first, and just
      // return NotImplemented when there are any user subclasses. Maybe that
      // would fix this problem?)
      continue;
    }

    // See https://github.com/pytorch/pytorch/issues/63767
    if (is_torch_function &&
        PyObject_FastGetAttrString(torch_function.ptr(), "__self__")
            .is(py::handle(arg)) &&
        torch_function.ptr() != torch::disabled_torch_function_impl()) {
      TORCH_WARN_ONCE(
          "Defining your `",
          torch_function_name_str,
          "` as a plain method is deprecated ",
          "and will be an error in future, please define it as a classmethod.");
    }

    if (!is_torch_function) {
      auto maybe_torch_dispatch_rule = maybe_get_registered_torch_dispatch_rule(
          torch_api_function, py::reinterpret_borrow<py::object>(arg));
      if (!maybe_torch_dispatch_rule.is_none()) {
        torch_function = maybe_torch_dispatch_rule;
        auto py_arg = py::reinterpret_borrow<py::object>(arg);
        ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
            torch_function.ptr(),
            py::type::handle_of(py_arg).ptr(),
            torch_api_function,
            py_types.ptr(),
            args,
            kwargs,
            NULL));
        if (ret.ptr() == nullptr) {
          throw python_error();
        }
        if (ret.ptr() != Py_NotImplemented) {
          break;
        }
      }
    }

    ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
        torch_function.ptr(),
        torch_api_function,
        py_types.ptr(),
        args,
        kwargs,
        NULL));
    if (ret.ptr() == nullptr) {
      throw python_error();
    }
    if (ret.ptr() != Py_NotImplemented) {
      // Return the reference to the result. This also covers the case where
      // ret is NULL and __torch_function__/__torch_dispatch raised an
      // exception, which we throw below
      break;
    }
  }
  return ret;
}

static std::tuple<py::object, py::object> dispatch_on_mode(
    PyObject* args,
    PyObject* kwargs,
    py::tuple py_types,
    PyObject* torch_api_function,
    bool is_torch_function,
    const char* torch_function_name_str) {
  // Disable mode on the inside; this makes for a more user-friendly
  // experience if you try to, e.g., print your tensors.
  std::optional<torch::overrides::StashTorchFunctionModeGuard> tf_g;
  std::optional<torch_dispatch_mode::StashTorchDispatchModeGuard> td_g;
  py::object mode_obj;
  // NB: We only really need keep the mode_obj live if the function call
  // fails for error reporting, but whatever, Python refcounts are cheap
  if (is_torch_function) {
    tf_g.emplace();
    mode_obj = py::reinterpret_borrow<py::object>(
        tf_g->get_cur_mode()->ptr(getPyInterpreter()));
  } else {
    td_g.emplace();
    mode_obj = py::reinterpret_borrow<py::object>(
        td_g->get_cur_mode()->ptr(getPyInterpreter()));
  }
  py::object torch_function =
      PyObject_FastGetAttrString(mode_obj.ptr(), torch_function_name_str);
  if (!torch_function) {
    TORCH_INTERNAL_ASSERT(0);
  }
  TORCH_INTERNAL_ASSERT(py_types.ptr() != nullptr);
  TORCH_INTERNAL_ASSERT(args != nullptr);

  TORCH_CHECK(
      PyObject_FastGetAttrString(torch_function.ptr(), "__self__").is(mode_obj),
      "Defining your mode's `",
      torch_function_name_str,
      "` as a classmethod is not supported, please make it a plain method");

  if (!is_torch_function) {
    auto maybe_torch_dispatch_rule =
        maybe_get_registered_torch_dispatch_rule(torch_api_function, mode_obj);
    if (!maybe_torch_dispatch_rule.is_none()) {
      auto ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
          maybe_torch_dispatch_rule.ptr(),
          mode_obj.ptr(),
          torch_api_function,
          py_types.ptr(),
          args,
          kwargs,
          NULL));
      if (ret.ptr() == nullptr) {
        throw python_error();
      }
      return std::make_tuple(ret, mode_obj);
    }
  }

  // Blegh.  This accidentally works in PyObject_CallFunctionObjArgs below
  // because the nullptr terminates the argument list ick ick ick.
  py::object ret;
  if (kwargs == nullptr) {
    ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
        mode_obj.ptr(),
        torch_function_name_str,
        "OOO",
        torch_api_function,
        py_types.ptr(),
        args));
  } else {
    ret = py::reinterpret_steal<py::object>(PyObject_CallMethod(
        mode_obj.ptr(),
        torch_function_name_str,
        "OOOO",
        torch_api_function,
        py_types.ptr(),
        args,
        kwargs));
  }
  if (ret.ptr() == nullptr) {
    throw python_error();
  }
  return std::make_tuple(ret, mode_obj);
}

// See Note: [Overloaded args] for what they hold
auto handle_torch_function_no_python_arg_parser(
    at::ArrayRef<PyObject*> overloaded_args,
    PyObject* args,
    PyObject* kwargs,
    const char* func_name,
    PyObject* torch_api_function,
    const char* module_name,
    TorchFunctionName torch_function_name) -> PyObject* {
  const char* torch_function_name_str = nullptr;
  switch (torch_function_name) {
    case TorchFunctionName::TorchFunction:
      torch_function_name_str = "__torch_function__";
      break;
    case TorchFunctionName::TorchDispatch:
      torch_function_name_str = "__torch_dispatch__";
      break;
    default:
      TORCH_INTERNAL_ASSERT(0, static_cast<int>(torch_function_name));
  }
  // overloaded_args already all have unique types
  // nb: modes don't go in the overloaded types list, as they are not
  // necessarily types
  std::vector<py::object> overloaded_types;
  overloaded_types.reserve(overloaded_args.size());
  for (auto& arg : overloaded_args) {
    overloaded_types.push_back(
        py::reinterpret_borrow<py::object>(get_type_of_overloaded_arg(arg)));
  }
  py::tuple py_types = py::cast(overloaded_types);
  py::object ret;
  py::object mode_obj;

  // Step 1: Try to dispatch based on the mode stack, *ignoring* infra
  // torch_dispatch modes.
  const bool is_torch_function =
      torch_function_name == TorchFunctionName::TorchFunction;
  const auto is_mode_active = [&]() {
    return is_torch_function
        ? at::impl::torch_function_mode_enabled()
        // Check if any *user* torch_dispatch modes are active (not including
        // fake and proxy modes, which are special)
        : c10::impl::dispatch_mode_enabled();
  };
  // Note [__torch_dispatch__ dispatching order]
  // The high-level idea motivating the dispatching
  // order below is that: (1) modes get higher dispatch precedence over
  // subclasses (2) "user" modes/subclasses get higher dispatch precedence over
  // "infra" modes/subclasses.
  //
  // To give a complete example: let's say we are running torch.compile, with
  // the following "user" modes and subclasses:
  //   mode_stack: [ModeA]
  //   user_args: [MyWrapperSubclassB(torchTensor)]

  // During tracing in AOTAutograd tracing, we use some additional infra modes
  // and subclasses to perform tracing:
  //   FunctionalTensorMode, ProxyTorchDispatchMode, FakeTensorMode,
  //   FunctionalTensor, FakeTensor
  // The modified mode stack and tracing arguments will look like this:
  //   mode_stack (user modes): [ModeA]
  //   mode_stack (infra modes): [
  //     FunctionalTensorMode, ProxyTorchDispatchMode, FakeTensorMode
  //   ]
  //   tracing_args: [
  //     MyWrapperSubclassB(FunctionalTensor(_to_functional_tensor(FakeTensor)))
  //   ]

  // And the dispatching order that we want is as follows:
  // (1) ModeA.__torch_dispatch__ (user modes highest)
  // (2) MyWrapperSubclassB.__torch_dispatch__ (user subclasses next highest)
  // (3) FunctionalTensorMode.__torch_dispatch__ (infra modes next highest)
  // (4) ProxyTorchDispatchMode.__torch_dispatch__ (infra modes next highest)
  // (5) FakeTensorMode.__torch_dispatch__ (infra modes next highest)
  // (6) FakeTensor.__torch_fake_dispatch__ (infra subclasses next highest)

  // Why does do FunctionalTensor and FakeTensor even need to be special-cased
  // in the ordering?
  // In theory we could remove their __torch_dispatch__, but both of these
  // subclasses override sizes/strides metadata calls with __torch_dispatch__,
  // which would mean a mode would be **required** to access their metadata.

  if (is_mode_active()) {
    // Step 1: Try to dispatch on any user TorchDispatchModes (including infra
    // modes, which will always be at the bottom of the mode stack).
    std::tie(ret, mode_obj) = dispatch_on_mode(
        args,
        kwargs,
        py_types,
        torch_api_function,
        is_torch_function,
        torch_function_name_str);
  }

  // Step 2: Try to dispatch based on any user subclasses,
  // ignoring any subclasses that have a _mode_key field
  // (corresponding to infra subclasses)
  // Note: user subclasses should always run *before* infra modes like
  // proxy/fake. This is handles by having proxy/fake modes return
  // NotImplemented when they see a user subclass that they don't understand.
  if (ret.ptr() == nullptr || ret.ptr() == Py_NotImplemented) {
    auto curr_ret = dispatch_on_subclass(
        args,
        kwargs,
        overloaded_args,
        py_types,
        torch_api_function,
        is_torch_function,
        torch_function_name_str);
    if (curr_ret.ptr() != nullptr) {
      ret = curr_ret;
    }
  }

  if (ret.ptr() == nullptr) {
    // if an exception occurred in a user's implementation of
    // __torch_function__, throw it
    throw python_error();
  } else if (ret.ptr() == Py_NotImplemented) {
    // all __torch_function__ implementations in overloaded_args
    // returned NotImplemented, so we raise a TypeError.
    std::stringstream ss;
    ss << "Multiple dispatch failed for '";
    if (module_name && func_name) {
      ss << module_name << "." << func_name;
    } else {
      py::handle fn = torch_api_function;
      ss << py::str(fn.attr("__module__")) << "."
         << py::str(fn.attr("__name__"));
    }
    ss << "'; all " << torch_function_name_str
       << " handlers returned NotImplemented:\n\n";
    if (mode_obj) {
      ss << "  - mode object " << py::repr(mode_obj) << "\n";
    }
    for (auto& arg : overloaded_args) {
      ss << "  - tensor subclass " << py::repr(get_type_of_overloaded_arg(arg))
         << "\n";
    }
    ss << "\nFor more information, try re-running with TORCH_LOGS=not_implemented";
    const std::string& tmp = ss.str();
    PyErr_SetString(PyExc_TypeError, tmp.c_str());
    throw python_error();
  }
  return ret.release().ptr();
}

auto handle_torch_function(
    PythonArgs& r,
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const char* module_name,
    const char* func_name_override) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(
      torch_api,
      (char*)(func_name_override ? func_name_override
                                 : r.get_func_name().c_str()));
  TORCH_INTERNAL_ASSERT(
      torch_api_function.ptr() != nullptr, "torch API function must exist");
  py::tuple args_ = combine_self_args(self, args);
  return handle_torch_function_no_python_arg_parser(
      r.overloaded_args,
      args_.ptr(),
      kwargs,
      r.get_func_name().c_str(),
      torch_api_function.ptr(),
      module_name);
}

auto handle_torch_function(
    PythonArgs& r,
    PyObject* args,
    PyObject* kwargs,
    PyObject* torch_api,
    const char* module_name,
    const char* func_name_override) -> PyObject* {
  return handle_torch_function(
      r, nullptr, args, kwargs, torch_api, module_name, func_name_override);
}

auto handle_torch_function_indexing(
    PyObject* self,
    PyObject* index,
    PyObject* val) -> PyObject* {
  const char* func_name = (val == nullptr) ? "__getitem__" : "__setitem__";
  py::object index_tup;
  if (PyTuple_Check(index)) {
    index_tup = py::reinterpret_borrow<py::object>(index);
  } else {
    index_tup = py::make_tuple(py::handle(index));
  }
  std::vector<PyObject*> overridable_args;
  is_tensor_and_append_overloaded(self, &overridable_args);
  auto size = PyTuple_GET_SIZE(index_tup.ptr());
  for (auto i : c10::irange(size)) {
    auto* obj = PyTuple_GetItem(index_tup.ptr(), i);
    is_tensor_and_append_overloaded(obj, &overridable_args);
  }
  if (val != nullptr) {
    is_tensor_and_append_overloaded(val, &overridable_args);
  }
  py::object func =
      PyObject_FastGetAttrString(THPVariableClass, (char*)func_name);
  py::object args = (val == nullptr)
      ? py::make_tuple(py::handle(self), py::handle(index))
      : py::make_tuple(py::handle(self), py::handle(index), py::handle(val));
  return handle_torch_function_no_python_arg_parser(
      overridable_args,
      args.ptr(),
      nullptr,
      func_name,
      func.ptr(),
      "torch.Tensor");
}

/*
 *  obj has a __torch_function__ implementation and may either be a
 *  subclass of Tensor or a Tensor-like duck type. We may need to
 *  append this object to the overloaded_args vector, which tracks all
 *  of the arguments with distinct __torch_function__ implementations
 *  we've seen so far.
 *
 *  If this is the first argument we've seen with __torch_function__
 *  defined, we unconditionally add obj to the overloaded_args vector.
 *
 *  If we've already seen arguments with __torch_function__ defined,
 *  then we first need to check if obj is the same type as any of the
 *  entries in overloaded_args.  If so, we can ignore obj since we
 *  already have an entry in overloaded_args with the same
 *  __torch_function__ implementation.
 *
 *  If it's a different type, we then need to check if it's a subclass
 *  of one of the types we've already seen. If so, we need to insert an
 *  entry in overloaded_args for this type with higher precedence than
 *  the superclass.
 *
 *  See torch._overrides._get_overloaded_args for the equivalent
 *  function in the Python __torch_function__ implementation.
 *
 *  The precedence-determining algorithm implemented in this function is
 *  described in NEP-0018:
 *  https://numpy.org/neps/nep-0018-array-function-protocol.html
 *
 *  'overloaded_args' is a raw pointer to a vector of pybind11 handles
 *  that have distinct __torch_function__ implementations, in order of calling
 *  precedence.
 *
 *  'obj' is an object to check for a __torch_function__ implementation
 *
 * If changing this file in a way that can affect the __torch_function__
 * overhead, please report the benchmarks in 'benchmarks/overrides_benchmark'.
 * See the instructions in the 'README.md' in that directory.
 *
 */

static void append_overloaded_arg(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj,
    bool obj_is_type) {
  bool class_not_seen_yet = true;
  PyObject* obj_type = obj_is_type ? obj : (PyObject*)Py_TYPE(obj);
  for (auto& arg : *overloaded_args) {
    if (obj_type == get_type_of_overloaded_arg(arg)) {
      // obj is the same type as another parameter we've seen in a prior
      // iteration of the loop over parameters so we already have an entry
      // with the proper __torch_function__ implementation to call, so skip
      // this parameter
      class_not_seen_yet = false;
      break;
    }
  }
  if (class_not_seen_yet) {
    auto arg_index = overloaded_args->size();
    for (const auto j : c10::irange(arg_index)) {
      if (PyObject_IsSubclass(
              obj_type, get_type_of_overloaded_arg((*overloaded_args)[j]))) {
        // obj is a subclass of another object we've seen already so its
        // __torch_function__ should be called first, therefore we
        // insert it into overloaded_args before the superclass
        arg_index = j;
        break;
      }
    }
    // add object to overloaded_args. If it's a subclass of another class
    // we've already seen it will be inserted before the superclass,
    // otherwise it will be inserted at the end of the array
    overloaded_args->insert(
        overloaded_args->begin() + static_cast<long>(arg_index), obj);
  }
}

void append_overloaded_tensor(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj) {
  append_overloaded_arg(overloaded_args, obj, /*obj_is_type*/ false);
}

void append_overloaded_type(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj) {
  append_overloaded_arg(overloaded_args, obj, /*obj_is_type*/ true);
}

bool is_tensor_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args) {
  if (THPVariable_CheckExact(obj)) {
    // torch.Tensor instances (not subclasses, except for Parameter)
    return true;
  }

  if (check_has_torch_function(obj, /*ignore_mode*/ true)) {
    // tensor subclasses and unrelated objects with __torch_function__
    append_overloaded_tensor(overloaded_args, obj);
    return true;
  } else if (THPVariable_Check(obj)) {
    // tensor subclasses without __torch_function__
    return true;
  }

  return false;
}

static bool is_scalar_list(PyObject* obj) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (const auto idx : c10::irange(size)) {
    PyObject* iobj =
        tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!THPUtils_checkScalar(iobj)) {
      return false;
    }
  }
  return true;
}

bool is_tensor_list_and_append_overloaded(
    PyObject* obj,
    std::vector<PyObject*>* overloaded_args,
    size_t argnum,
    bool throw_error) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (long idx = 0; idx < size; idx++) {
    PyObject* iobj =
        tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!is_tensor_and_append_overloaded(iobj, overloaded_args)) {
      if (throw_error) {
        TORCH_CHECK_TYPE(
            false,
            "expected Tensor as element ",
            idx,
            " in argument ",
            argnum,
            ", but got ",
            Py_TYPE(iobj)->tp_name);
      }
      return false;
    }
  }
  return true;
}

static bool is_float_or_symfloat(PyObject* obj) {
  if (torch::is_symfloat(py::handle(obj))) {
    return true;
  }

  if (THPUtils_checkDouble(obj)) {
    return true;
  }

  return false;
}

static bool is_float_or_complex_list(PyObject* obj) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-branch-clone)
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size > 0) {
    PyObject* iobj = tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
    if (!is_float_or_symfloat(iobj) && !PyComplex_Check(iobj)) {
      return false;
    }
  }

  return true;
}

static bool is_int_or_symint(PyObject* obj) {
  // THPUtils_checkIndex may call __index__ or __int__
  // which may have side effects if obj is a symint node
  // so we do `is_symint` check first
  // TODO: maybe we should be using checkLong here?
  if (torch::is_symint(py::handle(obj))) {
    return true;
  }

  // FakeTensor(..., size=()) is qualified for SymInt param,
  // but we can't go via __index__ (below) as we would normally
  // do for regular tensors, because __index__ first forces a
  // conversion into an int, which in general you cannot do
  // if you have an unbacked SymInt.  So this fastpath ensures
  // that we still allow for fake tensors in this case, but
  // for regular tensors it's redundant with the test below.
  if (THPVariable_Check(obj)) {
    auto& var = THPVariable_Unpack(obj);
    if (TORCH_GUARD_OR_FALSE(var.sym_numel().sym_eq(1)) &&
        at::isIntegralType(var.dtype().toScalarType(), /*include_bool*/ true)) {
      return true;
    }
  }

  if (THPUtils_checkIndex(obj)) {
    return true;
  }

  return false;
}

static bool is_int_or_symint_list(
    PyObject* obj,
    int broadcast_size,
    int64_t* failed_idx = nullptr) {
  if (PyTuple_Check(obj) || PyList_Check(obj)) {
    if (PySequence_Size(obj) == 0) {
      return true;
    }
    auto item = py::reinterpret_steal<py::object>(PySequence_GetItem(obj, 0));

    if (is_int_or_symint(item.ptr())) {
      return true;
    }

    // NOTE: JIT tracer allows arbitrary scalar tensors to act as ints
    // in an intlist argument. Even float or complex scalar tensors.
    bool r =
        (jit::tracer::isTracing() && THPVariable_Check(item.ptr()) &&
         THPVariable_Unpack(item.ptr()).sizes().empty());
    if (!r && failed_idx != nullptr) {
      *failed_idx = 0;
    }
    return r;
  }

  // if a size is specified (e.g. IntArrayRef[2]) we also allow passing a single
  // int
  return broadcast_size > 0 && is_int_or_symint(obj);
}

// argnum is needed for raising the TypeError, it's used in the error message.
auto FunctionParameter::check(
    PyObject* obj,
    std::vector<PyObject*>& overloaded_args,
    int argnum,
    int64_t* failed_idx) -> bool {
  switch (type_) {
    case ParameterType::TENSOR: {
      if (is_tensor_and_append_overloaded(obj, &overloaded_args)) {
        return true;
      }
      if (allow_numbers_as_tensors) {
        return THPUtils_checkScalar(obj);
      }
      return false;
    }
    case ParameterType::SCALAR:
      if (THPUtils_checkScalar(obj)) {
        return true;
      }
      [[fallthrough]];
    case ParameterType::COMPLEX:
      if (PyComplex_Check(obj)) {
        return true;
      }
      [[fallthrough]];
    case ParameterType::DOUBLE: {
      if (is_float_or_symfloat(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        const auto& var = THPVariable_Unpack(obj);
        return !var.requires_grad() && var.dim() == 0;
      }
      if (torch::is_symfloat(py::handle(obj)) ||
          torch::is_symint(py::handle(obj))) {
        // This will induce a guard
        return true;
      }
      return false;
    }
    case ParameterType::INT64: {
      if (THPUtils_checkLong(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        const auto& var = THPVariable_Unpack(obj);
        return at::isIntegralType(var.scalar_type(), /*includeBool=*/false) &&
            !var.requires_grad() && var.dim() == 0;
      }
      if (torch::is_symint(py::handle(obj))) {
        // This will induce a guard
        return true;
      }
      return false;
    }
    case ParameterType::DIMNAME:
      return THPUtils_checkDimname(obj);
    case ParameterType::DIMNAME_LIST: {
      if (THPUtils_checkDimnameList(obj)) {
        return true;
      }
      // if a size is specified (e.g. DimnameList[1]) we also allow passing a
      // single Dimname
      return size == 1 && THPUtils_checkDimname(obj);
    }
    case ParameterType::TENSOR_LIST: {
      return is_tensor_list_and_append_overloaded(
          obj, &overloaded_args, argnum, true /* throw_error */);
    }
    case ParameterType::FLOAT_LIST:
      return is_float_or_complex_list(obj);
    case ParameterType::GENERATOR:
      return THPGenerator_Check(obj);
    case ParameterType::BOOL:
      return PyBool_Check(obj);
    case ParameterType::STORAGE:
      return isStorage(obj);
    case ParameterType::PYOBJECT:
      return true;
    case ParameterType::SCALARTYPE:
      if (THPDtype_Check(obj) || THPPythonScalarType_Check(obj)) {
        return true;
      }
      if (check_has_torch_function(obj, /*ignore_mode*/ true)) {
        // tensor subclasses and unrelated objects with __torch_function__
        append_overloaded_arg(&overloaded_args, obj, /*obj_is_type*/ false);
        return true;
      }
      return false;
    case ParameterType::LAYOUT:
      return THPLayout_Check(obj);
    case ParameterType::MEMORY_FORMAT:
      return THPMemoryFormat_Check(obj);
    case ParameterType::QSCHEME:
      return THPQScheme_Check(obj);
    case ParameterType::DEVICE:
      // Allow symint to be passed in as device, but we'll specialize and
      // guard in this case.
      return THPUtils_checkLong(obj) || THPUtils_checkString(obj) ||
          THPDevice_Check(obj) || torch::is_symint(py::handle(obj));
    case ParameterType::STREAM:
      return THPStream_Check(obj);
    case ParameterType::STRING:
      return THPUtils_checkString(obj);
    case ParameterType::SCALAR_LIST:
      return is_scalar_list(obj);
    case ParameterType::SYM_INT:
      return is_int_or_symint(obj);
    // Allow SymInt where int is expected; we'll guard in this case
    case ParameterType::INT_LIST:
    case ParameterType::SYM_INT_LIST:
      return is_int_or_symint_list(obj, size, failed_idx);
    case ParameterType::DISPATCH_KEY_SET:
      return py::isinstance<c10::DispatchKeySet>(py::handle(obj));
    default:
      throw std::runtime_error("unknown parameter type");
  }
}

// WARNING: these strings are parsed invalid_arguments.cpp
std::string FunctionParameter::type_name() const {
  switch (type_) {
    case ParameterType::TENSOR:
      return "Tensor";
    case ParameterType::SCALAR:
      return "Number";
    case ParameterType::INT64:
    // NB: SymInt is intentionally not mentioned here, as conventional user
    // use will only know about ints
    case ParameterType::SYM_INT:
      return "int";
    case ParameterType::DOUBLE:
      return "float";
    case ParameterType::COMPLEX:
      return "complex";
    case ParameterType::TENSOR_LIST:
      return "tuple of Tensors";
    case ParameterType::INT_LIST:
      return "tuple of ints";
    case ParameterType::FLOAT_LIST:
      return "tuple of floats";
    case ParameterType::GENERATOR:
      return "torch.Generator";
    case ParameterType::BOOL:
      return "bool";
    case ParameterType::STORAGE:
      return "torch.Storage";
    case ParameterType::PYOBJECT:
      return "object";
    case ParameterType::SCALARTYPE:
      return "torch.dtype";
    case ParameterType::LAYOUT:
      return "torch.layout";
    case ParameterType::MEMORY_FORMAT:
      return "torch.memory_format";
    case ParameterType::QSCHEME:
      return "torch.qscheme";
    case ParameterType::DEVICE:
      return "torch.device";
    case ParameterType::STRING:
      return "str";
    case ParameterType::DIMNAME:
      return "name";
    case ParameterType::DIMNAME_LIST:
      return "tuple of names";
    case ParameterType::SCALAR_LIST:
      return "tuple of Scalars";
    case ParameterType::SYM_INT_LIST:
      return "tuple of ints";
    case ParameterType::DISPATCH_KEY_SET:
      return "DispatchKeySet";
    default:
      throw std::runtime_error("unknown parameter type");
  }
}

static std::optional<int64_t> parse_as_integer(const std::string& s) {
  if (s.empty())
    return std::nullopt;
  char* str_end = nullptr;
  long ans = strtol(s.c_str(), &str_end, 0);
  // *str_end == 0 if the entire string was parsed as an integer.
  return (*str_end == 0) ? std::optional<int64_t>(ans) : std::nullopt;
}

/*
Parse default value of IntArrayRef declared at native_functions.yaml

There are two kinds of default values:
1. IntArrayRef[2] x=1 (where size=2, value={1,1}
2. IntArrayRef x={1,2,3} (where size=3, value={1,2,3}, note that there cannot be
space after comma since native_parse.py uses ', ' to split args)
*/
static std::vector<int64_t> parse_intlist_args(
    const std::string& s,
    int64_t size) {
  size_t n = s.size();

  if (s.empty())
    return std::vector<int64_t>();

  // case 1. s is an int (e.g., s=2)
  if (s[0] != '{') {
    TORCH_CHECK(size > 0, "Incorrect size of IntArrayRef: ", size);
    return std::vector<int64_t>(size, std::stol(s));
  }

  // case 2. s is a list of dims (e.g., s={1,2})

  // since already checked left brace '{' above, here only checks right brace
  // '}'
  TORCH_CHECK(
      s[n - 1] == '}',
      "Default value of IntArrayRef is missing right brace '}', found ",
      s[n - 1]);

  auto args = std::vector<int64_t>();
  std::istringstream ss(s.substr(1, s.length() - 2)); // exclude '{' and '}'
  std::string tok;

  while (std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;
}

// Parse a string literal to remove quotes and escape sequences
static std::string parse_string_literal(std::string_view str) {
  TORCH_CHECK(str.length() >= 2, "String defaults must be quoted");

  if (str.front() == '"') {
    TORCH_CHECK(
        str.back() == '"', "Mismatched quotes in string default: ", str);
  } else {
    TORCH_CHECK(
        str.front() == '\'' && str.back() == '\'',
        "Invalid quotes in string default: ",
        str)
  }

  std::string parsed;
  parsed.reserve(str.size());
  for (size_t i = 1; i < str.size() - 1;) {
    if (str[i] != '\\') {
      parsed.push_back(str[i]);
      ++i;
      continue;
    }

    // Handle escape sequences
    TORCH_CHECK(
        i < str.size() - 2, "String ends with escaped final quote: ", str)
    char c = str[i + 1];
    switch (c) {
      case '\\':
      case '\'':
      case '\"':
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'v':
        c = '\v';
        break;
      case 't':
        c = '\t';
        break;
      default:
        TORCH_CHECK(
            false,
            "Unsupported escape sequence in string default: \\",
            str[i + 1]);
    }
    parsed.push_back(c);
    i += 2;
  }
  return parsed;
}

void FunctionParameter::set_default_str(const std::string& str) {
  if (str == "None") {
    allow_none = true;
  }
  if (type_ == ParameterType::TENSOR ||
      type_ == ParameterType::DISPATCH_KEY_SET) {
    if (str != "None") {
      throw std::runtime_error(
          "default value for Tensor must be none, got: " + str);
    }
  } else if (type_ == ParameterType::INT64 || type_ == ParameterType::SYM_INT) {
    default_int = atol(str.c_str());
  } else if (type_ == ParameterType::BOOL) {
    default_bool = (str == "True" || str == "true");
  } else if (type_ == ParameterType::DOUBLE) {
    default_double = atof(str.c_str());
  } else if (type_ == ParameterType::COMPLEX) {
    default_complex[0] = atof(str.c_str()); // TODO: parse "x + xj"?
    default_complex[1] = 0;
  } else if (type_ == ParameterType::SCALAR) {
    if (str != "None") {
      // we sometimes rely on integer-vs-float values, e.g. with arange.
      const auto as_integer = parse_as_integer(str);
      default_scalar = as_integer.has_value() ? at::Scalar(as_integer.value())
                                              : at::Scalar(atof(str.c_str()));
    }
  } else if (
      type_ == ParameterType::INT_LIST ||
      type_ == ParameterType::SYM_INT_LIST) {
    if (str != "None") {
      default_intlist = parse_intlist_args(str, size);
    }
  } else if (type_ == ParameterType::FLOAT_LIST) {
    if (str != "None") {
      throw std::runtime_error("Defaults not supported for float[]");
    }
  } else if (type_ == ParameterType::SCALARTYPE) {
    if (str == "None") {
      default_scalartype = at::ScalarType::Undefined;
    } else if (str == "torch.int64") {
      default_scalartype = at::ScalarType::Long;
    } else {
      throw std::runtime_error("invalid default value for ScalarType: " + str);
    }
  } else if (type_ == ParameterType::LAYOUT) {
    if (str == "None") {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(allow_none);
    } else if (str == "torch.strided") {
      default_layout = at::Layout::Strided;
    } else if (str == "torch.sparse_coo") {
      default_layout = at::Layout::Sparse;
    } else {
      throw std::runtime_error("invalid default value for layout: " + str);
    }
  } else if (type_ == ParameterType::DEVICE) {
    if (str != "None") {
      throw std::runtime_error("invalid device: " + str);
    }
  } else if (type_ == ParameterType::STREAM) {
    if (str != "None") {
      throw std::runtime_error("invalid stream: " + str);
    }
  } else if (type_ == ParameterType::STRING) {
    if (str != "None") {
      default_string = parse_string_literal(str);
    }
  }
  // These types weren't handled here before. Adding a default error
  // led to a lot of test failures so adding this skip for now.
  // We should correctly handle these though because it might be causing
  // silent failures.
  else if (type_ == ParameterType::TENSOR_LIST) { // NOLINT
    // throw std::runtime_error("Invalid Tensor List");
  } else if (type_ == ParameterType::GENERATOR) { // NOLINT
    // throw std::runtime_error("ParameterType::GENERATOR");
  } else if (type_ == ParameterType::PYOBJECT) { // NOLINT
    // throw std::runtime_error("ParameterType::PYOBJECT");
  } else if (type_ == ParameterType::MEMORY_FORMAT) { // NOLINT
    // throw std::runtime_error("ParameterType::MEMORY_FORMAT");
  } else if (type_ == ParameterType::DIMNAME) { // NOLINT
    // throw std::runtime_error("ParameterType::DIMNAME");
  } else if (type_ == ParameterType::DIMNAME_LIST) { // NOLINT
    // throw std::runtime_error("ParameterType::DIMNAME_LIST");
  } else if (type_ == ParameterType::SCALAR_LIST) { // NOLINT
    // throw std::runtime_error("ParameterType::SCALAR_LIST");
  } else if (type_ == ParameterType::STORAGE) { // NOLINT
    // throw std::runtime_error("ParameterType::STORAGE");
  } else if (type_ == ParameterType::QSCHEME) { // NOLINT
    // throw std::runtime_error("ParameterType::QSCHEME");
  } else {
    throw std::runtime_error("unknown parameter type");
  }
  default_value = str;
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
FunctionSignature::FunctionSignature(const std::string& fmt, int index)
    : min_args(0),
      max_args(0),
      max_pos_args(0),
      index(index),
      hidden(false),
      deprecated(false) {
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);

  bool allow_numbers_as_tensors = should_allow_numbers_as_tensors(name);

  auto last_offset = open_paren + 1;
  bool keyword_only = false;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    auto next_offset = offset + 2;
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);
      done = true;
      next_offset = offset + 1;
      // this 'if' happens for an empty parameter list, i.e. fn().
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    }
    if (offset == std::string::npos) {
      throw std::runtime_error("missing closing parenthesis: " + fmt);
    }
    if (offset == last_offset) {
      throw std::runtime_error("malformed signature: " + fmt);
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);
    last_offset = next_offset;
    if (param_str == "*") {
      keyword_only = true;
    } else {
      params.emplace_back(param_str, keyword_only);
      params.back().allow_numbers_as_tensors = allow_numbers_as_tensors;
    }
  }

  if (fmt.substr(last_offset) == "|deprecated") {
    hidden = true;
    // TODO: raise warning when parsing deprecated signatures
    deprecated = true;
  } else if (fmt.substr(last_offset) == "|hidden") {
    hidden = true;
  }

  max_args = params.size();

  // count the number of non-optional args
  for (auto& param : params) {
    if (!param.optional) {
      min_args++;
    }
    if (!param.keyword_only) {
      max_pos_args++;
    }
  }
}

std::string FunctionSignature::toString() const {
  // optionals, etc.
  std::ostringstream ss;
  bool keyword_already = false;
  ss << "(";
  int i = 0;
  for (auto& param : params) {
    if (i != 0) {
      ss << ", ";
    }
    if (param.keyword_only && !keyword_already) {
      ss << "*, ";
      keyword_already = true;
    }
    ss << param.type_name() << " " << param.name;
    if (param.optional) {
      ss << " = " << param.default_value;
    }
    i++;
  }
  ss << ")";
  return ss.str();
}

[[noreturn]] static void extra_args(
    const FunctionSignature& signature,
    Py_ssize_t nargs) {
  const auto max_pos_args = signature.max_pos_args;
  const auto min_args = signature.min_args;
  const long nargs_ = nargs;
  if (min_args != max_pos_args) {
    throw TypeError(
        "%s() takes from %zu to %zu positional arguments but %ld were given",
        signature.name.c_str(),
        min_args,
        max_pos_args,
        nargs_);
  }
  throw TypeError(
      "%s() takes %zu positional argument%s but %ld %s given",
      signature.name.c_str(),
      max_pos_args,
      max_pos_args == 1 ? "" : "s",
      nargs_,
      nargs == 1 ? "was" : "were");
}

[[noreturn]] static void missing_args(
    const FunctionSignature& signature,
    int idx) {
  int num_missing = 0;
  std::stringstream ss;

  auto& params = signature.params;
  for (auto it = params.begin() + idx; it != params.end(); ++it) {
    if (!it->optional) {
      if (num_missing > 0) {
        ss << ", ";
      }
      ss << '"' << it->name << '"';
      num_missing++;
    }
  }

  throw TypeError(
      "%s() missing %d required positional argument%s: %s",
      signature.name.c_str(),
      num_missing,
      num_missing == 1 ? "s" : "",
      ss.str().c_str());
}

static Py_ssize_t find_param(FunctionSignature& signature, PyObject* name) {
  Py_ssize_t i = 0;
  for (auto& param : signature.params) {
    int cmp = PyObject_RichCompareBool(name, param.python_name, Py_EQ);
    if (cmp < 0) {
      throw python_error();
    } else if (cmp) {
      return i;
    }
    i++;
  }
  return -1;
}

[[noreturn]] static void extra_kwargs(
    FunctionSignature& signature,
    PyObject* kwargs,
    Py_ssize_t num_pos_args) {
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  Py_ssize_t pos = 0;

  // Note that this dict traversal is NoGil safe as the kwargs dict is only
  // accessible within this thread.
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    if (!THPUtils_checkString(key)) {
      throw TypeError("keywords must be strings");
    }

    auto param_idx = find_param(signature, key);
    if (param_idx < 0) {
      throw TypeError(
          "%s() got an unexpected keyword argument '%s'",
          signature.name.c_str(),
          THPUtils_unpackString(key).c_str());
    }

    if (param_idx < num_pos_args) {
      throw TypeError(
          "%s() got multiple values for argument '%s'",
          signature.name.c_str(),
          THPUtils_unpackString(key).c_str());
    }
  }

  // this should never be hit
  throw TypeError("invalid keyword arguments");
}

bool FunctionSignature::parse(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* dst[], // NOLINT
    std::vector<PyObject*>& overloaded_args,
    bool raise_exception) {
  Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;
  auto remaining_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  size_t arg_pos = 0;
  bool allow_varargs_intlist = false;

  // if there is a single positional IntArrayRef argument, i.e. expand(..),
  // view(...), allow a var-args style IntArrayRef, so expand(5,3) behaves as
  // expand((5,3))
  if (max_pos_args == 1 &&
      (params[0].type_ == ParameterType::INT_LIST ||
       params[0].type_ == ParameterType::SYM_INT_LIST)) {
    allow_varargs_intlist = true;
  }

  if (static_cast<size_t>(nargs) > max_pos_args && !allow_varargs_intlist) {
    if (raise_exception) {
      // foo() takes takes 2 positional arguments but 3 were given
      extra_args(*this, nargs);
    }
    return false;
  }

  int i = 0;
  if (self != nullptr && check_has_torch_function(self, /*ignore_mode*/ true)) {
    append_overloaded_tensor(&overloaded_args, self);
  }
  for (auto& param : params) {
    PyObject* obj = nullptr;
    bool is_kwd = false;
    if (arg_pos < static_cast<size_t>(nargs)) {
      // extra positional args given after single positional IntArrayRef arg
      if (param.keyword_only) {
        if (raise_exception) {
          extra_args(*this, nargs);
        }
        return false;
      }
      obj = PyTuple_GET_ITEM(args, arg_pos);
    } else if (kwargs) {
      // Note that this call is NoGil safe as it works on kwargs which are local
      // to the current function call.
      obj = PyDict_GetItem(kwargs, param.python_name);
      for (PyObject* numpy_name : param.numpy_python_names) {
        if (obj) {
          break;
        }
        obj = PyDict_GetItem(kwargs, numpy_name);
      }
      is_kwd = true;
    }

    int64_t failed_idx = -1;
    bool varargs_eligible = allow_varargs_intlist && arg_pos == 0 && !is_kwd;
    if ((!obj && param.optional) || (obj == Py_None && param.allow_none)) {
      dst[i++] = nullptr;
    } else if (!obj) {
      if (raise_exception) {
        // foo() missing 1 required positional argument: "b"
        missing_args(*this, i);
      }
      return false;
    } else if (param.check(obj, overloaded_args, i, &failed_idx)) {
      dst[i++] = obj;
      // XXX: the Variable check is necessary because sizes become tensors when
      // tracer is enabled. This behavior easily leads to ambiguities, and we
      // should avoid having complex signatures that make use of it...
    } else if (
        varargs_eligible &&
        (is_int_or_symint_list(args, param.size, &failed_idx))) {
      // take all positional arguments as this parameter
      // e.g. permute(1, 2, 3) -> permute((1, 2, 3))
      dst[i++] = args;
      arg_pos = nargs;
      continue;
    } else if (raise_exception) {
      if (is_kwd) {
        // foo(): argument 'other' must be str, not int
        throw TypeError(
            "%s(): argument '%s' must be %s, not %s",
            name.c_str(),
            param.name.c_str(),
            param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      } else {
        // foo(): argument 'other' (position 2) must be str, not int
        if (failed_idx != -1) {
          if (!(PyTuple_Check(obj) || PyList_Check(obj))) {
            TORCH_INTERNAL_ASSERT(varargs_eligible);
            obj = args;
          }
          TORCH_INTERNAL_ASSERT(failed_idx < PySequence_Size(obj));
          throw TypeError(
              "%s(): argument '%s' (position %ld) must be %s, but found element of type %s at pos %ld",
              name.c_str(),
              param.name.c_str(),
              static_cast<long>(arg_pos + 1),
              param.type_name().c_str(),
              Py_TYPE(py::reinterpret_steal<py::object>(
                          PySequence_GetItem(obj, failed_idx))
                          .ptr())
                  ->tp_name,
              static_cast<long>(failed_idx));
        }
        throw TypeError(
            "%s(): argument '%s' (position %ld) must be %s, not %s",
            name.c_str(),
            param.name.c_str(),
            static_cast<long>(arg_pos + 1),
            param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      }
    } else {
      return false;
    }

    if (!is_kwd) {
      arg_pos++;
    } else if (obj) {
      remaining_kwargs--;
    }
  }

  if (remaining_kwargs > 0) {
    if (raise_exception) {
      // foo() got an unexpected keyword argument "b"
      extra_kwargs(*this, kwargs, nargs);
    }
    return false;
  }
  return true;
}

PythonArgParser::PythonArgParser(
    const std::vector<std::string>& fmts,
    bool traceable)
    : max_args(0), traceable(traceable) {
  int index = 0;
  for (auto& fmt : fmts) {
    signatures_.emplace_back(fmt, index);
    ++index;
  }
  for (auto& signature : signatures_) {
    if (signature.max_args > max_args) {
      max_args = signature.max_args;
    }
  }
  if (!signatures_.empty()) {
    function_name = signatures_[0].name;
  }

  // Check deprecated signatures last
  std::stable_partition(
      signatures_.begin(), signatures_.end(), [](const FunctionSignature& sig) {
        return !sig.deprecated;
      });
}

void PythonArgParser::check_deprecated(const FunctionSignature& signature) {
  if (signature.deprecated) {
    auto msg = c10::str(
        "This overload of ",
        signature.name,
        " is deprecated:\n\t",
        signature.name,
        signature.toString());
    auto signatures = get_signatures();
    if (!signatures.empty()) {
      msg += "\nConsider using one of the following signatures instead:";
      for (const auto& sig : signatures) {
        msg += "\n\t";
        msg += signature.name;
        msg += sig;
      }
    }
    TORCH_WARN_ONCE(msg);
  }
}

PythonArgs PythonArgParser::raw_parse(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* parsed_args[]) { // NOLINT
  if (signatures_.size() == 1) {
    auto& signature = signatures_[0];
    std::vector<PyObject*> overloaded_args;
    signature.parse(self, args, kwargs, parsed_args, overloaded_args, true);
    check_deprecated(signature);
    return PythonArgs(
        traceable, signature, parsed_args, std::move(overloaded_args));
  }

  for (auto& signature : signatures_) {
    std::vector<PyObject*> overloaded_args;
    if (signature.parse(
            self, args, kwargs, parsed_args, overloaded_args, false)) {
      check_deprecated(signature);
      return PythonArgs(
          traceable, signature, parsed_args, std::move(overloaded_args));
    }
  }

  print_error(self, args, kwargs, parsed_args);
}

void PythonArgParser::print_error(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs,
    PyObject* parsed_args[]) { // NOLINT
  size_t num_args =
      (args ? PyTuple_GET_SIZE(args) : 0) + (kwargs ? PyDict_Size(kwargs) : 0);
  std::vector<unsigned> plausible_idxs;
  unsigned i = 0;
  for (auto& signature : signatures_) {
    if (num_args >= signature.min_args && num_args <= signature.max_args &&
        !signature.hidden) {
      plausible_idxs.push_back(i);
    }
    i++;
  }

  if (plausible_idxs.size() == 1) {
    auto& signature = signatures_[plausible_idxs[0]];
    std::vector<PyObject*> overloaded_args;
    signature.parse(self, args, kwargs, parsed_args, overloaded_args, true);
  }

  auto options = get_signatures();
  auto msg =
      torch::format_invalid_args(args, kwargs, function_name + "()", options);
  throw TypeError("%s", msg.c_str());
}

std::vector<std::string> PythonArgParser::get_signatures() const {
  std::vector<std::string> options;
  for (auto& signature : signatures_) {
    if (!signature.hidden) {
      options.push_back(signature.toString());
    }
  }
  return options;
}

at::Tensor PythonArgs::tensor_slow(int i) {
  PyObject* obj = args[i];
  if (!obj) {
    return at::Tensor();
  }
  if (THPVariable_Check(obj)) {
    return THPVariable_Unpack(obj);
  }

  bool save_symint = false;
  at::Scalar scalar;
  if (PyBool_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackBool(obj));
  } else if (THPUtils_checkLong(obj)) {
    int overflow = -1;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (value == -1 && PyErr_Occurred()) {
      throw python_error();
    }
    if (overflow != 0) {
      // try unsigned
      unsigned long long value = PyLong_AsUnsignedLongLong(obj);
      if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        throw python_error();
      }
      scalar = at::Scalar(static_cast<uint64_t>(value));
    } else {
      scalar = at::Scalar(static_cast<int64_t>(value));
    }
  } else if (PyComplex_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackComplexDouble(obj));
  } else if (THPUtils_checkDouble(obj)) {
    scalar = at::Scalar(THPUtils_unpackDouble(obj));
    // NB: we DO NOT put symbolic ints/floats into the Scalar itself,
    // because although Scalar supports SymInt/SymFloat, the subsequent
    // conversion to Tensor does not.  Instead, do it out of band.
  } else if (torch::is_symint(py::handle(obj))) {
    save_symint = true;
    // This scalar value doesn't matter, it shouldn't ever actually
    // get read out.  Make it a big and weird looking number to help
    // people figure out if there's aproblem.
    scalar = at::Scalar(7777777);
  } else if (torch::is_symfloat(py::handle(obj))) {
    save_symint = true;
    scalar = at::Scalar(std::numeric_limits<double>::quiet_NaN());
  } else if (torch::is_symbool(py::handle(obj))) {
    save_symint = true;
    scalar = at::Scalar(true);
  } else {
    // NB: Are you here because you passed None to a Variable method,
    // and you expected an undefined tensor to be returned?   Don't add
    // a test for Py_None here; instead, you need to mark the argument
    // as *allowing none*; you can do this by writing 'Tensor?' instead
    // of 'Tensor' in the ATen metadata.
    throw TypeError(
        "expected Tensor as argument %d, but got %s", i, Py_TYPE(obj)->tp_name);
  }
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  at::Tensor tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);

  if (save_symint) {
    auto py_tensor = py::cast(tensor);
    if (PyObject_SetAttrString(py_tensor.ptr(), "_wrapped_number", obj) < 0) {
      throw python_error();
    }
  }

  return tensor;
}

at::Scalar PythonArgs::scalar_slow(int i) {
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::NumberType::get());
  }

  return scalar_slow(args[i]);
}

at::Scalar PythonArgs::scalar_slow(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    int overflow = -1;
    long long value = PyLong_AsLongLongAndOverflow(arg, &overflow);
    if (value == -1 && PyErr_Occurred()) {
      throw python_error();
    }
    if (overflow != 0) {
      // try unsigned
      unsigned long long value = PyLong_AsUnsignedLongLong(arg);
      if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        throw python_error();
      }
      return at::Scalar(static_cast<uint64_t>(value));
    } else {
      return at::Scalar(static_cast<int64_t>(value));
    }
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }

  if (torch::is_symint(arg)) {
    return at::Scalar(py::cast<c10::SymInt>(arg));
  }

  if (torch::is_symfloat(arg)) {
    return at::Scalar(py::cast<c10::SymFloat>(arg));
  }

  if (torch::is_symbool(arg)) {
    // Windows build fails with C2440: '<function-style-cast>'
    // when at:Scalar(py::cast<c10::SymBool>(arg))
    auto sym_bool = py::handle(arg).cast<c10::SymBool>();
    return at::Scalar(sym_bool);
  }

  return at::Scalar(THPUtils_unpackDouble(arg));
}

} // namespace torch
