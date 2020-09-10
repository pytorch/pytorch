#include <torch/csrc/utils/python_arg_parser.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/python_strings.h>

#include <ATen/ATen.h>
#include <ATen/TracerMode.h>

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
  {"double", ParameterType::DOUBLE},
  {"complex", ParameterType::COMPLEX},
  {"TensorList", ParameterType::TENSOR_LIST},
  {"IntArrayRef", ParameterType::INT_LIST},
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
  {"std::string", ParameterType::STRING},
  {"Dimname", ParameterType::DIMNAME},
  {"DimnameList", ParameterType::DIMNAME_LIST},
  {"ScalarList", ParameterType::SCALAR_LIST},
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
// In particular, NumPy sometimes uses "x" and sometimes "a" for the main input tensor.
// Rather than annotate each function separately with whether it should take "x" or "a",
// just try both.
//
// TODO: Allow individual functions to specify non-default translations:
// For example, `torch.pow` should translate "exponent" to "x2".
static const std::unordered_map<std::string, std::vector<std::string>> numpy_compatibility_arg_names = {
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
static bool should_allow_numbers_as_tensors(const std::string& name) {
  static std::unordered_set<std::string> allowed = {
    "add", "add_", "add_out",
    "div", "div_", "div_out",
    "mul", "mul_", "mul_out",
    "sub", "sub_", "sub_out",
    "true_divide", "true_divide_", "true_divide_out",
    "floor_divide", "floor_divide_", "floor_divide_out"
  };
  return allowed.find(name) != allowed.end();
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
FunctionParameter::FunctionParameter(const std::string& fmt, bool keyword_only)
  : optional(false)
  , allow_none(false)
  , keyword_only(keyword_only)
  , size(0)
  , default_scalar(0)
{
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
    auto size_str = type_str.substr(bracket + 1, type_str.length() - bracket - 2);
    size = atoi(size_str.c_str());
    type_str = type_str.substr(0, bracket);
  }

  auto name_str = fmt.substr(space + 1);
  auto it = type_map.find(type_str);
  if (it == type_map.end()) {
    throw std::runtime_error("FunctionParameter(): invalid type string: " + type_str);
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
    for (const auto& str: np_compat_it->second) {
      numpy_python_names.push_back(THPUtils_internString(str));
    }
  }
}

auto handle_torch_function_getter(THPVariable* self, const std::string& property_name) -> PyObject* {
  py::object torch_api = PyObject_FastGetAttrString(THPVariableClass, (char*)property_name.c_str());
  std::string module_name = "torch.Tensor." + property_name;
  return handle_torch_function((PyObject *)self, "__get__", nullptr, torch_api.ptr(), module_name);
}

auto handle_torch_function_setter(THPVariable* self, const std::string& property_name, PyObject* value) -> int {
  py::object torch_api = PyObject_FastGetAttrString(THPVariableClass, (char*)property_name.c_str());
  std::string module_name = "torch.Tensor." + property_name;
  if (value != nullptr)
  {
    py::tuple args_ = py::make_tuple(py::handle(value));
    handle_torch_function((PyObject *)self, "__set__", args_.ptr(), torch_api.ptr(), module_name);
  }
  else {
    handle_torch_function((PyObject *)self, "__delete__", nullptr, torch_api.ptr(), module_name);
  }
  return 0;
}

// Combines self and args into one tuple.
auto combine_self_args(PyObject *self, PyObject *args) -> py::tuple {
  if (args == nullptr) {
    return py::make_tuple(py::handle(self));
  }
  else if (self == nullptr) {
    return py::reinterpret_borrow<py::tuple>(args);
  }

  auto py_args = py::reinterpret_borrow<py::tuple>(args);
  size_t n = py_args.size();
  auto args_ = py::tuple(n + 1);
  args_[0] = py::handle(self);
  for (size_t i = 0; i < n; i++) {
    args_[i+1] = py_args[i];
  }
  return args_;
}

auto handle_torch_function(PyObject* self, const std::string& func_name, PyObject* args, PyObject* torch_api, const std::string& module_name) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(torch_api, (char*)func_name.c_str());
  TORCH_INTERNAL_ASSERT(torch_api_function.ptr() != nullptr, "torch API function must exist");
  py::tuple args_ = combine_self_args(self, args);
  py::tuple py_types = py::make_tuple(py::handle(PyObject_Type(self)));
  py::object torch_function = PyObject_FastGetAttrString(self, "__torch_function__");
  py::object ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(torch_function.ptr(), torch_api_function.ptr(), py_types.ptr(), args_.ptr(), NULL));
  if (ret.ptr() == nullptr) {
    // if an exception occurred in a user's implementation of
    // __torch_function__, throw it
    throw python_error();
  }
  if (ret.ptr() == Py_NotImplemented) {
    std::string error_msg = "no implementation found for " + module_name + "." + func_name + "' on types that implement __torch_function__: [" + self->ob_type->tp_name + "]";
    PyErr_SetString(PyExc_TypeError, error_msg.c_str());
    throw python_error();
  }
  return ret.release().ptr();
}

auto handle_torch_function_no_python_arg_parser(const std::vector<py::handle> &overloaded_args, PyObject* args, PyObject* kwargs, const char* func_name, PyObject* torch_api_function, const char* module_name) -> PyObject* {
  // overloaded_args already all have unique types
  std::vector<py::object> overloaded_types;
  overloaded_types.reserve(overloaded_args.size());
  for (auto &arg : overloaded_args) {
    overloaded_types.push_back(py::reinterpret_borrow<py::object>((PyObject *) Py_TYPE(arg.ptr())));
  }
  py::tuple py_types = py::cast(overloaded_types);
  py::object ret;
  for (auto &arg : overloaded_args) {
    py::object torch_function = PyObject_FastGetAttrString(arg.ptr(), "__torch_function__");
    ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(torch_function.ptr(), torch_api_function, py_types.ptr(), args, kwargs, NULL));
    if (ret.ptr() != Py_NotImplemented) {
      // Return the reference to the result. This also covers the case where ret
      // is NULL and __torch_function__ raised an exception, which we throw below
      break;
    }
  }
  if (ret.ptr() == nullptr) {
    // if an exception occurred in a user's implementation of
    // __torch_function__, throw it
    throw python_error();
  }
  else if (ret.ptr() == Py_NotImplemented) {
    // all __torch_function__ implementations in overloaded_args
    // returned NotImplemented, so we raise a TypeError.
    std::stringstream ss;
    ss << "no implementation found for '" << module_name << "." << func_name
       << "' on types that implement __torch_function__: [";
    for (auto &arg : overloaded_args) {
      ss << arg.ptr()->ob_type->tp_name;
      if (!arg.is(overloaded_args.back())) {
        ss << ", ";
      }
      else {
        ss << "]";
      }
    }
    const std::string& tmp = ss.str();
    PyErr_SetString(PyExc_TypeError, tmp.c_str());
    throw python_error();
  }
  return ret.release().ptr();
}

auto handle_torch_function(PythonArgs &r, PyObject* self, PyObject* args, PyObject* kwargs, PyObject* torch_api, const char* module_name) -> PyObject* {
  py::object torch_api_function = PyObject_FastGetAttrString(torch_api, (char*)r.get_func_name().c_str());
  TORCH_INTERNAL_ASSERT(torch_api_function.ptr() != nullptr, "torch API function must exist");
  py::object ret;
  py::tuple args_ = combine_self_args(self, args);
  // overloaded_args already all have unique types
  std::vector<py::object> overloaded_types;
  overloaded_types.reserve(r.signature.overloaded_args.size());
  for (auto &arg : r.signature.overloaded_args) {
    overloaded_types.push_back(py::reinterpret_borrow<py::object>((PyObject *) Py_TYPE(arg.ptr())));
  }
  py::tuple py_types = py::cast(overloaded_types);
  return handle_torch_function_no_python_arg_parser(r.signature.overloaded_args, args_.ptr(), kwargs, r.get_func_name().c_str(), torch_api_function.ptr(), module_name);
}

auto handle_torch_function(PythonArgs &r, PyObject* args, PyObject* kwargs, PyObject* torch_api, const char* module_name) -> PyObject*
{
  return handle_torch_function(r, nullptr, args, kwargs, torch_api, module_name);
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
 *  See torch._overrides._get_overloaded_types_and_args for the equivalent
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

void append_overloaded_arg(std::vector<py::handle>* overloaded_args, PyObject* obj) {
  bool class_not_seen_yet = true;
  for (auto &arg : *overloaded_args) {
    if (Py_TYPE(obj) == Py_TYPE(arg.ptr())) {
      // obj is the same type as another parameter we've seen in a prior
      // iteration of the loop over parameters so we already have an entry
      // with the proper __torch_function__ implementation to call, so skip
      // this parameter
      class_not_seen_yet = false;
      break;
    }
  }
  if (class_not_seen_yet) {
    int arg_index = overloaded_args->size();
    for (int j = 0; j < arg_index; j++) {
      if (PyObject_IsInstance(obj, (PyObject*)(Py_TYPE((*overloaded_args)[j].ptr())))) {
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
    overloaded_args->insert(overloaded_args->begin() + arg_index, obj);
  }
}

bool is_tensor_and_append_overloaded(PyObject* obj, std::vector<py::handle>* overloaded_args) {
  if (THPVariable_CheckExact(obj)) {
    // torch.Tensor instances (not subclasses)
    return true;
  }

  if (check_has_torch_function(obj)) {
    // tensor subclasses and unrelated objects with __torch_function__
    append_overloaded_arg(overloaded_args, obj);
    return true;
  } else if (THPVariable_Check(obj)) {
    // tensor subclasses without __torch_function__
    return true;
  }

  return false;
}

bool is_scalar_list_and_append_overloaded(PyObject* obj, int argnum) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (size_t idx = 0; idx < size; idx++) {
    PyObject* iobj = tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!THPUtils_checkScalar(iobj)) {
      return false;
    }
  }
  return true;
}

bool is_tensor_list_and_append_overloaded(PyObject* obj, std::vector<py::handle>* overloaded_args, int argnum, bool throw_error) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return false;
  }
  auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (size_t idx = 0; idx < size; idx++) {
    PyObject* iobj = tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (!is_tensor_and_append_overloaded(iobj, overloaded_args)) {
      if (throw_error) {
        throw TypeError("expected Tensor as element %d in argument %d, but got %s",
            static_cast<int>(idx), argnum, Py_TYPE(iobj)->tp_name);
      }
      return false;
    }
  }
  return true;
}

// argnum is needed for raising the TypeError, it's used in the error message.
auto FunctionParameter::check(PyObject* obj, std::vector<py::handle> &overloaded_args, int argnum) -> bool
{
  switch (type_) {
    case ParameterType::TENSOR: {
      if (is_tensor_and_append_overloaded(obj, &overloaded_args)) {
        return true;
      }
      return allow_numbers_as_tensors && THPUtils_checkScalar(obj);
    }
    case ParameterType::SCALAR:
    case ParameterType::COMPLEX: 
      if (PyComplex_Check(obj)) {
        return true;
      }
      // fallthrough
    
    case ParameterType::DOUBLE: {
      if (THPUtils_checkDouble(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        auto& var = ((THPVariable*)obj)->cdata;
        return !var.requires_grad() && var.dim() == 0;
      }
      return false;
    }
    case ParameterType::INT64: {
      if (THPUtils_checkLong(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        auto& var = ((THPVariable*)obj)->cdata;
        return at::isIntegralType(var.scalar_type(), /*includeBool=*/false) && !var.requires_grad() && var.dim() == 0;
      }
      return false;
    }
    case ParameterType::DIMNAME: return THPUtils_checkDimname(obj);
    case ParameterType::DIMNAME_LIST: {
      if (THPUtils_checkDimnameList(obj)) {
        return true;
      }
      // if a size is specified (e.g. DimnameList[1]) we also allow passing a single Dimname
      return size == 1 && THPUtils_checkDimname(obj);
    }
    case ParameterType::TENSOR_LIST: {
      return is_tensor_list_and_append_overloaded(obj, &overloaded_args, argnum, true /* throw_error */);
    }
    case ParameterType::INT_LIST: {
      if (PyTuple_Check(obj) || PyList_Check(obj)) {
        return true;
      }
      // if a size is specified (e.g. IntArrayRef[2]) we also allow passing a single int
      return size > 0 && THPUtils_checkLong(obj);
    }
    case ParameterType::FLOAT_LIST: return (PyTuple_Check(obj) || PyList_Check(obj));
    case ParameterType::GENERATOR: return THPGenerator_Check(obj);
    case ParameterType::BOOL: return PyBool_Check(obj);
    case ParameterType::STORAGE: return isStorage(obj);
    case ParameterType::PYOBJECT: return true;
    case ParameterType::SCALARTYPE: return THPDtype_Check(obj) || THPPythonScalarType_Check(obj);
    case ParameterType::LAYOUT: return THPLayout_Check(obj);
    case ParameterType::MEMORY_FORMAT: return THPMemoryFormat_Check(obj);
    case ParameterType::QSCHEME: return THPQScheme_Check(obj);
    case ParameterType::DEVICE:
      return THPUtils_checkLong(obj) || THPUtils_checkString(obj) || THPDevice_Check(obj);
    case ParameterType::STRING: return THPUtils_checkString(obj);
    case ParameterType::SCALAR_LIST: {
      return is_scalar_list_and_append_overloaded(obj, argnum);
    }
    default: throw std::runtime_error("unknown parameter type");
  }
}

std::string FunctionParameter::type_name() const {
  switch (type_) {
    case ParameterType::TENSOR: return "Tensor";
    case ParameterType::SCALAR: return "Number";
    case ParameterType::INT64: return "int";
    case ParameterType::DOUBLE: return "float";
    case ParameterType::COMPLEX: return "complex";
    case ParameterType::TENSOR_LIST: return "tuple of Tensors";
    case ParameterType::INT_LIST: return "tuple of ints";
    case ParameterType::FLOAT_LIST: return "tuple of floats";
    case ParameterType::GENERATOR: return "torch.Generator";
    case ParameterType::BOOL: return "bool";
    case ParameterType::STORAGE: return "torch.Storage";
    case ParameterType::PYOBJECT: return "object";
    case ParameterType::SCALARTYPE: return "torch.dtype";
    case ParameterType::LAYOUT: return "torch.layout";
    case ParameterType::MEMORY_FORMAT: return "torch.memory_format";
    case ParameterType::QSCHEME: return "torch.qscheme";
    case ParameterType::DEVICE: return "torch.device";
    case ParameterType::STRING: return "str";
    case ParameterType::DIMNAME: return "name";
    case ParameterType::DIMNAME_LIST: return "tuple of names";
    case ParameterType::SCALAR_LIST: return "tuple of Scalars";
    default: throw std::runtime_error("unknown parameter type");
  }
}

static inline c10::optional<int64_t> parse_as_integer(const std::string& s) {
  if (s.empty())
    return c10::nullopt;
  char *str_end;
  long ans = strtol(s.c_str(), &str_end, 0);
  // *str_end == 0 if the entire string was parsed as an integer.
  return (*str_end == 0) ? c10::optional<int64_t>(ans) : c10::nullopt;
}

/*
Parse default value of IntArrayRef declared at native_functions.yaml

There are two kinds of default values:
1. IntArrayRef[2] x=1 (where size=2, value={1,1}
2. IntArrayRef x={1,2,3} (where size=3, value={1,2,3}, note that there cannot be space after comma since native_parse.py uses ', ' to split args)
*/
static inline std::vector<int64_t> parse_intlist_args(const std::string& s, int64_t size) {
  size_t n = s.size();

  if (s.empty()) return std::vector<int64_t>();

  // case 1. s is an int (e.g., s=2)
  if (s[0] != '{') {
    return std::vector<int64_t>(size, std::stol(s));
  }

  // case 2. s is a list of dims (e.g., s={1,2})

  // since already checked left brace '{' above, here only checks right brace '}'
  TORCH_CHECK(s[n - 1] == '}', "Default value of IntArrayRef is missing right brace '}', found ", s[n - 1]);

  auto args = std::vector<int64_t>();
  std::istringstream ss(s.substr(1, s.length() - 2)); // exclude '{' and '}'
  std::string tok;

  while(std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;
}

void FunctionParameter::set_default_str(const std::string& str) {
  if (str == "None") {
    allow_none = true;
  }
  if (type_ == ParameterType::TENSOR) {
    if (str != "None") {
      throw std::runtime_error("default value for Tensor must be none, got: " + str);
    }
  } else if (type_ == ParameterType::INT64) {
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
      default_scalar = as_integer.has_value() ? at::Scalar(as_integer.value()) :
                                                at::Scalar(atof(str.c_str()));
    }
  } else if (type_ == ParameterType::INT_LIST) {
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
  } else if (type_ == ParameterType::STRING) {
    if (str != "None" && str != "") {
      throw std::runtime_error("invalid default string: " + str);
    }
  }
}

FunctionSignature::FunctionSignature(const std::string& fmt, int index)
  : min_args(0)
  , max_args(0)
  , max_pos_args(0)
  , index(index)
  , hidden(false)
  , deprecated(false)
{
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);

  bool allow_numbers_as_tensors = should_allow_numbers_as_tensors(name);

  auto last_offset = open_paren + 1;
  auto next_offset = last_offset;
  bool keyword_only = false;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);
      done = true;
      next_offset = offset+ 1;
      // this 'if' happens for an empty parameter list, i.e. fn().
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    } else {
      next_offset = offset + 2;
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
  // TODO: consider printing more proper schema strings with defaults, optionals, etc.
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
    i++;
  }
  ss << ")";
  return ss.str();
}

[[noreturn]]
static void extra_args(const FunctionSignature& signature, ssize_t nargs) {
  const long max_pos_args = signature.max_pos_args;
  const long min_args = signature.min_args;
  const long nargs_ = nargs;
  if (min_args != max_pos_args) {
    throw TypeError("%s() takes from %ld to %ld positional arguments but %ld were given",
        signature.name.c_str(), min_args, max_pos_args, nargs_);
  }
  throw TypeError("%s() takes %ld positional argument%s but %ld %s given",
      signature.name.c_str(),
      max_pos_args, max_pos_args == 1 ? "" : "s",
      nargs_, nargs == 1 ? "was" : "were");
}

[[noreturn]]
static void missing_args(const FunctionSignature& signature, int idx) {
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

  throw TypeError("%s() missing %d required positional argument%s: %s",
      signature.name.c_str(),
      num_missing,
      num_missing == 1 ? "s" : "",
      ss.str().c_str());
}

static ssize_t find_param(FunctionSignature& signature, PyObject* name) {
  ssize_t i = 0;
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

[[noreturn]]
static void extra_kwargs(FunctionSignature& signature, PyObject* kwargs, ssize_t num_pos_args) {
  PyObject *key, *value;
  ssize_t pos = 0;

  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    if (!THPUtils_checkString(key)) {
      throw TypeError("keywords must be strings");
    }

    auto param_idx = find_param(signature, key);
    if (param_idx < 0) {
      throw TypeError("%s() got an unexpected keyword argument '%s'",
          signature.name.c_str(), THPUtils_unpackString(key).c_str());
    }

    if (param_idx < num_pos_args) {
      throw TypeError("%s() got multiple values for argument '%s'",
          signature.name.c_str(), THPUtils_unpackString(key).c_str());
    }
  }

  // this should never be hit
  throw TypeError("invalid keyword arguments");
}

bool FunctionSignature::parse(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* dst[],  // NOLINT
                              bool raise_exception) {
  auto nargs = args ? PyTuple_GET_SIZE(args) : 0;
  ssize_t remaining_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  ssize_t arg_pos = 0;
  bool allow_varargs_intlist = false;

  // if there is a single positional IntArrayRef argument, i.e. expand(..), view(...),
  // allow a var-args style IntArrayRef, so expand(5,3) behaves as expand((5,3))
  if (max_pos_args == 1 && params[0].type_ == ParameterType::INT_LIST) {
    allow_varargs_intlist = true;
  }

  if (nargs > max_pos_args && !allow_varargs_intlist) {
    if (raise_exception) {
      // foo() takes takes 2 positional arguments but 3 were given
      extra_args(*this, nargs);
    }
    return false;
  }

  if (!overloaded_args.empty()) {
    overloaded_args.clear();
  }

  int i = 0;
  if (self != nullptr && !THPVariable_CheckExact(self) && check_has_torch_function(self)) {
    append_overloaded_arg(&this->overloaded_args, self);
  }
  for (auto& param : params) {
    PyObject* obj = nullptr;
    bool is_kwd = false;
    if (arg_pos < nargs) {
      // extra positional args given after single positional IntArrayRef arg
      if (param.keyword_only) {
        if (raise_exception) {
          extra_args(*this, nargs);
        }
        return false;
      }
      obj = PyTuple_GET_ITEM(args, arg_pos);
    } else if (kwargs) {
      obj = PyDict_GetItem(kwargs, param.python_name);
      for (PyObject *numpy_name: param.numpy_python_names) {
        if (obj) {
          break;
        }
        obj = PyDict_GetItem(kwargs, numpy_name);
      }
      is_kwd = true;
    }

    if ((!obj && param.optional) || (obj == Py_None && param.allow_none)) {
      dst[i++] = nullptr;
    } else if (!obj) {
      if (raise_exception) {
        // foo() missing 1 required positional argument: "b"
        missing_args(*this, i);
      }
      return false;
    } else if (param.check(obj, this->overloaded_args, i)) {
      dst[i++] = obj;
    // XXX: the Variable check is necessary because sizes become tensors when
    // tracer is enabled. This behavior easily leads to ambiguities, and we
    // should avoid having complex signatures that make use of it...
    } else if (allow_varargs_intlist && arg_pos == 0 && !is_kwd &&
               THPUtils_checkIndex(obj)) {
      // take all positional arguments as this parameter
      // e.g. permute(1, 2, 3) -> permute((1, 2, 3))
      dst[i++] = args;
      arg_pos = nargs;
      continue;
    } else if (raise_exception) {
      if (is_kwd) {
        // foo(): argument 'other' must be str, not int
        throw TypeError("%s(): argument '%s' must be %s, not %s",
            name.c_str(), param.name.c_str(), param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      } else {
        // foo(): argument 'other' (position 2) must be str, not int
        throw TypeError("%s(): argument '%s' (position %ld) must be %s, not %s",
            name.c_str(), param.name.c_str(), static_cast<long>(arg_pos + 1),
            param.type_name().c_str(), Py_TYPE(obj)->tp_name);
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

PythonArgParser::PythonArgParser(std::vector<std::string> fmts, bool traceable)
 : max_args(0)
 , traceable(traceable)
{
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
  if (signatures_.size() > 0) {
    function_name = signatures_[0].name;
  }

  // Check deprecated signatures last
  std::stable_partition(signatures_.begin(), signatures_.end(),
    [](const FunctionSignature & sig) {
      return !sig.deprecated;
    });
}

void PythonArgParser::check_deprecated(const FunctionSignature & signature) {
  if (signature.deprecated) {
    auto msg = c10::str(
      "This overload of ", signature.name, " is deprecated:\n\t",
      signature.name, signature.toString());
    auto signatures = get_signatures();
    if (!signatures.empty()) {
      msg += "\nConsider using one of the following signatures instead:";
      for (const auto & sig : signatures) {
        msg += "\n\t";
        msg += signature.name;
        msg += sig;
      }
    }
    TORCH_WARN_ONCE(msg);
  }
}

PythonArgs PythonArgParser::raw_parse(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {  // NOLINT
  if (signatures_.size() == 1) {
    auto& signature = signatures_[0];
    signature.parse(self, args, kwargs, parsed_args, true);
    check_deprecated(signature);
    return PythonArgs(traceable, signature, parsed_args);
  }

  for (auto& signature : signatures_) {
    if (signature.parse(self, args, kwargs, parsed_args, false)) {
      check_deprecated(signature);
      return PythonArgs(traceable, signature, parsed_args);
    }
  }

  print_error(self, args, kwargs, parsed_args);
}

void PythonArgParser::print_error(PyObject* self, PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {  // NOLINT
  auto num_args = PyTuple_GET_SIZE(args) + (kwargs ? PyDict_Size(kwargs) : 0);
  std::vector<int> plausible_idxs;
  ssize_t i = 0;
  for (auto& signature : signatures_) {
    if (num_args >= signature.min_args && num_args <= signature.max_args && !signature.hidden) {
      plausible_idxs.push_back(i);
    }
    i++;
  }

  if (plausible_idxs.size() == 1) {
    auto& signature = signatures_[plausible_idxs[0]];
    signature.parse(self, args, kwargs, parsed_args, true);
  }

  auto options = get_signatures();
  auto msg = torch::format_invalid_args(args, kwargs, function_name + "()", options);
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
    return reinterpret_cast<THPVariable*>(obj)->cdata;
  }

  at::Scalar scalar;
  if (PyBool_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackBool(obj));
  } else if (THPUtils_checkLong(obj)) {
    scalar = at::Scalar(THPUtils_unpackLong(obj));
  } else if (PyComplex_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackComplexDouble(obj));
  } else if (THPUtils_checkDouble(obj)) {
    scalar = at::Scalar(THPUtils_unpackDouble(obj));
  } else {
    // NB: Are you here because you passed None to a Variable method,
    // and you expected an undefined tensor to be returned?   Don't add
    // a test for Py_None here; instead, you need to mark the argument
    // as *allowing none*; you can do this by writing 'Tensor?' instead
    // of 'Tensor' in the ATen metadata.
    throw TypeError("expected Tensor as argument %d, but got %s", i,
        Py_TYPE(obj)->tp_name);
  }
  at::AutoNonVariableTypeMode guard;  // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  at::Tensor tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

at::Scalar PythonArgs::scalar_slow(int i) {
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, jit::NumberType::get());
  }

  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't currently
  // handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(args[i])) {
    return ((THPVariable*)args[i])->cdata.item();
  }

  if (THPUtils_checkLong(args[i])) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(args[i])));
  }

  if (PyBool_Check(args[i])) {
    return at::Scalar(THPUtils_unpackBool(args[i]));
  }

  if (PyComplex_Check(args[i])) {
    return at::Scalar(THPUtils_unpackComplexDouble(args[i]));
  }
  return at::Scalar(THPUtils_unpackDouble(args[i]));
}

at::Scalar PythonArgs::scalar_slow(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't currently
  // handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return ((THPVariable*)arg)->cdata.item();
  }

  if (THPUtils_checkLong(arg)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  return at::Scalar(THPUtils_unpackDouble(arg));
}

} // namespace torch
