#include <torch/csrc/utils/python_arg_parser.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/python_strings.h>
#include <ATen/core/EnableNamedTensor.h>

#include <ATen/ATen.h>

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
// If you modify this, you will need to adjust the blacklist in
// tools/pyi/gen_pyi.py (and add hardcoded signatures for these
// functions.)
static bool should_allow_numbers_as_tensors(const std::string& name) {
  static std::unordered_set<std::string> allowed = {
    "add", "add_", "add_out",
    "div", "div_", "div_out",
    "mul", "mul_", "mul_out",
    "sub", "sub_", "sub_out",
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

bool FunctionParameter::check(PyObject* obj) {
  switch (type_) {
    case ParameterType::TENSOR: {
      return THPVariable_Check(obj) || (allow_numbers_as_tensors && THPUtils_checkScalar(obj));
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
#ifdef BUILD_NAMEDTENSOR
    case ParameterType::DIMNAME: return THPUtils_checkDimname(obj);
    case ParameterType::DIMNAME_LIST: {
      if (THPUtils_checkDimnameList(obj)) {
        return true;
      }
      // if a size is specified (e.g. DimnameList[1]) we also allow passing a single Dimname
      return size == 1 && THPUtils_checkDimname(obj);
    }
#endif
    case ParameterType::TENSOR_LIST: return six::isTuple(obj) || PyList_Check(obj);
    case ParameterType::INT_LIST: {
      if (PyTuple_Check(obj) || PyList_Check(obj)) {
        return true;
      }
      // if a size is specified (e.g. IntArrayRef[2]) we also allow passing a single int
      return size > 0 && THPUtils_checkLong(obj);
    }
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
#ifdef BUILD_NAMEDTENSOR
    case ParameterType::DIMNAME: return "name";
    case ParameterType::DIMNAME_LIST: return "tuple of names";
#endif
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
      default_layout = nullptr;
    } else if (str == "torch.strided") {
      default_layout = torch::getLayout(at::Backend::CPU);
    } else if (str == "torch.sparse_coo") {
      default_layout = torch::getLayout(at::Backend::SparseCPU);
    } else {
      throw std::runtime_error("invalid default value for layout: " + str);
    }
  } else if (type_ == ParameterType::DEVICE) {
    if (str != "None") {
      throw std::runtime_error("invalid device: " + str);
    }
  } else if (type_ == ParameterType::STRING) {
    if (str != "None" || str != "") {
      throw std::runtime_error("invalid default string: " + str);
    }
  }
}

FunctionSignature::FunctionSignature(const std::string& fmt)
  : min_args(0)
  , max_args(0)
  , max_pos_args(0)
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
      next_offset = offset + 1;
    } else {
      next_offset = offset + 2;
    }
    if (offset == std::string::npos) {
      throw std::runtime_error("missing closing parenthesis: " + fmt);
    }
    if (offset == last_offset) {
      break;
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
  std::ostringstream ss;
  ss << "(";
  int i = 0;
  for (auto& param : params) {
    if (i != 0) {
      ss << ", ";
    }
    ss << param.type_name() << " " << param.name;
    i++;
  }
  ss << ")";
  return ss.str();
}

[[noreturn]]
static void extra_args(const FunctionSignature& signature, ssize_t nargs) {
  auto max_pos_args = signature.max_pos_args;
  auto min_args = signature.min_args;
  if (min_args != max_pos_args) {
    throw TypeError("%s() takes from %d to %d positional arguments but %d were given",
        signature.name.c_str(), min_args, max_pos_args, nargs);
  }
  throw TypeError("%s() takes %d positional argument%s but %d %s given",
      signature.name.c_str(),
      max_pos_args, max_pos_args == 1 ? "" : "s",
      nargs, nargs == 1 ? "was" : "were");
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

bool FunctionSignature::parse(PyObject* args, PyObject* kwargs, PyObject* dst[],
                              bool raise_exception) {
  auto nargs = PyTuple_GET_SIZE(args);
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

  int i = 0;
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
    } else if (param.check(obj)) {
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
        throw TypeError("%s(): argument '%s' (position %d) must be %s, not %s",
            name.c_str(), param.name.c_str(), arg_pos + 1,
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
  for (auto& fmt : fmts) {
    signatures_.emplace_back(fmt);
  }
  for (auto& signature : signatures_) {
    if (signature.max_args > max_args) {
      max_args = signature.max_args;
    }
  }
  if (signatures_.size() > 0) {
    function_name = signatures_[0].name;
  }
}

PythonArgs PythonArgParser::raw_parse(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {
  if (signatures_.size() == 1) {
    auto& signature = signatures_[0];
    signature.parse(args, kwargs, parsed_args, true);
    return PythonArgs(0, traceable, signature, parsed_args);
  }

  int i = 0;
  for (auto& signature : signatures_) {
    if (signature.parse(args, kwargs, parsed_args, false)) {
      return PythonArgs(i, traceable, signature, parsed_args);
    }
    i++;
  }

  print_error(args, kwargs, parsed_args);
}

void PythonArgParser::print_error(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {
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
    signature.parse(args, kwargs, parsed_args, true);
  }

  std::vector<std::string> options;
  for (auto& signature : signatures_) {
    if (!signature.hidden) {
      options.push_back(signature.toString());
    }
  }

  auto msg = torch::format_invalid_args(args, kwargs, function_name + "()", options);
  throw TypeError("%s", msg.c_str());
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
  }else if (PyComplex_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackComplexDouble(obj));
  }else if (THPUtils_checkDouble(obj)) {
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
  at::AutoNonVariableTypeMode guard;

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

} // namespace torch
