#include "torch/csrc/utils/python_arg_parser.h"

#include <stdexcept>
#include <unordered_map>
#include <sstream>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/invalid_arguments.h"

using namespace at;

namespace torch {

static std::unordered_map<std::string, ParameterType> type_map = {
  {"Tensor", ParameterType::TENSOR},
  {"Scalar", ParameterType::SCALAR},
  {"int64_t", ParameterType::INT64},
  {"double", ParameterType::DOUBLE},
  {"TensorList", ParameterType::TENSOR_LIST},
  {"IntList", ParameterType::INT_LIST},
  {"Generator", ParameterType::GENERATOR},
  {"bool", ParameterType::BOOL},
  {"Storage", ParameterType::STORAGE},
  {"PyObject*", ParameterType::PYOBJECT}
};

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
#if PY_MAJOR_VERSION == 2
  python_name = PyString_InternFromString(name.c_str());
#else
  python_name = PyUnicode_InternFromString(name.c_str());
#endif
}

bool FunctionParameter::check(PyObject* obj) {
  switch (type_) {
    case ParameterType::TENSOR: {
      return THPVariable_Check(obj);
    }
    case ParameterType::SCALAR: return THPUtils_checkDouble(obj);
    case ParameterType::INT64: return THPUtils_checkLong(obj);
    case ParameterType::DOUBLE: return THPUtils_checkDouble(obj);
    case ParameterType::TENSOR_LIST: return PyTuple_Check(obj) || PyList_Check(obj);
    case ParameterType::INT_LIST: {
      if (PyTuple_Check(obj) || PyList_Check(obj)) {
        return true;
      }
      // if a size is specified (e.g. IntList[2]) we also allow passing a single int
      return size > 0 && THPUtils_checkLong(obj);
    }
    case ParameterType::GENERATOR: return THPGenerator_Check(obj);
    case ParameterType::BOOL: return PyBool_Check(obj);
    case ParameterType::STORAGE: return false;
    case ParameterType::PYOBJECT: return true;
    default: throw std::runtime_error("unknown parameter type");
  }
}

std::string FunctionParameter::type_name() const {
  switch (type_) {
    case ParameterType::TENSOR: return "Variable";
    case ParameterType::SCALAR: return "float";
    case ParameterType::INT64: return "int";
    case ParameterType::DOUBLE: return "float";
    case ParameterType::TENSOR_LIST: return "tuple of Variables";
    case ParameterType::INT_LIST: return "tuple of ints";
    case ParameterType::GENERATOR: return "torch.Generator";
    case ParameterType::BOOL: return "bool";
    case ParameterType::STORAGE: return "torch.Storage";
    case ParameterType::PYOBJECT: return "object";
    default: throw std::runtime_error("unknown parameter type");
  }
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
  } else if (type_ == ParameterType::SCALAR) {
    if (str == "None") {
      // This is a bit awkward, but convenient for clamp which takes Scalars,
      // but allows None.
      default_scalar = Scalar(NAN);
    } else {
      default_scalar = Scalar(atof(str.c_str()));
    }
  } else if (type_ == ParameterType::INT_LIST) {
    if (str != "None") {
      default_intlist.assign(size, std::stoi(str));
    }
  }
}

FunctionSignature::FunctionSignature(const std::string& fmt)
  : min_args(0)
  , max_args(0)
  , max_pos_args(0)
  , deprecated(false)
{
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);

  auto last_offset = open_paren + 1;
  auto next_offset = last_offset;
  bool keyword_only = false;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    if (offset == std::string::npos) {
      offset = fmt.find(")", last_offset);
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
    }
  }

  if (fmt.substr(last_offset) == "|deprecated") {
    deprecated = true;
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
void type_error(const char *format, ...) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];

  va_list fmt_args;
  va_start(fmt_args, format);
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);
  va_end(fmt_args);

  throw type_exception(error_buf);
}

[[noreturn]]
static void extra_args(const FunctionSignature& signature, ssize_t nargs) {
  auto max_pos_args = signature.max_pos_args;
  auto min_args = signature.min_args;
  if (min_args != max_pos_args) {
    type_error("%s() takes from %d to %d positional arguments but %d were given",
        signature.name.c_str(), min_args, max_pos_args, nargs);
  }
  type_error("%s() takes %d positional argument%s but %d %s given",
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

  type_error("%s() missing %d required positional argument%s: %s",
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
      type_error("keywords must be strings");
    }

    auto param_idx = find_param(signature, key);
    if (param_idx < 0) {
      type_error("%s() got an unexpected keyword argument '%s'",
          signature.name.c_str(), THPUtils_unpackString(key).c_str());
    }

    if (param_idx < num_pos_args) {
      type_error("%s() got multiple values for argument '%s'",
          signature.name.c_str(), THPUtils_unpackString(key).c_str());
    }
  }

  // this should never be hit
  type_error("invalid keyword arguments");
}

bool FunctionSignature::parse(PyObject* args, PyObject* kwargs, PyObject* dst[],
                              bool raise_exception) {
  auto nargs = PyTuple_GET_SIZE(args);
  ssize_t remaining_kwargs = kwargs ? PyDict_Size(kwargs) : 0;
  ssize_t arg_pos = 0;
  bool allow_varargs_intlist = false;

  // if there is a single positional IntList argument, i.e. expand(..), view(...),
  // allow a var-args style IntList, so expand(5,3) behaves as expand((5,3))
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
      obj = PyTuple_GET_ITEM(args, arg_pos);
    } else if (kwargs) {
      obj = PyDict_GetItem(kwargs, param.python_name);
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
    } else if (allow_varargs_intlist && arg_pos == 0 && !is_kwd) {
      // take all positional arguments as this parameter
      // e.g. permute(1, 2, 3) -> permute((1, 2, 3))
      dst[i++] = args;
      arg_pos = nargs;
      continue;
    } else if (raise_exception) {
      if (is_kwd) {
        // foo(): argument 'other' must be str, not int
        type_error("%s(): argument '%s' must be %s, not %s",
            name.c_str(), param.name.c_str(), param.type_name().c_str(),
            Py_TYPE(obj)->tp_name);
      } else {
        // foo(): argument 'other' (position 2) must be str, not int
        type_error("%s(): argument '%s' (position %d) must be %s, not %s",
            name.c_str(), param.name.c_str(), arg_pos + 1,
            param.type_name().c_str(), Py_TYPE(obj)->tp_name);
      }
    } else {
      return false;
    }

    if (!is_kwd) {
      arg_pos++;
    } else {
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

PythonArgParser::PythonArgParser(std::vector<std::string> fmts)
 : max_args(0)
{
  for (auto& fmt : fmts) {
    signatures_.push_back(FunctionSignature(fmt));
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

PythonArgs PythonArgParser::parse(PyObject* args, PyObject* kwargs, PyObject* parsed_args[]) {
  if (signatures_.size() == 1) {
    auto& signature = signatures_[0];
    signature.parse(args, kwargs, parsed_args, true);
    return PythonArgs(0, signature, parsed_args);
  }

  int i = 0;
  for (auto& signature : signatures_) {
    if (signature.parse(args, kwargs, parsed_args, false)) {
      return PythonArgs(i, signature, parsed_args);
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
    if (num_args >= signature.min_args && num_args <= signature.max_args && !signature.deprecated) {
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
    if (!signature.deprecated) {
      options.push_back(signature.toString());
    }
  }

  auto msg = torch::format_invalid_args(args, kwargs, function_name + "()", options);
  type_error("%s", msg.c_str());
}


} // namespace torch
