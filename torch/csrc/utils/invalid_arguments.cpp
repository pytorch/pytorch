#include <torch/csrc/utils/invalid_arguments.h>

#include <torch/csrc/utils/python_strings.h>

#include <torch/csrc/utils/memory.h>

#include <algorithm>
#include <unordered_map>
#include <memory>

namespace torch {

namespace {

std::string py_typename(PyObject *object) {
  return Py_TYPE(object)->tp_name;
}

struct Type {
  virtual bool is_matching(PyObject *object) = 0;
  virtual ~Type() = default;
};

struct SimpleType: public Type {
  SimpleType(std::string& name): name(name) {};

  bool is_matching(PyObject *object) override {
    return py_typename(object) == name;
  }

  std::string name;
};

struct MultiType: public Type {
  MultiType(std::initializer_list<std::string> accepted_types):
    types(accepted_types) {};

  bool is_matching(PyObject *object) override {
    auto it = std::find(types.begin(), types.end(), py_typename(object));
    return it != types.end();
  }

  std::vector<std::string> types;
};

struct NullableType: public Type {
  NullableType(std::unique_ptr<Type> type): type(std::move(type)) {};

  bool is_matching(PyObject *object) override {
    return object == Py_None || type->is_matching(object);
  }

  std::unique_ptr<Type> type;
};

struct TupleType: public Type {
  TupleType(std::vector<std::unique_ptr<Type>> types):
    types(std::move(types)) {};

  bool is_matching(PyObject *object) override {
    if (!PyTuple_Check(object)) return false;
    auto num_elements = PyTuple_GET_SIZE(object);
    if (num_elements != (long)types.size()) return false;
    for (int i = 0; i < num_elements; i++) {
      if (!types[i]->is_matching(PyTuple_GET_ITEM(object, i)))
        return false;
    }
    return true;
  }

  std::vector<std::unique_ptr<Type>> types;
};

struct SequenceType: public Type {
  SequenceType(std::unique_ptr<Type> type):
    type(std::move(type)) {};

  bool is_matching(PyObject *object) override {
    if (!PySequence_Check(object)) return false;
    auto num_elements = PySequence_Length(object);
    for (int i = 0; i < num_elements; i++) {
      if (!type->is_matching(PySequence_GetItem(object, i)))
        return false;
    }
    return true;
  }

  std::unique_ptr<Type> type;
};

struct Argument {
  Argument(std::string name, std::unique_ptr<Type> type):
      name(std::move(name)), type(std::move(type)) {};

  std::string name;
  std::unique_ptr<Type> type;
};

struct Option {
  Option(std::vector<Argument> arguments, bool is_variadic, bool has_out):
      arguments(std::move(arguments)), is_variadic(is_variadic), has_out(has_out) {};
  Option(bool is_variadic, bool has_out):
      arguments(), is_variadic(is_variadic), has_out(has_out) {};
  Option(const Option&) = delete;
  Option(Option&& other):
    arguments(std::move(other.arguments)), is_variadic(other.is_variadic),
    has_out(other.has_out) {};

  std::vector<Argument> arguments;
  bool is_variadic;
  bool has_out;
};

std::vector<std::string> _splitString(const std::string &s, const std::string& delim) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end;
  while((end = s.find(delim, start)) != std::string::npos) {
    tokens.push_back(s.substr(start, end-start));
    start = end + delim.length();
  }
  tokens.push_back(s.substr(start));
  return tokens;
}

std::unique_ptr<Type> _buildType(std::string type_name, bool is_nullable) {
  std::unique_ptr<Type> result;
  if (type_name == "float") {
    result = torch::make_unique<MultiType>(MultiType{"float", "int", "long"});
  } else if (type_name == "int") {
    result = torch::make_unique<MultiType>(MultiType{"int", "long"});
  } else if (type_name.find("tuple[") == 0) {
    auto type_list = type_name.substr(6);
    type_list.pop_back();
    std::vector<std::unique_ptr<Type>> types;
    for (auto& type: _splitString(type_list, ","))
      types.emplace_back(_buildType(type, false));
    result = torch::make_unique<TupleType>(std::move(types));
  } else if (type_name.find("sequence[") == 0) {
    auto subtype = type_name.substr(9);
    subtype.pop_back();
    result = torch::make_unique<SequenceType>(_buildType(subtype, false));
  } else {
    result = torch::make_unique<SimpleType>(type_name);
  }
  if (is_nullable)
    result = torch::make_unique<NullableType>(std::move(result));
  return result;
}

std::pair<Option, std::string> _parseOption(const std::string& _option_str,
    const std::unordered_map<std::string, PyObject*>& kwargs)
{
  if (_option_str == "no arguments")
    return std::pair<Option, std::string>(Option(false, false), _option_str);
  bool has_out = false;
  std::vector<Argument> arguments;
  std::string printable_option = _option_str;
  std::string option_str = _option_str.substr(1, _option_str.length()-2);

  /// XXX: this is a hack only for the out arg in TensorMethods
  auto out_pos = printable_option.find('#');
  if (out_pos != std::string::npos) {
    if (kwargs.count("out") > 0) {
      std::string kwonly_part = printable_option.substr(out_pos+1);
      printable_option.erase(out_pos);
      printable_option += "*, ";
      printable_option += kwonly_part;
    } else if (out_pos >= 2) {
      printable_option.erase(out_pos-2);
      printable_option += ")";
    } else {
      printable_option.erase(out_pos);
      printable_option += ")";
    }
    has_out = true;
  }

  for (auto& arg: _splitString(option_str, ", ")) {
    bool is_nullable = false;
    auto type_start_idx = 0;
    if (arg[type_start_idx] == '#') {
      type_start_idx++;
    }
    if (arg[type_start_idx] == '[') {
      is_nullable = true;
      type_start_idx++;
      arg.erase(arg.length() - std::string(" or None]").length());
    }

    auto type_end_idx = arg.find_last_of(' ');
    auto name_start_idx = type_end_idx + 1;

    // "type ... name" => "type ... name"
    //          ^              ^
    auto dots_idx = arg.find("...");
    if (dots_idx != std::string::npos)
        type_end_idx -= 4;

    std::string type_name =
      arg.substr(type_start_idx, type_end_idx-type_start_idx);
    std::string name =
        arg.substr(name_start_idx);

    arguments.emplace_back(name, _buildType(type_name, is_nullable));
  }

  bool is_variadic = option_str.find("...") != std::string::npos;
  return std::pair<Option, std::string>(
    Option(std::move(arguments), is_variadic, has_out),
    std::move(printable_option)
  );
}

bool _argcountMatch(
    const Option& option,
    const std::vector<PyObject*>& arguments,
    const std::unordered_map<std::string, PyObject*>& kwargs)
{
  auto num_expected = option.arguments.size();
  auto num_got = arguments.size() + kwargs.size();
  // Note: variadic functions don't accept kwargs, so it's ok
  if (option.has_out && kwargs.count("out") == 0)
    num_expected--;
  return num_got == num_expected ||
    (option.is_variadic && num_got > num_expected);
}

std::string _formattedArgDesc(
    const Option& option,
    const std::vector<PyObject*>& arguments,
    const std::unordered_map<std::string, PyObject*>& kwargs)
{
  std::string red;
  std::string reset_red;
  std::string green;
  std::string reset_green;
  if (isatty(1) && isatty(2)) {
    red = "\33[31;1m";
    reset_red = "\33[0m";
    green = "\33[32;1m";
    reset_green = "\33[0m";
  } else {
    red = "!";
    reset_red = "!";
    green = "";
    reset_green = "";
  }

  auto num_args = arguments.size() + kwargs.size();
  std::string result = "(";
  for (size_t i = 0; i < num_args; i++) {
    bool is_kwarg = i >= arguments.size();
    PyObject *arg = is_kwarg ? kwargs.at(option.arguments[i].name) : arguments[i];

    bool is_matching = false;
    if (i < option.arguments.size()) {
      is_matching = option.arguments[i].type->is_matching(arg);
    } else if (option.is_variadic) {
      is_matching = option.arguments.back().type->is_matching(arg);
    }

    if (is_matching)
      result += green;
    else
      result += red;
    if (is_kwarg) result += option.arguments[i].name + "=";
    result += py_typename(arg);
    if (is_matching)
        result += reset_green;
    else
        result += reset_red;
    result += ", ";
  }
  if (arguments.size() > 0)
    result.erase(result.length()-2);
  result += ")";
  return result;
}

std::string _argDesc(const std::vector<PyObject *>& arguments,
    const std::unordered_map<std::string, PyObject *>& kwargs)
{
  std::string result = "(";
  for (auto& arg: arguments)
    result += std::string(py_typename(arg)) + ", ";
  for (auto& kwarg: kwargs)
    result += kwarg.first + "=" + py_typename(kwarg.second) + ", ";
  if (arguments.size() > 0)
    result.erase(result.length()-2);
  result += ")";
  return result;
}

std::vector<std::string> _tryMatchKwargs(const Option& option,
    const std::unordered_map<std::string, PyObject*>& kwargs) {
  std::vector<std::string> unmatched;
  int start_idx = option.arguments.size() - kwargs.size();
  if (option.has_out && kwargs.count("out") == 0)
    start_idx--;
  if (start_idx < 0)
    start_idx = 0;
  for (auto& entry: kwargs) {
    bool found = false;
    for (unsigned int i = start_idx; i < option.arguments.size(); i++) {
      if (option.arguments[i].name == entry.first) {
        found = true;
        break;
      }
    }
    if (!found)
      unmatched.push_back(entry.first);
  }
  return unmatched;
}

} // anonymous namespace

std::string format_invalid_args(
    PyObject *given_args, PyObject *given_kwargs, const std::string& function_name,
    const std::vector<std::string>& options)
{
  std::vector<PyObject *> args;
  std::unordered_map<std::string, PyObject *> kwargs;
  std::string error_msg;
  error_msg.reserve(2000);
  error_msg += function_name;
  error_msg += " received an invalid combination of arguments - ";

  Py_ssize_t num_args = PyTuple_Size(given_args);
  for (int i = 0; i < num_args; i++) {
    PyObject *arg = PyTuple_GET_ITEM(given_args, i);
    args.push_back(arg);
  }

  bool has_kwargs = given_kwargs && PyDict_Size(given_kwargs) > 0;
  if (has_kwargs) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(given_kwargs, &pos, &key, &value)) {
      kwargs.emplace(THPUtils_unpackString(key), value);
    }
  }

  if (options.size() == 1) {
    auto pair = _parseOption(options[0], kwargs);
    auto& option = pair.first;
    auto& option_str = pair.second;
    std::vector<std::string> unmatched_kwargs;
    if (has_kwargs)
      unmatched_kwargs = _tryMatchKwargs(option, kwargs);
    if (unmatched_kwargs.size()) {
      error_msg += "got unrecognized keyword arguments: ";
      for (auto& kwarg: unmatched_kwargs)
        error_msg += kwarg + ", ";
      error_msg.erase(error_msg.length()-2);
    } else {
      error_msg += "got ";
      if (_argcountMatch(option, args, kwargs)) {
        error_msg += _formattedArgDesc(option, args, kwargs);
      } else {
        error_msg += _argDesc(args, kwargs);
      }
      error_msg += ", but expected ";
      error_msg += option_str;
    }
  } else {
    error_msg += "got ";
    error_msg += _argDesc(args, kwargs);
    error_msg += ", but expected one of:\n";
    for (auto &option_str: options) {
      auto pair = _parseOption(option_str, kwargs);
      auto& option = pair.first;
      auto& printable_option_str = pair.second;
      error_msg += " * ";
      error_msg += printable_option_str;
      error_msg += "\n";
      if (_argcountMatch(option, args, kwargs)) {
        std::vector<std::string> unmatched_kwargs;
        if (has_kwargs)
          unmatched_kwargs = _tryMatchKwargs(option, kwargs);
        if (unmatched_kwargs.size() > 0) {
          error_msg += "      didn't match because some of the keywords were incorrect: ";
          for (auto& kwarg: unmatched_kwargs)
            error_msg += kwarg + ", ";
          error_msg.erase(error_msg.length()-2);
          error_msg += "\n";
        } else {
          error_msg += "      didn't match because some of the arguments have invalid types: ";
          error_msg += _formattedArgDesc(option, args, kwargs);
          error_msg += "\n";
        }
      }
    }
  }
  return error_msg;
}


} // namespace torch
