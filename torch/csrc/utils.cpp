#include <Python.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include "THP.h"

#include "generic/utils.cpp"
#include <TH/THGenerateAllTypes.h>

int THPUtils_getCallable(PyObject *arg, PyObject **result) {
  if (!PyCallable_Check(arg))
    return 0;
  *result = arg;
  return 1;
}


THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first) {
  Py_ssize_t length = PyTuple_Size(args);
  if (length < ignore_first+1)
    throw std::runtime_error("Provided " + std::to_string(length) +
        " arguments, but expected at least " + std::to_string(ignore_first+1));

  // Maybe there's a LongStorage
  PyObject *first_arg = PyTuple_GET_ITEM(args, ignore_first);
  if (length == ignore_first+1 && THPLongStorage_Check(first_arg)) {
    THPLongStorage *storage = (THPLongStorage*)first_arg;
    THLongStorage_retain(storage->cdata);
    return storage->cdata;
  }

  // If not, let's try to parse the numbers
  THLongStoragePtr result = THLongStorage_newWithSize(length-ignore_first);
  for (Py_ssize_t i = ignore_first; i < length; ++i) {
    PyObject *arg = PyTuple_GET_ITEM(args, i);
    if (!THPUtils_checkLong(arg))
      throw std::invalid_argument("Expected an int argument, but got " +
          std::string(THPUtils_typename(arg)) + "at position " +
          std::to_string(i));
    result->data[i-ignore_first] = THPUtils_unpackLong(arg);
  }
  return result.release();
}


void THPUtils_setError(const char *format, ...)
{
  static const size_t ERROR_BUFFER_SIZE = 1000;
  char buffer[ERROR_BUFFER_SIZE];
  va_list fmt_args;

  va_start(fmt_args, format);
  vsnprintf(buffer, ERROR_BUFFER_SIZE, format, fmt_args);
  va_end(fmt_args);
  PyErr_SetString(PyExc_RuntimeError, buffer);
}

std::string _THPUtils_typename(PyObject *object)
{
  std::string type_name = Py_TYPE(object)->tp_name;
  std::string result;
  if (type_name.find("Storage") != std::string::npos ||
          type_name.find("Tensor") != std::string::npos) {
    PyObject *module_name = PyObject_GetAttrString(object, "__module__");
#if PY_MAJOR_VERSION == 2
    if (module_name && PyString_Check(module_name)) {
      result = PyString_AS_STRING(module_name);
    }
#else
    if (module_name && PyUnicode_Check(module_name)) {
      PyObject *module_name_bytes = PyUnicode_AsASCIIString(module_name);
      if (module_name_bytes) {
        result = PyBytes_AS_STRING(module_name_bytes);
        Py_DECREF(module_name_bytes);
      }
    }
#endif
    Py_XDECREF(module_name);
    result += ".";
    result += type_name;
  } else {
    result = std::move(type_name);
  }
  return result;
}

struct Argument {
  Argument(): type(), is_nullable(false), is_variadic(false) {};
  std::string type;
  bool is_nullable;
  bool is_variadic;
};

std::vector<Argument> THPUtils_parseOption(const std::string &option)
{
  std::vector<Argument> arguments;
  if (option == "no arguments")
    return arguments;
  long current_index = -1;
  do {
    Argument arg;
    auto type_end_idx = option.find_first_of(' ', current_index+2);
    // Last argument - there are no more spaces
    if (type_end_idx == std::string::npos)
      type_end_idx = option.length()-2;
    arg.type = option.substr(current_index+2, type_end_idx-current_index-2);
    // A dumb check for [type or None]
    if (arg.type[0] == '[')
      arg.is_nullable = true;
    current_index++;
    current_index = option.find_first_of(',', current_index+1);
    arguments.push_back(std::move(arg));
  } while ((size_t)current_index != std::string::npos);
  // Look for "type name...)" at the end
  if (option.substr(option.length()-4, 3) == "...")
    arguments.back().is_variadic = true;
  return arguments;
}

bool THPUtils_argumentCountMatches(
    const std::vector<Argument> expected_arguments,
    const std::vector<std::string> argument_types)
{
  return argument_types.size() == expected_arguments.size() ||
    (expected_arguments.size() && expected_arguments.back().is_variadic &&
     expected_arguments.size() < argument_types.size());
}

std::string THPUtils_formattedTupleDesc(
    const std::vector<Argument> expected_arguments,
    const std::vector<std::string> argument_types)
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

  std::string result = "(";
  bool is_variadic = expected_arguments.size() &&
    expected_arguments.back().is_variadic;
  for (size_t i = 0; i < argument_types.size(); i++) {
    bool is_matching = false;
    if (i < expected_arguments.size()) {
      if (expected_arguments[i].type == "float") {
          is_matching = argument_types[i] == "float" ||
                        argument_types[i] == "int" ||
                        argument_types[i] == "long";
      } else if (expected_arguments[i].type == "int") {
          is_matching = argument_types[i] == "int" ||
                        argument_types[i] == "long";
      } else {
          is_matching = expected_arguments[i].type == argument_types[i];
      }
      if (expected_arguments[i].is_nullable && argument_types[i] == "NoneType")
          is_matching = true;
    } else if (is_variadic) {
      is_matching = argument_types[i] == expected_arguments.back().type;
    }
    if (is_matching)
      result += green;
    else
      result += red;
    result += argument_types[i];
    if (is_matching)
        result += reset_green;
    else
        result += reset_red;
    result += ", ";
  }
  result.erase(result.length()-2);
  result += ")";
  return result;
}

std::string THPUtils_tupleDesc(const std::vector<std::string> argument_types)
{
  std::string result = "(";
  for (auto &type: argument_types)
    result += type + ", ";
  result.erase(result.length()-2);
  result += ")";
  return result;
}

void THPUtils_invalidArguments(PyObject *given_args,
        const char *function_name, size_t num_options, ...) {
  std::vector<std::string> option_strings;
  std::vector<std::string> argument_types;
  std::string error_msg;
  error_msg.reserve(2000);
  error_msg += function_name;
  error_msg += " recieved an invalid combination of argument types - got ";
  va_list option_list;
  va_start(option_list, num_options);
  for (size_t i = 0; i < num_options; i++)
    option_strings.push_back(va_arg(option_list, const char*));
  va_end(option_list);

  // TODO: assert that args is a tuple?
  Py_ssize_t num_args = PyTuple_Size(given_args);
  for (int i = 0; i < num_args; i++) {
    PyObject *arg = PyTuple_GET_ITEM(given_args, i);
    argument_types.push_back(_THPUtils_typename(arg));
  }

  if (num_options == 1) {
    auto expected_arguments = THPUtils_parseOption(option_strings[0]);
    if (THPUtils_argumentCountMatches(expected_arguments, argument_types)) {
      error_msg += THPUtils_formattedTupleDesc(expected_arguments,
          argument_types);
    } else {
      error_msg += THPUtils_tupleDesc(argument_types);
    }
    error_msg += ", but expected ";
    error_msg += option_strings[0];
  } else {
    error_msg += THPUtils_tupleDesc(argument_types);
    error_msg += ", but expected one of:\n";
    for (auto &option: option_strings) {
      error_msg += " * ";
      error_msg += option;
      error_msg += "\n";
      auto expected_args = THPUtils_parseOption(option);
      if (THPUtils_argumentCountMatches(expected_args, argument_types)) {
        error_msg += "      didn't match because some of the arguments have invalid types: ";
        error_msg += THPUtils_formattedTupleDesc(expected_args, argument_types);
        error_msg += "\n";
      }
    }
  }

  PyErr_SetString(PyExc_ValueError, error_msg.c_str());
}



bool THPUtils_parseSlice(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength)
{
  Py_ssize_t start, stop, step, slicelength;
  if (PySlice_GetIndicesEx(
// https://bugsfiles.kde.org/attachment.cgi?id=61186
#if PY_VERSION_HEX >= 0x03020000
         slice,
#else
         (PySliceObject *)slice,
#endif
         len, &start, &stop, &step, &slicelength) < 0) {
    return false;
  }
  if (step != 1) {
    THPUtils_setError("Trying to slice with a step of %ld, but only a step of "
        "1 is supported", (long)step);
    return false;
  }
  *ostart = start;
  *ostop = stop;
  if(oslicelength)
    *oslicelength = slicelength;
  return true;
}

template<>
void THPPointer<THPGenerator>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template<>
void THPPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<THPGenerator>;
template class THPPointer<PyObject>;
