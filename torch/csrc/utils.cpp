#include <Python.h>
#include <stdarg.h>
#include <string>
#include <vector>
#include "THP.h"

#include "generic/utils.cpp"
#include <TH/THGenerateAllTypes.h>

bool THPUtils_checkLong(PyObject *index) {
    return PyLong_Check(index) || PyInt_Check(index);
}

long THPUtils_unpackLong(PyObject *index) {
  if (PyLong_Check(index)) {
    return PyLong_AsLong(index);
  } else if (PyInt_Check(index)) {
    return PyInt_AsLong(index);
  } else {
    throw std::exception();
  }
}

int THPUtils_getLong(PyObject *index, long *result) {
  try {
    *result = THPUtils_unpackLong(index);
  } catch(...) {
    char err_string[512];
    snprintf (err_string, 512, "%s %s",
      "getLong expected int or long, but got type: ",
      index->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return 0;
  }
  return 1;
}

int THPUtils_getCallable(PyObject *arg, PyObject **result) {
  if (!PyCallable_Check(arg))
    return 0;
  *result = arg;
  return 1;
}

THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first) {
// TODO: error messages
  long value;

  Py_ssize_t length = PyTuple_Size(args);
  if (length < ignore_first+1)
    throw std::logic_error("Provided too few arguments");

  // Maybe there's a LongStorage
  PyObject *first_arg = PyTuple_GET_ITEM(args, ignore_first);
  if (length == ignore_first+1 && THPLongStorage_IsSubclass(first_arg)) {
    THPLongStorage *storage = (THPLongStorage*)first_arg;
    THLongStorage_retain(storage->cdata);
    return storage->cdata;
  }

  // If not, let's try to parse the numbers
  THLongStoragePtr result = THLongStorage_newWithSize(length-ignore_first);
  for (Py_ssize_t i = ignore_first; i < length; ++i) {
    PyObject *arg = PyTuple_GET_ITEM(args, i);
    if (!THPUtils_getLong(arg, &value))
        throw std::invalid_argument("Expected a numeric argument, but got " + std::string(Py_TYPE(arg)->tp_name));
    result->data[i-ignore_first] = value;
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


void THPUtils_invalidArguments(PyObject *given_args, const char *function_name, size_t num_options, ...) {
  std::vector<std::string> option_strings;
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
  if (num_args == 0) {
    error_msg += "no arguments";
  } else {
    error_msg += "(";
    for (int i = 0; i < num_args; i++) {
      PyObject *arg = PyTuple_GET_ITEM(given_args, i);
      if (i > 0)
        error_msg += ", ";
      std::string type_name = Py_TYPE(arg)->tp_name;
      if (type_name.find("Storage") != std::string::npos ||
              type_name.find("Tensor") != std::string::npos) {
        PyObject *module_name = PyObject_GetAttrString(arg, "__module__");
#if PY_MAJOR_VERSION == 2
        if (module_name && PyString_Check(module_name)) {
          error_msg += PyString_AS_STRING(module_name);
          error_msg += ".";
        }
#else
        if (module_name && PyUnicode_Check(module_name)) {
          THPObjectPtr module_name_bytes = PyUnicode_AsASCIIString(module_name);
          if (module_name_bytes.get()) {
            error_msg += PyBytes_AS_STRING(module_name_bytes.get());
            error_msg += ".";
          }
        }
#endif
      }
      error_msg += type_name;
    }
    error_msg += ")";
  }
  error_msg += ", but expected";

  if (num_options == 1) {
    error_msg += " ";
    error_msg += option_strings[0];
  } else {
    error_msg += " one of:\n";
    for (auto &option: option_strings) {
      error_msg += " * ";
      error_msg += option;
      error_msg += "\n";
    }
  }

  PyErr_SetString(PyExc_ValueError, error_msg.c_str());
}

PyObject * THPUtils_bytesFromString(const char *b)
{
#if PY_MAJOR_VERSION == 2
    return PyString_FromString(b);
#else
    return PyBytes_FromString(b);
#endif
}

bool THPUtils_checkBytes(PyObject *obj)
{
#if PY_MAJOR_VERSION == 2
    return PyString_Check(obj);
#else
    return PyBytes_Check(obj);
#endif
}

const char * THPUtils_bytesAsString(PyObject *bytes)
{
#if PY_MAJOR_VERSION == 2
    return PyString_AS_STRING(bytes);
#else
    return PyBytes_AS_STRING(bytes);
#endif
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
