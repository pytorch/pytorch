#include "torch/csrc/python_headers.h"
#include <stdarg.h>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include "THP.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/invalid_arguments.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/DynamicTypes.h"

#include "generic/utils.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/utils.cpp"
#include <TH/THGenerateHalfType.h>

#ifdef WITH_CUDA
#include "torch/csrc/cuda/THCP.h"
#endif

int THPUtils_getCallable(PyObject *arg, PyObject **result) {
  if (!PyCallable_Check(arg))
    return 0;
  *result = arg;
  return 1;
}

THLongStoragePtr THPUtils_unpackSize(PyObject *arg) {
  THLongStoragePtr result;
  if (!THPUtils_tryUnpackLongs(arg, result)) {
    std::string msg = "THPUtils_unpackSize() expects a torch.Size (got '";
    msg += Py_TYPE(arg)->tp_name;
    msg += "')";
    throw std::runtime_error(msg);
  }
  return result;
}

bool THPUtils_tryUnpackLongs(PyObject *arg, THLongStoragePtr& result) {
  bool tuple = PyTuple_Check(arg);
  bool list = PyList_Check(arg);
  if (tuple || list) {
    int nDim = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
    THLongStoragePtr storage(THLongStorage_newWithSize(nDim));
    for (int i = 0; i != nDim; ++i) {
      PyObject* item = tuple ? PyTuple_GET_ITEM(arg, i) : PyList_GET_ITEM(arg, i);
      if (!THPUtils_checkLong(item)) {
        return false;
      }
      THLongStorage_set(storage, i, THPUtils_unpackLong(item));
    }
    result  = std::move(storage);
    return true;
  }
  return false;
}

std::vector<int64_t> THPUtils_unpackLongs(PyObject *arg) {
  bool tuple = PyTuple_Check(arg);
  bool list = PyList_Check(arg);
  if (tuple || list) {
    int nDim = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
    std::vector<int64_t> sizes(nDim);
    for (int i = 0; i != nDim; ++i) {
      PyObject* item = tuple ? PyTuple_GET_ITEM(arg, i) : PyList_GET_ITEM(arg, i);
      if (!THPUtils_checkLong(item)) {
        std::ostringstream oss;
        oss << "expected int at position " << i << ", but got: " << THPUtils_typename(item);
        throw std::runtime_error(oss.str());
      }
      sizes[i] = THPUtils_unpackLong(item);
    }
    return sizes;
  }
  throw std::runtime_error("Expected tuple or list");
}

bool THPUtils_tryUnpackLongVarArgs(PyObject *args, int ignore_first, THLongStoragePtr& result) {
  Py_ssize_t length = PyTuple_Size(args) - ignore_first;
  if (length < 1) {
    return false;
  }

  PyObject *first_arg = PyTuple_GET_ITEM(args, ignore_first);
  if (length == 1 && THPUtils_tryUnpackLongs(first_arg, result)) {
    return true;
  }

  // Try to parse the numbers
  result = THLongStorage_newWithSize(length);
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject *arg = PyTuple_GET_ITEM(args, i + ignore_first);
    if (!THPUtils_checkLong(arg)) {
      return false;
    }
    THLongStorage_set(result, i, THPUtils_unpackLong(arg));
  }
  return true;
}

bool THPUtils_checkIntTuple(PyObject *arg)
{
  if (!PyTuple_Check(arg)) {
    return false;
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(arg); ++i) {
    if (!THPUtils_checkLong(PyTuple_GET_ITEM(arg, i))) {
      return false;
    }
  }
  return true;
}

std::vector<int> THPUtils_unpackIntTuple(PyObject *arg)
{
  if (!THPUtils_checkIntTuple(arg)) {
    throw std::runtime_error("Couldn't unpack int tuple");
  }
  std::vector<int> values(PyTuple_GET_SIZE(arg));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(arg); ++i) {
    values[i] = (int)THPUtils_unpackLong(PyTuple_GET_ITEM(arg, i));
  }
  return values;
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

void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
  if (!vector.empty()) {
    // remove NULL terminator
    vector.pop_back();
  }
  while (1) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

static const char* classOrTypename(PyObject* obj) {
  if (PyType_Check(obj)) {
    return ((PyTypeObject*)obj)->tp_name;
  }
  return Py_TYPE(obj)->tp_name;
}

PyObject * THPUtils_dispatchStateless(
    PyObject *tensor, const char *name, PyObject *args, PyObject *kwargs)
{
  THPObjectPtr methods(PyObject_GetAttrString(tensor, THP_STATELESS_ATTRIBUTE_NAME));
  if (!methods) {
    return PyErr_Format(
        PyExc_TypeError,
        "Type %s doesn't implement stateless methods",
        classOrTypename(tensor));
  }
  THPObjectPtr method(PyObject_GetAttrString(methods, name));
  if (!method) {
    return PyErr_Format(
        PyExc_TypeError,
        "Type %s doesn't implement stateless method %s",
        classOrTypename(tensor),
        name);
  }
  return PyObject_Call(method.get(), args, kwargs);
}

void THPUtils_invalidArguments(PyObject *given_args, PyObject *given_kwargs,
        const char *function_name, size_t num_options, ...) {
  std::vector<std::string> option_strings;
  va_list option_list;
  va_start(option_list, num_options);
  for (size_t i = 0; i < num_options; i++)
    option_strings.push_back(va_arg(option_list, const char*));
  va_end(option_list);

  PyErr_SetString(PyExc_TypeError, torch::format_invalid_args(
      given_args, given_kwargs, function_name, option_strings).c_str());
}

template<>
void THPPointer<THPGenerator>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<THPGenerator>;

static bool backCompatBroadcastWarn = false;

void setBackCompatBroadcastWarn(bool warn) {
  backCompatBroadcastWarn = warn;
}

bool getBackCompatBroadcastWarn() {
  return backCompatBroadcastWarn;
}

static bool backCompatKeepdimWarn = false;

void setBackCompatKeepdimWarn(bool warn) {
  backCompatKeepdimWarn = warn;
}

bool getBackCompatKeepdimWarn() {
  return backCompatKeepdimWarn;
}

bool maybeThrowBackCompatKeepdimWarn(char *func) {
  if(getBackCompatKeepdimWarn()) {
     std::ostringstream ss;
     ss << "backwards compatibility: call to \"" << func
        << "\" uses default value for keepdim which has changed default to False.  Consider passing as kwarg.",
    PyErr_WarnEx(PyExc_UserWarning, ss.str().c_str(), 1);
  }
  return true;
}

#ifdef WITH_CUDA
std::vector <THCStream*> THPUtils_PySequence_to_THCStreamList(PyObject *obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error("Expected a sequence in THPUtils_PySequence_to_THCStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, NULL));
  if (seq.get() == NULL) {
    throw std::runtime_error("expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<THCStream*> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject *stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THCPStreamClass)) {
      streams.push_back( ((THCPStream *)stream)->cdata);
    } else if (stream == Py_None) {
      streams.push_back(NULL);
    } else {
      std::runtime_error("Unknown data type found in stream list. Need THCStream or None");
    }
  }
  return streams;
}
#endif
