#pragma once

// Parse arguments to Python functions implemented in C++
// This is similar to PyArg_ParseTupleAndKeywords(), but specifically handles
// the types relevant to PyTorch and distinguishes between overloaded function
// signatures.
//
// Example:
//
//   static PythonArgParser parser({
//     "norm(Scalar p, int64_t dim, bool keepdim=False)",
//     "norm(Scalar p=2)",
//   });
//   PyObject* parsed_args[3];
//   auto r = parser.parse(args, kwargs, parsed_args);
//   if (r.idx == 0) {
//     norm(r.scalar(0), r.int64(1), r.bool(0));
//   } else {
//     norm(r.scalar(0));
//   }


#include <Python.h>
#include <string>
#include <sstream>
#include <vector>
#include <ATen/ATen.h>

#include "torch/csrc/THP.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/DynamicTypes.h"

namespace torch {

enum class ParameterType {
  TENSOR, SCALAR, INT64, DOUBLE, TENSOR_LIST, INT_LIST, GENERATOR,
  BOOL, STORAGE, PYOBJECT
};

struct FunctionParameter;
struct FunctionSignature;
struct PythonArgs;

struct type_exception : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]]
void type_error(const char *format, ...);

struct PythonArgParser {
  explicit PythonArgParser(std::vector<std::string> fmts);

  PythonArgs parse(PyObject* args, PyObject* kwargs, PyObject* dst[]);

private:
  [[noreturn]]
  void print_error(PyObject* args, PyObject* kwargs, PyObject* dst[]);

  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  ssize_t max_args;
};

struct PythonArgs {
  PythonArgs(int idx, const FunctionSignature& signature, PyObject** args)
    : idx(idx)
    , signature(signature)
    , args(args) {}

  int idx;
  const FunctionSignature& signature;
  PyObject** args;

  inline at::Tensor tensor(int i);
  inline at::Scalar scalar(int i);
  inline at::Scalar scalarWithDefault(int i, at::Scalar default_scalar);
  inline std::vector<at::Tensor> tensorlist(int i);
  template<int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  inline std::vector<int64_t> intlistWithDefault(int i, std::vector<int64_t> default_intlist);
  inline at::Generator* generator(int i);
  inline std::unique_ptr<at::Storage> storage(int i);
  inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  inline int64_t toInt64WithDefault(int i, int64_t default_int);
  inline double toDouble(int i);
  inline double toDoubleWithDefault(int i, double default_double);
  inline bool toBool(int i);
  inline bool toBoolWithDefault(int i, bool default_bool);
  inline bool isNone(int i);
};

struct FunctionSignature {
  explicit FunctionSignature(const std::string& fmt);

  bool parse(PyObject* args, PyObject* kwargs, PyObject* dst[], bool raise_exception);
  std::string toString() const;

  std::string name;
  std::vector<FunctionParameter> params;
  ssize_t min_args;
  ssize_t max_args;
  ssize_t max_pos_args;
  bool hidden;
  bool deprecated;
};

struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(PyObject* obj);
  void set_default_str(const std::string& str);
  std::string type_name() const;

  ParameterType type_;
  bool optional;
  bool allow_none;
  bool keyword_only;
  int size;
  std::string name;
  // having this as a raw PyObject * will presumably leak it, but these are only held by static objects
  // anyway, and Py_Finalize can already be called when this is destructed.
  PyObject *python_name;
  at::Scalar default_scalar;
  std::vector<int64_t> default_intlist;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
  };
};

inline at::Tensor PythonArgs::tensor(int i) {
  if (!args[i]) return at::Tensor();
  if (!THPVariable_Check(args[i])) {
    // NB: Are you here because you passed None to a Variable method,
    // and you expected an undefined tensor to be returned?   Don't add
    // a test for Py_None here; instead, you need to mark the argument
    // as *allowing none*; you can do this by writing 'Tensor?' instead
    // of 'Tensor' in the ATen metadata.
    type_error("expected Variable as argument %d, but got %s", i, THPUtils_typename(args[i]));
  }
  return reinterpret_cast<THPVariable*>(args[i])->cdata;
}

inline at::Scalar PythonArgs::scalar(int i) {
  return scalarWithDefault(i, signature.params[i].default_scalar);
}

inline at::Scalar PythonArgs::scalarWithDefault(int i, at::Scalar default_scalar) {
  if (!args[i]) return default_scalar;
  if (PyFloat_Check(args[i])) {
    return at::Scalar(THPUtils_unpackDouble(args[i]));
  }
  return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(args[i])));
}

inline std::vector<at::Tensor> PythonArgs::tensorlist(int i) {
  if (!args[i]) return std::vector<at::Tensor>();
  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<at::Tensor> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    if (!THPVariable_Check(obj)) {
      type_error("expected Variable as element %d in argument %d, but got %s",
                 idx, i, THPUtils_typename(args[i]));
    }
    res[idx] = reinterpret_cast<THPVariable*>(obj)->cdata;
  }
  return res;
}

template<int N>
inline std::array<at::Tensor, N> PythonArgs::tensorlist_n(int i) {
  auto res = std::array<at::Tensor, N>();
  PyObject* arg = args[i];
  if (!arg) return res;
  auto tuple = PyTuple_Check(arg);
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  if (size != N) {
    type_error("expected tuple of %d elements but got %d", N, (int)size);
  }
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    if (!THPVariable_Check(obj)) {
      type_error("expected Variable as element %d in argument %d, but got %s",
                 idx, i, THPUtils_typename(args[i]));
    }
    res[idx] = reinterpret_cast<THPVariable*>(obj)->cdata;
  }
  return res;
}

inline std::vector<int64_t> PythonArgs::intlist(int i) {
  return intlistWithDefault(i, signature.params[i].default_intlist);
}

inline std::vector<int64_t> PythonArgs::intlistWithDefault(int i, std::vector<int64_t> default_intlist) {
  if (!args[i]) return default_intlist;
  PyObject* arg = args[i];
  auto size = signature.params[i].size;
  if (size > 0 && THPUtils_checkLong(arg)) {
    return std::vector<int64_t>(size, THPUtils_unpackLong(arg));
  }
  auto tuple = PyTuple_Check(arg);
  size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<int64_t> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    try {
      res[idx] = THPUtils_unpackLong(obj);
    } catch (std::runtime_error &e) {
      type_error("%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), Py_TYPE(obj)->tp_name, idx + 1);
    }
  }
  return res;
}

inline int64_t PythonArgs::toInt64(int i) {
  return toInt64WithDefault(i, signature.params[i].default_int);
}

inline int64_t PythonArgs::toInt64WithDefault(int i, int64_t default_int) {
  if (!args[i]) return default_int;
  return THPUtils_unpackLong(args[i]);
}

inline double PythonArgs::toDouble(int i) {
  return toDoubleWithDefault(i, signature.params[i].default_double);
}

inline double PythonArgs::toDoubleWithDefault(int i, double default_double) {
  if (!args[i]) return default_double;
  return THPUtils_unpackDouble(args[i]);
}

inline bool PythonArgs::toBool(int i) {
  return toBoolWithDefault(i, signature.params[i].default_bool);
}

inline bool PythonArgs::toBoolWithDefault(int i, bool default_bool) {
  if (!args[i]) return default_bool;
  return args[i] == Py_True;
}

inline bool PythonArgs::isNone(int i) {
  return args[i] == nullptr;
}

inline at::Generator* PythonArgs::generator(int i) {
  if (!args[i]) return nullptr;
  if (!THPGenerator_Check(args[i])) {
    type_error("expected Generator as argument %d, but got %s", i, THPUtils_typename(args[i]));
  }
  return reinterpret_cast<THPGenerator*>(args[i])->cdata;
}

inline std::unique_ptr<at::Storage> PythonArgs::storage(int i) {
  if (!args[i]) return nullptr;
  return createStorage(args[i]);
}

inline PyObject* PythonArgs::pyobject(int i) {
  if (!args[i]) return Py_None;
  return args[i];
}

} // namespace torch
