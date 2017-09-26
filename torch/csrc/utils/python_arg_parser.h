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
#include <vector>
#include <ATen/ATen.h>

#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/utils/python_numbers.h"

namespace torch {

enum class ParameterType {
  TENSOR, SCALAR, INT64, DOUBLE, TENSOR_LIST, INT_LIST, GENERATOR,
  BOOL, STORAGE
};

struct FunctionParameter;
struct FunctionSignature;
struct PythonArgs;

struct type_exception : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct PythonArgParser {
  explicit PythonArgParser(std::vector<std::string> fmts);

  PythonArgs parse(PyObject* args, PyObject* kwargs, PyObject* dst[]);

private:
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
  inline std::vector<int64_t> intlist(int i);
  inline int64_t toInt64(int i);
  inline double toDouble(int i);
  inline bool toBool(int i);
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
  bool deprecated;
};

struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(PyObject* obj);
  void set_default_str(const std::string& str);
  std::string type_name() const;

  ParameterType type_;
  bool optional;
  bool keyword_only;
  std::string name;
  THPObjectPtr python_name;
  at::Scalar default_scalar;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
  };
};

inline at::Tensor PythonArgs::tensor(int i) {
  if (!args[i]) return at::Tensor();
  return reinterpret_cast<THPVariable*>(args[i])->cdata;
}

inline at::Scalar PythonArgs::scalar(int i) {
  if (!args[i]) return signature.params[i].default_scalar;
  if (PyFloat_Check(args[i])) {
    return at::Scalar(THPUtils_unpackDouble(args[i]));
  }
  return at::Scalar(THPUtils_unpackLong(args[i]));
}

inline std::vector<int64_t> PythonArgs::intlist(int i) {
  if (!args[i]) return std::vector<int64_t>();
  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<int64_t> res(size);
  for (int idx = 0; idx < size; idx++) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    res[idx] = THPUtils_unpackLong(obj);
  }
  return res;
}

inline int64_t PythonArgs::toInt64(int i) {
  if (!args[i]) return signature.params[i].default_int;
  return THPUtils_unpackLong(args[i]);
}

inline double PythonArgs::toDouble(int i) {
  if (!args[i]) return signature.params[i].default_double;
  return THPUtils_unpackDouble(args[i]);
}

inline bool PythonArgs::toBool(int i) {
  if (!args[i]) return signature.params[i].default_bool;
  return args[i] == Py_True;
}

} // namespace torch
