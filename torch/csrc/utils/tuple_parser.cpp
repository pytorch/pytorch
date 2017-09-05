#include "tuple_parser.h"

#include <string>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/autograd/python_variable.h"
#include "python_strings.h"
#include "python_numbers.h"

namespace torch {

TupleParser::TupleParser(PyObject* args, int num_args) : args(args), idx(0) {
   int size = PyTuple_GET_SIZE(args);
   if (num_args >= 0 && size != num_args) {
     std::string msg("missing required arguments (expected ");
     msg += std::to_string(num_args) + " got " + std::to_string(size) + ")";
     throw std::runtime_error(msg);
   }
 }

auto TupleParser::parse(bool& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  if (!PyBool_Check(obj)) {
    throw invalid_type("bool", param_name);
  }
  x = (obj == Py_True);
}

auto TupleParser::parse(int& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  if (!THPUtils_checkLong(obj)) {
    throw invalid_type("int", param_name);
  }
  x = THPUtils_unpackLong(obj);
}

auto TupleParser::parse(double& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  if (!THPUtils_checkDouble(obj)) {
    throw invalid_type("float", param_name);
  }
  x = THPUtils_unpackDouble(obj);
}

auto TupleParser::parse(at::Tensor& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  x = torch::createTensor(obj);
}

auto TupleParser::parse(std::vector<int>& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  if (!PyTuple_Check(obj)) {
    throw invalid_type("tuple of int", param_name);
  }
  int size = PyTuple_GET_SIZE(obj);
  x.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(obj, i);
    if (!THPUtils_checkLong(item)) {
      throw invalid_type("tuple of int", param_name);
    }
    x[i] = THPUtils_unpackLong(item);
  }
}

auto TupleParser::parse(std::string& x, const std::string& param_name) -> void {
  PyObject* obj = next_arg();
  if (!THPUtils_checkString(obj)) {
    throw invalid_type("bytes/str", param_name);
  }
  x = THPUtils_unpackString(obj);
}

auto TupleParser::next_arg() -> PyObject* {
  if (idx >= PyTuple_GET_SIZE(args)) {
    throw std::runtime_error("out of range");
  }
  return PyTuple_GET_ITEM(args, idx++);
}

auto TupleParser::invalid_type(const std::string& expected, const std::string& param_name) -> std::runtime_error {
  std::string msg("argument ");
  msg += std::to_string(idx - 1);
  msg += " (";
  msg += param_name;
  msg += ") ";
  msg += "must be ";
  msg += expected;

  PyObject* obj = PyTuple_GET_ITEM(args, idx -1);
  if (PyTuple_Check(obj)){
    msg += " but got tuple of (";
    int size = PyTuple_GET_SIZE(obj);
    for (int i = 0; i < size; ++i) {
      msg += Py_TYPE(PyTuple_GET_ITEM(obj, i))->tp_name;
      if (i != size - 1){
        msg += ", ";
      }
    }
    msg += ")";
  }
  else{
    msg += ", not ";
    msg += Py_TYPE(PyTuple_GET_ITEM(args, idx - 1))->tp_name;
  }
  return std::runtime_error(msg);
}

} // namespace torch
