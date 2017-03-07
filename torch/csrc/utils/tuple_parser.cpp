#include "tuple_parser.h"

#include <string>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/autograd/python_variable.h"
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

auto TupleParser::parse(bool& x) -> void {
  PyObject* obj = next_arg();
  if (!PyBool_Check(obj)) {
    throw invalid_type("bool");
  }
  x = (obj == Py_True);
}

auto TupleParser::parse(int& x) -> void {
  PyObject* obj = next_arg();
  if (!THPUtils_checkLong(obj)) {
    throw invalid_type("int");
  }
  x = THPUtils_unpackLong(obj);
}

auto TupleParser::parse(double& x) -> void {
  PyObject* obj = next_arg();
  if (!THPUtils_checkDouble(obj)) {
    throw invalid_type("float");
  }
  x = THPUtils_unpackDouble(obj);
}

auto TupleParser::parse(std::unique_ptr<thpp::Tensor>& x) -> void {
  PyObject* obj = next_arg();
  x = torch::createTensor(obj);
}

auto TupleParser::parse(std::shared_ptr<thpp::Tensor>& x) -> void {
  PyObject* obj = next_arg();
  x.reset(torch::createTensor(obj)->clone_shallow());
}

auto TupleParser::parse(std::vector<int>& x) -> void {
  PyObject* obj = next_arg();
  if (!PyTuple_Check(obj)) {
    throw invalid_type("tuple of int");
  }
  int size = PyTuple_GET_SIZE(obj);
  x.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* item = PyTuple_GET_ITEM(obj, i);
    if (!THPUtils_checkLong(item)) {
      throw invalid_type("tuple of int");
    }
    x[i] = THPUtils_unpackLong(item);
  }
}

auto TupleParser::next_arg() -> PyObject* {
  if (idx >= PyTuple_GET_SIZE(args)) {
    throw std::runtime_error("out of range");
  }
  return PyTuple_GET_ITEM(args, idx++);
}

auto TupleParser::invalid_type(const char* expected) -> std::runtime_error {
  std::string msg("argument ");
  msg += std::to_string(idx - 1);
  msg += " must be ";
  msg += expected;
  msg += ", not ";
  msg += Py_TYPE(PyTuple_GET_ITEM(args, idx - 1))->tp_name;
  return std::runtime_error(msg);
}

} // namespace torch
