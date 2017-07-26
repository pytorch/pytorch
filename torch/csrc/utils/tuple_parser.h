#pragma once

#include <Python.h>
#include <memory>
#include <vector>
#include <string>
#include <ATen/ATen.h>

#include "python_numbers.h"

#ifdef WITH_C2ISL
#include "tvm/node.h"
#endif

namespace torch {

struct TupleParser {
  TupleParser(PyObject* args, int num_args=-1);

  void parse(bool& x, const std::string& param_name);
  void parse(int& x, const std::string& param_name);
  void parse(double& x, const std::string& param_name);
  void parse(at::Tensor& x, const std::string& param_name);
  void parse(std::vector<int>& x, const std::string& param_name);
  void parse(std::string& x, const std::string& param_name);

#ifdef WITH_C2ISL
  // The input is interpreted as a list of void* pointers, which is actually
  // a std::shared_ptr<Node>, which can be passed as an argument to T
  // to construct a NodeRef.
  template <typename T>
  auto parseNodeRefs(std::vector<T>& x, const std::string& param_name) -> void {
    PyObject* obj = next_arg();
    if (!PyTuple_Check(obj)) {
      throw invalid_type("tuple of long (void*)", param_name);
    }
    int size = PyTuple_GET_SIZE(obj);
    x.resize(size);
    for (int i = 0; i < size; ++i) {
      PyObject* item = PyTuple_GET_ITEM(obj, i);
      if (!THPUtils_checkLong_(item)) {
        throw invalid_type("tuple of long (void*)", param_name);
      }
      auto p = static_cast<std::shared_ptr<tvm::Node>*>(PyLong_AsVoidPtr(item));
      x[i] = T(*p);
    }
  }
#endif

protected:
  std::runtime_error invalid_type(const std::string& expected, const std::string& param_name);
  PyObject* next_arg();

private:
  PyObject* args;
  int idx;
};

} // namespace torch
