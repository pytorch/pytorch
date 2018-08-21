#pragma once

#include <pybind11/pybind11.h>
#include "caffe2/core/workspace.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

namespace python_detail {

struct Func {
  py::object py_func;
  bool needs_workspace;
};

// Python Op implementations.
using FuncRegistry = std::unordered_map<std::string, Func>;

FuncRegistry& gRegistry();

const Func& getOpFunc(const std::string& token);

const Func& getGradientFunc(const std::string& token);

py::object fetchBlob(Workspace* ws, const std::string& name);

} // namespace python_detail

} // namespace python
} // namespace caffe2
