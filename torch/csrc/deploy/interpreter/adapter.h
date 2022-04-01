#include <Python.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/jit/python/pybind_utils.h>
// #include <torch/csrc/deploy/interpreter/adapter.h>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>

using at::IValue;
using torch::deploy::Obj;

namespace torch {
namespace deploy {

template<typename T>
py::object toPyObj(T);

template<typename T>
T fromPyObj(py::handle);

} // namespace deploy
} // namespace torch
