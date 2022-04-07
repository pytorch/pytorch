#include <Python.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>
#include <torch/csrc/deploy/interpreter/adapter.h>

using at::IValue;
using torch::deploy::Obj;

namespace torch {
namespace deploy {

inline py::object toPyObj(at::IValue value){
    return torch::jit::toPyObject(value);
}

py::object toPyObj(IValue value){
    return torch::jit::toPyObject(value);
}

inline IValue fromPyObj(py::handle obj){
    return torch::jit::toTypeInferredIValue(obj);
}

} // namespace deploy
} // namespace torch
