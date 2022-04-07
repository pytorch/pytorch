#include <Python.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/jit/python/pybind_utils.h>
// #include <torch/csrc/deploy/interpreter/interpreter_impl.h>

using at::IValue;
using torch::deploy::Obj;

namespace torch {
namespace deploy {

template<typename T>
inline py::object toPyObj(T value){
    return py::cast(value);
}

template<>
inline py::object toPyObj<IValue>(IValue value){
    return torch::jit::toPyObject(value);
}

template<typename T>
inline T fromPyObj(py::handle obj){
    return obj.cast<T>();
}

// inline template<typename T>
template<>
inline IValue fromPyObj<IValue>(py::handle obj){
    return torch::jit::toTypeInferredIValue(obj);
}


} // namespace deploy
} // namespace torch
