#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <typeinfo>

#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_engine.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {
namespace jit {

namespace {

// Note: const_cast is used twice below to acquire a handle to a pyobject.
Operation createPythonOperation(const Node* op_) {
  pybind11::gil_scoped_acquire gil;
  const ConcretePythonOp* op = static_cast<const ConcretePythonOp*>(op_);
  const py::function func = py::reinterpret_borrow<const py::function>(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      py::handle(const_cast<ConcretePythonOp*>(op)->pyobj.get()));

  size_t num_inputs = 0;
  for (auto arg_type : op->cconv) {
    if (arg_type == 'd')
      num_inputs++;
  }

  AT_ASSERT(op->outputs().size() == 1);

  return [=](Stack& stack) {
    pybind11::gil_scoped_acquire gil;
    py::tuple py_inputs(op->cconv.size());
    size_t i = 0;
    size_t next_scalar = 0;
    size_t next_tensor = 0;
    for (auto arg_type : op->cconv) {
      if (arg_type == 'c') {
        py_inputs[i] = py::reinterpret_borrow<const py::object>(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<ConcretePythonOp*>(op)
                ->scalar_args[next_scalar++]
                .get());
      } else if (arg_type == 'd') {
        py_inputs[i] =
            toPyObject(std::move(peek(stack, next_tensor, num_inputs)));
        next_tensor++;
      }
      i++;
    }
    drop(stack, num_inputs);
    try {
      py::object py_output(func(*py_inputs));
      stack.push_back(returnToIValue(op->output()->type(), py_output));
    } catch (py::error_already_set& e) {
      throw std::runtime_error(e.what());
    }
  };
}

c10::AliasAnalysisKind aliasAnalysisIsSpecialCase() {
  return AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}

RegisterOperators reg({Operator(
    prim::PythonOp,
    createPythonOperation,
    aliasAnalysisIsSpecialCase())});

} // namespace
} // namespace jit
} // namespace torch
