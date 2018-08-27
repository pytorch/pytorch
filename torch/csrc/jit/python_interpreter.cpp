#include "torch/csrc/python_headers.h"
#include "torch/csrc/jit/interpreter.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"

#include "torch/csrc/variable_tensor_functions.h"

#include <typeinfo>

#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/utils/auto_gil.h"

namespace py = pybind11;

namespace torch { namespace jit {

namespace {

Operation createPythonOperation(Node* op_) {
  PythonOp* op = static_cast<PythonOp*>(op_);
  py::function func = py::reinterpret_borrow<py::function>(py::handle(op->pyobj.get()));
  size_t num_inputs = 0;
  for(auto arg_type : op->cconv) {
    if(arg_type == 't')
      num_inputs++;
  }
  return [=](Stack & stack) {
    AutoGIL gil;
    py::tuple py_inputs(op->cconv.size());
    size_t i = 0;
    size_t next_scalar = 0;
    size_t next_tensor = 0;
    for (auto arg_type : op->cconv) {
      if (arg_type == 's') {
        py_inputs[i] = py::reinterpret_borrow<py::object>(
            op->scalar_args[next_scalar++].get());
      } else if (arg_type == 't') {
        auto var = std::move(peek(stack, next_tensor, num_inputs)).toTensor();
        py_inputs[i] =
            py::reinterpret_steal<py::object>(THPVariable_Wrap(var));
        next_tensor++;
      }
      i++;
    }
    drop(stack, num_inputs);
    py::object py_outputs(func(*py_inputs));

    auto num_outputs = op->outputs().size();
    auto addOutput = [&](py::handle entry) {
      if (!THPVariable_Check(entry.ptr())) {
        throw std::runtime_error(
            "Function application returned a non-Variable output");
      }
      THPVariable* var = (THPVariable*)entry.ptr();
      auto cdata = var->cdata;
      stack.push_back(std::move(cdata));
    };

    if (!PyTuple_Check(py_outputs.ptr())) {
      if (num_outputs != 1) {
        throw std::runtime_error(
            "Function.apply returned the wrong number of outputs.");
      }
      addOutput(py_outputs);
    } else {
      auto output_tuple = py::tuple(py_outputs);
      if (output_tuple.size() != num_outputs) {
        throw std::runtime_error(
            "Function application returned the wrong number of outputs.");
      }
      for (py::handle entry : py::tuple(py_outputs)) {
        addOutput(entry);
      }
    }
    return 0;
  };
}


RegisterOperators reg({
  Operator(prim::PythonOp, createPythonOperation)
});

}}} // torch::jit::anon
