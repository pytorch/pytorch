#include "torch/csrc/python_headers.h"
#include "torch/csrc/jit/interpreter.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/pybind_utils.h"

#include "torch/csrc/variable_tensor_functions.h"

#include <typeinfo>

#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/Exceptions.h"

namespace py = pybind11;

namespace torch { namespace jit {

namespace {

// Note: const_cast is used twice below to acquire a handle to a pyobject.
Operation createPythonOperation(const Node* op_) {
  AutoGIL gil;
  const PythonOp* op = static_cast<const PythonOp*>(op_);
  const py::function func =
    py::reinterpret_borrow<const py::function>(py::handle(const_cast<PythonOp*>(op)->pyobj.get()));

  size_t num_inputs = 0;
  for(auto arg_type : op->cconv) {
    if(arg_type == 'd')
      num_inputs++;
  }

  JIT_ASSERT(op->outputs().size() == 1);

  return [=](Stack & stack) {
    AutoGIL gil;
    py::tuple py_inputs(op->cconv.size());
    size_t i = 0;
    size_t next_scalar = 0;
    size_t next_tensor = 0;
    for (auto arg_type : op->cconv) {
      if (arg_type == 'c') {
        py_inputs[i] =
          py::reinterpret_borrow<const py::object>(const_cast<PythonOp*>(op)->scalar_args[next_scalar++].get());
      } else if (arg_type == 'd') {
        py_inputs[i] = toPyObject(std::move(peek(stack, next_tensor, num_inputs)));
        next_tensor++;
      }
      i++;
    }
    drop(stack, num_inputs);
    try {
      py::object py_output(func(*py_inputs));
      stack.push_back(returnToIValue(op->output()->type(), py_output));
    } catch (py::error_already_set & e) {
      throw std::runtime_error(e.what());
    }
    return 0;
  };
}


RegisterOperators reg({
  Operator(prim::PythonOp, createPythonOperation)
});

}}} // torch::jit::anon
