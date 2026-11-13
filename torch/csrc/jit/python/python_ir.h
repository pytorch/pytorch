#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::jit {

void initPythonIRBindings(PyObject* module);

// execute a Python function, used for Ops we can't optimize but that we want to
// optimize around
struct ConcretePythonOp : public PythonOp {
  static Symbol Kind;

  ConcretePythonOp(Graph* graph) : PythonOp(graph, ::c10::prim::PythonOp) {}
  ConcretePythonOp* init(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      pyobj_list&& scalar_args) {
    this->pyobj = std::move(pyobj);
    this->scalar_args = std::move(scalar_args);
    this->cconv = cconv;
    return this;
  }
  // The Python object which contains the implementation of this function.
  // This is either a class (non-legacy) or an object (legacy).  See
  // TraceInterpreterState for execution semantics.
  THPObjectPtr pyobj;
  // The calling convention for the Python function.
  // 'c' -- constant argument
  // 'd' -- dynamic argument
  std::string cconv;
  // Scalar arguments to the Python function.  Not necessarily passed to
  // the function in this order; see cconv for the correct order.
  std::vector<THPObjectPtr> scalar_args;

  std::string name() const override;
  void cloneFrom(Node* other_) override;
  Node* allocNewInstance(Graph* g) override {
    return new ConcretePythonOp(g);
  }
  // recover the autograd.Function instance, if this PythonOp's function
  // was originally SomeFunction.apply
  // used in ONNX for discovering symbolics
  std::optional<THPObjectPtr> autogradFunction() const override;
  void writeScalars(std::ostream& out) const override;
  void lint_python() const override;
};

} // namespace torch::jit
