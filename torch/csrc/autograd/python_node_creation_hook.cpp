#include <torch/csrc/autograd/python_node_creation_hook.h>

#include <ATen/NodeCreationHooks.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::autograd {

PyNodeCreationHook::PyNodeCreationHook(c10::SafePyObject hook)
    : hook_(std::move(hook)) {}

// We don't use pybind for the call to avoid
// https://github.com/pytorch/pytorch/issues/34172
void PyNodeCreationHook::operator()(const c10::intrusive_ptr<Node>& node) {
  pybind11::gil_scoped_acquire gil;
  THPObjectPtr py_node(functionToPyObject(node));
  if (!py_node) {
    throw python_error();
  }
  THPObjectPtr result(PyObject_CallFunctionObjArgs(
      hook_.ptr(getPyInterpreter()), py_node.get(), nullptr));
  if (!result) {
    throw python_error();
  }
  // Reserve non-None returns for future semantics.
  TORCH_CHECK(
      result.get() == Py_None,
      "node creation hook must return None, got ",
      Py_TYPE(result.get())->tp_name);
}

std::vector<std::unique_ptr<NodeCreationHook>> PyDefaultNodeCreationHooks::
    get_hooks() {
  const auto& state = at::impl::NodeCreationHooks::get_tls_state();
  std::vector<std::unique_ptr<NodeCreationHook>> hooks;
  hooks.reserve(state.stack.size());
  for (const auto& hook : state.stack) {
    hooks.push_back(std::make_unique<PyNodeCreationHook>(hook));
  }
  return hooks;
}

} // namespace torch::autograd
