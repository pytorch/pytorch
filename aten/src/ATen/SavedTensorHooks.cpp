#include <ATen/SavedTensorHooks.h>
#include <c10/util/Exception.h>
#include <stack>

namespace at {

namespace {
  // PyObject is defined in c10/util/python_stub.h
  thread_local std::stack<std::pair<PyObject*, PyObject*>> stack;

  // This flag is set to true the first time default hooks are registered
  // and left at true for the rest of the execution.
  // It's an optimization so that users who never use default hooks don't need to
  // read the thread_local variables pack_hook_ and unpack_hook_.
  static bool is_enabled(false);
}

void SavedTensorDefaultHooks::enable() {
  is_enabled = true;
}

void SavedTensorDefaultHooks::push_hooks(PyObject* pack_hook, PyObject* unpack_hook) {
  // Reference counting is handled by the caller of `push_hooks`
  TORCH_INTERNAL_ASSERT(is_enabled);
  TORCH_INTERNAL_ASSERT(pack_hook != nullptr && unpack_hook != nullptr);
  stack.push(std::make_pair(pack_hook, unpack_hook));
}

void SavedTensorDefaultHooks::pop_hooks() {
  // Reference counting is handled by the caller of `pop_hooks`
  TORCH_INTERNAL_ASSERT(is_enabled && !stack.empty());
  stack.pop();
}

std::pair<PyObject*, PyObject*> SavedTensorDefaultHooks::get_hooks() {
  if (!is_enabled || stack.empty()) {
    return std::make_pair(nullptr, nullptr);
  }
  return stack.top();
}

std::stack<std::pair<PyObject*, PyObject*>> SavedTensorDefaultHooks::get_stack() {
  return stack;
}

void SavedTensorDefaultHooks::set_stack(std::stack<std::pair<PyObject*, PyObject*>> stack_) {
  stack = stack_;
}

}
