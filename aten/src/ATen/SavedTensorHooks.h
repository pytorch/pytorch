#pragma once

#include <c10/macros/Export.h>
#include <c10/util/python_stub.h>
#include <stack>

#include <utility>

namespace at {

struct TORCH_API SavedTensorDefaultHooks {
  static void push_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  static void pop_hooks();
  static std::pair<PyObject*, PyObject*> get_hooks();
  static void enable();
  static std::stack<std::pair<PyObject*, PyObject*>> get_stack();
  static void set_stack(std::stack<std::pair<PyObject*, PyObject*>>);
};

} // namespace at
