#pragma once

#include <c10/macros/Export.h>
#include <c10/util/python_stub.h>

#include <utility>

namespace at {

struct TORCH_API SavedTensorDefaultHooks {
  static void set_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  static std::pair<PyObject*, PyObject*> get_hooks();
  static void enable();
};

} // namespace at
