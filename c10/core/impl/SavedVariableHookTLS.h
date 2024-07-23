#pragma once

#include <c10/macros/Export.h>
#include <c10/util/python_stub.h>

#include <vector>
#include <optional>
#include <string>


namespace c10::impl {

struct C10_API SavedTensorDefaultHooksTLS {
  // PyObject is defined in c10/util/python_stub.h
  std::vector<std::pair<PyObject*, PyObject*>> stack;

  // See NOTE: [Disabling SavedTensorDefaultHooks] for context
  // NOTE: [disabled_error_message invariant]
  // disabled_error_message is nullopt IFF Saved Tensor hooks is enabled
  // We did this for efficiency (so we didn't have to keep a separate bool
  // around)
  std::optional<std::string> disabled_error_message;

  // See NOTE: [Deferring tensor pack/unpack hooks until runtime]
  bool is_tracing = false;
};

} // namespace c10::impl
