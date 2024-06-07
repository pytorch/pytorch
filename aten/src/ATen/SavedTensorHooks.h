#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <c10/util/python_stub.h>
#include <stack>
#include <string>

#include <utility>

namespace at {

namespace impl {

struct TORCH_API SavedTensorDefaultHooksTLS {
  // PyObject is defined in c10/util/python_stub.h
  std::stack<std::pair<PyObject*, PyObject*>> stack;

  // See NOTE: [Disabling SavedTensorDefaultHooks] for context
  // NOTE: [disabled_error_message invariant]
  // disabled_error_message is nullopt IFF Saved Tensor hooks is enabled
  // We did this for efficiency (so we didn't have to keep a separate bool
  // around)
  std::optional<std::string> disabled_error_message;

  // See NOTE: [Deferring tensor pack/unpack hooks until runtime]
  bool is_tracing = false;
};

} // namespace impl

struct TORCH_API SavedTensorDefaultHooks {
  static void push_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  static std::pair<PyObject*, PyObject*> pop_hooks();
  static std::pair<PyObject*, PyObject*> get_hooks();
  static void lazy_initialize();

  static const impl::SavedTensorDefaultHooksTLS& get_tls_state();
  static void set_tls_state(const impl::SavedTensorDefaultHooksTLS& tls);

  // NOTE: [Disabling SavedTensorDefaultHooks]
  // A developer of a PyTorch feature may choose to disable SavedTensorDefault
  // hooks, especially if their feature does not work with it. If they are
  // disabled, then the following will raise an error:
  // - Attempting to push_hooks
  // - calling disable(message) with a non-zero stack (hooks) size
  static void disable(const std::string& error_message);
  static void enable();
  static bool is_enabled();
  static const std::optional<std::string>& get_disabled_error_message();

  // NOTE: [Deferring tensor pack/unpack hooks until runtime]
  // To preserve eager semantics of pack/unpack hooks firing only once per saved
  // variable, Dynamo/AOTAutograd need to defer hook firing until runtime. Using
  // disable() would loud error at trace time, and pushing a no-op hook would
  // fail when the traced code is wrapped in a disable_saved_tensors_hooks ctx.
  // To do so, we disable these hooks during tracing. See
  // https://github.com/pytorch/pytorch/issues/113263.
  static bool set_tracing(bool is_tracing);
};

} // namespace at
