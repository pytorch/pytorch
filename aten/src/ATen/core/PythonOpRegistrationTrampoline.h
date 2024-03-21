#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: this can probably live in c10


namespace at::impl {

class TORCH_API PythonOpRegistrationTrampoline final {
  static std::atomic<c10::impl::PyInterpreter*> interpreter_;

public:
  //  Returns true if you successfully registered yourself (that means
  //  you are in the hot seat for doing the operator registrations!)
  static bool registerInterpreter(c10::impl::PyInterpreter*);

  // Returns nullptr if no interpreter has been registered yet.
  static c10::impl::PyInterpreter* getInterpreter();
};

} // namespace at::impl
